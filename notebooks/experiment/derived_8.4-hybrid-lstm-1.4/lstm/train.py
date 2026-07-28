from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
EXP_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = EXP_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from .dataset import TARGET, build_datasets
from .model import BiLSTMAttn

TOP25_FEATURES = [
    "V_rollmin_LST_modis_kobs30", "C_lag_LST_modis_kobs30", "C_lag_LST_modis_kobs6",
    "V_rollmean_G_API_kobs14", "K_slope_sin", "D_sin_DOY", "V_ema_G_API_kobs14",
    "V_rollmin_G_API_kobs30", "G_rain_sum_3d", "V_ema_LST_modis_kobs7",
    "G_rain_sum_7d", "G_DSLR", "cos_year", "SMAP_x_year", "G_API",
    "V_ema_G_API_kobs7", "V_rollmax_G_API_kobs30", "C_lag_G_API_kobs1", "elev",
    "V_rollmean_G_API_kobs7", "V_rollstd_SMAP_sm_interp_kobs30", "V_ema_G_API_kobs30",
    "V_rollmean_s2_b11_kobs7", "A_d_E_SAR_diff_kobs14", "K_slope_cos",
]

SEQ_LEN = 30
TRAIN_STRIDE = 1
HIDDEN_SIZE = 80
NUM_LAYERS = 2
DROPOUT = 0.3
PROJ_SIZE = 56
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 2e-3
HUBER_DELTA = 0.05
MAX_EPOCHS = 300
PATIENCE = 60
GRAD_CLIP = 1.0
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 8
SEEDS = [42, 7, 123, 2024, 31337]

PCA_LEVELS = [
    ("pca95", 0.95),
    ("pca64", 64),
    ("pca32", 32),
    ("pca16", 16),
]


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_inf(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    return X


def fit_preprocessors(train_df: pd.DataFrame, cols: list[str]):
    X = _clean_inf(train_df[cols].to_numpy(dtype=np.float32))
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    scaler.fit(X)
    return imputer, scaler


def apply_preprocessors(df: pd.DataFrame, cols: list[str], imputer, scaler) -> pd.DataFrame:
    out = df.copy()
    X = _clean_inf(out[cols].to_numpy(dtype=np.float32))
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = np.clip(X, -5, 5)
    out[cols] = X
    return out


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return dict(r2=float("nan"), rmse=float("nan"), ubrmse=float("nan"),
                    bias=float("nan"), mae=float("nan"), q90=float("nan"), n=0)
    err = yp - yt
    bias = float(err.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ubrmse = float(np.sqrt(np.mean((err - bias) ** 2)))
    mae = float(np.mean(np.abs(err)))
    q90 = float(np.quantile(np.abs(err), 0.90))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return dict(r2=r2, rmse=rmse, ubrmse=ubrmse, bias=bias, mae=mae, q90=q90, n=int(yt.size))


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    pr, tr = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        pr.append(model(Xb).cpu().numpy())
        tr.append(yb.numpy())
    return np.concatenate(tr), np.concatenate(pr)


@torch.no_grad()
def extract_representations(model, loader, device, flag):
    model.eval()
    pr, tr, reps = [], [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        pred, rep = model(Xb, **{flag: True})
        pr.append(pred.cpu().numpy())
        tr.append(yb.numpy())
        reps.append(rep.cpu().numpy())
    return np.concatenate(tr), np.concatenate(pr), np.concatenate(reps, axis=0)


def train_one_seed(seed, feature_cols, seq_len, data_dir, models_dir, preprocessors):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns for LSTM: {missing}")

    imputer, scaler = preprocessors
    train_prep = apply_preprocessors(train_df, feature_cols, imputer, scaler)
    val_prep = apply_preprocessors(val_df, feature_cols, imputer, scaler)
    test_prep = apply_preprocessors(test_df, feature_cols, imputer, scaler)

    ds_train, ds_val, ds_test = build_datasets(
        train_prep, val_prep, test_prep,
        feature_cols=feature_cols, seq_len=seq_len, train_stride=TRAIN_STRIDE,
    )

    pin = device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    model = BiLSTMAttn(n_features=len(feature_cols), hidden_size=HIDDEN_SIZE,
                        num_layers=NUM_LAYERS, dropout=DROPOUT, proj_size=PROJ_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_rmse = math.inf
    best_epoch = -1
    patience_ctr = 0
    checkpoint_path = models_dir / f"lstm_seed{seed}.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for Xb, yb in loader_train:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += loss.item() * len(yb)
        train_loss = running / len(ds_train)

        y_true_val, y_pred_val = predict_loader(model, loader_val, device)
        val_mse = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)
        scheduler.step(val_mse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_ctr += 1

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [Seed{seed} Ep{epoch:3d}] loss={train_loss:.5f} val_rmse={val_rmse:.5f} best={best_val_rmse:.5f} (ep{best_epoch}) lr={lr_now:.2e}", flush=True)

        if patience_ctr >= PATIENCE:
            print(f"  [Seed{seed}] Early stop at ep{epoch}", flush=True)
            break

    print(f"  [Seed{seed} Done] Best ep{best_epoch} val_rmse={best_val_rmse:.5f}", flush=True)
    return best_val_rmse


def train_v23_lstm(data_dir: Path, models_dir: Path):
    print(f"[V23] {len(TOP25_FEATURES)} features, seq_len={SEQ_LEN}, {len(SEEDS)} seeds", flush=True)
    train_df = pd.read_csv(data_dir / "train.csv")
    imputer, scaler = fit_preprocessors(train_df, TOP25_FEATURES)

    results = []
    for seed in SEEDS:
        val_rmse = train_one_seed(seed, TOP25_FEATURES, SEQ_LEN, data_dir, models_dir, (imputer, scaler))
        results.append({"seed": seed, "val_rmse": val_rmse})

    best = min(results, key=lambda r: r["val_rmse"])
    print(f"[Best Seed] seed={best['seed']} val_rmse={best['val_rmse']:.5f}", flush=True)
    best_checkpoint = models_dir / f"lstm_seed{best['seed']}.pt"
    return best_checkpoint, (imputer, scaler), best


def load_best_model(checkpoint_path, n_features, device):
    model = BiLSTMAttn(n_features=n_features, hidden_size=HIDDEN_SIZE,
                        num_layers=NUM_LAYERS, dropout=DROPOUT, proj_size=PROJ_SIZE).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model


def extract_all_representations(model, data_dir, artifacts_dir, feature_cols, seq_len):
    device = next(model.parameters()).device
    set_seed(42)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    imputer, scaler = fit_preprocessors(train_df, feature_cols)
    train_prep = apply_preprocessors(train_df, feature_cols, imputer, scaler)
    val_prep = apply_preprocessors(val_df, feature_cols, imputer, scaler)
    test_prep = apply_preprocessors(test_df, feature_cols, imputer, scaler)

    ds_train, ds_val, ds_test = build_datasets(
        train_prep, val_prep, test_prep,
        feature_cols=feature_cols, seq_len=seq_len, train_stride=1,
    )

    pin = device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    def extract(flag_name):
        _, _, tr = extract_representations(model, loader_train, device, flag_name)
        _, _, va = extract_representations(model, loader_val, device, flag_name)
        _, _, te = extract_representations(model, loader_test, device, flag_name)
        return tr, va, te

    ctx_tr, ctx_va, ctx_te = extract("return_ctx")
    hh_tr, hh_va, hh_te = extract("return_head_hidden")
    hp_tr, hp_va, hp_te = extract("return_head_pre_relu")

    np.save(artifacts_dir / "ctx_train.npy", ctx_tr)
    np.save(artifacts_dir / "ctx_val.npy", ctx_va)
    np.save(artifacts_dir / "ctx_test.npy", ctx_te)
    np.save(artifacts_dir / "head_hidden_train.npy", hh_tr)
    np.save(artifacts_dir / "head_hidden_val.npy", hh_va)
    np.save(artifacts_dir / "head_hidden_test.npy", hh_te)
    np.save(artifacts_dir / "head_pre_relu_train.npy", hp_tr)
    np.save(artifacts_dir / "head_pre_relu_val.npy", hp_va)
    np.save(artifacts_dir / "head_pre_relu_test.npy", hp_te)

    return {
        "ctx": (ctx_tr, ctx_va, ctx_te),
        "hh": (hh_tr, hh_va, hh_te),
        "hp": (hp_tr, hp_va, hp_te),
    }


def compute_pca_all(reps_map: dict, artifacts_dir: Path):
    _OUT_NAMES = {"ctx": "ctx", "hh": "head_hidden", "hp": "head_pre_relu"}
    pca_info = []
    for name, (tr, va, te) in reps_map.items():
        out_name = _OUT_NAMES[name]
        for level_name, n_components in PCA_LEVELS:
            suffix = f"_{level_name}"
            pca = PCA(n_components=n_components, random_state=42, svd_solver="auto")
            tr_t = pca.fit_transform(tr)
            va_t = pca.transform(va)
            te_t = pca.transform(te)
            n_actual = pca.n_components_
            var_expl = pca.explained_variance_ratio_.sum()

            np.save(artifacts_dir / f"{out_name}{suffix}_train.npy", tr_t)
            np.save(artifacts_dir / f"{out_name}{suffix}_val.npy", va_t)
            np.save(artifacts_dir / f"{out_name}{suffix}_test.npy", te_t)
            joblib.dump(pca, artifacts_dir / f"pca_{out_name}{suffix}.pkl")

            pca_info.append({
                "representation": name,
                "level": level_name,
                "target_components": n_components if isinstance(n_components, int) else float(str(n_components)),
                "actual_components": int(n_actual),
                "variance_explained": round(float(var_expl), 4),
            })
            print(f"  [PCA {name}{suffix}] {tr.shape[1]}->{n_actual} comps, var={var_expl:.4f}", flush=True)

    with open(artifacts_dir / "pca_info.json", "w") as f:
        json.dump(pca_info, f, indent=2)
    return pca_info


def compute_lstm_only_metrics(model, data_dir, artifacts_dir, feature_cols, seq_len):
    device = next(model.parameters()).device
    set_seed(42)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    imputer, scaler = fit_preprocessors(train_df, feature_cols)
    train_prep = apply_preprocessors(train_df, feature_cols, imputer, scaler)
    val_prep = apply_preprocessors(val_df, feature_cols, imputer, scaler)
    test_prep = apply_preprocessors(test_df, feature_cols, imputer, scaler)
    ds_train, ds_val, ds_test = build_datasets(
        train_prep, val_prep, test_prep,
        feature_cols=feature_cols, seq_len=seq_len, train_stride=1,
    )
    pin = device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    y_tr_t, y_tr_p = predict_loader(model, loader_train, device)
    y_va_t, y_va_p = predict_loader(model, loader_val, device)
    y_te_t, y_te_p = predict_loader(model, loader_test, device)

    metrics = {
        "train": compute_metrics(y_tr_t, y_tr_p),
        "val": compute_metrics(y_va_t, y_va_p),
        "test": compute_metrics(y_te_t, y_te_p),
    }
    with open(artifacts_dir / "lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[LSTM-only] Test R2={metrics['test']['r2']:.4f} RMSE={metrics['test']['rmse']:.5f}", flush=True)
    return metrics


def train_v23_and_extract_all(data_dir: Path, artifacts_dir: Path, models_dir: Path):
    print("\n" + "=" * 75, flush=True)
    print("Phase 1: V23 LSTM Training (5 seeds)", flush=True)
    print("=" * 75, flush=True)

    best_checkpoint, preprocessors, best_seed_info = train_v23_lstm(data_dir, models_dir)

    print("\n" + "=" * 75, flush=True)
    print("Phase 2: Extract ctx/hh/hp from best seed", flush=True)
    print("=" * 75, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_best_model(best_checkpoint, len(TOP25_FEATURES), device)

    print("\n--- LSTM-only evaluation ---", flush=True)
    lstm_metrics = compute_lstm_only_metrics(model, data_dir, artifacts_dir, TOP25_FEATURES, SEQ_LEN)

    print("\n--- Representation extraction ---", flush=True)
    reps_map = extract_all_representations(model, data_dir, artifacts_dir, TOP25_FEATURES, SEQ_LEN)

    print("\n" + "=" * 75, flush=True)
    print("Phase 3: PCA reduction", flush=True)
    print("=" * 75, flush=True)

    pca_info = compute_pca_all(reps_map, artifacts_dir)
    return best_seed_info, pca_info, lstm_metrics
