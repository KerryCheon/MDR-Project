"""
LSTM Training and Frozen Representation Extraction Module (strict feature split).

derived_8.4-hybrid-lstm-2.0: the LSTM is trained on TEMPORAL features ONLY
(79 dims). The 13 static station attributes that 1.6 fed to the LSTM are moved
to XGBoost (config.yaml `static_features`), and the 34 temporal backbone
features that 1.6 did not feed to the LSTM are added, so the frozen context
vector carries ALL temporal dynamics and XGBoost carries the station statics.

Phase 1: Trains BiLSTMAttn on derived_8.4 sequence dataset (1 seed per hidden size).
Phase 2: Freezes best checkpoint and extracts raw (non-PCA) representations:
  - ctx (2H-dim): attention-pooled bidirectional hidden state
  - head_hidden (H-dim): after head Linear(2H->H)->ReLU
  - head_pre_relu (H-dim): after head Linear(2H->H) BEFORE ReLU
for train, val, test splits, one `artifacts/h{H}/` subdir per hidden size.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
EXP_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = EXP_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from .dataset import TARGET, build_datasets
from .model import BiLSTMAttn

# ---------------------------------------------------------------------------
# Strict static/temporal split (derived_8.4-hybrid-lstm-2.0).
#
# The feature sets below are derived programmatically from the 1.6 provenance
# lists (NOT hand-typed), so the split is auditable:
#   TEMPORAL_FEATURES = (1.6 LSTM input minus its 13 static station attrs)
#                     + (54-feature backbone temporal members not already in)
#   STATIC_FEATURES   = backbone static subset (5) + the 13 moved statics
# Both lists are verified in the notebook: length, disjointness, and
# constant-per-station for every static feature.
# ---------------------------------------------------------------------------

# --- 1.6 provenance: the full LSTM input used by derived_8.4-hybrid-lstm-1.6 ---
ALL_FEATURES_V1_6 = [
    # Jakob 38 features
    "SMAP_sm_pm_interp_ema02", "V_rollmin_LST_modis_kobs30", "D_sin_DOY", "G_rain_sum_3d",
    "V_ema_G_API_kobs7", "V_rollmin_G_API_kobs30", "G_rain_sum_7d", "C_lag_LST_modis_kobs30",
    "C_lag_G_API_kobs1", "V_ema_G_API_kobs14", "V_rollmean_G_API_kobs14", "G_API",
    "G_DSLR", "SMAP_ampm_diff_interp", "V_rollmax_G_API_kobs30", "V_ema_G_API_kobs30",
    "V_rollmean_s2_b11_kobs7", "V_ema_LST_modis_kobs7", "V_rollmean_G_API_kobs7",
    "C_lag_s2_b11_kobs30", "A_d_E_SAR_diff_kobs14", "C_lag_LST_modis_kobs6",
    "A_d_LST_modis_kobs14", "A_d_SMAP_sm_interp_kobs14", "V_rollstd_SMAP_sm_interp_kobs30",
    "SMAP_sm_interp_grad7", "year_frac", "sin_year", "cos_year", "API_x_year",
    "SMAP_x_year", "slope", "elev", "K_slope_sin", "K_slope_cos", "K_aspect_cos",
    "J_clay_wfrac_b0", "J_sand_wfrac_b0",
    # V9-unique extras not in Jakob
    "precip_mm", "s1_vv", "s1_vh", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "LST_modis", "F_NDVI", "F_NDMI", "E_SAR_ratio",
    "SMAP_sm_am_interp", "SMAP_sm_pm_interp", "SMAP_sm_interp_mask",
    "latitude", "longitude", "aspect",
    "K_sand_clay_ratio_b0", "K_clay_plus_sand_b0", "K_aspect_sin",
]

# --- 1.6 provenance: the 54-feature tabular backbone (config `shared_backbone_54`) ---
BACKBONE_54 = [
    "precip_mm", "s2_b4", "s2_b8", "SMAP_sm_pm_interp", "D_sin_DOY", "D_cos_DOY",
    "E_SAR_ratio", "G_API", "G_DSLR", "G_rain_sum_3d", "G_rain_sum_7d",
    "SMAP_sm_pm_interp_lag7", "SMAP_sm_pm_interp_lag30", "SMAP_sm_pm_interp_rollrange7",
    "SMAP_sm_pm_interp_rollmean30", "SMAP_sm_pm_interp_rollrange30",
    "SMAP_sm_interp_rollrange7", "SMAP_ampm_diff_interp", "V_rollrng_G_API_kobs7",
    "V_rollmax_F_NDMI_kobs30", "A_d_E_SAR_ratio_kobs30", "V_rollmax_E_SAR_ratio_kobs7",
    "V_rollmin_E_SAR_ratio_kobs30", "V_rollmax_E_SAR_ratio_kobs30",
    "V_rollmin_LST_modis_kobs30", "V_ema_LST_modis_kobs30", "V_rollmax_F_NDVI_kobs14",
    "V_rollmax_F_NDVI_kobs30", "V_ema_F_NDVI_kobs30", "C_lag_F_NDVI_kobs30",
    "A_grad_E_SAR_diff_kobs30", "V_rollmax_E_SAR_diff_kobs14", "V_rollrng_E_SAR_diff_kobs30",
    "V_rollmax_E_SAR_diff_kobs30", "A_grad_s2_b11_kobs30", "V_rollrng_s2_b11_kobs30",
    "V_rollmin_s2_b11_kobs30", "V_rollmin_s2_b12_kobs30", "A_d_SMAP_sm_interp_kobs30",
    "V_rollmin_SMAP_sm_interp_kobs14", "V_rollmin_SMAP_sm_interp_kobs30",
    "E_rough_s1_vh_kobs14", "J_aspect_deg", "J_bio_bio02", "J_bio_bio13", "J_lc_code",
    "J_soil_texture_usda_b0", "sin_year", "cos_year", "SMAP_x_year", "D_z_F_NDMI",
    "D_z_LST_modis", "D_fft_dom_LST_modis_kobs30", "D_fft_ent_LST_modis_kobs30",
]

# Static station attributes that 1.6 fed to the LSTM; now exclusive to XGBoost.
LSTM_STATIC_EXCLUDED = [
    "longitude", "latitude", "elev", "slope", "aspect",
    "J_clay_wfrac_b0", "J_sand_wfrac_b0",
    "K_slope_sin", "K_slope_cos", "K_aspect_cos",
    "K_sand_clay_ratio_b0", "K_clay_plus_sand_b0", "K_aspect_sin",
]

# Static subset of the 54-feature backbone (constant per station).
BACKBONE_STATIC = [
    "J_aspect_deg", "J_bio_bio02", "J_bio_bio13", "J_lc_code", "J_soil_texture_usda_b0",
]

# 2.0: XGBoost's direct (tabular) feature set — static features only.
STATIC_FEATURES = BACKBONE_STATIC + LSTM_STATIC_EXCLUDED

# 2.0: LSTM input — temporal features only (79 = 45 from 1.6's set + 34 backbone
# temporal members not already present). All temporal/rolling dynamics.
TEMPORAL_FEATURES = (
    [f for f in ALL_FEATURES_V1_6 if f not in LSTM_STATIC_EXCLUDED]
    + [f for f in BACKBONE_54
       if f not in STATIC_FEATURES and f not in ALL_FEATURES_V1_6]
)

assert len(TEMPORAL_FEATURES) == 79, f"Expected 79 temporal features, got {len(TEMPORAL_FEATURES)}"
assert len(STATIC_FEATURES) == 18, f"Expected 18 static features, got {len(STATIC_FEATURES)}"
assert set(TEMPORAL_FEATURES).isdisjoint(STATIC_FEATURES), "Static/temporal split must be disjoint"

SEQ_LEN = 30
TRAIN_STRIDE = 1
# Hidden-size sweep. Each H produces ctx=2H dims and head_hidden/head_pre_relu=H dims.
HIDDEN_SIZES = [40, 20, 16, 8, 4]
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
SEEDS = [42]


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


def train_one_seed(seed, feature_cols, seq_len, data_dir, models_dir, preprocessors, hidden_size):
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

    model = BiLSTMAttn(n_features=len(feature_cols), hidden_size=hidden_size,
                        num_layers=NUM_LAYERS, dropout=DROPOUT, proj_size=PROJ_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_rmse = math.inf
    best_epoch = -1
    patience_ctr = 0
    checkpoint_path = models_dir / f"lstm_h{hidden_size}_seed{seed}.pt"

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
            print(f"  [H{hidden_size} Seed{seed} Ep{epoch:3d}] loss={train_loss:.5f} val_rmse={val_rmse:.5f} best={best_val_rmse:.5f} (ep{best_epoch}) lr={lr_now:.2e}", flush=True)

        if patience_ctr >= PATIENCE:
            print(f"  [H{hidden_size} Seed{seed}] Early stop at ep{epoch}", flush=True)
            break

    print(f"  [H{hidden_size} Seed{seed} Done] Best ep{best_epoch} val_rmse={best_val_rmse:.5f}", flush=True)
    return best_val_rmse


def train_lstm_hidden(hidden_size: int, data_dir: Path, models_dir: Path):
    """Train the BiLSTM+Attn with the given hidden size (1 seed), return best checkpoint info."""
    print(f"[V21 H{hidden_size}] {len(TEMPORAL_FEATURES)} temporal features, seq_len={SEQ_LEN}, {len(SEEDS)} seeds", flush=True)
    train_df = pd.read_csv(data_dir / "train.csv")
    imputer, scaler = fit_preprocessors(train_df, TEMPORAL_FEATURES)

    results = []
    for seed in SEEDS:
        val_rmse = train_one_seed(seed, TEMPORAL_FEATURES, SEQ_LEN, data_dir, models_dir, (imputer, scaler), hidden_size)
        results.append({"seed": seed, "val_rmse": val_rmse})

    best = min(results, key=lambda r: r["val_rmse"])
    print(f"[H{hidden_size} Best Seed] seed={best['seed']} val_rmse={best['val_rmse']:.5f}", flush=True)
    return best


def load_best_model(checkpoint_path, n_features, device, hidden_size):
    model = BiLSTMAttn(n_features=n_features, hidden_size=hidden_size,
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
    print(f"[H{model.hidden_size}] LSTM-only Test R2={metrics['test']['r2']:.4f} RMSE={metrics['test']['rmse']:.5f}", flush=True)
    return metrics


def train_lstm_for_hidden(hidden_size: int, data_dir: Path, artifacts_dir: Path, models_dir: Path):
    """Train LSTM for one hidden size, evaluate LSTM-only, and extract raw ctx/hh/hp."""
    print("\n" + "=" * 75, flush=True)
    print(f"Phase 1: V21 LSTM Training (H={hidden_size}, 1 seed)", flush=True)
    print("=" * 75, flush=True)

    best_seed_info = train_lstm_hidden(hidden_size, data_dir, models_dir)

    print("\n" + "=" * 75, flush=True)
    print(f"Phase 2: Extract ctx/hh/hp from best seed (H={hidden_size})", flush=True)
    print("=" * 75, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = models_dir / f"lstm_h{hidden_size}_seed{best_seed_info['seed']}.pt"
    model = load_best_model(checkpoint_path, len(TEMPORAL_FEATURES), device, hidden_size)

    print("\n--- LSTM-only evaluation ---", flush=True)
    lstm_metrics = compute_lstm_only_metrics(model, data_dir, artifacts_dir, TEMPORAL_FEATURES, SEQ_LEN)

    print("\n--- Representation extraction ---", flush=True)
    reps_map = extract_all_representations(model, data_dir, artifacts_dir, TEMPORAL_FEATURES, SEQ_LEN)

    for name, (tr, va, te) in reps_map.items():
        print(f"  [{name}] train={tr.shape} val={va.shape} test={te.shape}", flush=True)

    return best_seed_info, lstm_metrics
