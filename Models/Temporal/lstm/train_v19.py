"""
LSTM v19 - Forward-chain temporal cross-validation with expanding training windows.

Three folds, each trained from scratch with the same BiLSTMAttn architecture as v9:

  Fold A: train 2017-2020, val 2021, test 2022-2025  (baseline: same training as v9)
  Fold B: train 2017-2021, val 2022, test 2023-2025  (same test as v9/v18, comparable R^2)
  Fold C: train 2017-2022, val 2023, test 2024-2025  (maximum training data)

Fold B is identical in split to v18 -- if their test R^2 values are close, both scripts
are correct. Fold C shows whether more training data continues to help or plateaus.

Usage:
    python -m Models.Temporal.lstm.train_v19
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.dataset import TARGET, build_datasets

DATA_DIR = REPO_ROOT / "data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs_v19"
OUT_DIR.mkdir(exist_ok=True)


TIME_FEATURES = [
    "precip_mm",
    "s1_vv", "s1_vh",
    "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "LST_modis",
    "F_NDVI", "F_NDMI",
    "E_SAR_ratio",
    "SMAP_sm_am_interp", "SMAP_sm_pm_interp",
    "SMAP_sm_interp_mask",
    "sin_year", "cos_year",
    "SMAP_sm_pm_interp_ema02",
    "V_ema_LST_modis_kobs7",
    "V_rollmean_G_API_kobs14",
]

STATIC_FEATURES = [
    "latitude", "longitude",
    "elev", "slope", "aspect",
    "K_sand_clay_ratio_b0", "K_clay_plus_sand_b0",
    "K_slope_sin", "K_slope_cos", "K_aspect_sin", "K_aspect_cos",
]

ALL_FEATURES = TIME_FEATURES + STATIC_FEATURES


SEQ_LEN      = 10
TRAIN_STRIDE = 1
HIDDEN_SIZE  = 80
NUM_LAYERS   = 2
DROPOUT      = 0.3
PROJ_SIZE    = 56
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 2e-3
HUBER_DELTA  = 0.05
MAX_EPOCHS   = 250
PATIENCE     = 35
GRAD_CLIP    = 1.0
SEED         = 42

FOLDS = [
    dict(name="A", train_years=list(range(2017, 2021)), val_years=[2021], test_years=list(range(2022, 2026))),
    dict(name="B", train_years=list(range(2017, 2022)), val_years=[2022], test_years=list(range(2023, 2026))),
    dict(name="C", train_years=list(range(2017, 2023)), val_years=[2023], test_years=list(range(2024, 2026))),
]


class BiLSTMAttn(nn.Module):
    """BiLSTM with additive attention pooling over the time axis."""

    def __init__(self, n_features, hidden_size, num_layers, dropout, proj_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, proj_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        bi = hidden_size * 2
        self.attn = nn.Sequential(
            nn.Linear(bi, bi),
            nn.Tanh(),
            nn.Linear(bi, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(bi, bi // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bi // 2, 1),
        )

    def forward(self, x):
        b, s, f = x.shape
        x = self.proj(x.reshape(b * s, f)).reshape(b, s, -1)
        out, _ = self.lstm(x)
        scores = self.attn(out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(self.dropout(ctx)).squeeze(-1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_inf(X):
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    return X


def fit_preprocessors(train_df, cols):
    X = _clean_inf(train_df[cols].to_numpy(dtype=np.float32))
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    scaler.fit(X)
    return imputer, scaler


def apply_preprocessors(df, cols, imputer, scaler):
    out = df.copy()
    X = _clean_inf(out[cols].to_numpy(dtype=np.float32))
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = np.clip(X, -5, 5)
    out[cols] = X
    return out


def compute_metrics(y_true, y_pred):
    """R^2, RMSE, ubRMSE, Bias, MAE, Q90 on finite values."""
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
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        pr.append(model(X_batch).cpu().numpy())
        tr.append(y_batch.numpy())
    return np.concatenate(tr), np.concatenate(pr)


def run_fold(fold_cfg, all_df, device, fold_out_dir):
    fold_out_dir.mkdir(exist_ok=True)
    name = fold_cfg["name"]

    train_df = all_df[all_df["year"].isin(fold_cfg["train_years"])].copy()
    val_df   = all_df[all_df["year"].isin(fold_cfg["val_years"])].copy()
    test_df  = all_df[all_df["year"].isin(fold_cfg["test_years"])].copy()

    print(f"\n[fold {name}] train={train_df.shape}  val={val_df.shape}  test={test_df.shape}")
    print(f"            train {fold_cfg['train_years']} | val {fold_cfg['val_years']} | test {fold_cfg['test_years']}")

    imputer, scaler = fit_preprocessors(train_df, ALL_FEATURES)
    train_df = apply_preprocessors(train_df, ALL_FEATURES, imputer, scaler)
    val_df   = apply_preprocessors(val_df,   ALL_FEATURES, imputer, scaler)
    test_df  = apply_preprocessors(test_df,  ALL_FEATURES, imputer, scaler)

    ds_train, ds_val, ds_test = build_datasets(
        train_df, val_df, test_df,
        feature_cols=ALL_FEATURES,
        seq_len=SEQ_LEN,
        train_stride=TRAIN_STRIDE,
    )
    pin = device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=pin)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    set_seed(SEED)
    model = BiLSTMAttn(
        n_features=len(ALL_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj_size=PROJ_SIZE,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, epochs=MAX_EPOCHS, steps_per_epoch=len(loader_train),
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3,
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_rmse = math.inf
    best_epoch = -1
    patience_ctr = 0
    train_losses, val_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for X_batch, y_batch in loader_train:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y_batch)
        train_loss = running / len(ds_train)

        y_true_val, y_pred_val = predict_loader(model, loader_val, device)
        val_mse  = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)

        train_losses.append(train_loss)
        val_losses.append(val_mse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), fold_out_dir / "best_model.pt")
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {epoch:3d}  train={train_loss:.5f}  val_rmse={val_rmse:.5f}  "
                  f"best={best_val_rmse:.5f} (ep{best_epoch})  lr={lr_now:.2e}")

        if patience_ctr >= PATIENCE:
            print(f"\n[early stop] no improvement for {PATIENCE} epochs at epoch {epoch}")
            break

    print(f"\n[fold {name} eval] best epoch {best_epoch}  val_rmse={best_val_rmse:.5f}")
    model.load_state_dict(torch.load(fold_out_dir / "best_model.pt", map_location=device, weights_only=True))

    fold_results = {}
    for split_name, loader in [("train", loader_train), ("val", loader_val), ("test", loader_test)]:
        y_true, y_pred = predict_loader(model, loader, device)
        m = compute_metrics(y_true, y_pred)
        fold_results[split_name] = m
        print(f"  {split_name:5s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.5f}  ubRMSE={m['ubrmse']:.5f}  "
              f"Bias={m['bias']:+.5f}  MAE={m['mae']:.5f}  Q90={m['q90']:.5f}  n={m['n']}")

    fold_results["config"] = dict(
        fold=name,
        train_years=fold_cfg["train_years"],
        val_years=fold_cfg["val_years"],
        test_years=fold_cfg["test_years"],
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
    )

    with open(fold_out_dir / "metrics.json", "w") as f:
        json.dump(fold_results, f, indent=2)

    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        train_label = f"{fold_cfg['train_years'][0]}-{fold_cfg['train_years'][-1]}"
        ax.set_title(f"v19 fold {name}: train {train_label} / val {fold_cfg['val_years']} / test {fold_cfg['test_years']}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fold_out_dir / "loss_curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped ({e})")

    return fold_results


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    all_df = pd.concat([
        pd.read_csv(DATA_DIR / "train.csv"),
        pd.read_csv(DATA_DIR / "val.csv"),
        pd.read_csv(DATA_DIR / "test.csv"),
    ], ignore_index=True)
    all_df["year"] = pd.to_datetime(all_df["date"]).dt.year
    print(f"[load] total rows={all_df.shape[0]}  years={sorted(all_df['year'].unique())}")

    missing = [c for c in ALL_FEATURES if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    all_results = {}
    for fold_cfg in FOLDS:
        fold_out_dir = OUT_DIR / f"fold_{fold_cfg['name']}"
        fold_results = run_fold(fold_cfg, all_df, device, fold_out_dir)
        all_results[f"fold_{fold_cfg['name']}"] = fold_results

    # Save combined results
    with open(OUT_DIR / "fold_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {OUT_DIR / 'fold_metrics.json'}")

    # Summary table
    print("\n" + "=" * 80)
    print("FORWARD-CHAIN TEMPORAL CV SUMMARY")
    print("=" * 80)
    print(f"{'Version':<10} {'Train years':<18} {'Val':<8} {'Test years':<16} {'Test R2':>8}  {'Test RMSE':>10}  {'n':>6}")
    print("-" * 80)
    print(f"{'v9'::<10} {'2017-2020':<18} {'2021-22':<8} {'2023-2025':<16} {'0.7470':>8}  {'(ref)':>10}  {'(ref)':>6}")

    fold_labels = {
        "A": ("2017-2020", "2021",    "2022-2025"),
        "B": ("2017-2021", "2022",    "2023-2025"),
        "C": ("2017-2022", "2023",    "2024-2025"),
    }
    for fold_name, (tr_label, va_label, te_label) in fold_labels.items():
        key = f"fold_{fold_name}"
        if key not in all_results:
            continue
        m = all_results[key].get("test", {})
        r2   = m.get("r2", float("nan"))
        rmse = m.get("rmse", float("nan"))
        n    = m.get("n", 0)
        version = f"v19-{fold_name}"
        print(f"{version:<10} {tr_label:<18} {va_label:<8} {te_label:<16} {r2:>8.4f}  {rmse:>10.5f}  {n:>6}")

    print("=" * 80)
    print("\nNote: fold B uses same split as v18 -- R^2 values should be similar.")
    print("      fold A test includes 2022 (val year in v9); not directly comparable.")
    print("      fold C test is only 2024-2025; shorter window may inflate R^2.")


if __name__ == "__main__":
    main()
