"""
LSTM training pipeline — v6: regularisation + Gaussian noise augmentation.

Changes from baseline (train.py):
  - Import from model_v6 (with noise injection)
  - DROPOUT = 0.4  (was 0.3)
  - WEIGHT_DECAY = 5e-3  (was 3e-3)
  - noise_std = 0.1 passed to model constructor
  - ReduceLROnPlateau patience=25
  - MAX_EPOCHS=300, PATIENCE=60
  - Outputs to Models/Temporal/lstm/outputs_v6/

Usage:
    python -m Models.Temporal.lstm.train_v6
  or
    python Models/Temporal/lstm/train_v6.py

Outputs (written to Models/Temporal/lstm/outputs_v6/):
    best_model.pt   — best checkpoint (lowest val RMSE)
    metrics.json    — final train / val / test metrics
    loss_curve.png  — training curve
"""

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

# path setup
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.dataset import TARGET, build_datasets
from Models.Temporal.lstm.model_v6 import LSTMRawSeries

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "Temporal/Pipeline/data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs_v6"
OUT_DIR.mkdir(exist_ok=True)

# Raw daily observations fed as the sequence to the LSTM.
TIME_FEATURES = [
    "precip_mm",
    "s1_vv", "s1_vh",
    "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "LST_modis",
    "F_NDVI", "F_NDMI", "F_MSI",
    "E_SAR_ratio", "E_SAR_diff",
    "SMAP_sm_am_interp", "SMAP_sm_pm_interp", "SMAP_ampm_diff_interp",
    "SMAP_sm_am_interp_mask", "SMAP_sm_pm_interp_mask", "SMAP_sm_interp_mask",
    "sin_year", "cos_year",
]

STATIC_FEATURES = [
    "latitude", "longitude",
    "elev", "slope", "aspect",
    "K_sand_clay_ratio_b0", "K_clay_plus_sand_b0",
    "K_slope_sin", "K_slope_cos", "K_aspect_sin", "K_aspect_cos",
    "lia_mean_asc_deg", "lia_std_asc_deg", "lia_mean_desc_deg", "lia_std_desc_deg",
]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEQ_LEN          = 60
TRAIN_STRIDE     = 1
TIME_PROJ_SIZE   = 32
STATIC_PROJ_SIZE = 32
HIDDEN_SIZE      = 128
NUM_LAYERS       = 2
DROPOUT          = 0.4     # v6: increased from 0.3
NOISE_STD        = 0.1     # v6: Gaussian noise injection
BATCH_SIZE       = 256
LR               = 1e-4
WEIGHT_DECAY     = 5e-3    # v6: increased from 3e-3
HUBER_DELTA      = 0.05
MAX_EPOCHS       = 300
PATIENCE         = 60
GRAD_CLIP        = 1.0
TEMPORAL_BETA    = 0.2
SEED             = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    print(f"[load] train={train.shape}  val={val.shape}  test={test.shape}")
    return train, val, test


def _clean_inf(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    return X


def fit_preprocessors(train_df: pd.DataFrame, all_feature_cols: list):
    """Fit imputer + scaler on train features only."""
    X = _clean_inf(train_df[all_feature_cols].to_numpy(dtype=np.float32))
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    scaler.fit(X)
    return imputer, scaler


def apply_preprocessors(df: pd.DataFrame, all_feature_cols: list, imputer, scaler) -> pd.DataFrame:
    out = df.copy()
    X = _clean_inf(out[all_feature_cols].to_numpy(dtype=np.float32))
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = np.clip(X, -5, 5)
    out[all_feature_cols] = X
    return out


def compute_temporal_weights(years: torch.Tensor, year_max: int) -> torch.Tensor:
    """Exponential year weights (Eq. 2 in paper), normalised to unit mean."""
    w = torch.exp(TEMPORAL_BETA * (years.float() - year_max))
    return w / w.mean()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return dict(r2=float("nan"), rmse=float("nan"), mae=float("nan"), bias=float("nan"), n=0)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae  = float(np.mean(np.abs(yt - yp)))
    bias = float(np.mean(yp - yt))
    return dict(r2=r2, rmse=rmse, mae=mae, bias=bias, n=int(yt.size))


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple:
    model.eval()
    all_pred, all_true = [], []
    for x_time, x_static, _years, y_batch in loader:
        x_time   = x_time.to(device)
        x_static = x_static.to(device)
        preds    = model(x_time, x_static).cpu().numpy()
        all_pred.append(preds)
        all_true.append(y_batch.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def save_loss_curve(train_losses: list, val_losses: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="train loss")
        ax.plot(val_losses,   label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted Huber Loss")
        ax.set_title("LSTM v6 — Regularisation + Noise Augmentation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "loss_curve.png", dpi=120)
        plt.close(fig)
        print(f"[plot] saved -> {OUT_DIR / 'loss_curve.png'}")
    except Exception as e:
        print(f"[plot] skipped ({e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_df, val_df, test_df = load_data()

    # Validate that every declared feature exists in the data
    all_cols = TIME_FEATURES + STATIC_FEATURES
    missing = [c for c in all_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Features missing from dataset: {missing}")
    print(f"[features] {len(TIME_FEATURES)} time  +  {len(STATIC_FEATURES)} static")

    # Fit preprocessors on train split only
    imputer, scaler = fit_preprocessors(train_df, all_cols)
    train_df = apply_preprocessors(train_df, all_cols, imputer, scaler)
    val_df   = apply_preprocessors(val_df,   all_cols, imputer, scaler)
    test_df  = apply_preprocessors(test_df,  all_cols, imputer, scaler)

    # Build sequence datasets
    ds_train, ds_val, ds_test = build_datasets(
        train_df, val_df, test_df,
        time_cols=TIME_FEATURES,
        static_cols=STATIC_FEATURES,
        seq_len=SEQ_LEN,
        train_stride=TRAIN_STRIDE,
    )

    year_max = int(ds_train.years.max())
    print(f"[temporal weighting] beta={TEMPORAL_BETA}  year_max={year_max}")

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=device.type == "cuda")
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    # Model — v6 with noise injection
    model = LSTMRawSeries(
        n_time=len(TIME_FEATURES),
        n_static=len(STATIC_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        time_proj_size=TIME_PROJ_SIZE,
        static_proj_size=STATIC_PROJ_SIZE,
        noise_std=NOISE_STD,
    ).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")
    print(f"[model] noise_std={NOISE_STD}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA, reduction="none")

    # Training loop
    best_val_rmse = math.inf
    best_epoch    = -1
    patience_ctr  = 0
    train_losses, val_losses = [], []

    print(
        f"\n[train] seq_len={SEQ_LEN}  stride={TRAIN_STRIDE}  "
        f"time_proj={TIME_PROJ_SIZE}  static_proj={STATIC_PROJ_SIZE}  "
        f"hidden={HIDDEN_SIZE}  layers={NUM_LAYERS}  dropout={DROPOUT}"
    )
    print(f"        noise_std={NOISE_STD}  batch={BATCH_SIZE}  lr={LR}  wd={WEIGHT_DECAY}  "
          f"max_epochs={MAX_EPOCHS}  patience={PATIENCE}\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x_time, x_static, yr_batch, y_batch in loader_train:
            x_time   = x_time.to(device)
            x_static = x_static.to(device)
            yr_batch = yr_batch.to(device)
            y_batch  = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_time, x_static)

            per_sample = criterion(pred, y_batch)
            w          = compute_temporal_weights(yr_batch, year_max).to(device)
            loss       = (per_sample * w).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        train_loss = running_loss / len(ds_train)

        y_true_val, y_pred_val = predict_loader(model, loader_val, device)
        val_mse  = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)

        scheduler.step(val_mse)
        train_losses.append(train_loss)
        val_losses.append(val_mse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch    = epoch
            patience_ctr  = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch:3d}/{MAX_EPOCHS}  "
                f"train_loss={train_loss:.5f}  val_rmse={val_rmse:.5f}  "
                f"best={best_val_rmse:.5f} (ep{best_epoch})  lr={lr_now:.2e}"
            )

        if patience_ctr >= PATIENCE:
            print(f"\n[early stop] patience={PATIENCE} reached at epoch {epoch}")
            break

    # Evaluate best checkpoint
    print(f"\n[eval] loading best checkpoint (epoch {best_epoch}, val_rmse={best_val_rmse:.5f})")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device, weights_only=True))

    results = {}
    for name, loader in [("train", loader_train), ("val", loader_val), ("test", loader_test)]:
        y_true, y_pred = predict_loader(model, loader, device)
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:5s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  bias={m['bias']:+.5f}  n={m['n']}")

    # Print train-test gap for easy comparison
    gap = results["train"]["r2"] - results["test"]["r2"]
    print(f"\n  train-test R² gap = {gap:.4f}  (baseline: 0.064)")

    results["config"] = dict(
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        time_proj_size=TIME_PROJ_SIZE, static_proj_size=STATIC_PROJ_SIZE,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        noise_std=NOISE_STD,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        huber_delta=HUBER_DELTA, temporal_beta=TEMPORAL_BETA,
        time_features=TIME_FEATURES, static_features=STATIC_FEATURES,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
    )

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] metrics -> {metrics_path}")
    print(f"[saved] model   -> {OUT_DIR / 'best_model.pt'}")

    save_loss_curve(train_losses, val_losses)


if __name__ == "__main__":
    main()
