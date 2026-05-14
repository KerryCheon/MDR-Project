"""
Transformer-encoder training pipeline for soil moisture prediction.

Mirrors the LSTM pipeline — same splits, same preprocessing (StandardScaler
+ SimpleImputer fit on train only, clipped to ±5), same temporal
year-weighted Huber loss, same early stopping.  Differences:

* Model is a pure Transformer encoder with CLS-token pooling
  (see model.py for the design rationale).
* Longer look-back (SEQ_LEN=90) — literature suggests Transformers benefit
  from more context than LSTMs.
* Time-feature list augmented with a handful of engineered memory-state
  signals (G_API, G_DSLR, G_rain_sum_{3,7,30}d, SMAP rollmean7/30) so the
  attention layers have easier-to-digest summaries on this tiny dataset.
* Cosine schedule with 5% linear warmup (Transformers dislike plateau LR).
* Higher weight decay (1e-2) — standard Transformer regularisation.

Usage:
    python Models/Temporal/transformer/train.py

Outputs (Models/Temporal/transformer/outputs/):
    best_model.pt
    metrics.json
    loss_curve.png
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

from Models.Spatial.v0.transformer.dataset import TARGET, build_datasets
from Models.Spatial.v0.transformer.model import TransformerSoilMoisture

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "Temporal/Pipeline/data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Raw daily observations + a small set of engineered memory-state features.
# With only ~7k training rows the attention layers struggle to rediscover
# rain-accumulation and drying-time statistics from scratch, so we feed
# them the pre-computed summaries (G_API, G_DSLR, rain sums, SMAP rolling
# means).  We keep the list to ~11 engineered cols out of 400+ — physical
# signals only.
TIME_FEATURES = [
    # precipitation (no gaps)
    "precip_mm",
    # Sentinel-1 SAR backscatter (imputed to daily)
    "s1_vv", "s1_vh",
    # Sentinel-2 surface reflectance (imputed to daily)
    "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    # MODIS land surface temperature (imputed to daily)
    "LST_modis",
    # Vegetation / water indices
    "F_NDVI", "F_NDMI", "F_MSI",
    # SAR-derived cross-pol indices
    "E_SAR_ratio", "E_SAR_diff",
    # SMAP (AM + PM + diff) with observation masks
    "SMAP_sm_am_interp", "SMAP_sm_pm_interp", "SMAP_ampm_diff_interp",
    "SMAP_sm_am_interp_mask", "SMAP_sm_pm_interp_mask", "SMAP_sm_interp_mask",
    # Seasonality
    "sin_year", "cos_year",
    # --- engineered memory-state features (study #8: helpful on small data) ---
    "G_API",                # antecedent precipitation index
    "G_DSLR",               # days since last rain
    "G_rain_sum_3d",
    "G_rain_sum_7d",
    "G_rain_sum_30d",
    "SMAP_sm_am_interp_rollmean7",
    "SMAP_sm_am_interp_rollmean30",
    "SMAP_sm_pm_interp_rollmean7",
    "SMAP_sm_pm_interp_rollmean30",
    "SMAP_sm_interp_rollmean7",
    "SMAP_sm_interp_rollmean30",
]

# Fixed per-station features (not time-varying) — same as LSTM baseline.
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
SEQ_LEN          = 90
TRAIN_STRIDE     = 1
D_MODEL          = 96
NHEAD            = 4
NUM_LAYERS       = 3
DIM_FF           = 256
DROPOUT          = 0.2
STATIC_PROJ_SIZE = 32
BATCH_SIZE       = 128
LR               = 3e-4
WEIGHT_DECAY     = 1e-2
WARMUP_FRAC      = 0.05
HUBER_DELTA      = 0.05
MAX_EPOCHS       = 200
PATIENCE         = 40
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
    X = _clean_inf(train_df[all_feature_cols].to_numpy(dtype=np.float32))
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    scaler.fit(X)
    return imputer, scaler


def apply_preprocessors(df, all_feature_cols, imputer, scaler):
    out = df.copy()
    X = _clean_inf(out[all_feature_cols].to_numpy(dtype=np.float32))
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = np.clip(X, -5, 5)
    out[all_feature_cols] = X
    return out


def compute_temporal_weights(years: torch.Tensor, year_max: int) -> torch.Tensor:
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
def predict_loader(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for x_time, x_static, _years, y_batch in loader:
        x_time   = x_time.to(device)
        x_static = x_static.to(device)
        preds    = model(x_time, x_static).cpu().numpy()
        all_pred.append(preds)
        all_true.append(y_batch.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def build_lr_lambda(total_steps: int, warmup_frac: float):
    """Linear warmup over warmup_frac of steps, then cosine decay to 0."""
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return lr_lambda


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
        ax.set_title("Transformer Soil Moisture Training Curve")
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

    all_cols = TIME_FEATURES + STATIC_FEATURES
    missing = [c for c in all_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Features missing from dataset: {missing}")
    print(f"[features] {len(TIME_FEATURES)} time  +  {len(STATIC_FEATURES)} static")

    imputer, scaler = fit_preprocessors(train_df, all_cols)
    train_df = apply_preprocessors(train_df, all_cols, imputer, scaler)
    val_df   = apply_preprocessors(val_df,   all_cols, imputer, scaler)
    test_df  = apply_preprocessors(test_df,  all_cols, imputer, scaler)

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

    model = TransformerSoilMoisture(
        n_time=len(TIME_FEATURES),
        n_static=len(STATIC_FEATURES),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        seq_len=SEQ_LEN,
        static_proj_size=STATIC_PROJ_SIZE,
    ).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = MAX_EPOCHS * max(1, len(loader_train))
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_lr_lambda(total_steps, WARMUP_FRAC)
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA, reduction="none")

    best_val_rmse = math.inf
    best_epoch    = -1
    patience_ctr  = 0
    train_losses, val_losses = [], []

    print(
        f"\n[train] seq_len={SEQ_LEN}  stride={TRAIN_STRIDE}  "
        f"d_model={D_MODEL}  nhead={NHEAD}  layers={NUM_LAYERS}  "
        f"ff={DIM_FF}  dropout={DROPOUT}"
    )
    print(f"        batch={BATCH_SIZE}  lr={LR}  wd={WEIGHT_DECAY}  "
          f"warmup_frac={WARMUP_FRAC}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}\n")

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
            scheduler.step()   # per-step cosine warmup
            running_loss += loss.item() * len(y_batch)

        train_loss = running_loss / len(ds_train)

        y_true_val, y_pred_val = predict_loader(model, loader_val, device)
        val_mse  = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)

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

    print(f"\n[eval] loading best checkpoint (epoch {best_epoch}, val_rmse={best_val_rmse:.5f})")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device, weights_only=True))

    results = {}
    for name, loader in [("train", loader_train), ("val", loader_val), ("test", loader_test)]:
        y_true, y_pred = predict_loader(model, loader, device)
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:5s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  bias={m['bias']:+.5f}  n={m['n']}")

    results["config"] = dict(
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        dropout=DROPOUT, static_proj_size=STATIC_PROJ_SIZE,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, warmup_frac=WARMUP_FRAC,
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
