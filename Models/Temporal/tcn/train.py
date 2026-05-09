"""
TCN training pipeline for soil moisture with a physics-inspired mass-balance
soft constraint (inspired by PGDL literature).

Loss = Huber(pred, y) * year_weights  +  λ_phys * physics_penalty(pred, prev_sm, precip_today)

physics_penalty:
    Δ = pred - prev_sm
    penalise Δ > +0.02 when precip_today < 1 mm  (can't gain moisture w/o rain)
    penalise Δ < -0.10 when precip_today > 1 mm  (a big drop on a rainy day is
                                                   implausible)
Samples with NaN prev_sm are masked out of the physics term.

Usage:
    python Models/Temporal/tcn/train.py

Outputs → Models/Temporal/tcn/outputs/:
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
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# --- path setup ---
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.tcn.dataset import TARGET, PRECIP_COL, build_datasets
from Models.Temporal.tcn.model   import TCNRegressor

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "Temporal/Pipeline/data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Same feature set as the LSTM baseline — we isolate the architectural +
# physics contributions rather than adding new features.
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
CHANNELS         = 64
KERNEL_SIZE      = 3
DILATIONS        = (1, 2, 4, 8, 16)
DROPOUT          = 0.2
STATIC_PROJ_SIZE = 32
HEAD_HIDDEN      = 64
POOL             = "mean_max"
BATCH_SIZE       = 128
LR               = 5e-4
WEIGHT_DECAY     = 3e-3
HUBER_DELTA      = 0.05
LAMBDA_PHYS      = 0.1
PRECIP_DRY_MM    = 1.0     # below this → "no rain"
GAIN_TOL         = 0.02    # allowed noise ΔSM without rain
DROP_TOL         = 0.10    # a bigger single-day drop with rain is implausible
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


def apply_preprocessors(df: pd.DataFrame, all_feature_cols: list, imputer, scaler) -> pd.DataFrame:
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


def physics_penalty(
    pred:          torch.Tensor,   # (B,)
    prev_sm:       torch.Tensor,   # (B,)  may contain NaN
    precip_today:  torch.Tensor,   # (B,)  raw mm
    gain_tol: float = GAIN_TOL,
    drop_tol: float = DROP_TOL,
    dry_mm:   float = PRECIP_DRY_MM,
):
    """Mass-balance soft constraint.

    Returns
    -------
    penalty  : scalar tensor (MSE of violations across valid samples; 0 if none)
    n_valid  : int  — how many samples contributed
    """
    valid = torch.isfinite(prev_sm)
    if valid.sum().item() == 0:
        return pred.new_zeros(()), 0

    pred_v   = pred[valid]
    prev_v   = prev_sm[valid]
    precip_v = precip_today[valid]

    delta = pred_v - prev_v
    is_dry = (precip_v < dry_mm).float()
    is_wet = (precip_v >= dry_mm).float()

    # gained moisture on a dry day → penalty
    no_rain_gain = F.relu(delta - gain_tol) * is_dry
    # big drop on a rainy day → penalty
    wet_drop     = F.relu(-delta - drop_tol) * is_wet

    violations = torch.cat([no_rain_gain, wet_drop], dim=0)
    penalty = (violations ** 2).mean()
    return penalty, int(valid.sum().item())


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for batch in loader:
        x_time, x_static, _precip, _prev_sm, _yrs, y_batch = batch
        x_time   = x_time.to(device)
        x_static = x_static.to(device)
        preds    = model(x_time, x_static).cpu().numpy()
        all_pred.append(preds)
        all_true.append(y_batch.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def save_loss_curve(train_losses, val_losses, phys_losses):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="train total loss")
        ax.plot(val_losses,   label="val MSE")
        ax.plot(phys_losses,  label="train physics penalty", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("TCN + Physics Training Curve")
        ax.set_yscale("log")
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
    for col in [TARGET, PRECIP_COL, "station_id", "date"]:
        if col not in train_df.columns:
            raise ValueError(f"Required column `{col}` missing from train.csv")
    missing = [c for c in all_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Features missing from dataset: {missing}")
    print(f"[features] {len(TIME_FEATURES)} time  +  {len(STATIC_FEATURES)} static")

    # Raw copies — used for un-normalised precip and the prev_sm series.
    train_raw = train_df.copy()
    val_raw   = val_df.copy()
    test_raw  = test_df.copy()

    imputer, scaler = fit_preprocessors(train_df, all_cols)
    train_df = apply_preprocessors(train_df, all_cols, imputer, scaler)
    val_df   = apply_preprocessors(val_df,   all_cols, imputer, scaler)
    test_df  = apply_preprocessors(test_df,  all_cols, imputer, scaler)

    ds_train, ds_val, ds_test = build_datasets(
        train_scaled=train_df, val_scaled=val_df, test_scaled=test_df,
        train_raw=train_raw,   val_raw=val_raw,   test_raw=test_raw,
        time_cols=TIME_FEATURES, static_cols=STATIC_FEATURES,
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
    )

    year_max = int(ds_train.years.max())
    print(f"[temporal weighting] beta={TEMPORAL_BETA}  year_max={year_max}")

    pin = device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    model = TCNRegressor(
        n_time=len(TIME_FEATURES),
        n_static=len(STATIC_FEATURES),
        channels=CHANNELS,
        kernel_size=KERNEL_SIZE,
        dilations=DILATIONS,
        dropout=DROPOUT,
        static_proj_size=STATIC_PROJ_SIZE,
        head_hidden=HEAD_HIDDEN,
        pool=POOL,
    ).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA, reduction="none")

    best_val_rmse = math.inf
    best_epoch    = -1
    patience_ctr  = 0
    train_losses, val_losses, phys_losses = [], [], []

    print(
        f"\n[train] seq_len={SEQ_LEN}  stride={TRAIN_STRIDE}  "
        f"channels={CHANNELS}  kernel={KERNEL_SIZE}  dilations={DILATIONS}  "
        f"dropout={DROPOUT}  pool={POOL}"
    )
    print(
        f"        batch={BATCH_SIZE}  lr={LR}  wd={WEIGHT_DECAY}  "
        f"λ_phys={LAMBDA_PHYS}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}\n"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_total  = 0.0
        running_data   = 0.0
        running_phys   = 0.0
        n_seen         = 0

        for batch in loader_train:
            x_time, x_static, precip_seq, prev_sm, yr_batch, y_batch = batch
            x_time     = x_time.to(device)
            x_static   = x_static.to(device)
            precip_seq = precip_seq.to(device)
            prev_sm    = prev_sm.to(device)
            yr_batch   = yr_batch.to(device)
            y_batch    = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_time, x_static)

            per_sample = criterion(pred, y_batch)
            w          = compute_temporal_weights(yr_batch, year_max).to(device)
            data_loss  = (per_sample * w).mean()

            precip_today = precip_seq[:, -1]
            phys_loss, _n_valid = physics_penalty(pred, prev_sm, precip_today)

            total = data_loss + LAMBDA_PHYS * phys_loss

            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            bsz = len(y_batch)
            running_total += total.item()     * bsz
            running_data  += data_loss.item() * bsz
            running_phys  += phys_loss.item() * bsz
            n_seen        += bsz

        train_loss = running_total / n_seen
        data_loss_epoch = running_data / n_seen
        phys_loss_epoch = running_phys / n_seen

        y_true_val, y_pred_val = predict_loader(model, loader_val, device)
        val_mse  = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)

        scheduler.step(val_mse)
        train_losses.append(train_loss)
        val_losses.append(val_mse)
        phys_losses.append(phys_loss_epoch)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch    = epoch
            patience_ctr  = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch:3d}/{MAX_EPOCHS}  "
                f"total={train_loss:.5f}  data={data_loss_epoch:.5f}  "
                f"phys={phys_loss_epoch:.2e}  val_rmse={val_rmse:.5f}  "
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

    # Final physics-penalty sanity check on training set using best model
    model.eval()
    total_phys = 0.0
    total_n    = 0
    with torch.no_grad():
        for batch in loader_train:
            x_time, x_static, precip_seq, prev_sm, _yr, _y = batch
            x_time     = x_time.to(device)
            x_static   = x_static.to(device)
            precip_seq = precip_seq.to(device)
            prev_sm    = prev_sm.to(device)
            pred = model(x_time, x_static)
            p, n = physics_penalty(pred, prev_sm, precip_seq[:, -1])
            total_phys += p.item() * max(n, 1)
            total_n    += max(n, 1)
    final_phys = total_phys / max(total_n, 1)
    print(f"[physics] final penalty (train, best model) = {final_phys:.3e}")

    results["config"] = dict(
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        channels=CHANNELS, kernel_size=KERNEL_SIZE, dilations=list(DILATIONS),
        dropout=DROPOUT, static_proj_size=STATIC_PROJ_SIZE, head_hidden=HEAD_HIDDEN,
        pool=POOL, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        huber_delta=HUBER_DELTA, lambda_phys=LAMBDA_PHYS,
        precip_dry_mm=PRECIP_DRY_MM, gain_tol=GAIN_TOL, drop_tol=DROP_TOL,
        temporal_beta=TEMPORAL_BETA,
        time_features=TIME_FEATURES, static_features=STATIC_FEATURES,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
        final_physics_penalty=final_phys,
    )

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] metrics -> {metrics_path}")
    print(f"[saved] model   -> {OUT_DIR / 'best_model.pt'}")

    save_loss_curve(train_losses, val_losses, phys_losses)


if __name__ == "__main__":
    main()
