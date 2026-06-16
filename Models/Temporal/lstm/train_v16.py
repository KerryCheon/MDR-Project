"""
LSTM v16 - BiLSTM+Attention with comprehensive engineered lag features.

Hypothesis: v9 (best so far, test R^2 = 0.747) plateaued because it only
exposes 3 hand-picked lag features. Wang 2024 emphasized that even with
attention-based architectures, explicit lag/rolling features remain a
strong source of signal -- provided they are the *right* ones (well-
transformed, low-skew, and physically meaningful).

We keep the v9 architecture exactly (BiLSTM + additive temporal attention)
so the comparison isolates feature engineering. The only change vs v9 is
proj_size 56 -> 64 to accommodate the larger input dimension.

Engineered features (computed per-station, in-script):
  - feat_log1p_rain_7d   : log1p of 7-day rolling precip sum
                           (recent wetting; log avoids G_ right-tail collapse)
  - feat_log1p_rain_30d  : log1p of 30-day rolling precip sum
                           (seasonal wetness state)
  - feat_days_since_rain : days since last day with precip > 1mm, capped 30
                           (direct drying-state proxy; log1p before scaling
                            because still right-skewed even capped)
  - feat_ndmi_14d        : 14-day mean of F_NDMI
                           (slow vegetation moisture buffer)
  - feat_smap_pm_7d      : 7-day mean of SMAP_sm_pm_interp
                           (smooths SMAP noise into recent-state summary)
  - feat_smap_pm_30d     : 30-day mean of SMAP_sm_pm_interp
                           (longer moisture memory)
  - feat_smap_am_pm_diff_7d : 7-day mean of (AM - PM) SMAP soil moisture
                              (diurnal moisture change averaged)

Lit references: MDR Report 4/29 Table 3 ("lag features we could try").

Pre-computed features carried over from v5 that v9 didn't include:
  V_rollmin_LST_modis_kobs30, V_rollmean_s2_b11_kobs7,
  V_rollstd_F_NDMI_kobs7, SMAP_sm_interp_rollstd7

Lag features from v9 kept as-is:
  SMAP_sm_pm_interp_ema02, V_ema_LST_modis_kobs7, V_rollmean_G_API_kobs14

Total time features: 30 (was 19 in v9).

Usage:
    python -m Models.Temporal.lstm.train_v16
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.dataset import TARGET, build_datasets

DATA_DIR = REPO_ROOT / "Temporal/Pipeline/data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs_v16"
OUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Engineered feature names (kept in one place for reproducibility)
# ----------------------------------------------------------------------
ENGINEERED_FEATURES = [
    "feat_log1p_rain_7d",
    "feat_log1p_rain_30d",
    "feat_days_since_rain",
    "feat_ndmi_14d",
    "feat_smap_pm_7d",
    "feat_smap_pm_30d",
    "feat_smap_am_pm_diff_7d",
]

# Pre-computed lag/rolling features used by v9
V9_LAG_FEATURES = [
    "SMAP_sm_pm_interp_ema02",
    "V_ema_LST_modis_kobs7",
    "V_rollmean_G_API_kobs14",
]

# Additional pre-computed lag features that v5 used but v9 didn't
V5_EXTRA_LAG_FEATURES = [
    "V_rollmin_LST_modis_kobs30",
    "V_rollmean_s2_b11_kobs7",
    "V_rollstd_F_NDMI_kobs7",
    "SMAP_sm_interp_rollstd7",
]

# Base v9 time features (no lag features yet)
BASE_TIME_FEATURES = [
    "precip_mm",
    "s1_vv", "s1_vh",
    "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "LST_modis",
    "F_NDVI", "F_NDMI",
    "E_SAR_ratio",
    "SMAP_sm_am_interp", "SMAP_sm_pm_interp",
    "SMAP_sm_interp_mask",
    "sin_year", "cos_year",
]

TIME_FEATURES = (
    BASE_TIME_FEATURES
    + V9_LAG_FEATURES
    + V5_EXTRA_LAG_FEATURES
    + ENGINEERED_FEATURES
)

STATIC_FEATURES = [
    "latitude", "longitude",
    "elev", "slope", "aspect",
    "K_sand_clay_ratio_b0", "K_clay_plus_sand_b0",
    "K_slope_sin", "K_slope_cos", "K_aspect_sin", "K_aspect_cos",
]

ALL_FEATURES = TIME_FEATURES + STATIC_FEATURES


# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
SEQ_LEN       = 10
TRAIN_STRIDE  = 1
HIDDEN_SIZE   = 80
NUM_LAYERS    = 2
DROPOUT       = 0.3
PROJ_SIZE     = 64           # was 56 in v9; slightly larger for more features
BATCH_SIZE    = 256
LR            = 7e-4
WEIGHT_DECAY  = 2e-3
HUBER_DELTA   = 0.05
MAX_EPOCHS    = 250
PATIENCE      = 40
GRAD_CLIP     = 1.0
SEED          = 42


# ----------------------------------------------------------------------
# Architecture (identical to v9)
# ----------------------------------------------------------------------
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
        out, _ = self.lstm(x)                  # (B, S, 2H)
        scores = self.attn(out).squeeze(-1)    # (B, S)
        weights = torch.softmax(scores, dim=-1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)   # (B, 2H)
        return self.head(self.dropout(ctx)).squeeze(-1)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _days_since_rain(precip: np.ndarray, threshold: float = 1.0, cap: int = 30) -> np.ndarray:
    """Per-station scan: days since the last day with precip > threshold.

    Looks only at PAST values (today's precip resets to 0 if it rains today).
    Initialized at 0 on the first day of the station record (no prior history).
    Capped at `cap` to suppress right-tail blowups during dry stations.
    """
    n = len(precip)
    out = np.zeros(n, dtype=np.float32)
    counter = 0
    for i in range(n):
        out[i] = min(counter, cap)
        # update counter for next step: if it rained today, reset; else +1
        if np.isfinite(precip[i]) and precip[i] > threshold:
            counter = 0
        else:
            counter += 1
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the engineered rolling/lag features per station, sorted by date.

    Rolling windows look ONLY backwards (pandas default), so the t-th value
    of each engineered feature uses precip/SMAP/NDMI from days <= t. This
    matches the LSTM's causal sequence usage and avoids leakage. Grouping
    by station_id avoids any cross-station contamination.

    NOTE: We assume the splits are already station-disjoint OR contiguous-in-
    time per station; either way the rolling is local to (station, split).
    """
    df = df.copy()
    # Ensure deterministic order: sort by (station, date) once.
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    g = df.groupby("station_id", sort=False, group_keys=False)

    # Rain features (log1p to tame heavy right tail)
    rain_sum_7  = g["precip_mm"].apply(lambda s: s.rolling(window=7,  min_periods=1).sum())
    rain_sum_30 = g["precip_mm"].apply(lambda s: s.rolling(window=30, min_periods=1).sum())
    df["feat_log1p_rain_7d"]  = np.log1p(rain_sum_7.to_numpy())
    df["feat_log1p_rain_30d"] = np.log1p(rain_sum_30.to_numpy())

    # Days since last rain (per-station scan, capped at 30)
    dsr_parts = []
    for _, grp in df.groupby("station_id", sort=False):
        precip = grp["precip_mm"].to_numpy(dtype=np.float32)
        dsr_parts.append(pd.Series(
            _days_since_rain(precip, threshold=1.0, cap=30),
            index=grp.index,
        ))
    df["feat_days_since_rain"] = pd.concat(dsr_parts).sort_index().to_numpy()

    # NDMI 14d mean
    ndmi_14 = g["F_NDMI"].apply(lambda s: s.rolling(window=14, min_periods=1).mean())
    df["feat_ndmi_14d"] = ndmi_14.to_numpy()

    # SMAP rolling means
    smap_pm_7  = g["SMAP_sm_pm_interp"].apply(lambda s: s.rolling(window=7,  min_periods=1).mean())
    smap_pm_30 = g["SMAP_sm_pm_interp"].apply(lambda s: s.rolling(window=30, min_periods=1).mean())
    df["feat_smap_pm_7d"]  = smap_pm_7.to_numpy()
    df["feat_smap_pm_30d"] = smap_pm_30.to_numpy()

    # SMAP diurnal difference, 7d mean
    am_pm_diff = df["SMAP_sm_am_interp"] - df["SMAP_sm_pm_interp"]
    df["_tmp_am_pm_diff"] = am_pm_diff
    diff_7 = (df.groupby("station_id", sort=False, group_keys=False)["_tmp_am_pm_diff"]
                .apply(lambda s: s.rolling(window=7, min_periods=1).mean()))
    df["feat_smap_am_pm_diff_7d"] = diff_7.to_numpy()
    df = df.drop(columns=["_tmp_am_pm_diff"])

    # Pre-scale transform: log1p the (still right-skewed) days-since-rain
    # so the StandardScaler doesn't get pulled by the long dry stretches.
    df["feat_days_since_rain"] = np.log1p(df["feat_days_since_rain"].to_numpy())

    return df


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


def save_loss_curve(train_losses, val_losses):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"BiLSTM+Attn v16 (seq_len={SEQ_LEN}, engineered features)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "loss_curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped ({e})")


def _summarize_engineered(df: pd.DataFrame, tag: str):
    print(f"[engineered:{tag}] NaN/inf and range summary:")
    for c in ENGINEERED_FEATURES:
        x = df[c].to_numpy(dtype=np.float64)
        finite = np.isfinite(x)
        n_nan = int(np.isnan(x).sum())
        n_inf = int(np.isinf(x).sum())
        if finite.any():
            xf = x[finite]
            print(f"  {c:28s}  nan={n_nan:5d}  inf={n_inf:3d}  "
                  f"min={xf.min():.4f}  mean={xf.mean():.4f}  max={xf.max():.4f}")
        else:
            print(f"  {c:28s}  nan={n_nan:5d}  inf={n_inf:3d}  (no finite values)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    print(f"[load] train={train_df.shape}  val={val_df.shape}  test={test_df.shape}")

    # Engineer rolling/lag features per split (within-station, look-back only).
    # Doing it per split is safe because rolling windows are station-local;
    # there is no cross-split or cross-station contamination.
    print("[v16] engineering rolling/lag features per split (groupby station, sort by date)")
    train_df = engineer_features(train_df)
    val_df   = engineer_features(val_df)
    test_df  = engineer_features(test_df)

    _summarize_engineered(train_df, "train")
    _summarize_engineered(val_df,   "val")
    _summarize_engineered(test_df,  "test")

    # Safety check: every required column must be present after engineering.
    missing_time   = [c for c in TIME_FEATURES   if c not in train_df.columns]
    missing_static = [c for c in STATIC_FEATURES if c not in train_df.columns]
    missing = missing_time + missing_static
    assert not missing, f"Missing columns after engineering: {missing}"

    print(f"[v16] BiLSTM+Attn  {len(TIME_FEATURES)} time + {len(STATIC_FEATURES)} static  "
          f"seq_len={SEQ_LEN}")

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

    model = BiLSTMAttn(
        n_features=len(ALL_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj_size=PROJ_SIZE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] BiLSTMAttn  params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6,
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
            running += loss.item() * len(y_batch)
        train_loss = running / len(ds_train)
        scheduler.step()

        y_true_val, y_pred_val = predict_loader(model, loader_val, device)
        val_mse  = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)

        train_losses.append(train_loss)
        val_losses.append(val_mse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {epoch:3d}  train={train_loss:.5f}  val_rmse={val_rmse:.5f}  "
                  f"best={best_val_rmse:.5f} (ep{best_epoch})  lr={lr_now:.2e}")

        if patience_ctr >= PATIENCE:
            print(f"\n[early stop] no improvement for {PATIENCE} epochs at epoch {epoch}")
            break

    print(f"\n[eval] best epoch {best_epoch}  val_rmse={best_val_rmse:.5f}")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device, weights_only=True))

    results = {}
    print("\n[eval] split metrics:")
    for name, loader in [("train", loader_train), ("val", loader_val), ("test", loader_test)]:
        y_true, y_pred = predict_loader(model, loader, device)
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:5s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.5f}  ubRMSE={m['ubrmse']:.5f}  "
              f"Bias={m['bias']:+.5f}  MAE={m['mae']:.5f}  Q90={m['q90']:.5f}  n={m['n']}")

    results["config"] = dict(
        variant="v16_bilstm_attn_engineered_lags",
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        proj_size=PROJ_SIZE, batch_size=BATCH_SIZE, lr=LR,
        weight_decay=WEIGHT_DECAY, huber_delta=HUBER_DELTA,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, grad_clip=GRAD_CLIP, seed=SEED,
        scheduler="cosine_annealing(T_max=max_epochs, eta_min=1e-6)",
        time_features=TIME_FEATURES,
        static_features=STATIC_FEATURES,
        engineered_features=ENGINEERED_FEATURES,
        v9_lag_features=V9_LAG_FEATURES,
        v5_extra_lag_features=V5_EXTRA_LAG_FEATURES,
        base_time_features=BASE_TIME_FEATURES,
        n_features=len(ALL_FEATURES), n_params=n_params,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
    )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {OUT_DIR / 'metrics.json'}")
    save_loss_curve(train_losses, val_losses)


if __name__ == "__main__":
    main()
