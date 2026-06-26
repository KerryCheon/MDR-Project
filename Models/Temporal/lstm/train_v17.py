"""
LSTM v17 — per-station target normalization on top of v9 architecture.

Hypothesis: The ~10-point val→test R² gap (v9: 0.845→0.747) is partly
driven by station-level mean shifts between training years (2017-2020) and
test years (2023-2025). Subtracting each station's train-period mean from
the target and predicting the residual should reduce the systematic
component of the error.

Architecture: identical to v9 (BiLSTM + additive temporal attention pooling,
10-day window, 19 time + 11 static features).

Key differences from v9:
  - Compute station_means from train split only (no leakage).
  - Subtract station mean from TARGET in all splits before building datasets.
  - Train on residuals; reconstruct at eval by adding mean back.
  - Save station_means.json for inference-time use.
  - Metrics reported in both residual space and reconstructed (absolute) space.

Usage:
    python -m Models.Temporal.lstm.train_v17
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

DATA_DIR = REPO_ROOT / "data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs_v17"
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

SEQ_LEN       = 10
TRAIN_STRIDE  = 1
HIDDEN_SIZE   = 80
NUM_LAYERS    = 2
DROPOUT       = 0.3
PROJ_SIZE     = 56
BATCH_SIZE    = 256
LR            = 1e-3
WEIGHT_DECAY  = 2e-3
HUBER_DELTA   = 0.05
MAX_EPOCHS    = 250
PATIENCE      = 35
GRAD_CLIP     = 1.0
SEED          = 42


# ── model (identical to v9) ──────────────────────────────────────────────────

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


# ── utilities ────────────────────────────────────────────────────────────────

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
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        return dict(r2=float("nan"), rmse=float("nan"), ubrmse=float("nan"),
                    bias=float("nan"), mae=float("nan"), q90=float("nan"), n=0)
    err = yp - yt
    bias   = float(err.mean())
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    ubrmse = float(np.sqrt(np.mean((err - bias) ** 2)))
    mae    = float(np.mean(np.abs(err)))
    q90    = float(np.quantile(np.abs(err), 0.90))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
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
        ax.plot(val_losses,   label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (residual)")
        ax.set_title(f"BiLSTM+Attn v17 — per-station normalization (seq_len={SEQ_LEN})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "loss_curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped ({e})")


# ── per-station normalization helpers ────────────────────────────────────────

def compute_station_means(train_df: pd.DataFrame) -> dict:
    """Return {station_id: mean_target} computed on the train split only."""
    return train_df.groupby("station_id")[TARGET].mean().to_dict()


def apply_station_demean(df: pd.DataFrame, station_means: dict) -> pd.DataFrame:
    """Subtract per-station mean from TARGET column. Unknown stations get 0."""
    out = df.copy()
    offsets = out["station_id"].map(station_means).fillna(0.0)
    out[TARGET] = out[TARGET] - offsets
    return out


def build_offset_array(df: pd.DataFrame, station_means: dict,
                       seq_len: int, stride: int = 1) -> np.ndarray:
    """Build array of per-sequence station offsets matching _build_sequences order.

    Must mirror the exact groupby + sort logic in dataset._build_sequences so
    that offset[i] corresponds to sequence i in the Dataset.
    """
    offsets = []
    for station, grp in df.groupby("station_id", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        mean = station_means.get(station, 0.0)
        n_seqs = len(range(seq_len, len(grp), stride))
        offsets.extend([mean] * n_seqs)
    return np.array(offsets, dtype=np.float32)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    print(f"[load] train={train_df.shape}  val={val_df.shape}  test={test_df.shape}")

    missing = [c for c in ALL_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ── per-station normalization (train means only) ──────────────────────────
    station_means = compute_station_means(train_df)
    n_stations = len(station_means)
    overall_mean = float(np.mean(list(station_means.values())))
    print(f"[v17] station means computed: {n_stations} stations  "
          f"grand mean={overall_mean:.4f}  "
          f"range=[{min(station_means.values()):.4f}, {max(station_means.values()):.4f}]")

    # Save station means for inference-time reconstruction
    with open(OUT_DIR / "station_means.json", "w") as f:
        json.dump(station_means, f, indent=2)
    print(f"[saved] station_means.json")

    # Build offset arrays BEFORE de-meaning (original target values needed)
    off_tr = build_offset_array(train_df, station_means, SEQ_LEN, stride=TRAIN_STRIDE)
    off_va = build_offset_array(val_df,   station_means, SEQ_LEN, stride=1)
    off_te = build_offset_array(test_df,  station_means, SEQ_LEN, stride=1)

    # De-mean targets
    train_df = apply_station_demean(train_df, station_means)
    val_df   = apply_station_demean(val_df,   station_means)
    test_df  = apply_station_demean(test_df,  station_means)

    # ── feature preprocessing (unchanged from v9) ─────────────────────────────
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

    # Sanity-check offset array lengths match dataset sizes
    assert len(off_tr) == len(ds_train), \
        f"Offset/dataset mismatch: {len(off_tr)} vs {len(ds_train)}"
    assert len(off_va) == len(ds_val), \
        f"Offset/dataset mismatch: {len(off_va)} vs {len(ds_val)}"
    assert len(off_te) == len(ds_test), \
        f"Offset/dataset mismatch: {len(off_te)} vs {len(ds_test)}"
    print(f"[offsets] train={len(off_tr)}  val={len(off_va)}  test={len(off_te)}  OK")

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
    print(f"[v17] BiLSTM+Attn  {len(TIME_FEATURES)} time + {len(STATIC_FEATURES)} static  "
          f"seq_len={SEQ_LEN}  per-station normalization")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, epochs=MAX_EPOCHS, steps_per_epoch=len(loader_train),
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3,
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_rmse = math.inf
    best_epoch    = -1
    patience_ctr  = 0
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
        val_rmse = math.sqrt(float(np.mean((y_true_val - y_pred_val) ** 2)))

        train_losses.append(train_loss)
        val_losses.append(val_rmse ** 2)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch    = epoch
            patience_ctr  = 0
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

    # ── evaluate in both residual and reconstructed space ────────────────────
    results = {}
    loaders_offsets = [
        ("train", loader_train, off_tr),
        ("val",   loader_val,   off_va),
        ("test",  loader_test,  off_te),
    ]

    print("\n  --- Residual space (what the model trained on) ---")
    for name, loader, offsets in loaders_offsets:
        y_true_res, y_pred_res = predict_loader(model, loader, device)
        m_res = compute_metrics(y_true_res, y_pred_res)
        results[f"{name}_residual"] = m_res
        print(f"  {name:5s} [res]   R2={m_res['r2']:.4f}  RMSE={m_res['rmse']:.5f}  "
              f"ubRMSE={m_res['ubrmse']:.5f}  Bias={m_res['bias']:+.5f}  "
              f"MAE={m_res['mae']:.5f}  Q90={m_res['q90']:.5f}")

    print("\n  --- Reconstructed space (absolute, comparable to v7–v16) ---")
    for name, loader, offsets in loaders_offsets:
        y_true_res, y_pred_res = predict_loader(model, loader, device)
        y_true_abs = y_true_res + offsets
        y_pred_abs = y_pred_res + offsets
        m_abs = compute_metrics(y_true_abs, y_pred_abs)
        results[f"{name}_reconstructed"] = m_abs
        print(f"  {name:5s} [abs]   R2={m_abs['r2']:.4f}  RMSE={m_abs['rmse']:.5f}  "
              f"ubRMSE={m_abs['ubrmse']:.5f}  Bias={m_abs['bias']:+.5f}  "
              f"MAE={m_abs['mae']:.5f}  Q90={m_abs['q90']:.5f}")

    print(f"\n  v9 baseline:  test R2=0.7446  RMSE=0.04760  Bias=-0.01090")

    results["config"] = dict(
        variant="v17_bilstm_attn_per_station_norm",
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        proj_size=PROJ_SIZE, batch_size=BATCH_SIZE, lr=LR,
        weight_decay=WEIGHT_DECAY, huber_delta=HUBER_DELTA,
        scheduler="onecycle(max_lr=3*lr,pct_start=0.1)",
        time_features=TIME_FEATURES, static_features=STATIC_FEATURES,
        n_features=len(ALL_FEATURES), n_params=n_params,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
        n_station_means=n_stations, station_mean_grand=overall_mean,
        normalization="per_station_train_mean_subtracted",
    )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT_DIR / 'metrics.json'}")
    save_loss_curve(train_losses, val_losses)


if __name__ == "__main__":
    main()
