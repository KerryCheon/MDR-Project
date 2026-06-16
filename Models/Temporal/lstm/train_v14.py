"""
LSTM v14 - Multi-Scale Parallel Window BiLSTM with per-branch attention.

Hypothesis
----------
Every prior variant (v7-v12) commits to a single window length, forcing a
tradeoff between short-horizon dynamics and slower seasonal/baseline state.
Soil moisture is driven by both:

  * Immediate wet/dry dynamics: rainfall pulses, evapotranspiration on
    hot days, freeze/thaw -- best captured by a 5-day window.
  * Mid-range mass balance: cumulative recent forcing, antecedent
    wetness -- the 10-day window that v9 (best baseline, test R^2=0.747)
    is built around.
  * Slow seasonal / wetness baseline: vegetation phenology, deeper
    soil column memory -- captured by a 20-day window.

Architecture
------------
Three BiLSTM branches run in parallel, each with its own input projection
and its own additive temporal attention pool. All three branches anchor
on the same final day; the difference is how far back they look.

    x: (B, 20, F)
      |
      |-- last  5 -- proj_s -> BiLSTM_s -> attn_s -> ctx_s (2H)
      |-- last 10 -- proj_m -> BiLSTM_m -> attn_m -> ctx_m (2H)
      |-- last 20 -- proj_l -> BiLSTM_l -> attn_l -> ctx_l (2H)
      |
      concat -> Linear(6H, 3H) -> ReLU -> Dropout -> Linear(3H, 1)

The fusion head learns which timescale to lean on per prediction; the
attention pools learn *which days* matter within each timescale.

Branches do NOT share input projections: each scale can specialize its
feature compression (e.g. the 5d branch may weight precipitation more
aggressively, the 20d branch may weight EMA/rolling features more).

The dataset only needs to be built once at the largest window length
(20d). Each branch slices its own suffix from x in the forward pass --
no extra dataloading cost and the three contexts are guaranteed to
share the same anchor day.

Usage:
    python -m Models.Temporal.lstm.train_v14
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
OUT_DIR  = Path(__file__).parent / "outputs_v14"
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


# Window scales -- longest defines the dataset window.
WINDOWS       = (5, 10, 20)
SEQ_LEN       = max(WINDOWS)     # 20
TRAIN_STRIDE  = 1

# Per-branch architecture (small on purpose -- 3 branches multiply param cost).
HIDDEN_SIZE   = 32
PROJ_SIZE     = 40
NUM_LAYERS    = 1
DROPOUT       = 0.3
HEAD_HIDDEN   = 96               # 192 -> 96 -> 1

# Optimization
BATCH_SIZE    = 256
LR            = 7e-4
WEIGHT_DECAY  = 2e-3
HUBER_DELTA   = 0.05
MAX_EPOCHS    = 250
PATIENCE      = 35
GRAD_CLIP     = 1.0
SEED          = 42


class _ScaleBranch(nn.Module):
    """One BiLSTM branch with its own input projection and attention pool."""

    def __init__(self, n_features, hidden_size, proj_size, dropout, num_layers=1):
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

    def forward(self, x):
        # x: (B, S, F)
        b, s, f = x.shape
        x = self.proj(x.reshape(b * s, f)).reshape(b, s, -1)
        out, _ = self.lstm(x)                              # (B, S, 2H)
        scores = self.attn(out).squeeze(-1)                # (B, S)
        weights = torch.softmax(scores, dim=-1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)     # (B, 2H)
        return ctx


class MultiScaleBiLSTM(nn.Module):
    """Three parallel BiLSTM branches (5/10/20 day) fused by an MLP head."""

    def __init__(self, n_features, hidden_size, proj_size, dropout,
                 windows=WINDOWS, num_layers=1, head_hidden=HEAD_HIDDEN):
        super().__init__()
        self.windows = tuple(windows)
        # One independent branch per scale -- no parameter sharing.
        self.branches = nn.ModuleList([
            _ScaleBranch(n_features, hidden_size, proj_size, dropout, num_layers)
            for _ in self.windows
        ])
        bi = hidden_size * 2
        fused_dim = bi * len(self.windows)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        # x: (B, SEQ_LEN, F)   SEQ_LEN must be >= max(self.windows).
        contexts = []
        for w, branch in zip(self.windows, self.branches):
            x_w = x[:, -w:, :]                # slice the suffix for this scale
            contexts.append(branch(x_w))
        fused = torch.cat(contexts, dim=-1)   # (B, 2H * n_scales)
        return self.head(self.dropout(fused)).squeeze(-1)


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


def save_loss_curve(train_losses, val_losses, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(title); ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped ({e})")


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
    print(f"[v14] MultiScaleBiLSTM  {len(TIME_FEATURES)} time + {len(STATIC_FEATURES)} static  "
          f"windows={WINDOWS}  seq_len={SEQ_LEN}")

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

    n_features = len(ALL_FEATURES)
    model = MultiScaleBiLSTM(
        n_features=n_features,
        hidden_size=HIDDEN_SIZE,
        proj_size=PROJ_SIZE,
        dropout=DROPOUT,
        windows=WINDOWS,
        num_layers=NUM_LAYERS,
        head_hidden=HEAD_HIDDEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MultiScaleBiLSTM  branches={len(WINDOWS)}  hidden={HIDDEN_SIZE}  "
          f"proj={PROJ_SIZE}  head={HEAD_HIDDEN}  params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
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
        variant="v14_multiscale_parallel_bilstm",
        windows=list(WINDOWS), seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        proj_size=PROJ_SIZE, head_hidden=HEAD_HIDDEN,
        batch_size=BATCH_SIZE, lr=LR,
        weight_decay=WEIGHT_DECAY, huber_delta=HUBER_DELTA,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, grad_clip=GRAD_CLIP, seed=SEED,
        scheduler="cosine_annealing",
        time_features=TIME_FEATURES, static_features=STATIC_FEATURES,
        n_features=n_features, n_params=n_params,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
        branches_share_input_projection=False,
    )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {OUT_DIR / 'metrics.json'}")
    save_loss_curve(
        train_losses, val_losses,
        OUT_DIR / "loss_curve.png",
        f"v14 MultiScaleBiLSTM  windows={WINDOWS}  h={HIDDEN_SIZE}  proj={PROJ_SIZE}",
    )


if __name__ == "__main__":
    main()
