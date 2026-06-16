"""
LSTM v11 - v9 + multi-head temporal attention, feature attention, and
two extra rolling features.

v9 won test R^2 0.747 with single-head additive attention. v11 explores
whether more expressive attention helps:

  - Multi-head temporal attention pooling (4 heads) over the BiLSTM
    output. Each head learns a different sequence-level summary; the
    concatenated context is fed to the head.
  - A feature-attention layer (FA) before the LSTM, ported from v8.
    Wang 2024 reports FA + temporal attention together is more stable
    than either alone.
  - Two extra rolling features that ranked just below v9's three in
    the XGBoost importance: V_rollmin_LST_modis_kobs30 (rank 2) and
    SMAP_sm_am_interp_rollrange7 (rank 38). Still nowhere near the
    496-feature collapse threshold.
  - hidden_size 80 -> 96 to give the multi-head split (96/4=24 per
    head) reasonable capacity.

Same 10-day window as v9 to keep that variable controlled.

Usage:
    python -m Models.Temporal.lstm.train_v11
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
OUT_DIR  = Path(__file__).parent / "outputs_v11"
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
    # v9's three lag features:
    "SMAP_sm_pm_interp_ema02",
    "V_ema_LST_modis_kobs7",
    "V_rollmean_G_API_kobs14",
    # v11 additions (next two top XGBoost features):
    "V_rollmin_LST_modis_kobs30",
    "SMAP_sm_am_interp_rollrange7",
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
HIDDEN_SIZE   = 96
NUM_LAYERS    = 2
DROPOUT       = 0.35
PROJ_SIZE     = 64
N_HEADS       = 4
ATTN_HIDDEN   = 32
BATCH_SIZE    = 256
LR            = 7e-4
WEIGHT_DECAY  = 3e-3
HUBER_DELTA   = 0.05
MAX_EPOCHS    = 250
PATIENCE      = 40
GRAD_CLIP     = 1.0
SEED          = 42


class FeatureAttention(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=-1)
        # multiply by F so feature mass averages 1 (preserves magnitudes)
        return x * weights * x.size(-1)


class MultiHeadTemporalAttn(nn.Module):
    """Per-head additive attention scores; concat head contexts."""

    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0, "dim must divide evenly across heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.score = nn.Linear(self.head_dim, 1)
        self.proj  = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape
        h = x.view(B, S, self.n_heads, self.head_dim)         # (B,S,H,d)
        scores  = self.score(h).squeeze(-1)                   # (B,S,H)
        weights = torch.softmax(scores, dim=1)                # over time
        ctx = (h * weights.unsqueeze(-1)).sum(dim=1)          # (B,H,d)
        ctx = ctx.reshape(B, D)
        return self.proj(ctx)


class BiLSTMMultiHead(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout,
                 proj_size, n_heads, attn_hidden):
        super().__init__()
        self.feat_attn = FeatureAttention(n_features, attn_hidden)
        self.proj = nn.Sequential(
            nn.Linear(n_features, proj_size),
            nn.LayerNorm(proj_size),
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
        self.mh_attn = MultiHeadTemporalAttn(bi, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(bi, bi // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bi // 2, 1),
        )

    def forward(self, x):
        x = self.feat_attn(x)
        b, s, f = x.shape
        x = self.proj(x.reshape(b * s, f)).reshape(b, s, -1)
        out, _ = self.lstm(x)
        ctx = self.mh_attn(out)
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
        ax.set_title(f"BiLSTM MultiHead+FA v11 (seq_len={SEQ_LEN})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "loss_curve.png", dpi=120)
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
    print(f"[v11] BiLSTM+FA+MHA  {len(TIME_FEATURES)} time + {len(STATIC_FEATURES)} static  seq_len={SEQ_LEN}  heads={N_HEADS}")

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

    model = BiLSTMMultiHead(
        n_features=len(ALL_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj_size=PROJ_SIZE,
        n_heads=N_HEADS,
        attn_hidden=ATTN_HIDDEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] BiLSTMMultiHead  params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
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
            print(f"  ep {epoch:3d}  train={train_loss:.5f}  val_rmse={val_rmse:.5f}  best={best_val_rmse:.5f} (ep{best_epoch})  lr={lr_now:.2e}")

        if patience_ctr >= PATIENCE:
            print(f"\n[early stop] no improvement for {PATIENCE} epochs at epoch {epoch}")
            break

    print(f"\n[eval] best epoch {best_epoch}  val_rmse={best_val_rmse:.5f}")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device, weights_only=True))

    results = {}
    for name, loader in [("train", loader_train), ("val", loader_val), ("test", loader_test)]:
        y_true, y_pred = predict_loader(model, loader, device)
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:5s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  bias={m['bias']:+.5f}  n={m['n']}")

    results["config"] = dict(
        variant="v11_bilstm_multihead_fa",
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        proj_size=PROJ_SIZE, n_heads=N_HEADS, attn_hidden=ATTN_HIDDEN,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        huber_delta=HUBER_DELTA, scheduler="cosine_annealing",
        time_features=TIME_FEATURES, static_features=STATIC_FEATURES,
        n_features=len(ALL_FEATURES), n_params=n_params,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
    )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {OUT_DIR / 'metrics.json'}")
    save_loss_curve(train_losses, val_losses)


if __name__ == "__main__":
    main()
