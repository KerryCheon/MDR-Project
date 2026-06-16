"""
LSTM v15 - Bidirectional GRU encoder + temporal attention + multi-task
auxiliary supervision.

Hypothesis
----------
Previous variants (v7-v12, all LSTM-based) plateau around test R^2=0.747,
with a persistent ~10-point val->test gap that suggests the encoder is
overfitting to per-station idiosyncrasies in the training split. Two
changes are introduced jointly to attack this:

1. GRU instead of LSTM. East Java 2025 reported R^2=0.80 on day-10 soil
   moisture forecasting with a multivariate GRU. GRUs have 2 gates instead
   of LSTM's 4, fewer parameters per hidden unit, and tend to generalize
   better on small temporal datasets like ours (~6.9K training rows).

2. Multi-task auxiliary head. Following Wang 2024 (HESS) which showed that
   auxiliary supervision on related signals regularizes a shared recurrent
   encoder, we add a second head that predicts SMAP_sm_pm_interp at the
   anchor day. SMAP is a noisy but well-behaved proxy for surface soil
   moisture; predicting it back from a feature set that already contains
   it functions as a denoising-autoencoder-style regularizer. The shared
   GRU backbone is forced to learn representations of wetness state and
   recent rainfall history that explain *both* targets, rather than
   memorizing station-specific quirks of the primary target.

Total loss
----------
    L = huber(pred_primary, y_sm) + 0.3 * huber(pred_aux, y_smap)

Validation/test metrics are reported on the primary head only -- the
auxiliary head exists purely as a regularizer.

Usage:
    python -m Models.Temporal.lstm.train_v15
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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.dataset import TARGET   # "soil_moisture_5cm"

DATA_DIR = REPO_ROOT / "Temporal/Pipeline/data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs_v15"
OUT_DIR.mkdir(exist_ok=True)


# Auxiliary target: SMAP pm interpolated. Same unit space as primary, so
# it does NOT get scaled (kept in raw soil-moisture units, like the primary).
AUX_TARGET = "SMAP_sm_pm_interp"


TIME_FEATURES = [
    "precip_mm",
    "s1_vv", "s1_vh",
    "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "LST_modis",
    "F_NDVI", "F_NDMI",
    "E_SAR_ratio",
    "SMAP_sm_am_interp", "SMAP_sm_pm_interp",   # aux target is also an input -- DAE-style
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
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
PROJ_SIZE     = 48
BATCH_SIZE    = 256
LR            = 7e-4
WEIGHT_DECAY  = 2e-3
HUBER_DELTA   = 0.05
AUX_WEIGHT    = 0.3
MAX_EPOCHS    = 250
PATIENCE      = 35
GRAD_CLIP     = 1.0
SEED          = 42


# ---------------------------------------------------------------------------
# Multi-task dataset
# ---------------------------------------------------------------------------

def _build_multitask_sequences(df: pd.DataFrame,
                                feature_cols: list,
                                primary_col: str,
                                aux_col: str,
                                seq_len: int,
                                stride: int = 1):
    """Slide a (seq_len) window over each station, emit (X, y_primary, y_aux).

    Mirrors the logic in dataset._build_sequences but emits two targets
    (both at the anchor day = index `i`).
    """
    X_seqs, y_prim, y_aux = [], [], []

    for station, grp in df.groupby("station_id", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        X  = grp[feature_cols].to_numpy(dtype=np.float32)
        yp = grp[primary_col].to_numpy(dtype=np.float32)
        ya = grp[aux_col].to_numpy(dtype=np.float32)

        for i in range(seq_len, len(grp), stride):
            X_seqs.append(X[i - seq_len : i])   # (seq_len, n_features)
            y_prim.append(yp[i])
            y_aux.append(ya[i])

    if not X_seqs:
        raise ValueError("No sequences built -- check seq_len vs. data length.")

    return (
        np.stack(X_seqs, axis=0),
        np.asarray(y_prim, dtype=np.float32),
        np.asarray(y_aux,  dtype=np.float32),
    )


class SoilMoistureMultiTaskDataset(Dataset):
    """PyTorch Dataset returning (X, y_primary, y_aux) triples."""

    def __init__(self, X: np.ndarray, y_primary: np.ndarray, y_aux: np.ndarray):
        assert len(X) == len(y_primary) == len(y_aux)
        self.X = X
        self.y_primary = y_primary
        self.y_aux = y_aux

    def __len__(self):
        return len(self.y_primary)

    def __getitem__(self, idx):
        return self.X[idx], self.y_primary[idx], self.y_aux[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiGRUAttnMultiTask(nn.Module):
    """Bidirectional GRU + temporal attention pool + two heads sharing
    the same pooled context vector.

    Forward returns (pred_primary, pred_aux).
    """

    def __init__(self, n_features, hidden_size, num_layers, dropout, proj_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, proj_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        bi = hidden_size * 2

        # Additive temporal attention (same shape as v9).
        self.attn = nn.Sequential(
            nn.Linear(bi, bi),
            nn.Tanh(),
            nn.Linear(bi, 1),
        )
        self.dropout = nn.Dropout(dropout)

        # Two heads, separate weights, identical structure, shared context.
        self.head_primary = nn.Sequential(
            nn.Linear(bi, bi // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bi // 2, 1),
        )
        self.head_aux = nn.Sequential(
            nn.Linear(bi, bi // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bi // 2, 1),
        )

    def forward(self, x):
        b, s, f = x.shape
        x = self.proj(x.reshape(b * s, f)).reshape(b, s, -1)
        out, _ = self.gru(x)                       # (B, S, 2H)
        scores = self.attn(out).squeeze(-1)        # (B, S)
        weights = torch.softmax(scores, dim=-1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)   # (B, 2H)
        ctx = self.dropout(ctx)
        pred_primary = self.head_primary(ctx).squeeze(-1)
        pred_aux     = self.head_aux(ctx).squeeze(-1)
        return pred_primary, pred_aux


# ---------------------------------------------------------------------------
# Utilities (copied/adapted from v9/v12 conventions)
# ---------------------------------------------------------------------------

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
    """Median imputer + StandardScaler fit on TRAIN features only."""
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
    """Run model over a loader, return (y_true_primary, y_pred_primary).

    Auxiliary outputs are discarded -- validation/test report primary only.
    """
    model.eval()
    pr, tr = [], []
    for X_batch, y_prim, _y_aux in loader:
        X_batch = X_batch.to(device)
        pred_primary, _ = model(X_batch)
        pr.append(pred_primary.cpu().numpy())
        tr.append(y_prim.numpy())
    return np.concatenate(tr), np.concatenate(pr)


def save_loss_curve(train_losses, val_losses, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="train (total)")
        ax.plot(val_losses,   label="val (primary mse)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped ({e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        raise ValueError(f"Missing feature columns: {missing}")
    for col in (TARGET, AUX_TARGET):
        if col not in train_df.columns:
            raise ValueError(f"Missing target column: {col}")
    print(f"[v15] BiGRU+Attn+MultiTask  {len(TIME_FEATURES)} time + "
          f"{len(STATIC_FEATURES)} static  seq_len={SEQ_LEN}  aux={AUX_TARGET}  "
          f"aux_weight={AUX_WEIGHT}")

    # Snapshot raw aux target BEFORE scaling -- aux target stays in raw soil-moisture units.
    train_aux_raw = train_df[AUX_TARGET].copy()
    val_aux_raw   = val_df[AUX_TARGET].copy()
    test_aux_raw  = test_df[AUX_TARGET].copy()

    imputer, scaler = fit_preprocessors(train_df, ALL_FEATURES)
    train_df = apply_preprocessors(train_df, ALL_FEATURES, imputer, scaler)
    val_df   = apply_preprocessors(val_df,   ALL_FEATURES, imputer, scaler)
    test_df  = apply_preprocessors(test_df,  ALL_FEATURES, imputer, scaler)

    # Restore the auxiliary target column to its raw (unscaled) values --
    # SMAP_sm_pm_interp is both an input (scaled) and an output (raw units,
    # same space as primary target). We snapshot before scaling and write
    # the raw values back under a parallel name used only by the dataset
    # builder. The scaled copy in ALL_FEATURES remains untouched.
    train_df["_aux_target_raw"] = train_aux_raw.to_numpy()
    val_df["_aux_target_raw"]   = val_aux_raw.to_numpy()
    test_df["_aux_target_raw"]  = test_aux_raw.to_numpy()

    # Patch AUX_TARGET reference for the dataset builder by reading "_aux_target_raw".
    # Simplest way: do it inline here rather than mutate global AUX_TARGET.
    def _build(df, stride):
        return _build_multitask_sequences(
            df, ALL_FEATURES, TARGET, "_aux_target_raw", SEQ_LEN, stride=stride)

    X_tr, yp_tr, ya_tr = _build(train_df, TRAIN_STRIDE)
    X_va, yp_va, ya_va = _build(val_df,   1)
    X_te, yp_te, ya_te = _build(test_df,  1)
    print(f"[dataset] train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}  "
          f"(train_stride={TRAIN_STRIDE})  primary={TARGET}  aux={AUX_TARGET}")

    ds_train = SoilMoistureMultiTaskDataset(X_tr, yp_tr, ya_tr)
    ds_val   = SoilMoistureMultiTaskDataset(X_va, yp_va, ya_va)
    ds_test  = SoilMoistureMultiTaskDataset(X_te, yp_te, ya_te)

    pin = device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=pin)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    model = BiGRUAttnMultiTask(
        n_features=len(ALL_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj_size=PROJ_SIZE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] BiGRUAttnMultiTask  params={n_params:,}")

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
        for X_batch, yp_batch, ya_batch in loader_train:
            X_batch  = X_batch.to(device)
            yp_batch = yp_batch.to(device)
            ya_batch = ya_batch.to(device)

            optimizer.zero_grad()
            pred_primary, pred_aux = model(X_batch)

            loss_primary = criterion(pred_primary, yp_batch)

            # The auxiliary target (SMAP) can contain NaNs; mask them out so
            # they don't poison the gradient. Primary target is assumed clean
            # in the splits (filtered upstream), matching v9/v12 behaviour.
            aux_mask = torch.isfinite(ya_batch)
            if aux_mask.any():
                loss_aux = criterion(pred_aux[aux_mask], ya_batch[aux_mask])
            else:
                loss_aux = torch.zeros((), device=device)

            loss = loss_primary + AUX_WEIGHT * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += loss.item() * len(yp_batch)

        train_loss = running / len(ds_train)
        scheduler.step()

        # Val metric: primary-head RMSE only.
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
    print("\n[eval] split metrics (primary head only):")
    for name, loader in [("train", loader_train), ("val", loader_val), ("test", loader_test)]:
        y_true, y_pred = predict_loader(model, loader, device)
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:5s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.5f}  ubRMSE={m['ubrmse']:.5f}  "
              f"Bias={m['bias']:+.5f}  MAE={m['mae']:.5f}  Q90={m['q90']:.5f}  n={m['n']}")

    results["config"] = dict(
        variant="v15_bigru_attn_multitask",
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        proj_size=PROJ_SIZE, batch_size=BATCH_SIZE, lr=LR,
        weight_decay=WEIGHT_DECAY, huber_delta=HUBER_DELTA,
        aux_target=AUX_TARGET, aux_weight=AUX_WEIGHT,
        scheduler="cosine_annealing",
        max_epochs=MAX_EPOCHS, patience=PATIENCE, grad_clip=GRAD_CLIP, seed=SEED,
        time_features=TIME_FEATURES, static_features=STATIC_FEATURES,
        n_features=len(ALL_FEATURES), n_params=n_params,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
    )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {OUT_DIR / 'metrics.json'}")
    save_loss_curve(train_losses, val_losses,
                    OUT_DIR / "loss_curve.png",
                    f"BiGRU+Attn+MultiTask v15 (seq_len={SEQ_LEN}, aux_w={AUX_WEIGHT})")


if __name__ == "__main__":
    main()
