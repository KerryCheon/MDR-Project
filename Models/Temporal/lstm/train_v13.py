"""
LSTM v13 - Residual-over-SMAP target reframing.

Hypothesis
----------
v9 (test R^2 = 0.747) and its descendants all show a consistent val->test
gap of 10-20 R^2 points, suggesting the LSTM is wasting capacity
re-learning what SMAP already provides, which leaves it sensitive to
station-level distribution shift between splits.

This variant reframes the regression target as the residual from the
SMAP satellite signal:

    y_residual = soil_moisture_5cm - SMAP_sm_pm_interp

The LSTM predicts that residual. At inference, we reconstruct

    soil_moisture_5cm_hat = predicted_residual + SMAP_sm_pm_interp

This should:
  - Focus model capacity on what is hard (deviation from satellite
    baseline) instead of the easy bulk signal.
  - Reduce sensitivity to station-level offsets if SMAP captures them.
  - Act as a strong prior: if the LSTM learns nothing useful, predictions
    still hover near the SMAP baseline rather than collapsing to the
    training-set mean.

Architecture matches v9: BiLSTM (2 layers, hidden=80) + additive temporal
attention pooling, projection=56, dropout=0.3, Huber(delta=0.05), 10-day
window. SMAP_sm_pm_interp is RETAINED as an input feature (it is both the
prior used for reconstruction and a genuinely informative signal).

Implementation notes
--------------------
1. We do NOT modify dataset.py. We define a tiny custom sequence builder
   inline that returns (X, y_residual, smap_anchor) per window, where
   smap_anchor is the raw (un-standardized) SMAP value at the anchor
   timestep used for reconstruction.
2. The SMAP anchor is captured BEFORE preprocessing/scaling, so it is in
   the natural soil-moisture units.
3. Target residual is NOT scaled (HuberLoss with delta=0.05 is calibrated
   to raw soil-moisture-scale residuals).
4. All metrics (R^2, RMSE, ubRMSE, Bias, MAE, Q90) are reported on the
   RECONSTRUCTED predictions, not the residuals.
5. Sanity check: at startup we report the residual MAE of a zero-residual
   baseline (pred=SMAP). A clean signal should yield MAE ~0.04. A value
   near zero would indicate a target leak.

Usage
-----
    python -m Models.Temporal.lstm.train_v13
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

from Models.Temporal.lstm.dataset import TARGET  # "soil_moisture_5cm"

DATA_DIR = REPO_ROOT / "Temporal/Pipeline/data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs_v13"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Features (identical to v9)
# ---------------------------------------------------------------------------

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

# Column used as the residual anchor / reconstruction prior.
SMAP_COL = "SMAP_sm_pm_interp"


# ---------------------------------------------------------------------------
# Hyperparameters (v9 baseline; LR lowered slightly because residuals are
# smaller-magnitude than the raw target, so a milder schedule trains more
# stably).
# ---------------------------------------------------------------------------

SEQ_LEN       = 10
TRAIN_STRIDE  = 1
HIDDEN_SIZE   = 80
NUM_LAYERS    = 2
DROPOUT       = 0.3
PROJ_SIZE     = 56
BATCH_SIZE    = 256
LR            = 7e-4
WEIGHT_DECAY  = 2e-3
HUBER_DELTA   = 0.05
MAX_EPOCHS    = 250
PATIENCE      = 35
GRAD_CLIP     = 1.0
SEED          = 42


# ---------------------------------------------------------------------------
# Model (identical to v9 BiLSTMAttn)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Custom sequence builder (returns smap_anchor in parallel for reconstruction)
# ---------------------------------------------------------------------------

class ResidualSeqDataset(Dataset):
    """Holds (X, y_residual, smap_anchor) triples.

    smap_anchor is the raw (un-standardized) SMAP_sm_pm_interp value at the
    anchor timestep (index i, matching the target row of the window).
    """

    def __init__(self, X: np.ndarray, y_res: np.ndarray, smap_anchor: np.ndarray, y_true: np.ndarray):
        assert len(X) == len(y_res) == len(smap_anchor) == len(y_true)
        self.X = X                # (N, seq_len, n_features)  preprocessed
        self.y_res = y_res        # (N,) residual target (raw units)
        self.smap_anchor = smap_anchor  # (N,) raw SMAP at anchor row
        self.y_true = y_true      # (N,) raw soil_moisture_5cm at anchor row

    def __len__(self):
        return len(self.y_res)

    def __getitem__(self, idx):
        # Only X and y_res are needed during training/backprop. Anchors and
        # truth are looked up via the parallel arrays in eval.
        return self.X[idx], self.y_res[idx]


def build_residual_sequences(
    df_processed: pd.DataFrame,
    df_raw: pd.DataFrame,
    feature_cols: list,
    seq_len: int,
    stride: int = 1,
):
    """Build sliding-window sequences with residual targets and SMAP anchors.

    df_processed : DataFrame with input features already imputed + scaled.
                   Its `feature_cols` are read for X. Its TARGET column
                   should already be the residual (y_residual).
    df_raw       : DataFrame BEFORE preprocessing, used to look up the raw
                   SMAP_sm_pm_interp at the anchor and the raw
                   soil_moisture_5cm ground truth.

    Returns numpy arrays (X_seqs, y_res, smap_anchor, y_true).
    Drops windows where smap_anchor is non-finite (cannot reconstruct).
    """
    X_seqs, y_res, smap_anchor, y_true = [], [], [], []

    # Align by (station_id, date) so processed and raw stay in lockstep.
    for station, grp_p in df_processed.groupby("station_id", sort=False):
        grp_p = grp_p.sort_values("date").reset_index(drop=True)
        grp_r = (
            df_raw[df_raw["station_id"] == station]
            .sort_values("date")
            .reset_index(drop=True)
        )
        if len(grp_p) != len(grp_r):
            raise RuntimeError(
                f"Station {station}: processed/raw length mismatch "
                f"({len(grp_p)} vs {len(grp_r)})."
            )

        X = grp_p[feature_cols].to_numpy(dtype=np.float32)
        y_residual = grp_p[TARGET].to_numpy(dtype=np.float32)        # residual
        smap_raw = grp_r[SMAP_COL].to_numpy(dtype=np.float32)        # anchor source
        y_raw = grp_r[TARGET].to_numpy(dtype=np.float32)             # raw truth

        for i in range(seq_len, len(grp_p), stride):
            anchor = smap_raw[i]
            truth = y_raw[i]
            # Skip if we can't reconstruct (no SMAP) or no truth (missing target).
            if not np.isfinite(anchor) or not np.isfinite(truth):
                continue
            # Residual must also be finite (it is truth - anchor; both finite here).
            r = y_residual[i]
            if not np.isfinite(r):
                continue
            X_seqs.append(X[i - seq_len : i])
            y_res.append(r)
            smap_anchor.append(anchor)
            y_true.append(truth)

    if not X_seqs:
        raise ValueError("No sequences built -- check seq_len vs. data length.")

    return (
        np.stack(X_seqs, axis=0),
        np.array(y_res, dtype=np.float32),
        np.array(smap_anchor, dtype=np.float32),
        np.array(y_true, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Preprocessing utilities (identical to v9 / v12)
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


# ---------------------------------------------------------------------------
# Metrics (copied from train_v12.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Inference (returns predicted residuals only; reconstruction is done outside)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_residuals(model, loader, device):
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
        ax.set_ylabel("Loss (residual scale)")
        ax.set_title(f"BiLSTM+Attn v13 residual-over-SMAP (seq_len={SEQ_LEN})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "loss_curve.png", dpi=120)
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
    for needed in (SMAP_COL, TARGET, "station_id", "date"):
        if needed not in train_df.columns:
            raise ValueError(f"Missing required column: {needed}")

    print(f"[v13] residual-over-SMAP  {len(TIME_FEATURES)} time + "
          f"{len(STATIC_FEATURES)} static  seq_len={SEQ_LEN}")

    # Keep raw frames around (BEFORE scaling) for residual + reconstruction.
    raw_train, raw_val, raw_test = train_df.copy(), val_df.copy(), test_df.copy()

    # ---- Sanity check on raw values: residual MAE of a zero-residual model
    #      (i.e. predict SMAP directly). Should be ~0.04 on a reasonable
    #      sensor dataset. Near-zero would mean the target leaks into SMAP.
    for name, df in [("train", raw_train), ("val", raw_val), ("test", raw_test)]:
        m = np.isfinite(df[TARGET]) & np.isfinite(df[SMAP_COL])
        err = (df.loc[m, TARGET] - df.loc[m, SMAP_COL]).to_numpy()
        baseline_mae = float(np.mean(np.abs(err))) if err.size else float("nan")
        baseline_r2 = float(
            1.0 - np.sum(err ** 2) /
            np.sum((df.loc[m, TARGET] - df.loc[m, TARGET].mean()) ** 2)
        ) if err.size else float("nan")
        print(f"[sanity:{name}] SMAP-as-prediction  MAE={baseline_mae:.5f}  "
              f"R2={baseline_r2:+.4f}  n={int(m.sum())}")
        if baseline_mae < 1e-6:
            raise RuntimeError(
                f"[LEAK?] {name} residual is identically zero -- "
                f"check that SMAP_sm_pm_interp is not the target itself."
            )

    # ---- Fit preprocessors on training input features only.
    imputer, scaler = fit_preprocessors(raw_train, ALL_FEATURES)
    train_proc = apply_preprocessors(raw_train, ALL_FEATURES, imputer, scaler)
    val_proc   = apply_preprocessors(raw_val,   ALL_FEATURES, imputer, scaler)
    test_proc  = apply_preprocessors(raw_test,  ALL_FEATURES, imputer, scaler)

    # ---- Replace the target column in processed frames with the residual.
    #      Residual uses the RAW soil_moisture_5cm and RAW SMAP, both in
    #      natural units. Where either is missing, residual becomes NaN
    #      and the sequence builder will drop that window.
    for name, proc, raw in [
        ("train", train_proc, raw_train),
        ("val",   val_proc,   raw_val),
        ("test",  test_proc,  raw_test),
    ]:
        residual = raw[TARGET].astype(np.float32) - raw[SMAP_COL].astype(np.float32)
        proc[TARGET] = residual
        # Helpful summary for sanity.
        finite = residual[np.isfinite(residual)]
        if finite.size:
            print(f"[residual:{name}] n_finite={finite.size}  "
                  f"mean={finite.mean():+.5f}  std={finite.std():.5f}  "
                  f"min={finite.min():+.5f}  max={finite.max():+.5f}")

    # ---- Build datasets with residual target + raw SMAP anchor.
    X_tr, y_tr, anchor_tr, ytrue_tr = build_residual_sequences(
        train_proc, raw_train, ALL_FEATURES, SEQ_LEN, stride=TRAIN_STRIDE,
    )
    X_va, y_va, anchor_va, ytrue_va = build_residual_sequences(
        val_proc, raw_val, ALL_FEATURES, SEQ_LEN, stride=1,
    )
    X_te, y_te, anchor_te, ytrue_te = build_residual_sequences(
        test_proc, raw_test, ALL_FEATURES, SEQ_LEN, stride=1,
    )
    print(f"[dataset] train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}  "
          f"(train_stride={TRAIN_STRIDE})")

    ds_train = ResidualSeqDataset(X_tr, y_tr, anchor_tr, ytrue_tr)
    ds_val   = ResidualSeqDataset(X_va, y_va, anchor_va, ytrue_va)
    ds_test  = ResidualSeqDataset(X_te, y_te, anchor_te, ytrue_te)

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
    print(f"[model] BiLSTMAttn (residual-target)  params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, epochs=MAX_EPOCHS, steps_per_epoch=len(loader_train),
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3,
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_rmse = math.inf   # measured on RECONSTRUCTED predictions
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

        # Validate on reconstructed-scale RMSE so early-stopping aligns with
        # the actual quantity we care about.
        y_res_true_val, y_res_pred_val = predict_residuals(model, loader_val, device)
        y_pred_val_reco = y_res_pred_val + ds_val.smap_anchor
        y_true_val_reco = ds_val.y_true
        val_mse  = float(np.mean((y_true_val_reco - y_pred_val_reco) ** 2))
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
            print(f"  ep {epoch:3d}  train(res)={train_loss:.5f}  "
                  f"val_rmse(reco)={val_rmse:.5f}  best={best_val_rmse:.5f} "
                  f"(ep{best_epoch})  lr={lr_now:.2e}")

        if patience_ctr >= PATIENCE:
            print(f"\n[early stop] no improvement for {PATIENCE} epochs at epoch {epoch}")
            break

    print(f"\n[eval] best epoch {best_epoch}  val_rmse(reco)={best_val_rmse:.5f}")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device, weights_only=True))

    results = {}
    print("\n[eval] split metrics (on reconstructed soil_moisture_5cm):")
    for name, loader, ds in [
        ("train", loader_train, ds_train),
        ("val",   loader_val,   ds_val),
        ("test",  loader_test,  ds_test),
    ]:
        _, y_res_pred = predict_residuals(model, loader, device)
        y_pred = y_res_pred + ds.smap_anchor
        y_true = ds.y_true
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:5s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.5f}  "
              f"ubRMSE={m['ubrmse']:.5f}  Bias={m['bias']:+.5f}  "
              f"MAE={m['mae']:.5f}  Q90={m['q90']:.5f}  n={m['n']}")

        # Also surface the residual-scale metrics and the SMAP-only baseline
        # on the same windows, for comparison.
        m_smap = compute_metrics(y_true, ds.smap_anchor)
        print(f"  {name:5s}  [SMAP-only baseline]  R2={m_smap['r2']:.4f}  "
              f"RMSE={m_smap['rmse']:.5f}  MAE={m_smap['mae']:.5f}")
        results[f"{name}_smap_baseline"] = m_smap

    results["config"] = dict(
        variant="v13_residual_over_smap",
        target_reframing=f"y_residual = {TARGET} - {SMAP_COL}",
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
        proj_size=PROJ_SIZE, batch_size=BATCH_SIZE, lr=LR,
        weight_decay=WEIGHT_DECAY, huber_delta=HUBER_DELTA,
        scheduler="onecycle(max_lr=3*lr,pct_start=0.1)",
        time_features=TIME_FEATURES, static_features=STATIC_FEATURES,
        smap_anchor_col=SMAP_COL,
        n_features=len(ALL_FEATURES), n_params=n_params,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
    )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {OUT_DIR / 'metrics.json'}")
    save_loss_curve(train_losses, val_losses)


if __name__ == "__main__":
    main()
