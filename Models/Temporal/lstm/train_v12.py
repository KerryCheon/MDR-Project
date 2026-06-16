"""
LSTM v12 - Smaller v9 + grid search + ubRMSE / Q90 metrics.

Three changes from v9:

1. Reduced architecture: 1-layer BiLSTM (was 2), hidden=48 (was 80),
   proj=40 (was 56). About 50K params, roughly 5-6x smaller than v9.
   Motivation: v9 already overfits hard (train R^2 0.93 vs test 0.75),
   so trimming capacity is a more direct lever than adding regularization
   (which v10 tried and failed at).

2. New metrics on every split:
     - ubRMSE = sqrt(RMSE^2 - bias^2)  (standard in SMAP / SM literature;
       measures error variance after removing systematic bias)
     - Q90 = 90th percentile of |residuals|  (worst-decile error magnitude)
     - Bias = mean(pred - true)  (kept, surfaced explicitly)

3. Hyperparameter grid scan (a manual analogue to sklearn's
   GridSearchCV -- sklearn's version doesn't work on PyTorch models,
   and k-fold CV is too expensive for DL, so we use a single train/val
   split per config). We sweep hidden_size x dropout x lr (2x2x2 = 8
   configs), train each with shorter patience, pick the lowest val
   RMSE, then retrain the winner with full max_epochs.

Usage:
    python -m Models.Temporal.lstm.train_v12
"""

import json
import math
import sys
from pathlib import Path
from itertools import product

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
OUT_DIR  = Path(__file__).parent / "outputs_v12"
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


# Fixed hyperparameters
SEQ_LEN       = 10
TRAIN_STRIDE  = 1
NUM_LAYERS    = 1            # was 2 in v9
PROJ_SIZE     = 40           # was 56 in v9
WEIGHT_DECAY  = 2e-3
HUBER_DELTA   = 0.05
BATCH_SIZE    = 256
GRAD_CLIP     = 1.0
SEED          = 42

# Grid search axes (8 configs)
GRID = {
    "hidden_size": [40, 56],
    "dropout":     [0.25, 0.35],
    "lr":          [5e-4, 1e-3],
}
GRID_MAX_EPOCHS = 60
GRID_PATIENCE   = 15

# Final training (best config)
FINAL_MAX_EPOCHS = 250
FINAL_PATIENCE   = 35


class SmallBiLSTMAttn(nn.Module):
    """Single-layer BiLSTM with additive attention pooling."""

    def __init__(self, n_features, hidden_size, dropout, proj_size, num_layers=1):
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
    # ubRMSE = sqrt(mean((err - mean(err))^2)) -- error stddev after debiasing
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


def train_one(cfg, n_features, ds_train, loader_train, loader_val,
              device, max_epochs, patience, verbose, save_path=None):
    """Train one config; return dict with best val_rmse, best epoch, loss histories."""
    set_seed(SEED)
    model = SmallBiLSTMAttn(
        n_features=n_features,
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        proj_size=PROJ_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_rmse = math.inf
    best_epoch = -1
    best_state = None
    patience_ctr = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
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
        val_mse = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)
        train_losses.append(train_loss)
        val_losses.append(val_mse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    ep {epoch:3d}  train={train_loss:.5f}  val_rmse={val_rmse:.5f}  best={best_val_rmse:.5f} (ep{best_epoch})  lr={lr_now:.2e}")

        if patience_ctr >= patience:
            if verbose:
                print(f"    [early stop] at epoch {epoch}")
            break

    if save_path is not None and best_state is not None:
        torch.save(best_state, save_path)

    return dict(
        best_val_rmse=best_val_rmse,
        best_epoch=best_epoch,
        n_params=n_params,
        train_losses=train_losses,
        val_losses=val_losses,
        best_state=best_state,
    )


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
    print(f"[v12] SmallBiLSTM+Attn  {len(ALL_FEATURES)} features  seq_len={SEQ_LEN}")

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

    # Grid search
    keys = list(GRID.keys())
    grid_combos = [dict(zip(keys, vals)) for vals in product(*[GRID[k] for k in keys])]
    print(f"\n[grid] scanning {len(grid_combos)} configs  max_epochs={GRID_MAX_EPOCHS}  patience={GRID_PATIENCE}")
    print(f"       grid: {GRID}")

    grid_results = []
    for i, cfg in enumerate(grid_combos, 1):
        print(f"\n  [{i}/{len(grid_combos)}] {cfg}")
        out = train_one(cfg, n_features, ds_train, loader_train, loader_val,
                        device, GRID_MAX_EPOCHS, GRID_PATIENCE, verbose=False)
        print(f"    -> best_val_rmse={out['best_val_rmse']:.5f}  best_epoch={out['best_epoch']}  params={out['n_params']:,}")
        grid_results.append(dict(cfg=cfg, best_val_rmse=out["best_val_rmse"],
                                  best_epoch=out["best_epoch"], n_params=out["n_params"]))

    grid_results.sort(key=lambda r: r["best_val_rmse"])
    best = grid_results[0]
    print("\n[grid] ranking:")
    for r in grid_results:
        print(f"  val_rmse={r['best_val_rmse']:.5f}  ep={r['best_epoch']:3d}  params={r['n_params']:,}  cfg={r['cfg']}")
    print(f"\n[grid] winner: {best['cfg']}  val_rmse={best['best_val_rmse']:.5f}")

    # Final training
    print(f"\n[final] retraining winner with max_epochs={FINAL_MAX_EPOCHS}  patience={FINAL_PATIENCE}")
    final_out = train_one(
        best["cfg"], n_features, ds_train, loader_train, loader_val, device,
        FINAL_MAX_EPOCHS, FINAL_PATIENCE, verbose=True,
        save_path=OUT_DIR / "best_model.pt",
    )
    print(f"\n[final] best epoch {final_out['best_epoch']}  val_rmse={final_out['best_val_rmse']:.5f}  params={final_out['n_params']:,}")

    # Evaluate
    model = SmallBiLSTMAttn(
        n_features=n_features,
        hidden_size=best["cfg"]["hidden_size"],
        dropout=best["cfg"]["dropout"],
        proj_size=PROJ_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device, weights_only=True))

    results = {}
    print("\n[eval] split metrics:")
    for name, loader in [("train", loader_train), ("val", loader_val), ("test", loader_test)]:
        y_true, y_pred = predict_loader(model, loader, device)
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:5s}  R2={m['r2']:.4f}  RMSE={m['rmse']:.5f}  ubRMSE={m['ubrmse']:.5f}  "
              f"Bias={m['bias']:+.5f}  MAE={m['mae']:.5f}  Q90={m['q90']:.5f}  n={m['n']}")

    results["grid_search"] = grid_results
    results["best_config"] = best["cfg"]
    results["config"] = dict(
        variant="v12_small_bilstm_grid",
        seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE,
        num_layers=NUM_LAYERS, proj_size=PROJ_SIZE,
        weight_decay=WEIGHT_DECAY, huber_delta=HUBER_DELTA, batch_size=BATCH_SIZE,
        scheduler="cosine_annealing", grid=GRID,
        grid_max_epochs=GRID_MAX_EPOCHS, grid_patience=GRID_PATIENCE,
        final_max_epochs=FINAL_MAX_EPOCHS, final_patience=FINAL_PATIENCE,
        time_features=TIME_FEATURES, static_features=STATIC_FEATURES,
        n_features=n_features, n_params=final_out["n_params"],
        best_epoch=final_out["best_epoch"], best_val_rmse=final_out["best_val_rmse"],
    )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT_DIR / 'metrics.json'}")
    save_loss_curve(final_out["train_losses"], final_out["val_losses"],
                    OUT_DIR / "loss_curve.png",
                    f"v12 SmallBiLSTM (h={best['cfg']['hidden_size']}, dr={best['cfg']['dropout']}, lr={best['cfg']['lr']})")


if __name__ == "__main__":
    main()
