"""
LSTM training pipeline for derived_8.0 soil moisture dataset.

Usage:
    python -m Models.Temporal.lstm.train
  or
    python Models/Temporal/lstm/train.py

Outputs (written to Models/Temporal/lstm/outputs/):
    best_model.pt   — best checkpoint (lowest val RMSE)
    metrics.json    — final train / val / test metrics
    loss_curve.png  — training curve
"""

import json
import math
import os
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

from Models.Temporal.lstm.dataset import TARGET, DROP_COLS, build_datasets
from Models.Temporal.lstm.model import LSTMRegressor

# config
DATA_DIR   = REPO_ROOT / "Temporal/Pipeline/data/splits/derived_8.0"
OUT_DIR    = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEQ_LEN       = 30       # look-back window (days)
TRAIN_STRIDE  = 1        # step between training sequences
PROJ_SIZE     = 496      # 496 features passed to LSTM
HIDDEN_SIZE   = 64
NUM_LAYERS    = 1        # single layer
DROPOUT       = 0.5
BATCH_SIZE    = 256      # large batches
LR            = 1e-4
WEIGHT_DECAY  = 3e-3
HUBER_DELTA   = 0.05
MAX_EPOCHS    = 300
PATIENCE      = 40
GRAD_CLIP     = 1.0
SEED          = 42


# helpers
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


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS]


def _clean_inf(X: np.ndarray) -> np.ndarray:
    """Replace ±inf with NaN so the imputer handles them uniformly."""
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    return X


def fit_preprocessors(train_df: pd.DataFrame, feature_cols: list):
    """Fit imputer + scaler on train features only."""
    X_tr = _clean_inf(train_df[feature_cols].to_numpy(dtype=np.float32))
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr)
    scaler = StandardScaler()
    scaler.fit(X_tr)
    return imputer, scaler


def apply_preprocessors(df: pd.DataFrame, feature_cols: list, imputer, scaler) -> pd.DataFrame:
    """Return df with feature columns imputed and scaled (in-place copy)."""
    out = df.copy()
    X = _clean_inf(out[feature_cols].to_numpy(dtype=np.float32))
    X = imputer.transform(X)
    X = scaler.transform(X)
    out[feature_cols] = X
    return out


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
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        all_pred.append(preds)
        all_true.append(y_batch.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def save_loss_curve(train_losses: list, val_losses: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="train MSE")
        ax.plot(val_losses,   label="val MSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("LSTM Training Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "loss_curve.png", dpi=120)
        plt.close(fig)
        print(f"[plot] saved -> {OUT_DIR / 'loss_curve.png'}")
    except Exception as e:
        print(f"[plot] skipped ({e})")


# main
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Load
    train_df, val_df, test_df = load_data()
    feature_cols = get_feature_cols(train_df)
    n_features = len(feature_cols)
    print(f"[features] {n_features} feature columns")

    # Preprocess (fit only on train)
    imputer, scaler = fit_preprocessors(train_df, feature_cols)
    train_df = apply_preprocessors(train_df, feature_cols, imputer, scaler)
    val_df   = apply_preprocessors(val_df,   feature_cols, imputer, scaler)
    test_df  = apply_preprocessors(test_df,  feature_cols, imputer, scaler)

    #Build sequence datasets
    ds_train, ds_val, ds_test = build_datasets(
        train_df, val_df, test_df, feature_cols, SEQ_LEN, train_stride=TRAIN_STRIDE
    )
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=device.type == "cuda")
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    # Model
    model = LSTMRegressor(
        n_features=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj_size=PROJ_SIZE,
    ).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    #  Training loop
    best_val_rmse = math.inf
    best_epoch    = -1
    patience_ctr  = 0
    train_losses, val_losses = [], []

    print(f"\n[train] starting — seq_len={SEQ_LEN}  stride={TRAIN_STRIDE}  proj={PROJ_SIZE}  hidden={HIDDEN_SIZE}  layers={NUM_LAYERS}  dropout={DROPOUT}")
    print(f"        batch={BATCH_SIZE}  lr={LR}  wd={WEIGHT_DECAY}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        # train
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in loader_train:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * len(y_batch)
        train_mse = running_loss / len(ds_train)

        # val 
        y_true_val, y_pred_val = predict_loader(model, loader_val, device)
        val_mse  = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)

        scheduler.step(val_mse)
        train_losses.append(train_mse)
        val_losses.append(val_mse)

        # early stopping 
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
                f"train_mse={train_mse:.5f}  val_rmse={val_rmse:.5f}  "
                f"best={best_val_rmse:.5f} (ep{best_epoch})  lr={lr_now:.2e}"
            )

        if patience_ctr >= PATIENCE:
            print(f"\n[early stop] no improvement for {PATIENCE} epochs — stopping at epoch {epoch}")
            break

    # Evaluate best checkpoint on all splits
    print(f"\n[eval] loading best checkpoint (epoch {best_epoch}, val_rmse={best_val_rmse:.5f})")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device, weights_only=True))

    results = {}
    for name, loader in [("train", loader_train), ("val", loader_val), ("test", loader_test)]:
        y_true, y_pred = predict_loader(model, loader, device)
        m = compute_metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:5s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  bias={m['bias']:+.5f}  n={m['n']}")

    results["config"] = dict(
        seq_len=SEQ_LEN, proj_size=PROJ_SIZE, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT, batch_size=BATCH_SIZE,
        lr=LR, weight_decay=WEIGHT_DECAY, train_stride=TRAIN_STRIDE, huber_delta=HUBER_DELTA,
        best_epoch=best_epoch, best_val_rmse=best_val_rmse,
        n_features=n_features,
    )

    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] metrics  -> {metrics_path}")
    print(f"[saved] model    -> {OUT_DIR / 'best_model.pt'}")

    save_loss_curve(train_losses, val_losses)


if __name__ == "__main__":
    main()
