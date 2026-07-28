"""
LSTM Training and Frozen Context Vector (CTX) + Head Hidden Extraction Module.

Phase 1: Trains BiLSTMAttn model on derived_8.4 sequence dataset until convergence.
Phase 2: Freezes best_model.pt (eval mode, torch.no_grad()) and extracts:
  - 160-dim ctx vectors (attention-pooled hidden state)
  - 80-dim head_hidden vectors (intermediate after head Linear→ReLU)
for train, val, test splits.
"""

from __future__ import annotations

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

CURRENT_DIR = Path(__file__).resolve().parent
EXP_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = EXP_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from .dataset import TARGET, build_datasets
from .model import BiLSTMAttn

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

SEQ_LEN = 10
TRAIN_STRIDE = 1
HIDDEN_SIZE = 80
NUM_LAYERS = 2
DROPOUT = 0.3
PROJ_SIZE = 56
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 2e-3
HUBER_DELTA = 0.05
MAX_EPOCHS = 250
PATIENCE = 35
GRAD_CLIP = 1.0
SEED = 42


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_inf(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    return X


def fit_preprocessors(train_df: pd.DataFrame, cols: list[str]):
    X = _clean_inf(train_df[cols].to_numpy(dtype=np.float32))
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    scaler.fit(X)
    return imputer, scaler


def apply_preprocessors(df: pd.DataFrame, cols: list[str], imputer, scaler) -> pd.DataFrame:
    out = df.copy()
    X = _clean_inf(out[cols].to_numpy(dtype=np.float32))
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = np.clip(X, -5, 5)
    out[cols] = X
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    pr, tr = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        pr.append(model(X_batch).cpu().numpy())
        tr.append(y_batch.numpy())
    return np.concatenate(tr), np.concatenate(pr)


@torch.no_grad()
def predict_and_extract_ctx(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Phase 2: Evaluates frozen model and extracts context vectors ctx (shape B, 160).
    """
    model.eval()
    pr, tr, ctx_list = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        pred, ctx = model(X_batch, return_ctx=True)
        pr.append(pred.cpu().numpy())
        tr.append(y_batch.numpy())
        ctx_list.append(ctx.cpu().numpy())
    return np.concatenate(tr), np.concatenate(pr), np.concatenate(ctx_list, axis=0)


@torch.no_grad()
def predict_and_extract_head_hidden(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Extract 80-dim head intermediate representations (after head Linear→ReLU).
    Returns (y_true, y_pred, head_hidden) where head_hidden shape is (N, 80).
    """
    model.eval()
    pr, tr, hh_list = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        pred, head_hidden = model(X_batch, return_head_hidden=True)
        pr.append(pred.cpu().numpy())
        tr.append(y_batch.numpy())
        hh_list.append(head_hidden.cpu().numpy())
    return np.concatenate(tr), np.concatenate(pr), np.concatenate(hh_list, axis=0)


@torch.no_grad()
def predict_and_extract_head_pre_relu(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Extract 80-dim head intermediate representations (after head Linear BEFORE ReLU).
    Returns (y_true, y_pred, head_pre_relu) where head_pre_relu shape is (N, 80).
    """
    model.eval()
    pr, tr, hp_list = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        pred, head_pre_relu = model(X_batch, return_head_pre_relu=True)
        pr.append(pred.cpu().numpy())
        tr.append(y_batch.numpy())
        hp_list.append(head_pre_relu.cpu().numpy())
    return np.concatenate(tr), np.concatenate(pr), np.concatenate(hp_list, axis=0)


def extract_head_pre_relu_only(data_dir: Path, artifacts_dir: Path, checkpoint_path: Path):
    """
    Load an existing BiLSTMAttn checkpoint and extract pre-ReLU vectors
    for train, val, test splits without re-training.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pre-ReLU Extraction] Device: {device}", flush=True)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    missing = [c for c in ALL_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns for LSTM: {missing}")

    imputer, scaler = fit_preprocessors(train_df, ALL_FEATURES)
    train_prep = apply_preprocessors(train_df, ALL_FEATURES, imputer, scaler)
    val_prep   = apply_preprocessors(val_df,   ALL_FEATURES, imputer, scaler)
    test_prep  = apply_preprocessors(test_df,  ALL_FEATURES, imputer, scaler)

    ds_train, ds_val, ds_test = build_datasets(
        train_prep, val_prep, test_prep,
        feature_cols=ALL_FEATURES,
        seq_len=SEQ_LEN,
        train_stride=1,
    )

    pin = device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    model = BiLSTMAttn(
        n_features=len(ALL_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj_size=PROJ_SIZE,
    ).to(device)

    print(f"[Pre-ReLU Extraction] Loading checkpoint: {checkpoint_path}", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    _, _, hp_tr = predict_and_extract_head_pre_relu(model, loader_train, device)
    _, _, hp_va = predict_and_extract_head_pre_relu(model, loader_val, device)
    _, _, hp_te = predict_and_extract_head_pre_relu(model, loader_test, device)

    print(f"[Pre-ReLU Extracted] Train: {hp_tr.shape}, Val: {hp_va.shape}, Test: {hp_te.shape}", flush=True)

    np.save(artifacts_dir / "head_pre_relu_train.npy", hp_tr)
    np.save(artifacts_dir / "head_pre_relu_val.npy", hp_va)
    np.save(artifacts_dir / "head_pre_relu_test.npy", hp_te)

    print(f"[Pre-ReLU Saved] artifacts/head_pre_relu_*.npy", flush=True)


def train_lstm_and_extract_ctx(data_dir: Path, artifacts_dir: Path):
    """
    Executes Phase 1 (LSTM training) & Phase 2 (Frozen CTX extraction).
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 1: LSTM Training] Device: {device}", flush=True)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir = EXP_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")
    print(f"[LSTM Load] Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}", flush=True)

    missing = [c for c in ALL_FEATURES if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns for LSTM: {missing}")

    imputer, scaler = fit_preprocessors(train_df, ALL_FEATURES)
    train_prep = apply_preprocessors(train_df, ALL_FEATURES, imputer, scaler)
    val_prep   = apply_preprocessors(val_df,   ALL_FEATURES, imputer, scaler)
    test_prep  = apply_preprocessors(test_df,  ALL_FEATURES, imputer, scaler)

    ds_train, ds_val, ds_test = build_datasets(
        train_prep, val_prep, test_prep,
        feature_cols=ALL_FEATURES,
        seq_len=SEQ_LEN,
        train_stride=TRAIN_STRIDE,
    )

    pin = device.type == "cuda"
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    loader_train_shuffle = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)
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
    print(f"[LSTM Model] BiLSTMAttn params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, epochs=MAX_EPOCHS, steps_per_epoch=len(loader_train_shuffle),
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3,
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_rmse = math.inf
    best_epoch = -1
    patience_ctr = 0
    train_losses, val_losses = [], []
    checkpoint_path = models_dir / "best_lstm_model.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for X_batch, y_batch in loader_train_shuffle:
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
        val_mse  = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)

        train_losses.append(train_loss)
        val_losses.append(val_mse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_ctr += 1

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [LSTM Ep {epoch:3d}] train_loss={train_loss:.5f} val_rmse={val_rmse:.5f} best={best_val_rmse:.5f} (ep{best_epoch}) lr={lr_now:.2e}", flush=True)

        if patience_ctr >= PATIENCE:
            print(f"\n[LSTM Early Stop] No improvement for {PATIENCE} epochs at epoch {epoch}", flush=True)
            break

    print(f"\n[Phase 1 Complete] Best epoch {best_epoch} val_rmse={best_val_rmse:.5f}", flush=True)

    # Phase 2: Load frozen converged model & extract ctx and head_hidden vectors
    print("[Phase 2: Frozen CTX + Head Hidden Extraction] Loading best checkpoint...", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    y_tr_t, y_tr_p, ctx_tr = predict_and_extract_ctx(model, loader_train, device)
    y_va_t, y_va_p, ctx_va = predict_and_extract_ctx(model, loader_val, device)
    y_te_t, y_te_p, ctx_te = predict_and_extract_ctx(model, loader_test, device)

    print(f"[CTX Extracted] Train: {ctx_tr.shape}, Val: {ctx_va.shape}, Test: {ctx_te.shape}", flush=True)

    np.save(artifacts_dir / "ctx_train.npy", ctx_tr)
    np.save(artifacts_dir / "ctx_val.npy", ctx_va)
    np.save(artifacts_dir / "ctx_test.npy", ctx_te)

    # Extract 80-dim head hidden (after head Linear→ReLU)
    _, _, hh_tr = predict_and_extract_head_hidden(model, loader_train, device)
    _, _, hh_va = predict_and_extract_head_hidden(model, loader_val, device)
    _, _, hh_te = predict_and_extract_head_hidden(model, loader_test, device)

    print(f"[Head Hidden Extracted] Train: {hh_tr.shape}, Val: {hh_va.shape}, Test: {hh_te.shape}", flush=True)

    np.save(artifacts_dir / "head_hidden_train.npy", hh_tr)
    np.save(artifacts_dir / "head_hidden_val.npy", hh_va)
    np.save(artifacts_dir / "head_hidden_test.npy", hh_te)

    lstm_metrics = {
        "train": compute_metrics(y_tr_t, y_tr_p),
        "val": compute_metrics(y_va_t, y_va_p),
        "test": compute_metrics(y_te_t, y_te_p),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
    }

    with open(artifacts_dir / "lstm_metrics.json", "w") as f:
        json.dump(lstm_metrics, f, indent=2)

    print(f"[LSTM Extracted & Saved] Test R2={lstm_metrics['test']['r2']:.4f} RMSE={lstm_metrics['test']['rmse']:.5f}", flush=True)
    return ctx_tr, ctx_va, ctx_te, lstm_metrics
