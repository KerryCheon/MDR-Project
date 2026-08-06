"""Training loop for one neural tabular model (global or a 2-regime specialist).

Protocol (data_version 4): train on `train`, early-stop on `val`, evaluate on
`test`. Extends mlp-1.1's trainer with an AUXILIARY 2020 holdout (aux): the
2020 slice of train is evaluated every epoch (same train-fitted imputer/scaler
— no leakage), and `best_aux_rmse` records the aux RMSE at the epoch that
achieved the best val RMSE. This is the second selection signal that exposes
val-period overfitting (mlp-1.1's deep residual nets fit 2021-22 but
generalized worst to 2023-25).

Keeps the mlp-1.1 checkpoint/resume contract (checkpoint.pt every
checkpoint_every epochs and at early stop) so interrupted jobs resume exactly.
curves.npy is now a 3-row stack: [val, aux, test].
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import build_model, make_scheduler

CKPT_NAME = "checkpoint.pt"
BEST_NAME = "best_model.pt"
PREDS_NAME = "preds.npy"
CURVES_NAME = "curves.npy"


@dataclass
class TrainResult:
    config_id: str
    val_rmse: float
    test_preds: np.ndarray
    test_metrics: dict[str, float]
    aux_rmse: float = float("nan")
    val_curve: list[float] = field(default_factory=list)
    aux_curve: list[float] = field(default_factory=list)
    test_curve: list[float] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = 0
    train_time_s: float = 0.0
    n_params: int = 0
    early_stopped: bool = False


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_loaders(fs: dict[str, Any], batch_size: int, device: torch.device):
    train_ds = TensorDataset(torch.from_numpy(fs["X_train"]), torch.from_numpy(fs["y_train"]))
    val_ds = TensorDataset(torch.from_numpy(fs["X_val"]), torch.from_numpy(fs["y_val"]))
    test_ds = TensorDataset(torch.from_numpy(fs["X_test"]), torch.from_numpy(fs["y_test"]))
    pin = device.type == "cuda"
    loaders = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin),
        DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=pin),
        DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=pin),
    )
    if fs.get("X_aux") is not None and fs["X_aux"].shape[0] > 0:
        aux_ds = TensorDataset(torch.from_numpy(fs["X_aux"]), torch.from_numpy(fs["y_aux"]))
        loaders = (*loaders, DataLoader(aux_ds, batch_size=1024, shuffle=False, pin_memory=pin))
    return loaders


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for X_batch, _ in loader:
        out.append(model(X_batch.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))


def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / CKPT_NAME


def _ema_update(ema: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for k, v in model.named_parameters():
            if v.requires_grad:
                ema[k] = ema.get(k, v.detach().clone()) * decay + v.detach().clone() * (1.0 - decay)


@torch.no_grad()
def _predict_with_ema(model: nn.Module, ema: dict[str, torch.Tensor], loader: DataLoader, device: torch.device) -> np.ndarray:
    """Predict using EMA weights while keeping the live model's BN buffers."""
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema, strict=False)  # params <- EMA; buffers stay live
    preds = _predict(model, loader, device)
    model.load_state_dict(saved)
    return preds


def train_one_config(
    cfg: dict[str, Any],
    fs: dict[str, Any],
    out_dir: Path,
    *,
    resume: bool = False,
) -> TrainResult:
    """Train a single model on a prepared feature set.

    cfg keys: architecture, hidden_sizes, ft_d, ft_layers, activation, norm,
    dropout, lr, weight_decay, batch_size, loss (mse|huber), huber_delta,
    warmup_frac, ema, max_epochs, patience, checkpoint_every, seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(int(cfg.get("seed", 42)))
    retrain_mode = int(cfg.get("retrain_epochs", 0)) > 0
    max_epochs = int(cfg.get("max_epochs", 400))
    patience = int(cfg.get("patience", 60))
    checkpoint_every = int(cfg.get("checkpoint_every", 20))
    use_ema = bool(cfg.get("ema", False))
    ema_decay = float(cfg.get("ema_decay", 0.999))

    if retrain_mode:
        # Train on trainval (train+val) for a fixed number of epochs (the
        # best-epoch count from the val-protocol sweep); no early stopping.
        max_epochs = int(cfg["retrain_epochs"])
        patience = 10 ** 9
        fs = dict(fs)
        fs["X_train"] = np.concatenate([fs["X_train"], fs["X_val"]], axis=0)
        fs["y_train"] = np.concatenate([fs["y_train"], fs["y_val"]], axis=0)

    model = build_model(cfg, fs["n_features"]).to(device)
    n_params = model.n_params()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = make_scheduler(optimizer, cfg, max_epochs)
    if cfg.get("loss", "mse") == "huber":
        criterion = nn.HuberLoss(delta=float(cfg.get("huber_delta", 0.05)))
    else:
        criterion = nn.MSELoss()

    train_loader, val_loader, test_loader, *aux_loader = _build_loaders(fs, int(cfg.get("batch_size", 512)), device)
    has_aux = len(aux_loader) == 1

    start_epoch = 1
    best_val_rmse = math.inf
    best_epoch = 0
    best_aux_rmse = float("nan")
    patience_ctr = 0
    val_curve: list[float] = []
    aux_curve: list[float] = []
    test_curve: list[float] = []
    train_time_elapsed = 0.0
    done = False
    epoch = 0
    ema: dict[str, torch.Tensor] = {}

    ckpt_path = _checkpoint_path(out_dir)
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_rmse = float(ckpt["best_val_rmse"])
        best_epoch = int(ckpt["best_epoch"])
        best_aux_rmse = float(ckpt.get("best_aux_rmse", float("nan")))
        patience_ctr = int(ckpt["patience_ctr"])
        val_curve = list(ckpt["val_curve"])
        aux_curve = list(ckpt.get("aux_curve", []))
        test_curve = list(ckpt["test_curve"])
        train_time_elapsed = float(ckpt["train_time_elapsed"])
        done = bool(ckpt.get("done", False))
        epoch = int(ckpt["epoch"])
        ema = {k: v.to(device) for k, v in ckpt.get("ema_state", {}).items()}
        print(f"[resume] {out_dir.name}: continuing from epoch {start_epoch} (done={done})", flush=True)

    t_start = time.perf_counter()
    grad_clip = float(cfg.get("grad_clip", 1.0))

    while not done and epoch < max_epochs:
        epoch += 1
        if epoch < start_epoch:
            continue
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # Standard EMA: update once per optimizer step (decay 0.999 per step).
            if use_ema:
                _ema_update(ema, model, ema_decay)
        scheduler.step()

        if retrain_mode:
            y_val_pred = None
            val_rmse = float("nan")
        elif use_ema:
            y_val_pred = _predict_with_ema(model, ema, val_loader, device)
            val_rmse = _rmse(fs["y_val"], y_val_pred)
        else:
            y_val_pred = _predict(model, val_loader, device)
            val_rmse = _rmse(fs["y_val"], y_val_pred)

        # aux2020 holdout (same weights that produced val_rmse this epoch)
        if has_aux and not retrain_mode:
            if use_ema:
                y_aux_pred = _predict_with_ema(model, ema, aux_loader[0], device)
            else:
                y_aux_pred = _predict(model, aux_loader[0], device)
            aux_rmse = _rmse(fs["y_aux"], y_aux_pred)
        else:
            aux_rmse = float("nan")
        aux_curve.append(aux_rmse)

        if use_ema:
            y_test_pred = _predict_with_ema(model, ema, test_loader, device)
        else:
            y_test_pred = _predict(model, test_loader, device)
        test_rmse = _rmse(fs["y_test"], y_test_pred)
        val_curve.append(val_rmse)
        test_curve.append(test_rmse)

        if not retrain_mode and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_aux_rmse = aux_rmse
            patience_ctr = 0
            if use_ema:
                torch.save(ema, out_dir / BEST_NAME)
            else:
                torch.save(model.state_dict(), out_dir / BEST_NAME)
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  ep {epoch:3d}/{max_epochs} val_rmse={val_rmse:.5f} "
                f"best={best_val_rmse:.5f} (ep{best_epoch}) aux={aux_rmse:.5f} test_rmse={test_rmse:.5f}",
                flush=True,
            )

        if patience_ctr >= patience:
            print(f"[early stop] no val improvement for {patience} epochs at epoch {epoch}", flush=True)
            done = True

        train_time_elapsed = time.perf_counter() - t_start
        if epoch % checkpoint_every == 0 or done:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_rmse": best_val_rmse,
                    "best_epoch": best_epoch,
                    "best_aux_rmse": best_aux_rmse,
                    "patience_ctr": patience_ctr,
                    "val_curve": val_curve,
                    "aux_curve": aux_curve,
                    "test_curve": test_curve,
                    "train_time_elapsed": train_time_elapsed,
                    "done": done,
                    "ema_state": {k: v.cpu() for k, v in ema.items()},
                    "config": cfg,
                },
                ckpt_path,
            )

    # --- finalize: best model -> test predictions ---
    if best_epoch > 0 and (out_dir / BEST_NAME).exists():
        if use_ema:
            best_weights = torch.load(out_dir / BEST_NAME, map_location=device, weights_only=False)
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_weights, strict=False)
            test_preds = _predict(model, test_loader, device)
            model.load_state_dict(saved)
        else:
            model.load_state_dict(torch.load(out_dir / BEST_NAME, map_location=device, weights_only=True))
            test_preds = _predict(model, test_loader, device)
    else:
        test_preds = _predict(model, test_loader, device)

    if retrain_mode:
        torch.save(model.state_dict(), out_dir / BEST_NAME)
        best_epoch = epoch

    np.save(out_dir / PREDS_NAME, test_preds)
    np.save(out_dir / CURVES_NAME, np.stack([np.asarray(val_curve), np.asarray(aux_curve), np.asarray(test_curve)]))

    result = TrainResult(
        config_id=str(cfg.get("id", out_dir.name)),
        val_rmse=float("nan") if retrain_mode else best_val_rmse,
        test_preds=test_preds,
        test_metrics={},
        aux_rmse=best_aux_rmse,
        val_curve=val_curve,
        aux_curve=aux_curve,
        test_curve=test_curve,
        epochs_run=epoch,
        best_epoch=best_epoch,
        train_time_s=train_time_elapsed,
        n_params=n_params,
        early_stopped=done and best_epoch < epoch,
    )
    print(
        f"[done] {out_dir.name}: best_val_rmse={best_val_rmse:.5f} (ep{best_epoch}) "
        f"aux={best_aux_rmse:.5f} epochs={epoch} time={train_time_elapsed:.1f}s params={n_params}",
        flush=True,
    )
    return result


def save_result_json(out_dir: Path, payload: dict[str, Any]) -> None:
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
