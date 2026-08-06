"""Training loop for one MLP (global model or a 2-regime specialist).

Conventions follow Models/Temporal/lstm/train_v9.py: AdamW + cosine LR,
grad clipping, early stopping on a held-out RMSE, seed 42, best state dict
saved. Adds full checkpointing (model + optimizer + scheduler + curves +
elapsed time) every `checkpoint_every` epochs and at early stop, so an
interrupted job can resume exactly where it left off (`resume=True`).
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

from .model import MLPRegressor

CKPT_NAME = "checkpoint.pt"
BEST_NAME = "best_model.pt"
PREDS_NAME = "preds.npy"
CURVES_NAME = "curves.npy"


@dataclass
class TrainResult:
    config_id: str
    hold_rmse: float
    test_preds: np.ndarray
    test_metrics: dict[str, float]
    hold_curve: list[float] = field(default_factory=list)
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


def _build_loaders(
    fs: dict[str, Any], batch_size: int, device: torch.device
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(fs["X_train"]), torch.from_numpy(fs["y_train"])
    )
    hold_ds = TensorDataset(
        torch.from_numpy(fs["X_hold"]), torch.from_numpy(fs["y_hold"])
    )
    test_ds = TensorDataset(
        torch.from_numpy(fs["X_test"]), torch.from_numpy(fs["y_test"])
    )
    pin = device.type == "cuda"
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin),
        DataLoader(hold_ds, batch_size=1024, shuffle=False, pin_memory=pin),
        DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=pin),
    )


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


def train_one_config(
    cfg: dict[str, Any],
    fs: dict[str, Any],
    out_dir: Path,
    *,
    resume: bool = False,
) -> TrainResult:
    """Train a single MLP on a prepared feature set.

    cfg keys: hidden_sizes, activation, dropout, use_bn, lr, weight_decay,
    batch_size, loss (mse|huber), max_epochs, patience, checkpoint_every, seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(int(cfg.get("seed", 42)))
    max_epochs = int(cfg.get("max_epochs", 200))
    patience = int(cfg.get("patience", 25))
    checkpoint_every = int(cfg.get("checkpoint_every", 10))

    model = MLPRegressor(
        n_features=fs["n_features"],
        hidden_sizes=cfg["hidden_sizes"],
        activation=cfg.get("activation", "silu"),
        dropout=float(cfg.get("dropout", 0.1)),
        use_bn=bool(cfg.get("use_bn", True)),
        seed=int(cfg.get("seed", 42)),
    ).to(device)
    n_params = model.n_params()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=float(cfg.get("lr", 3e-4)) / 100.0
    )
    if cfg.get("loss", "mse") == "huber":
        criterion = nn.HuberLoss(delta=0.05)
    else:
        criterion = nn.MSELoss()

    train_loader, hold_loader, test_loader = _build_loaders(fs, int(cfg.get("batch_size", 512)), device)

    # --- state that survives resume ---
    start_epoch = 1
    best_hold_rmse = math.inf
    best_epoch = 0
    patience_ctr = 0
    hold_curve: list[float] = []
    test_curve: list[float] = []
    train_time_elapsed = 0.0
    done = False
    epoch = 0

    ckpt_path = _checkpoint_path(out_dir)
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_hold_rmse = float(ckpt["best_hold_rmse"])
        best_epoch = int(ckpt["best_epoch"])
        patience_ctr = int(ckpt["patience_ctr"])
        hold_curve = list(ckpt["hold_curve"])
        test_curve = list(ckpt["test_curve"])
        train_time_elapsed = float(ckpt["train_time_elapsed"])
        done = bool(ckpt.get("done", False))
        epoch = int(ckpt["epoch"])
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
        scheduler.step()

        y_hold_pred = _predict(model, hold_loader, device)
        hold_rmse = _rmse(fs["y_hold"], y_hold_pred)
        y_test_pred = _predict(model, test_loader, device)
        test_rmse = _rmse(fs["y_test"], y_test_pred)
        hold_curve.append(hold_rmse)
        test_curve.append(test_rmse)

        if hold_rmse < best_hold_rmse:
            best_hold_rmse = hold_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), out_dir / BEST_NAME)
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  ep {epoch:3d}/{max_epochs} hold_rmse={hold_rmse:.5f} "
                f"best={best_hold_rmse:.5f} (ep{best_epoch}) test_rmse={test_rmse:.5f}",
                flush=True,
            )

        if patience_ctr >= patience:
            print(f"[early stop] no holdout improvement for {patience} epochs at epoch {epoch}", flush=True)
            done = True

        train_time_elapsed = time.perf_counter() - t_start
        if epoch % checkpoint_every == 0 or done:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_hold_rmse": best_hold_rmse,
                    "best_epoch": best_epoch,
                    "patience_ctr": patience_ctr,
                    "hold_curve": hold_curve,
                    "test_curve": test_curve,
                    "train_time_elapsed": train_time_elapsed,
                    "done": done,
                    "config": cfg,
                },
                ckpt_path,
            )

    # --- finalize: best model -> test predictions ---
    if best_epoch > 0 and (out_dir / BEST_NAME).exists():
        model.load_state_dict(
            torch.load(out_dir / BEST_NAME, map_location=device, weights_only=True)
        )
    test_preds = _predict(model, test_loader, device)
    np.save(out_dir / PREDS_NAME, test_preds)
    np.save(out_dir / CURVES_NAME, np.stack([np.asarray(hold_curve), np.asarray(test_curve)]))

    result = TrainResult(
        config_id=str(cfg.get("id", out_dir.name)),
        hold_rmse=best_hold_rmse,
        test_preds=test_preds,
        test_metrics={},
        hold_curve=hold_curve,
        test_curve=test_curve,
        epochs_run=epoch,
        best_epoch=best_epoch,
        train_time_s=train_time_elapsed,
        n_params=n_params,
        early_stopped=done and best_epoch < epoch,
    )
    print(
        f"[done] {out_dir.name}: best_hold_rmse={best_hold_rmse:.5f} (ep{best_epoch}) "
        f"epochs={epoch} time={train_time_elapsed:.1f}s params={n_params}",
        flush=True,
    )
    return result


def save_result_json(out_dir: Path, payload: dict[str, Any]) -> None:
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
