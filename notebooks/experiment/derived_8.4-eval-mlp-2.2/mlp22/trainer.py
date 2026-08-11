"""Training loop for one neural tabular model (global or a 2-regime specialist).

Protocol (data_version 9): train on `train`, early-stop on `val`, evaluate on
`test`. Identical to the mlp21 trainer (data_version 8) apart from the 2.2
val-preds save (below); the SWA machinery stays importable for parity but NO
2.2 config uses it (SWA is a documented negative — 0/152 deployments with the
RNG guard in place, mlp-2.1). Train on the OFFICIAL train split (2017-2020),
early-stop / select on the OFFICIAL val split (2021-2022), evaluate on the
untouched test split (2023-2025) — with the AUXILIARY 2020 holdout (aux) kept
as a DIAGNOSTIC ONLY (mlp-1.2 documented it measures train fit, not
generalization).

New in mlp-2.2:
  - the best-val predictions are saved to `val_preds.npy` (post-training,
    eval-mode forward with the SAME deployed weights that produced
    `preds.npy`, no RNG consumption) so the offline val-year (2021 vs 2022)
    selection-reliability diagnostic (`analyze_val_years.py`) can be computed
    without retraining. The training path is byte-identical to mlp21 — the
    anchors' val curves stay bit-identical across versions (stack check).

New in mlp-2.0 (kept):
  - swa=True: Stochastic Weight Averaging (Izmailov et al. 2018). From
    `swa_start_frac` of max_epochs onward, maintain a running average of the
    model state (params + BN buffers, excluding num_batches_tracked) updated
    once PER EPOCH. The SWA snapshot is evaluated on val (and aux/test for
    diagnostics) every `swa_eval_every` epochs; BN running stats are
    recalibrated on the train loader before each SWA evaluation
    (swa_bn_recal=True, standard SWA update_bn) so the val signal is honest.
    Early stopping keeps using the LIVE model's val with patience-60 (the
    1.3-confirmed rule); the deployed model is the SWA snapshot iff its best
    val RMSE beats the live best — an honest within-val comparison.

New in mlp-2.1 (the two documented fixes from the 2.0 SWA section):
  - RNG guard: the SWA snapshot evaluation (`_eval_swa_snapshot`) runs the
    train loader in train mode (BN recalibration) and iterates the val/aux/test
    loaders — ALL of which consume the shared torch RNG (DataLoader iteration
    consumes RNG even with shuffle=False), which in 2.0 made a swa=true job's
    live trajectory diverge from its swa=false anchor (the 2.0 README's "gains
    are live-trajectory artifacts" caveat). In 2.1 the whole snapshot
    evaluation runs inside `_rng_guard()`, which restores the torch / numpy /
    random RNG states on exit, so a swa=true job's LIVE trajectory is
    bit-identical to its swa=false anchor. Verified by a pre-sweep stack check
    (anchor val curve == swa_late live val curve).
  - `swa_start_frac` is a swept knob ({0.7, 0.75, 0.8, 0.85} in the 2.1
    configs; 2.0 hard-coded 0.6, which averaged over the 240-400 region and
    never beat the live best). 0.85 x 400 = 340 risks never starting before
    the patience-60 early stop, so the sweep tests all four values.

Keeps the mlp-1.3 checkpoint/resume contract (checkpoint.pt every
checkpoint_every epochs and at early stop) so interrupted jobs resume exactly.
curves.npy is the LIVE model's [val, aux, test] stack (1.3 semantics);
curves_swa.npy (when swa=True) is the SWA snapshot's [val, aux, test] stack.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
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
BEST_SWA_NAME = "best_swa.pt"
PREDS_NAME = "preds.npy"
CURVES_NAME = "curves.npy"
CURVES_SWA_NAME = "curves_swa.npy"

# Buffers that must NOT be averaged into the SWA snapshot (counters).
_SWA_EXCLUDE = ("num_batches_tracked",)


@contextlib.contextmanager
def _rng_guard(device: torch.device):
    """Restore every RNG state on exit (mlp-2.1 fix).

    The SWA BN-recalibration pass runs the TRAIN loader in train mode, so its
    shuffle=True RandomSampler consumes torch RNG at iterator creation and its
    dropout layers consume the shared torch RNG stream per forward (plus, in
    principle, the numpy/python streams via mixup / dataloader helpers). The
    val/aux/test loaders use SequentialSampler with num_workers=0 and are
    RNG-free. Without a guard the recalibration perturbs the live training
    trajectory of a swa=true job relative to its swa=false anchor — the 2.0
    README's documented "gains are live-trajectory artifacts" caveat. Guarding
    the whole snapshot evaluation makes the live trajectory bit-identical to
    the anchor, so any `_swa*` gain is attributable to SWA.
    """
    rng_saved = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_saved: dict[int, torch.Tensor] = {}
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        cuda_saved[idx] = torch.cuda.get_rng_state(idx)
    try:
        yield
    finally:
        random.setstate(rng_saved[0])
        np.random.set_state(rng_saved[1])
        torch.set_rng_state(rng_saved[2])
        for i, st in cuda_saved.items():
            torch.cuda.set_rng_state(st, i)


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
    # --- mlp-2.0 SWA bookkeeping ---
    deployed: str = "live"          # "live" | "swa" — which weights produced test_preds
    val_rmse_live: float = float("nan")
    val_rmse_swa: float = float("nan")
    swa_best_epoch: int = 0


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
    """Legacy per-step EMA from mlp-1.3 (documented failure; kept for 1.3 parity)."""
    with torch.no_grad():
        for k, v in model.named_parameters():
            if v.requires_grad:
                ema[k] = ema.get(k, v.detach().clone()) * decay + v.detach().clone() * (1.0 - decay)


@torch.no_grad()
def _predict_with_ema(model: nn.Module, ema: dict[str, torch.Tensor], loader: DataLoader, device: torch.device) -> np.ndarray:
    """Predict using EMA weights while keeping the live model's BN buffers (mlp-1.3 parity)."""
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema, strict=False)  # params <- EMA; buffers stay live
    preds = _predict(model, loader, device)
    model.load_state_dict(saved)
    return preds


@torch.no_grad()
def _recalibrate_bn(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    """Standard SWA update_bn: recompute BatchNorm running stats on the train loader.

    The SWA-averaged buffers are a crude average; a single no-grad forward pass
    over the training set with reset running stats gives the correct statistics
    for the averaged weights. No-op when the model has no BatchNorm.

    mlp-2.1: the forward pass runs in train mode, so dropout consumes the shared
    RNG stream; it is wrapped in `_rng_guard` so the live training trajectory
    stays bit-identical to the swa=false anchor (2.0 RNG-leak fix).
    """
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not bn_modules:
        return
    for m in bn_modules:
        m.reset_running_stats()
    model.train()
    with _rng_guard(device):
        for X_batch, _ in loader:
            model(X_batch.to(device))
    model.eval()


@torch.no_grad()
def _predict_with_swa(
    model: nn.Module,
    swa_state: dict[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    recal_loader: DataLoader | None = None,
    bn_recal: bool = False,
) -> np.ndarray:
    """Predict using the SWA snapshot; optionally recalibrate BN first; restore live state."""
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(swa_state)
    if bn_recal and recal_loader is not None:
        _recalibrate_bn(model, recal_loader, device)
    preds = _predict(model, loader, device)
    model.load_state_dict(saved)
    return preds


def _swa_update(swa_state: dict[str, torch.Tensor] | None, model: nn.Module, count: int) -> tuple[dict[str, torch.Tensor], int]:
    """Incremental running average of params + buffers (excl. counters), once per epoch."""
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sd = {k: v for k, v in sd.items() if not any(k.endswith(s) for s in _SWA_EXCLUDE)}
    if swa_state is None:
        return sd, 1
    n = count + 1
    for k in sd:
        swa_state[k] = swa_state[k] + (sd[k] - swa_state[k]) / n
    return swa_state, n


def train_one_config(
    cfg: dict[str, Any],
    fs: dict[str, Any],
    out_dir: Path,
    *,
    resume: bool = False,
) -> TrainResult:
    """Train a single model on a prepared feature set.

    cfg keys (mlp-1.3 set unchanged): architecture, hidden_sizes, ft_d,
    ft_layers, activation, norm, dropout, lr, weight_decay, batch_size, loss
    (mse|huber), huber_delta, warmup_frac, ema, max_epochs, patience,
    checkpoint_every, seed, stop_rule, mixup_alpha, center_target.

    New in mlp-2.0:
      - swa: bool (default False)
      - swa_start_frac: float (default 0.6) — SWA averaging starts at
        round(max_epochs * swa_start_frac)
      - swa_eval_every: int (default 10) — epochs between SWA-val evaluations
      - swa_bn_recal: bool (default True) — recalibrate BN before SWA eval

    Deployed model: SWA snapshot iff its best val RMSE < the live best
    (within-val comparison, honest). result.val_rmse / best_epoch report the
    DEPLOYED model; live/SWA numbers are kept in val_rmse_live / val_rmse_swa.
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
    stop_rule = str(cfg.get("stop_rule", "patience"))
    stop_plateau_eps = float(cfg.get("stop_plateau_eps", 1e-4))
    stop_plateau_window = int(cfg.get("stop_plateau_window", 40))
    mixup_alpha = float(cfg.get("mixup_alpha", 0.0))
    center_target = bool(cfg.get("center_target", False))

    # --- mlp-2.0 SWA knobs ---
    use_swa = bool(cfg.get("swa", False))
    swa_start_frac = float(cfg.get("swa_start_frac", 0.6))
    swa_eval_every = int(cfg.get("swa_eval_every", 10))
    swa_bn_recal = bool(cfg.get("swa_bn_recal", True))
    swa_start_epoch = int(round(max_epochs * swa_start_frac))

    if retrain_mode:
        # Train on trainval (train+val) for a fixed number of epochs (the
        # best-epoch count from the val-protocol sweep); no early stopping.
        # NOTE: retrain-on-trainval is a documented negative in mlp-1.2; kept
        # only for parity, no 2.0 config enables it.
        max_epochs = int(cfg["retrain_epochs"])
        patience = 10 ** 9
        stop_rule = "patience"
        fs = dict(fs)
        fs["X_train"] = np.concatenate([fs["X_train"], fs["X_val"]], axis=0)
        fs["y_train"] = np.concatenate([fs["y_train"], fs["y_val"]], axis=0)

    # --- optional target centering (debias at source; mlp-1.3 negative, kept for parity) ---
    target_shift = 0.0
    if center_target:
        target_shift = float(np.mean(fs["y_train"]))
        fs = dict(fs)
        for key in ("y_train", "y_val", "y_test", "y_aux"):
            if fs.get(key) is not None and fs[key].shape[0] > 0:
                fs[key] = np.asarray(fs[key], dtype=np.float32) - float(target_shift)

    model = build_model(cfg, fs["n_features"], fs.get("feature_names")).to(device)
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

    # --- SWA state ---
    swa_state: dict[str, torch.Tensor] | None = None
    swa_count = 0
    best_swa_val_rmse = math.inf
    best_swa_epoch = 0
    best_swa_aux_rmse = float("nan")
    swa_val_curve: list[float] = []
    swa_aux_curve: list[float] = []
    swa_test_curve: list[float] = []

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
        best_score = float(ckpt.get("best_score", best_val_rmse))
        patience_ctr = int(ckpt["patience_ctr"])
        val_curve = list(ckpt["val_curve"])
        aux_curve = list(ckpt.get("aux_curve", []))
        test_curve = list(ckpt["test_curve"])
        train_time_elapsed = float(ckpt["train_time_elapsed"])
        done = bool(ckpt.get("done", False))
        epoch = int(ckpt["epoch"])
        ema = {k: v.to(device) for k, v in ckpt.get("ema_state", {}).items()}
        target_shift = float(ckpt.get("target_shift", target_shift))
        stop_rule = str(ckpt.get("stop_rule", stop_rule))
        # SWA resume state
        swa_state = {k: v.to(device) for k, v in ckpt.get("swa_state", {}).items()} or None
        swa_count = int(ckpt.get("swa_count", 0))
        best_swa_val_rmse = float(ckpt.get("best_swa_val_rmse", math.inf))
        best_swa_epoch = int(ckpt.get("best_swa_epoch", 0))
        best_swa_aux_rmse = float(ckpt.get("best_swa_aux_rmse", float("nan")))
        swa_val_curve = list(ckpt.get("swa_val_curve", []))
        swa_aux_curve = list(ckpt.get("swa_aux_curve", []))
        swa_test_curve = list(ckpt.get("swa_test_curve", []))
        print(f"[resume] {out_dir.name}: continuing from epoch {start_epoch} (done={done})", flush=True)

    t_start = time.perf_counter()
    grad_clip = float(cfg.get("grad_clip", 1.0))
    best_score = math.inf  # selection score per stop_rule (patience/plateau: val; val_aux: (val+aux)/2)

    def _eval_swa_snapshot() -> None:
        """Evaluate the current SWA average on val/aux/test and track the best epoch."""
        nonlocal best_swa_val_rmse, best_swa_epoch, best_swa_aux_rmse
        if swa_state is None:
            return
        # mlp-2.1 RNG guard: EVERYTHING in the snapshot evaluation is SWA
        # bookkeeping and must not perturb the live trajectory. This covers the
        # train-loader recalibration forward pass AND the val/aux/test loader
        # iterations in _predict_with_swa — DataLoader iteration consumes torch
        # RNG even with shuffle=False, so the whole block is guarded.
        with _rng_guard(device):
            val_preds = _predict_with_swa(model, swa_state, val_loader, device,
                                          recal_loader=train_loader, bn_recal=swa_bn_recal)
            val_rmse_swa_now = _rmse(fs["y_val"], val_preds)
            swa_val_curve.append(val_rmse_swa_now)
            if has_aux and not retrain_mode:
                aux_preds = _predict_with_swa(model, swa_state, aux_loader[0], device,
                                              recal_loader=train_loader, bn_recal=swa_bn_recal)
                aux_rmse_swa_now = _rmse(fs["y_aux"], aux_preds)
            else:
                aux_rmse_swa_now = float("nan")
            swa_aux_curve.append(aux_rmse_swa_now)
            test_preds_swa = _predict_with_swa(model, swa_state, test_loader, device,
                                               recal_loader=train_loader, bn_recal=swa_bn_recal)
            test_rmse_swa_now = _rmse(fs["y_test"], test_preds_swa)
            swa_test_curve.append(test_rmse_swa_now)
            if val_rmse_swa_now < best_swa_val_rmse:
                best_swa_val_rmse = val_rmse_swa_now
                best_swa_epoch = epoch
                best_swa_aux_rmse = aux_rmse_swa_now
                torch.save({k: v.cpu() for k, v in swa_state.items()}, out_dir / BEST_SWA_NAME)
        print(f"  [swa] ep {epoch:3d} swa_val={val_rmse_swa_now:.5f} "
              f"best_swa={best_swa_val_rmse:.5f} (ep{best_swa_epoch}) "
              f"swa_aux={aux_rmse_swa_now:.5f} swa_test={test_rmse_swa_now:.5f}", flush=True)

    while not done and epoch < max_epochs:
        epoch += 1
        if epoch < start_epoch:
            continue
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if mixup_alpha > 0.0 and X_batch.shape[0] > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(X_batch.shape[0], device=device)
                X_batch = lam * X_batch + (1.0 - lam) * X_batch[idx]
                y_batch = lam * y_batch + (1.0 - lam) * y_batch[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # Legacy EMA (mlp-1.3 parity): update once per optimizer step.
            if use_ema:
                _ema_update(ema, model, ema_decay)
        scheduler.step()

        # --- SWA: update the running average once per epoch (after LR step) ---
        if use_swa and not retrain_mode and epoch >= swa_start_epoch:
            swa_state, swa_count = _swa_update(swa_state, model, swa_count)

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

        # SWA snapshot evaluation (cadence)
        if use_swa and not retrain_mode and swa_state is not None and (
                epoch % swa_eval_every == 0 or epoch == swa_start_epoch):
            _eval_swa_snapshot()

        # selection score for best-epoch + early stopping (per stop_rule)
        if stop_rule == "val_aux" and not math.isnan(aux_rmse):
            score = 0.5 * (val_rmse + aux_rmse)
        else:
            score = val_rmse

        if not retrain_mode and score < best_score:
            best_score = score
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
                f"best={best_val_rmse:.5f} (ep{best_epoch}) aux={aux_rmse:.5f} test_rmse={test_rmse:.5f} "
                f"rule={stop_rule}",
                flush=True,
            )

        if stop_rule == "plateau" and not retrain_mode:
            # Stop at the FIRST sustained plateau (mlp-1.3 rule; documented negative vs patience).
            window = max(1, stop_plateau_window)
            if epoch >= window:
                window_min = min(val_curve[-window:])
                if best_score - window_min < stop_plateau_eps:
                    print(f"[plateau stop] no >= {stop_plateau_eps:.1e} improvement over the last "
                          f"{window} epochs at epoch {epoch} (best {best_score:.5f} ep{best_epoch})",
                          flush=True)
                    done = True
        elif patience_ctr >= patience:
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
                    "best_score": best_score,
                    "stop_rule": stop_rule,
                    "target_shift": target_shift,
                    "patience_ctr": patience_ctr,
                    "val_curve": val_curve,
                    "aux_curve": aux_curve,
                    "test_curve": test_curve,
                    "train_time_elapsed": train_time_elapsed,
                    "done": done,
                    "ema_state": {k: v.cpu() for k, v in ema.items()},
                    "config": cfg,
                    # SWA resume state
                    "swa_state": {k: v.cpu() for k, v in swa_state.items()} if swa_state else {},
                    "swa_count": swa_count,
                    "best_swa_val_rmse": best_swa_val_rmse,
                    "best_swa_epoch": best_swa_epoch,
                    "best_swa_aux_rmse": best_swa_aux_rmse,
                    "swa_val_curve": swa_val_curve,
                    "swa_aux_curve": swa_aux_curve,
                    "swa_test_curve": swa_test_curve,
                },
                ckpt_path,
            )

    # --- finalize: best model -> test predictions ---
    val_rmse_live = best_val_rmse
    deployed = "live"
    if use_swa and best_swa_epoch > 0 and best_swa_val_rmse < best_val_rmse:
        deployed = "swa"
        print(f"[deploy] SWA snapshot (val {best_swa_val_rmse:.5f} @ ep{best_swa_epoch}) "
              f"beats live best (val {best_val_rmse:.5f} @ ep{best_epoch})", flush=True)
        best_val_rmse = best_swa_val_rmse
        best_epoch = best_swa_epoch
        best_aux_rmse = best_swa_aux_rmse

    val_preds: np.ndarray | None = None
    if best_epoch > 0 and (out_dir / BEST_NAME).exists():
        if use_ema:
            best_weights = torch.load(out_dir / BEST_NAME, map_location=device, weights_only=False)
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_weights, strict=False)
            test_preds = _predict(model, test_loader, device)
            val_preds = _predict(model, val_loader, device)
            model.load_state_dict(saved)
        elif deployed == "swa" and (out_dir / BEST_SWA_NAME).exists():
            swa_weights = torch.load(out_dir / BEST_SWA_NAME, map_location=device, weights_only=True)
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(swa_weights)
            if swa_bn_recal:
                _recalibrate_bn(model, train_loader, device)
            test_preds = _predict(model, test_loader, device)
            val_preds = _predict(model, val_loader, device)
            model.load_state_dict(saved)
        else:
            model.load_state_dict(torch.load(out_dir / BEST_NAME, map_location=device, weights_only=True))
            test_preds = _predict(model, test_loader, device)
            val_preds = _predict(model, val_loader, device)
    else:
        test_preds = _predict(model, test_loader, device)
        val_preds = _predict(model, val_loader, device)

    if retrain_mode:
        torch.save(model.state_dict(), out_dir / BEST_NAME)
        best_epoch = epoch

    # add back the target shift (target-centering configs predict the residual)
    if abs(target_shift) > 0.0:
        test_preds = np.asarray(test_preds, dtype=np.float64) + float(target_shift)
        val_preds = np.asarray(val_preds, dtype=np.float64) + float(target_shift)

    np.save(out_dir / PREDS_NAME, test_preds)
    # NEW in 2.2: best-val predictions (same deployed weights as preds.npy) —
    # enables the offline val-year selection-reliability diagnostic.
    np.save(out_dir / "val_preds.npy", val_preds)
    np.save(out_dir / CURVES_NAME, np.stack([np.asarray(val_curve), np.asarray(aux_curve), np.asarray(test_curve)]))
    if use_swa and (swa_val_curve or swa_test_curve):
        np.save(out_dir / CURVES_SWA_NAME,
                np.stack([np.asarray(swa_val_curve), np.asarray(swa_aux_curve), np.asarray(swa_test_curve)]))

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
        deployed=deployed,
        val_rmse_live=val_rmse_live,
        val_rmse_swa=best_swa_val_rmse if use_swa else float("nan"),
        swa_best_epoch=best_swa_epoch if use_swa else 0,
    )
    print(
        f"[done] {out_dir.name}: best_val_rmse={best_val_rmse:.5f} (ep{best_epoch}, {deployed}) "
        f"aux={best_aux_rmse:.5f} epochs={epoch} time={train_time_elapsed:.1f}s params={n_params}",
        flush=True,
    )
    return result


def save_result_json(out_dir: Path, payload: dict[str, Any]) -> None:
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
