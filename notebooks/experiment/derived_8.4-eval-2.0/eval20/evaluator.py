"""Per-fold MLP training + metric assembly for derived_8.4-eval-2.0.

One job = (family, config_id, station_or_full, seed). The worker loads the
prebuilt fold tensors (eval20.data.load_fold_tensors), trains the specialists
with the mlp-1.3 trainer, and assembles the eval-1.2-style metrics: pooled /
per-year / per-regime (r2, rmse, ubrmse, bias, mae, pearson).

An empty fold-train cluster (possible under LOSO when the refitted KMeans
router leaves a cluster with no rows from the 6 remaining stations) falls back
to predicting the fold-train target mean for that cluster's test rows —
mirroring eval-1.2's fallback.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
EVAL12_DIR = EXP_DIR.parent / "derived_8.4-eval-1.2"
MLP13_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-1.3"
for _d in (EVAL11_DIR, EVAL12_DIR, MLP13_DIR):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))

from eval12.evaluator import compute_metrics  # noqa: E402
from mlp13.trainer import train_one_config  # noqa: E402

CLUSTERS = ("0", "1")


@dataclass
class FoldTrainOutcome:
    """Outcome of training one (family x config x seed) on one fold."""

    preds: np.ndarray                 # fold-test predictions, fold-test row order
    per_cluster: dict[str, dict]      # cl -> {n_train, n_test, test_idx, preds,
                                      #        val_rmse, aux_rmse, best_epoch,
                                      #        epochs, train_time_s, n_params}
    train_time_s: float
    n_params: int
    fold_train_mean: float            # fallback constant (train-target mean)


def _load_fold_meta(artifacts: Path, family: str, station: str) -> dict:
    suffix = "__full" if station == "full" else f"__{station}"
    meta_path = artifacts / f"fold_{family}{suffix}_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def train_and_predict(
    cfg: dict,
    tensors: dict[str, dict],
    fold_meta: dict,
    out_dir: Path,
    *,
    resume: bool = False,
) -> FoldTrainOutcome:
    """Train the specialists for one fold and return fold-test predictions.

    ``tensors`` maps "" (global) or "0"/"1" (cluster) to a mlp13 feature set.
    ``out_dir`` is the job dir; cluster specialists write under spec_0/ spec_1/.
    """
    t0 = time.perf_counter()
    if "" in tensors:  # global family
        fs = tensors[""]
        res = train_one_config(cfg, fs, out_dir, resume=resume)
        preds = np.asarray(res.test_preds, dtype=np.float64)
        n_train = int(len(fs["y_train"]))
        per_cluster = {
            "0": {
                "n_train": n_train,
                "n_test": int(len(fs["y_test"])),
                "test_idx": np.arange(len(fs["y_test"]), dtype=np.int64),
                "preds": preds,
                "val_rmse": res.val_rmse,
                "aux_rmse": res.aux_rmse,
                "best_epoch": res.best_epoch,
                "epochs": res.epochs_run,
                "train_time_s": res.train_time_s,
                "n_params": res.n_params,
            }
        }
        n_params = res.n_params
    else:  # cluster family: per-cluster specialists
        n_test_total = int(fold_meta.get("n_test", 0))
        preds = np.full(n_test_total, np.nan, dtype=np.float64)
        per_cluster: dict[str, dict] = {}
        n_params = 0
        for cl in CLUSTERS:
            spec_dir = out_dir / f"spec_{cl}"
            if cl not in tensors:
                # Empty fold-train cluster -> fold-train-mean fallback (eval-1.2).
                test_idx = _empty_cluster_test_idx(tensors, cl, n_test_total)
                per_cluster[cl] = {
                    "n_train": 0,
                    "n_test": int(len(test_idx)),
                    "test_idx": test_idx,
                    "preds": np.full(len(test_idx), float(fold_meta.get("train_target_mean", np.nan))),
                    "val_rmse": float("nan"),
                    "aux_rmse": float("nan"),
                    "best_epoch": 0,
                    "epochs": 0,
                    "train_time_s": 0.0,
                    "n_params": 0,
                }
                preds[test_idx] = per_cluster[cl]["preds"]
                continue
            fs = tensors[cl]
            res = train_one_config(cfg, fs, spec_dir, resume=resume)
            test_idx = np.asarray(fs["test_idx"], dtype=np.int64)
            cl_preds = np.asarray(res.test_preds, dtype=np.float64)
            per_cluster[cl] = {
                "n_train": int(len(fs["y_train"])),
                "n_test": int(len(test_idx)),
                "test_idx": test_idx,
                "preds": cl_preds,
                "val_rmse": res.val_rmse,
                "aux_rmse": res.aux_rmse,
                "best_epoch": res.best_epoch,
                "epochs": res.epochs_run,
                "train_time_s": res.train_time_s,
                "n_params": res.n_params,
            }
            preds[test_idx] = cl_preds
            n_params += res.n_params
        if np.isnan(preds).any():
            raise RuntimeError(f"Uncovered test rows for fold (clusters {sorted(tensors)}): "
                               f"{int(np.isnan(preds).sum())} NaN")
    return FoldTrainOutcome(
        preds=preds,
        per_cluster=per_cluster,
        train_time_s=time.perf_counter() - t0,
        n_params=n_params,
        fold_train_mean=float(fold_meta.get("train_target_mean", np.nan)),
    )


def _empty_cluster_test_idx(tensors: dict[str, dict], missing_cl: str, n_test_total: int) -> np.ndarray:
    """Test rows NOT covered by the present cluster's test_idx (the missing cluster)."""
    present = [t for c, t in tensors.items() if c != missing_cl]
    if not present:
        return np.arange(n_test_total, dtype=np.int64)
    covered = np.concatenate([np.asarray(t["test_idx"], dtype=np.int64) for t in present])
    mask = np.ones(n_test_total, dtype=bool)
    mask[covered] = False
    return np.where(mask)[0].astype(np.int64)


def fold_metrics(
    y_test: np.ndarray,
    preds: np.ndarray,
    per_cluster: dict[str, dict],
    years: np.ndarray,
) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    """(pooled, yearly, cluster) metrics from fold-test targets/preds."""
    pooled = compute_metrics(y_test, preds)
    yearly: dict[str, dict] = {}
    for year in sorted(np.unique(years)):
        mask = years == year
        yearly[str(int(year))] = compute_metrics(y_test[mask], preds[mask])
    cluster: dict[str, dict] = {}
    for cl, pc in per_cluster.items():
        idx = pc["test_idx"]
        cluster[cl] = {
            "n_train": float(pc["n_train"]),
            "n_test": float(pc["n_test"]),
            **compute_metrics(y_test[idx], pc["preds"]),
        }
    return pooled, yearly, cluster
