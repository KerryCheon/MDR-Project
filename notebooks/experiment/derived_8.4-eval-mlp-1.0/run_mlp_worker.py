#!/usr/bin/env python3
"""Train a single MLP job (one sweep config in one family).

Job = family x config:
  - family "1regime": one global MLP on the 54-feature shared backbone.
  - family "2regime": two per-cluster specialists (c0 = 54 feats, c1 = 64 feats),
    pooled holdout RMSE = sqrt(sum_c (n_hold_c / n_hold_total) * rmse_c^2).

Usage:
    python run_mlp_worker.py --family <1regime|2regime> --config-id <id> \
        --artifacts <dir> --out <dir> [--resume]

Run by run_mlp_sweep.py as a subprocess; not meant to be called directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
sys.path.insert(0, str(EVAL11_DIR))
sys.path.insert(0, str(EXP_DIR))

from eval11.evaluator import compute_metrics  # noqa: E402
from mlp10.trainer import train_one_config  # noqa: E402

CLUSTERS = ("0", "1")


def load_job_configs(artifacts: Path) -> dict[str, dict]:
    with open(artifacts / "sweep_configs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["1regime", "2regime"], required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_job_configs(args.artifacts)[args.config_id]
    job_out = args.out / "models" / args.family / args.config_id
    job_out.mkdir(parents=True, exist_ok=True)

    test_meta = np.load(args.artifacts / "test_meta.npz")
    y_test_full = test_meta["y_test"]

    if args.family == "1regime":
        fs = _load_fs(args.artifacts / "tensors_global.npz")
        res = train_one_config(cfg, fs, job_out, resume=args.resume)
        test_metrics = compute_metrics(y_test_full, res.test_preds)
        payload = {
            "family": "1regime",
            "config_id": args.config_id,
            "config": cfg,
            "holdout_rmse": res.hold_rmse,
            "test": test_metrics,
            "epochs": res.epochs_run,
            "best_epoch": res.best_epoch,
            "train_time_s": res.train_time_s,
            "n_params": res.n_params,
            "device": "cuda" if _cuda() else "cpu",
            "status": "completed",
        }
        _write(payload, job_out)
        print(f"[job] 1regime/{args.config_id} hold_rmse={res.hold_rmse:.5f} "
              f"test_r2={test_metrics['r2']:.4f} time={res.train_time_s:.1f}s", flush=True)

    else:  # 2regime
        full_preds = np.zeros(len(y_test_full), dtype=np.float64)
        cluster_meta: dict[str, dict] = {}
        hold_rmses: list[float] = []
        hold_ns: list[int] = []

        for cl in CLUSTERS:
            fs = _load_fs(args.artifacts / f"tensors_cluster{cl}.npz")
            spec_out = job_out / f"spec_{cl}"
            res = train_one_config(cfg, fs, spec_out, resume=args.resume)
            y_test_cl = y_test_full[fs["test_idx"]]
            cm = compute_metrics(y_test_cl, res.test_preds)
            full_preds[fs["test_idx"]] = res.test_preds
            cluster_meta[cl] = {
                "n_train": int(len(fs["train_idx"])),
                "n_hold": int(len(fs["hold_idx"])),
                "n_test": int(len(fs["test_idx"])),
                "holdout_rmse": res.hold_rmse,
                "test": cm,
                "best_epoch": res.best_epoch,
                "epochs": res.epochs_run,
                "train_time_s": res.train_time_s,
                "n_params": res.n_params,
            }
            hold_rmses.append(res.hold_rmse)
            hold_ns.append(int(len(fs["hold_idx"])))

        pooled_hold = float(
            np.sqrt(
                sum((n / sum(hold_ns)) * r**2 for n, r in zip(hold_ns, hold_rmses))
            )
        )
        test_metrics = compute_metrics(y_test_full, full_preds)
        np.save(job_out / PREDS_NAME, full_preds)
        payload = {
            "family": "2regime",
            "config_id": args.config_id,
            "config": cfg,
            "holdout_rmse": pooled_hold,
            "test": test_metrics,
            "per_cluster": cluster_meta,
            "epochs": max(c["epochs"] for c in cluster_meta.values()),
            "best_epoch": max(c["best_epoch"] for c in cluster_meta.values()),
            "train_time_s": sum(c["train_time_s"] for c in cluster_meta.values()),
            "n_params": sum(c["n_params"] for c in cluster_meta.values()),
            "device": "cuda" if _cuda() else "cpu",
            "status": "completed",
        }
        _write(payload, job_out)
        print(f"[job] 2regime/{args.config_id} pooled_hold_rmse={pooled_hold:.5f} "
              f"test_r2={test_metrics['r2']:.4f} time={payload['train_time_s']:.1f}s", flush=True)


def _load_fs(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_hold": data["X_hold"],
        "y_hold": data["y_hold"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "train_idx": data["train_idx"],
        "hold_idx": data["hold_idx"],
        "test_idx": data["test_idx"],
        "feature_names": list(data["feature_names"]),
        "n_features": int(data["X_train"].shape[1]),
    }


def _write(payload: dict, job_out: Path) -> None:
    with open(job_out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


PREDS_NAME = "preds.npy"


if __name__ == "__main__":
    main()
