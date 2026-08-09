#!/usr/bin/env python3
"""Train a single MLP LOSO job: one (family x config_id x station x seed).

The driver (run_loso.py / run_full_baseline.py) prebuilds the fold tensors
(eval20.data.build_*_tensors) and spawns these workers as subprocesses. Each
worker loads its fold's persisted feature sets, trains the specialists with the
mlp-1.3 trainer (early-stop on the fold val split, patience 60), and writes:

    models/<family>/<config_id>/<station>/seed_<s>/{checkpoint.pt,
        best_model.pt, curves.npy, preds.npy, meta.json}   (spec_0/1 for 2-regime)

meta.json carries pooled / per-year / per-regime metrics for the fold so the
driver can aggregate seeds and build the eval-1.2-format CSVs. Resumable:
--resume continues interrupted jobs from checkpoint.pt; completed jobs are
skipped by the driver (data_version match).

Usage:
    python run_loso_worker.py --family <id> --config-id <id> --station <s|full> \
        --seed <s> --artifacts <dir> --out <dir> [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
for _d in (EXP_DIR,):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from eval20 import data as edata  # noqa: E402
from eval20.evaluator import _load_fold_meta, fold_metrics, train_and_predict  # noqa: E402


def _pooled_rmse(rmses: list[float], ns: list[int]) -> float:
    if not ns or sum(ns) == 0:
        return float("nan")
    return float(np.sqrt(sum((n / sum(ns)) * r**2 for n, r in zip(ns, rmses))))


def _load_fold_test(artifacts: Path, family: str, station: str) -> dict:
    suffix = "__full" if station == "full" else f"__{station}"
    data = np.load(artifacts / f"fold_{family}{suffix}_test.npz", allow_pickle=True)
    return {
        "y_test": data["y_test"],
        "year": data["year"],
        "station": data["station"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--station", required=True, help="held-out station id or 'full'")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = json.loads((args.artifacts / "sweep_configs.json").read_text(encoding="utf-8"))[args.config_id]
    cfg["seed"] = int(args.seed)
    cfg["id"] = args.config_id

    job_out = args.out / "models" / args.family / args.config_id / args.station / f"seed_{args.seed}"
    job_out.mkdir(parents=True, exist_ok=True)

    tensors = edata.load_fold_tensors(args.artifacts, args.family, args.station)
    fold_meta = _load_fold_meta(args.artifacts, args.family, args.station)
    test = _load_fold_test(args.artifacts, args.family, args.station)

    outcome = train_and_predict(cfg, tensors, fold_meta, job_out, resume=args.resume)
    np.save(job_out / "preds.npy", outcome.preds)

    pooled, yearly, cluster = fold_metrics(
        test["y_test"], outcome.preds, outcome.per_cluster, test["year"]
    )

    val_rmses = [pc["val_rmse"] for pc in outcome.per_cluster.values() if pc["val_rmse"] == pc["val_rmse"]]
    val_ns = [pc["n_train"] for pc in outcome.per_cluster.values() if pc["val_rmse"] == pc["val_rmse"]]
    aux_rmses = [pc["aux_rmse"] for pc in outcome.per_cluster.values() if pc["aux_rmse"] == pc["aux_rmse"]]
    aux_ns = [pc["n_train"] for pc in outcome.per_cluster.values() if pc["aux_rmse"] == pc["aux_rmse"]]

    payload = {
        "family": args.family,
        "config_id": args.config_id,
        "station": args.station,
        "seed": args.seed,
        "config": cfg,
        "n_train_total": int(fold_meta.get("n_train_total", 0)),
        "n_test": int(len(test["y_test"])),
        "val_rmse": _pooled_rmse(val_rmses, val_ns),
        "aux_rmse": _pooled_rmse(aux_rmses, aux_ns),
        "test": pooled,
        "yearly": {str(k): v for k, v in yearly.items()},
        "per_cluster": {
            cl: {
                "n_train": pc["n_train"],
                "n_test": pc["n_test"],
                "val_rmse": pc["val_rmse"],
                "aux_rmse": pc["aux_rmse"],
                "best_epoch": pc["best_epoch"],
                "epochs": pc["epochs"],
                "train_time_s": pc["train_time_s"],
                "n_params": pc["n_params"],
                "test": cluster[cl],
            }
            for cl, pc in outcome.per_cluster.items()
        },
        "epochs": max((pc["epochs"] for pc in outcome.per_cluster.values()), default=0),
        "best_epoch": max((pc["best_epoch"] for pc in outcome.per_cluster.values()), default=0),
        "train_time_s": outcome.train_time_s,
        "n_params": outcome.n_params,
        "device": "cuda" if _cuda() else "cpu",
        "status": "completed",
    }
    with open(job_out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[job] {args.family}/{args.config_id}/{args.station}/seed{args.seed} "
          f"r2={pooled['r2']:.4f} rmse={pooled['rmse']:.4f} val={payload['val_rmse']:.5f} "
          f"time={outcome.train_time_s:.1f}s", flush=True)


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
