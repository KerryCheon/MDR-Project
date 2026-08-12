#!/usr/bin/env python3
"""Train a single XGBoost LOSO job: one (config_id, station) fold.

The driver (run_loso.py / run_full_baseline.py) pins loso_configurations.json,
writes artifacts/runtime.json {data_version, device, n_estimators}, and spawns
these workers as subprocesses (n_parallel concurrent, derived_8.4-eval-2.0
worker format). Each worker loads the derived_8.4 split, refits the router on
the fold's trainval, trains the per-regime experts, evaluates on the fold's
test rows, and persists:

    predictions/<config_id>__<station>_preds.npy / _labels_te.npy  (LOSO fold)
    predictions_full/<config_id>__full_preds.npy / _labels_te.npy  (station="full")
    models/<config_id>__<station>_{meta,spec_0/1,reg,gating}.json  (LOSO fold)
    artifacts/jobs/<config_id>__<station>/meta.json                (job status)

The job meta.json carries pooled / per-year / per-regime (+ per-station for
"full") metrics so the driver can aggregate without re-reading predictions.
Completed jobs are skipped by the driver (data_version match); XGBoost folds
are atomic, so interrupted jobs are simply re-run fresh.

Usage:
    python run_loso_worker.py --config-id <id> --station <s|full> \
        --artifacts <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from eval13.data import load_experiment_data  # noqa: E402
from eval13.evaluator import LosoEvaluator  # noqa: E402

# Repo root: parents of the EXP_DIR directory (not of a file) -> [2].
PROJECT_ROOT = EXP_DIR.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--station", required=True, help="held-out station id or 'full'")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(EXP_DIR / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    runtime = json.loads((args.artifacts / "runtime.json").read_text(encoding="utf-8"))

    pinned = json.loads((args.out / "loso_configurations.json").read_text(encoding="utf-8"))
    cfg = next((c for c in pinned if c["config_id"] == args.config_id), None)
    if cfg is None:
        raise SystemExit(f"config_id {args.config_id} not found in loso_configurations.json")

    # Apply per-run overrides (data_version is only recorded in the job meta).
    params = dict(config["model"]["exact_params"])
    params["device"] = str(runtime.get("device", params.get("device", "cuda")))
    if runtime.get("n_estimators") is not None:
        params["n_estimators"] = int(runtime["n_estimators"])
    config["model"]["exact_params"] = params

    data = load_experiment_data(PROJECT_ROOT, config)

    is_full = args.station == "full"
    if is_full:
        save_weights = False
        save_predictions = bool(config["full_baseline"].get("save_predictions", True))
        predictions_dir = EXP_DIR / Path(config["full_baseline"].get("predictions_dir", "predictions_full"))
    else:
        save_weights = bool(config["loso"].get("save_weights", True))
        save_predictions = bool(config["loso"].get("save_predictions", True))
        predictions_dir = EXP_DIR / "predictions"

    evaluator = LosoEvaluator(
        data,
        config,
        config_id=cfg["config_id"],
        strategy_name=cfg["strategy_name"],
        global_features=cfg["global_features"],
        cluster_additions=cfg["cluster_additions"],
        models_dir=EXP_DIR / "models" if save_weights else None,
        predictions_dir=predictions_dir if save_predictions else None,
        save_weights=save_weights,
        save_predictions=save_predictions,
    )

    result = evaluator.evaluate_full() if is_full else evaluator.evaluate_station(args.station)

    if is_full:
        per_cluster = {
            str(cl): {
                "n_train": float(m.get("n_train", 0.0)),
                "n_test": float(m.get("n", 0.0)),
                **{k: m.get(k) for k in ("r2", "rmse", "ubrmse", "bias", "mae", "pearson")},
            }
            for cl, m in result.cluster_metrics.items()
        }
        per_station = {
            str(s): {
                "n_test": float(m.get("n", 0.0)),
                **{k: m.get(k) for k in ("r2", "rmse", "ubrmse", "bias", "mae", "pearson")},
            }
            for s, m in result.station_metrics.items()
        }
        pooled = dict(result.pooled)
        pooled.pop("n", None)
    else:
        per_cluster = {
            str(cl): {
                "n_train": float(m.get("n_train", 0.0)),
                "n_test": float(m.get("n_test", 0.0)),
                **{k: m.get(k) for k in ("r2", "rmse", "ubrmse", "bias", "mae", "pearson")},
            }
            for cl, m in result.cluster_metrics.items()
        }
        per_station = {}
        pooled = {
            "r2": result.r2, "rmse": result.rmse, "ubrmse": result.ubrmse,
            "bias": result.bias, "mae": result.mae, "pearson": result.pearson,
        }

    payload = {
        "config_id": cfg["config_id"],
        "strategy_name": cfg["strategy_name"],
        "station": args.station,
        "data_version": int(runtime.get("data_version", 0)),
        "device": params["device"],
        "n_train_total": int(result.n_train_total),
        "n_test": int(result.n_test),
        "train_time_s": float(result.train_time_s),
        "pooled": pooled,
        "yearly": {str(k): v for k, v in result.yearly_metrics.items()},
        "per_cluster": per_cluster,
        "per_station": per_station,
        "status": "completed",
    }
    job_dir = args.artifacts / "jobs" / f"{args.config_id}__{args.station}"
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[job] {args.config_id} @ {args.station} r2={pooled['r2']:.4f} "
          f"rmse={pooled['rmse']:.4f} time={result.train_time_s:.1f}s "
          f"device={params['device']}", flush=True)


if __name__ == "__main__":
    main()
