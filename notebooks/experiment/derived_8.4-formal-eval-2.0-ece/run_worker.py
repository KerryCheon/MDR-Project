#!/usr/bin/env python3
"""Train or evaluate a single derived_8.4-formal-eval-2.0-ece job: (config_id, seed, target).

target:
  - 'full': Temporal evaluation on Washington test set (2023-2025, 7 stations).
  - 'ece': Spatial evaluation on derived_8.4-ece (2026, 5 in-situ stations in WA).

Usage:
    python run_worker.py --config-id <id> --seed <s> --target <full|ece> \
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

from eval_formal.data import load_experiment_data  # noqa: E402
from eval_formal.evaluator import FormalEvaluator  # noqa: E402

PROJECT_ROOT = EXP_DIR.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", required=True, choices=["full", "ece"],
                        help="'full' for WA temporal or 'ece' for In-Situ ECE spatial")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(EXP_DIR / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    runtime = json.loads((args.artifacts / "runtime.json").read_text(encoding="utf-8"))

    pinned = json.loads((args.out / "pinned_configurations.json").read_text(encoding="utf-8"))
    cfg = next((c for c in pinned if c["config_id"] == args.config_id), None)
    if cfg is None:
        raise SystemExit(f"config_id {args.config_id} not found in pinned_configurations.json")

    # Apply per-run overrides
    params = dict(config["model"]["exact_params"])
    params["device"] = str(runtime.get("device", params.get("device", "cuda")))
    if runtime.get("n_estimators") is not None:
        params["n_estimators"] = int(runtime["n_estimators"])
    config["model"]["exact_params"] = params

    data = load_experiment_data(PROJECT_ROOT, config)

    is_full = args.target == "full"
    if is_full:
        predictions_dir = EXP_DIR / Path(config["temporal"]["predictions_dir"])
        save_weights = bool(config["temporal"].get("save_weights", True))
        save_predictions = bool(config["temporal"].get("save_predictions", True))
    else:
        predictions_dir = EXP_DIR / Path(config["spatial"]["predictions_dir"])
        save_weights = False
        save_predictions = bool(config["spatial"].get("save_predictions", True))
    models_dir = EXP_DIR / Path(config["temporal"]["models_dir"])

    evaluator = FormalEvaluator(
        data,
        config,
        config_id=cfg["config_id"],
        strategy_name=cfg["strategy_name"],
        seed=args.seed,
        global_features=cfg["global_features"],
        cluster_additions=cfg["cluster_additions"],
        models_dir=models_dir,
        predictions_dir=predictions_dir if save_predictions else None,
        save_weights=save_weights,
        save_predictions=save_predictions,
    )

    result = evaluator.evaluate_full() if is_full else evaluator.evaluate_spatial_ece()

    per_cluster = {
        str(cl): {
            "n_train": float(m.get("n_train", 0.0)),
            "n_test": float(m.get("n", m.get("n_test", 0.0))),
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

    payload = {
        "config_id": cfg["config_id"],
        "strategy_name": cfg["strategy_name"],
        "seed": args.seed,
        "target": args.target,
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
    job_dir = args.artifacts / "jobs" / f"{args.config_id}__s{args.seed}__{args.target}"
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[job] {args.config_id} s{args.seed} @ {args.target} r2={pooled['r2']:.4f} "
          f"rmse={pooled['rmse']:.4f} time={result.train_time_s:.1f}s "
          f"device={params['device']}", flush=True)


if __name__ == "__main__":
    main()
