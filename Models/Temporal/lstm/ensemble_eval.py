"""
LSTM Track A - seed ensembling.

Averages the 5 per-seed val/test predictions from train_v23_baseline.py
(no retraining) and compares the ensemble against the best single seed and
the 5-seed mean, for both configs (top25_seq30, full58_seq10).

Usage (run after train_v23_baseline.py):
    python -m Models.Temporal.lstm.ensemble_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.eval_common import evaluate
from Models.Temporal.lstm.train_v20 import _json_safe
from Models.Temporal.lstm.train_v23_baseline import CONFIGS, OUT_DIR, SEEDS


def _load_and_average(config_dir: Path, seeds: list[int], split: str):
    frames = []
    for seed in seeds:
        path = config_dir / f"seed{seed}" / f"{split}_preds.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.rename(columns={"y_pred": f"y_pred_seed{seed}"})
        frames.append(df)

    merged = frames[0][["station_id", "date", "y_true"]].copy()
    for seed, df in zip(seeds, frames):
        merged[f"y_pred_seed{seed}"] = df[f"y_pred_seed{seed}"]

    counts = merged.groupby(["station_id", "date"]).size()
    assert (counts == 1).all(), "duplicate (station_id, date) rows within a single seed's predictions"
    # each seed's df must cover the identical (station_id, date) row set,
    # since build_datasets_v2's row set is seed-independent
    for seed, df in zip(seeds, frames):
        assert len(df) == len(merged), f"seed {seed} row count {len(df)} != {len(merged)}"

    pred_cols = [f"y_pred_seed{seed}" for seed in seeds]
    merged["y_pred_ensemble"] = merged[pred_cols].mean(axis=1)
    return merged


def main():
    combined = {}
    for config_name in CONFIGS:
        config_dir = OUT_DIR / config_name
        print(f"\n[{config_name}]")

        test_merged = _load_and_average(config_dir, SEEDS, "test")
        ensemble_eval = evaluate(
            test_merged["y_true"].to_numpy(),
            test_merged["y_pred_ensemble"].to_numpy(),
            test_merged["station_id"].to_numpy(),
            test_merged["date"].to_numpy(),
        )

        per_seed_r2 = {}
        for seed in SEEDS:
            metrics_path = config_dir / f"seed{seed}" / "metrics.json"
            with open(metrics_path) as f:
                m = json.load(f)
            per_seed_r2[seed] = m["test"]["r2"]

        best_seed = max(per_seed_r2, key=per_seed_r2.get)
        mean_seed_r2 = float(np.mean(list(per_seed_r2.values())))
        ensemble_r2 = ensemble_eval["overall"]["r2"]

        print(f"  per-seed test R2: {per_seed_r2}")
        print(f"  best single seed: seed={best_seed} R2={per_seed_r2[best_seed]:.4f}")
        print(f"  5-seed mean R2:   {mean_seed_r2:.4f}")
        print(f"  ensemble R2:      {ensemble_r2:.4f}")
        print(f"  ensemble beats best single seed: {ensemble_r2 > per_seed_r2[best_seed]}")
        print(f"  ensemble beats 5-seed mean:       {ensemble_r2 > mean_seed_r2}")

        combined[config_name] = {
            "per_seed_test_r2": per_seed_r2,
            "best_single_seed": best_seed,
            "best_single_seed_test_r2": per_seed_r2[best_seed],
            "mean_seed_test_r2": mean_seed_r2,
            "ensemble_test_eval": ensemble_eval,
            "ensemble_beats_best_seed": bool(ensemble_r2 > per_seed_r2[best_seed]),
            "ensemble_beats_mean": bool(ensemble_r2 > mean_seed_r2),
        }

    out_path = OUT_DIR / "ensemble_metrics.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=_json_safe)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
