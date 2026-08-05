#!/usr/bin/env python3
"""Full-training baseline for derived_8.4-eval-1.2 (replicates derived_8.4-eval-1.1).

LOSO is an *addition* to the experiment, not a replacement: this script trains every
configuration WITHOUT leave-one-station-out — router fit and experts trained on the
entire trainval (all 7 stations), evaluated on the full test set — exactly the eval-1.1
protocol. Per-station test metrics are collected so station difficulty under full
training (the *intrinsic* difficulty) can be contrasted with LOSO difficulty (how much
a station suffers when held out).

Each configuration's pooled test R2 is validated against eval-1.1's tracked results
(`delta_grid_summary.csv` / `metrics_summary.csv`, i.e. `eval11_test_r2`) and must match
within ~1e-3, proving the baseline faithfully replicates 1.1.

Usage:
    python run_full_baseline.py                     # full run (47 configs)
    python run_full_baseline.py --max-configs 2     # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eval12.data import load_experiment_data
from eval12.evaluator import LosoEvaluator
from run_loso import build_config_frame, load_configurations

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.yaml"
PREDICTIONS_DIR = EXP_DIR / "predictions_full"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs (smoke test).")
    parser.add_argument("--config-id", action="append", default=None, help="Only run these config ids (repeatable).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial results.")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("Starting derived_8.4-eval-1.2 Full-Training Baseline (replicates eval-1.1)", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_experiment_data(PROJECT_ROOT, config)
    print(f"[Data] TrainVal={len(data.trainval)} samples, Test={len(data.test)} samples.", flush=True)

    configurations = load_configurations(data, config)
    if args.config_id:
        configurations = [c for c in configurations if c["config_id"] in args.config_id]
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    print(f"[Configs] Evaluating {len(configurations)} configs on full trainval (no LOSO).", flush=True)

    cfg_frame = build_config_frame(configurations)
    eval11_r2 = cfg_frame.set_index("config_id")["eval11_test_r2"].to_dict()

    # Resume: skip configurations already present in the partial output. Pooled
    # rows are rebuilt from the per-config meta JSONs (schema-stable) so resuming
    # never mixes old summary columns with new rows.
    done: set[str] = set()
    per_config_station: list[dict] = []
    pooled_rows: list[dict] = []
    if not args.no_resume and (EXP_DIR / "full_per_config_station.csv").exists():
        prev = pd.read_csv(EXP_DIR / "full_per_config_station.csv")
        done = set(prev["config_id"].unique())
        per_config_station = prev.to_dict(orient="records")
        for cid in done:
            meta_path = PREDICTIONS_DIR / f"{cid}__full_meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pooled_rows.append({
                "config_id": cid,
                "strategy_name": meta["strategy_name"],
                "n_train_total": meta["n_train_total"],
                "n_test": meta["n_test"],
                "full_pooled_r2": meta["pooled_r2"],
                "full_pooled_rmse": meta["pooled_rmse"],
                "full_pooled_ubrmse": meta.get("pooled_ubrmse", float("nan")),
                "full_pooled_bias": meta.get("pooled_bias", float("nan")),
                "full_pooled_mae": meta.get("pooled_mae", float("nan")),
                "train_time_s": meta["train_time_s"],
            })
        print(f"[Resume] Found {len(done)} completed configs; skipping those.", flush=True)

    total = len(configurations)
    completed = 0
    for cfg in configurations:
        config_id = cfg["config_id"]
        if config_id in done:
            completed += 1
            continue

        evaluator = LosoEvaluator(
            data,
            config,
            config_id=config_id,
            strategy_name=cfg["strategy_name"],
            global_features=cfg["global_features"],
            cluster_additions=cfg["cluster_additions"],
            models_dir=None,  # no weights for the full baseline (predictions suffice)
            predictions_dir=PREDICTIONS_DIR,
            save_weights=False,
            save_predictions=True,
        )
        res = evaluator.evaluate_full()
        completed += 1

        for station, m in res.station_metrics.items():
            per_config_station.append({
                "config_id": config_id,
                "station": station,
                "strategy_name": res.strategy_name,
                "n_train_total": res.n_train_total,
                "n_test": m["n"],
                "r2": m["r2"],
                "rmse": m["rmse"],
                "ubrmse": m["ubrmse"],
                "bias": m["bias"],
                "mae": m["mae"],
                "pearson": m["pearson"],
            })
        pooled_rows.append({
            "config_id": config_id,
            "strategy_name": res.strategy_name,
            "n_train_total": res.n_train_total,
            "n_test": res.pooled["n"],
            "full_pooled_r2": res.pooled["r2"],
            "full_pooled_rmse": res.pooled["rmse"],
            "full_pooled_ubrmse": res.pooled["ubrmse"],
            "full_pooled_bias": res.pooled["bias"],
            "full_pooled_mae": res.pooled["mae"],
            "train_time_s": res.train_time_s,
        })

        _write_partial(per_config_station, pooled_rows)
        print(f"[Config] Finished {config_id} ({completed}/{total}), pooled R2={res.pooled['r2']:.4f}", flush=True)

    df_pcs = pd.DataFrame(per_config_station)
    df_pooled = pd.DataFrame(pooled_rows)

    # ---- Per-configuration summary (pooled + per-station spread + validation) ----
    summary = df_pooled.copy()
    station_agg = df_pcs.groupby("config_id").agg(
        n_stations=("station", "count"),
        full_station_mean_r2=("r2", "mean"),
        full_station_median_r2=("r2", "median"),
        full_station_min_r2=("r2", "min"),
        full_station_max_r2=("r2", "max"),
    ).reset_index()
    summary = summary.merge(station_agg, on="config_id", how="left")
    summary = summary.merge(
        cfg_frame[["config_id", "config_label", "strategy_name", "is_baseline", "is_winner",
                   "eval11_test_r2"]],
        on="config_id", how="left",
    )
    summary["r2_diff"] = (summary["full_pooled_r2"] - summary["eval11_test_r2"]).abs()
    summary = summary.sort_values("full_pooled_r2", ascending=False).reset_index(drop=True)
    summary.to_csv(EXP_DIR / "full_config_summary.csv", index=False)

    # ---- Per-station summary (intrinsic difficulty under full training) ----
    station_summary = df_pcs.groupby("station").agg(
        n_configs=("config_id", "count"),
        total_test_n=("n_test", "sum"),
        mean_r2=("r2", "mean"),
        median_r2=("r2", "median"),
        std_r2=("r2", "std"),
        min_r2=("r2", "min"),
        max_r2=("r2", "max"),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
        mean_bias=("bias", "mean"),
    ).reset_index()
    neg = (
        df_pcs.groupby("station")["r2"]
        .apply(lambda r: int((r < 0).sum()))
        .rename("n_negative_r2")
        .reset_index()
    )
    station_summary = station_summary.merge(neg, on="station", how="left")
    station_summary = station_summary.sort_values("median_r2", ascending=False).reset_index(drop=True)
    station_summary.to_csv(EXP_DIR / "full_station_summary.csv", index=False)

    df_pcs.to_csv(EXP_DIR / "full_per_config_station.csv", index=False)
    print(f"[Artifacts] Wrote {EXP_DIR / 'full_per_config_station.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'full_config_summary.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'full_station_summary.csv'}", flush=True)

    # ---- Validation vs eval-1.1 (replication check) ----
    valid = summary.dropna(subset=["eval11_test_r2"])
    max_diff = float(valid["r2_diff"].max()) if len(valid) else float("nan")
    print("\nVALIDATION vs derived_8.4-eval-1.1 (pooled test R2):", flush=True)
    print(f"  configs compared: {len(valid)}, max |full_pooled_r2 - eval11_test_r2| = {max_diff:.6f}", flush=True)
    if max_diff < 1e-3:
        print("  PASS: full-training baseline replicates eval-1.1 within 1e-3.", flush=True)
    else:
        print("  WARNING: max diff >= 1e-3 — investigate (see r2_diff in full_config_summary.csv).", flush=True)

    print("\nSTATION DIFFICULTY UNDER FULL TRAINING (median R2 over configs; intrinsic difficulty)", flush=True)
    print(station_summary[["station", "median_r2", "mean_r2", "min_r2", "max_r2", "n_negative_r2"]].to_string(index=False), flush=True)


def _write_partial(per_config_station: list[dict], pooled_rows: list[dict]) -> None:
    """Crash-safety: flush accumulated rows to CSVs."""
    pd.DataFrame(per_config_station).to_csv(EXP_DIR / "full_per_config_station.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(EXP_DIR / "full_config_summary.csv", index=False)


if __name__ == "__main__":
    main()
