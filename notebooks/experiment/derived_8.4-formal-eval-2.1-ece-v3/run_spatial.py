#!/usr/bin/env python3
"""Spatial (In-Situ ECE) evaluation driver for derived_8.4-formal-eval-2.1-ece-v3 (30 seeds).

Evaluates the 20 pinned configurations x 30 seeds on derived_8.4_ece_v3 (5 in-situ stations:
ECE_BBG_Main_St, ECE_BBG_Lost_Meadow, ECE_Renton_Home, ECE_Renton_Garden_North, ECE_Renton_Garden_Shed;
150 rows across 2026-07-20 to 2026-08-19).

Aggregates per-job meta.json into:
    spatial_seed_summary.csv      pooled ECE metrics per (config, seed)
    spatial_seed_station.csv      per (config, seed, station) across 5 ECE stations
    spatial_seed_year.csv         per (config, seed, year) across 2026
    spatial_seed_cluster.csv      per (config, seed, cluster) on ECE

Usage:
    python run_spatial.py                                  # full run (20 configs x 30 seeds)
    python run_spatial.py --smoke --max-configs 2          # CPU smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from eval_formal.configs import config_frame, load_pinned_configs  # noqa: E402
from eval_formal.data import load_experiment_data  # noqa: E402
from eval_formal.jobs import (  # noqa: E402
    ARTIFACTS_DIR,
    expected_version,
    job_complete,
    load_job_meta,
    make_jobs,
    run_jobs,
    set_smoke,
    write_runtime,
)

PROJECT_ROOT = EXP_DIR.parents[2]
CONFIG_PATH = EXP_DIR / "config.yaml"

METRIC_KEYS = ("r2", "rmse", "ubrmse", "bias", "mae", "pearson")


def aggregate_spatial(config: dict, configurations: list[dict],
                      seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read per-job meta.json into the four spatial CSVs (only completed jobs)."""
    version = expected_version(config, "spatial")
    predictions_dir = EXP_DIR / Path(config["spatial"]["predictions_dir"])
    models_dir = EXP_DIR / Path(config["temporal"]["models_dir"])

    seed_rows, station_rows, year_rows, cluster_rows = [], [], [], []
    for cfg in configurations:
        for seed in seeds:
            if not job_complete(cfg["config_id"], seed, "ece", version,
                                predictions_dir=predictions_dir, models_dir=models_dir):
                continue
            meta = load_job_meta(cfg["config_id"], seed, "ece")
            pooled = meta["pooled"]
            seed_rows.append({
                "config_id": cfg["config_id"],
                "strategy_name": meta["strategy_name"],
                "seed": seed,
                "n_train_total": int(meta["n_train_total"]),
                "n_test": int(meta["n_test"]),
                "train_time_s": float(meta["train_time_s"]),
                **{k: pooled.get(k, float("nan")) for k in METRIC_KEYS},
            })
            for st, m in (meta.get("per_station") or {}).items():
                station_rows.append({
                    "config_id": cfg["config_id"],
                    "seed": seed,
                    "station": st,
                    "n_test": int(m.get("n_test", 0)),
                    **{k: m.get(k, float("nan")) for k in METRIC_KEYS},
                })
            for year, m in (meta.get("yearly") or {}).items():
                year_rows.append({
                    "config_id": cfg["config_id"],
                    "seed": seed,
                    "year": int(year),
                    **{k: m.get(k, float("nan")) for k in METRIC_KEYS},
                })
            for cl, m in (meta.get("per_cluster") or {}).items():
                cluster_rows.append({
                    "config_id": cfg["config_id"],
                    "seed": seed,
                    "cluster": int(cl),
                    "n_train": int(m.get("n_train", 0)),
                    "n_test": int(m.get("n_test", 0)),
                    **{k: m.get(k, float("nan")) for k in METRIC_KEYS},
                })
    return (pd.DataFrame(seed_rows), pd.DataFrame(station_rows),
            pd.DataFrame(year_rows), pd.DataFrame(cluster_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-configs", type=int, default=None, help="Limit configs (smoke).")
    parser.add_argument("--config-id", action="append", default=None,
                        help="Only run these config ids (repeatable).")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds to run.")
    parser.add_argument("--n-parallel", type=int, default=None, help="Concurrent GPU workers.")
    parser.add_argument("--device", default=None, help="Override model device (e.g. cpu).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing results.")
    parser.add_argument("--smoke", action="store_true",
                        help="data_version -1 + n_estimators 100 + CPU (never reused).")
    args = parser.parse_args()

    set_smoke(args.smoke)

    print("=" * 70, flush=True)
    print("Spatial (In-Situ ECE) evaluation — derived_8.4-formal-eval-2.1-ece-v3", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_experiment_data(PROJECT_ROOT, config)
    configurations = load_pinned_configs(data, config)
    if args.config_id:
        configurations = [c for c in configurations if c["config_id"] in args.config_id]
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    seeds = list(args.seeds) if args.seeds else list(config["seeds"]["spatial"])
    n_parallel = args.n_parallel or int(config["spatial"].get("n_parallel", 8))

    print(f"[Data] TrainVal (WA)={len(data.trainval)} ECE Test (5 stations)={len(data.ece_all)}", flush=True)
    print(f"[Configs] {len(configurations)} pinned configs", flush=True)
    print(f"[Seeds] {len(seeds)} spatial seeds: {seeds}", flush=True)
    print(f"[ECE Stations] {len(data.ece_stations)} stations: {data.ece_stations}", flush=True)

    # Pin configurations so workers resolve the same feature sets even when
    # run_spatial.py is executed standalone (mirrors run_temporal.py).
    # Never silently reuse a stale pin from a smoke run: rewrite unconditionally.
    pin_path = EXP_DIR / "pinned_configurations.json"
    with open(pin_path, "w", encoding="utf-8") as f:
        json.dump(configurations, f, indent=2)
    config_frame(configurations).to_csv(EXP_DIR / "pinned_configs.csv", index=False)

    write_runtime(config, device=args.device or ("cpu" if args.smoke else None),
                  n_estimators=100 if args.smoke else None,
                  version=expected_version(config, "spatial"))

    predictions_dir = EXP_DIR / Path(config["spatial"]["predictions_dir"])
    models_dir = EXP_DIR / Path(config["temporal"]["models_dir"])
    jobs = make_jobs([c["config_id"] for c in configurations], seeds, ["ece"], config,
                     "spatial", predictions_dir, models_dir, no_resume=args.no_resume)
    todo = [j for j in jobs if j[3] != "skip"]
    print(f"[Jobs] {len(jobs)} total ({len(jobs) - len(todo)} done, {len(todo)} to run) "
          f"with {n_parallel} parallel workers.", flush=True)
    if todo:
        run_jobs(todo, n_parallel)

    seed_df, station_df, year_df, cluster_df = aggregate_spatial(config, configurations, seeds)
    if seed_df.empty:
        print("[Error] No completed jobs — check artifacts/logs/*.log.", flush=True)
        raise SystemExit(1)
    expected = len(configurations) * len(seeds)
    if len(seed_df) != expected:
        done = {(r["config_id"], int(r["seed"])) for _, r in seed_df.iterrows()}
        missing = [(c["config_id"], s) for c in configurations for s in seeds
                   if (c["config_id"], s) not in done][:20]
        print(f"[Error] Incomplete spatial grid: got {len(seed_df)}/{expected}. "
              f"Missing e.g. {missing}. Check artifacts/logs/*.log.", flush=True)
        raise SystemExit(1)
    seed_df.to_csv(EXP_DIR / "spatial_seed_summary.csv", index=False)
    station_df.to_csv(EXP_DIR / "spatial_seed_station.csv", index=False)
    year_df.to_csv(EXP_DIR / "spatial_seed_year.csv", index=False)
    cluster_df.to_csv(EXP_DIR / "spatial_seed_cluster.csv", index=False)
    print(f"[Artifacts] Wrote spatial_seed_summary.csv ({len(seed_df)} rows), "
          f"spatial_seed_station.csv ({len(station_df)}), spatial_seed_year.csv "
          f"({len(year_df)}), spatial_seed_cluster.csv ({len(cluster_df)})", flush=True)


if __name__ == "__main__":
    main()
