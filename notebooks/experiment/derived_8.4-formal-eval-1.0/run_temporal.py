#!/usr/bin/env python3
"""Temporal evaluation driver for derived_8.4-formal-eval-1.0 (frozen split, 30 seeds).

Pins the 20 configurations, spawns one run_worker.py job per (config, seed) on the
GPU ("full" station = temporal full-baseline protocol: experts on trainval, test on
the frozen 2023-2025 split), then aggregates per-job meta.json into:

    temporal_seed_summary.csv     pooled metrics per (config, seed)
    temporal_seed_station.csv     per (config, seed, station)
    temporal_seed_year.csv        per (config, seed, year)
    temporal_seed_cluster.csv     per (config, seed, cluster)

Resume: completed jobs are skipped (artifacts/jobs/<config_id>__s<seed>__full/
meta.json + prediction/weight file presence, data_version match). `--smoke` uses
data_version=-1 + n_estimators=100 + CPU so smoke artifacts are never reused.

Usage:
    python run_temporal.py                                  # full run (20 configs x 30 seeds)
    python run_temporal.py --smoke --max-configs 2          # CPU smoke
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


def aggregate_temporal(config: dict, configurations: list[dict],
                       seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read per-job meta.json into the four temporal CSVs (only completed jobs)."""
    version = expected_version(config, "temporal")
    predictions_dir = EXP_DIR / Path(config["temporal"]["predictions_dir"])
    models_dir = EXP_DIR / Path(config["temporal"]["models_dir"])

    seed_rows, station_rows, year_rows, cluster_rows = [], [], [], []
    for cfg in configurations:
        for seed in seeds:
            if not job_complete(cfg["config_id"], seed, "full", version,
                                predictions_dir=predictions_dir, models_dir=models_dir):
                continue
            meta = load_job_meta(cfg["config_id"], seed, "full")
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
    print("Temporal evaluation (frozen split, multi-seed) — derived_8.4-formal-eval-1.0", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_experiment_data(PROJECT_ROOT, config)
    configurations = load_pinned_configs(data, config)
    if args.config_id:
        configurations = [c for c in configurations if c["config_id"] in args.config_id]
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    seeds = list(args.seeds) if args.seeds else list(config["seeds"]["temporal"])
    n_parallel = args.n_parallel or int(config["temporal"].get("n_parallel", 8))

    print(f"[Data] TrainVal={len(data.trainval)} Test={len(data.test)}", flush=True)
    print(f"[Configs] {len(configurations)} pinned configs", flush=True)
    print(f"[Seeds] {len(seeds)} temporal seeds: {seeds}", flush=True)

    # Pin ALL configurations (provenance / audit trail) before any run filter.
    pin_path = EXP_DIR / "pinned_configurations.json"
    with open(pin_path, "w", encoding="utf-8") as f:
        json.dump(configurations, f, indent=2)
    config_frame(configurations).to_csv(EXP_DIR / "pinned_configs.csv", index=False)

    write_runtime(config, device=args.device or ("cpu" if args.smoke else None),
                  n_estimators=100 if args.smoke else None,
                  version=expected_version(config, "temporal"))

    predictions_dir = EXP_DIR / Path(config["temporal"]["predictions_dir"])
    models_dir = EXP_DIR / Path(config["temporal"]["models_dir"])
    jobs = make_jobs([c["config_id"] for c in configurations], seeds, ["full"], config,
                     "temporal", predictions_dir, models_dir, no_resume=args.no_resume)
    todo = [j for j in jobs if j[3] != "skip"]
    print(f"[Jobs] {len(jobs)} total ({len(jobs) - len(todo)} done, {len(todo)} to run) "
          f"with {n_parallel} parallel workers.", flush=True)
    run_jobs(todo, n_parallel)

    seed_df, station_df, year_df, cluster_df = aggregate_temporal(config, configurations, seeds)
    if seed_df.empty:
        print("[Error] No completed jobs — check artifacts/logs/*.log.", flush=True)
        raise SystemExit(1)
    seed_df.to_csv(EXP_DIR / "temporal_seed_summary.csv", index=False)
    station_df.to_csv(EXP_DIR / "temporal_seed_station.csv", index=False)
    year_df.to_csv(EXP_DIR / "temporal_seed_year.csv", index=False)
    cluster_df.to_csv(EXP_DIR / "temporal_seed_cluster.csv", index=False)
    print(f"[Artifacts] Wrote temporal_seed_summary.csv ({len(seed_df)} rows), "
          f"temporal_seed_station.csv ({len(station_df)}), temporal_seed_year.csv "
          f"({len(year_df)}), temporal_seed_cluster.csv ({len(cluster_df)})", flush=True)

    # Replication anchors: seed-42 pooled R2 must match eval-1.1 (via eval-1.3 full baseline).
    print("\nSEED-42 REPLICATION CHECK (vs eval-1.1 pooled test R2)", flush=True)
    if args.smoke:
        print("  skipped in smoke mode (n_estimators=100 — values are not comparable)", flush=True)
    anchors = config.get("replication", {}).get("temporal_r2", {})
    for config_id, expected in anchors.items():
        row = seed_df[(seed_df["config_id"] == config_id) & (seed_df["seed"] == 42)]
        if row.empty:
            print(f"  {config_id}: MISSING (job not complete)", flush=True)
            continue
        got = float(row.iloc[0]["r2"])
        diff = abs(got - float(expected))
        ok = "OK" if diff < 1e-6 else "MISMATCH"
        print(f"  {config_id}: got={got:.6f} expected={expected:.6f} |diff|={diff:.2e} [{ok}]",
              flush=True)


if __name__ == "__main__":
    main()
