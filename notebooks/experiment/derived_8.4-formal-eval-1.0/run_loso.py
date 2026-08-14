#!/usr/bin/env python3
"""LOSO spatial evaluation driver for derived_8.4-formal-eval-1.0 (5 seeds).

Same pinned 20 configurations; for each held-out station the router is refitted on
the 6-station trainval (no held-out-station leakage into routing) and experts are
trained per regime cluster, evaluated on the held-out station's 2023-2025 test rows
(identical protocol to derived_8.4-eval-1.3). Spawns one run_worker.py job per
(config, seed, station), then aggregates per-job meta.json into:

    loso_seed_station.csv      per (config, seed, station) metrics
    loso_seed_year.csv         per (config, seed, station, year)
    loso_seed_cluster.csv      per (config, seed, station, cluster)

Resume: completed folds are skipped (meta.json + file presence, data_version match).
`--smoke` uses data_version=-1 + n_estimators=100 + CPU.

Usage:
    python run_loso.py                                    # full run (20 configs x 5 seeds x 7 stations)
    python run_loso.py --smoke --max-configs 1 --max-stations 1
"""

from __future__ import annotations

import argparse
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


def aggregate_loso(config: dict, configurations: list[dict], seeds: list[int],
                   stations: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    version = expected_version(config, "loso")
    predictions_dir = EXP_DIR / Path(config["loso"]["predictions_dir"])
    models_dir = EXP_DIR / Path(config["temporal"]["models_dir"])

    station_rows, year_rows, cluster_rows = [], [], []
    for cfg in configurations:
        for seed in seeds:
            for station in stations:
                if not job_complete(cfg["config_id"], seed, station, version,
                                    predictions_dir=predictions_dir, models_dir=models_dir):
                    continue
                meta = load_job_meta(cfg["config_id"], seed, station)
                pooled = meta["pooled"]
                station_rows.append({
                    "config_id": cfg["config_id"],
                    "strategy_name": meta["strategy_name"],
                    "seed": seed,
                    "station": station,
                    "n_train_total": int(meta["n_train_total"]),
                    "n_test": int(meta["n_test"]),
                    **{k: pooled.get(k, float("nan")) for k in METRIC_KEYS},
                })
                for year, m in (meta.get("yearly") or {}).items():
                    year_rows.append({
                        "config_id": cfg["config_id"],
                        "seed": seed,
                        "station": station,
                        "year": int(year),
                        **{k: m.get(k, float("nan")) for k in METRIC_KEYS},
                    })
                for cl, m in (meta.get("per_cluster") or {}).items():
                    cluster_rows.append({
                        "config_id": cfg["config_id"],
                        "seed": seed,
                        "station": station,
                        "cluster": int(cl),
                        "n_train": int(m.get("n_train", 0)),
                        "n_test": int(m.get("n_test", 0)),
                        **{k: m.get(k, float("nan")) for k in METRIC_KEYS},
                    })
    return (pd.DataFrame(station_rows), pd.DataFrame(year_rows), pd.DataFrame(cluster_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-configs", type=int, default=None, help="Limit configs (smoke).")
    parser.add_argument("--config-id", action="append", default=None,
                        help="Only run these config ids (repeatable).")
    parser.add_argument("--max-stations", type=int, default=None, help="Limit stations (smoke).")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds to run.")
    parser.add_argument("--n-parallel", type=int, default=None, help="Concurrent GPU workers.")
    parser.add_argument("--device", default=None, help="Override model device (e.g. cpu).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing results.")
    parser.add_argument("--smoke", action="store_true",
                        help="data_version -1 + n_estimators 100 + CPU (never reused).")
    args = parser.parse_args()

    set_smoke(args.smoke)

    print("=" * 70, flush=True)
    print("LOSO spatial evaluation (multi-seed) — derived_8.4-formal-eval-1.0", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_experiment_data(PROJECT_ROOT, config)
    configurations = load_pinned_configs(data, config)
    if args.config_id:
        configurations = [c for c in configurations if c["config_id"] in args.config_id]
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    stations = sorted(data.test["station_id"].unique())
    if args.max_stations:
        stations = stations[: args.max_stations]
    seeds = list(args.seeds) if args.seeds else list(config["seeds"]["loso"])
    n_parallel = args.n_parallel or int(config["loso"].get("n_parallel", 8))

    print(f"[Data] TrainVal={len(data.trainval)} Test={len(data.test)}", flush=True)
    print(f"[Configs] {len(configurations)} pinned configs", flush=True)
    print(f"[Seeds] {len(seeds)} LOSO seeds: {seeds}", flush=True)
    print(f"[Stations] {len(stations)} held-out stations: {stations}", flush=True)

    write_runtime(config, device=args.device or ("cpu" if args.smoke else None),
                  n_estimators=100 if args.smoke else None,
                  version=expected_version(config, "loso"))

    predictions_dir = EXP_DIR / Path(config["loso"]["predictions_dir"])
    models_dir = EXP_DIR / Path(config["temporal"]["models_dir"])
    jobs = make_jobs([c["config_id"] for c in configurations], seeds, stations, config,
                     "loso", predictions_dir, models_dir, no_resume=args.no_resume)
    todo = [j for j in jobs if j[3] != "skip"]
    print(f"[Jobs] {len(jobs)} total ({len(jobs) - len(todo)} done, {len(todo)} to run) "
          f"with {n_parallel} parallel workers.", flush=True)
    run_jobs(todo, n_parallel)

    station_df, year_df, cluster_df = aggregate_loso(config, configurations, seeds, stations)
    if station_df.empty:
        print("[Error] No completed jobs — check artifacts/logs/*.log.", flush=True)
        raise SystemExit(1)
    station_df.to_csv(EXP_DIR / "loso_seed_station.csv", index=False)
    year_df.to_csv(EXP_DIR / "loso_seed_year.csv", index=False)
    cluster_df.to_csv(EXP_DIR / "loso_seed_cluster.csv", index=False)
    print(f"[Artifacts] Wrote loso_seed_station.csv ({len(station_df)} rows), "
          f"loso_seed_year.csv ({len(year_df)}), loso_seed_cluster.csv ({len(cluster_df)})",
          flush=True)

    # Replication anchors: seed-42 loso_mean_r2 must match eval-1.2/-1.3.
    print("\nSEED-42 REPLICATION CHECK (LOSO mean R2 vs eval-1.2/-1.3)", flush=True)
    if args.smoke:
        print("  skipped in smoke mode (n_estimators=100 — values are not comparable)", flush=True)
    anchors = config.get("replication", {}).get("loso_mean_r2", {})
    for config_id, expected in anchors.items():
        sub = station_df[(station_df["config_id"] == config_id) & (station_df["seed"] == 42)]
        if len(sub) < 7:
            print(f"  {config_id}: INCOMPLETE ({len(sub)}/7 folds)", flush=True)
            continue
        got = float(sub["r2"].mean())
        diff = abs(got - float(expected))
        ok = "OK" if diff < 1e-3 else "MISMATCH"
        print(f"  {config_id}: got={got:.4f} expected={expected:.4f} |diff|={diff:.4f} [{ok}]",
              flush=True)


if __name__ == "__main__":
    main()
