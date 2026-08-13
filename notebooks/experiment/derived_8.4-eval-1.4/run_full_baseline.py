#!/usr/bin/env python3
"""Full-training baseline for derived_8.4-eval-1.4 (replicates derived_8.4-eval-1.1).

LOSO is an *addition* to the experiment, not a replacement: this script trains every
configuration WITHOUT leave-one-station-out — router fit and experts trained on the
entire trainval (all 7 stations), evaluated on the full test set — exactly the eval-1.1
protocol. Per-station test metrics are collected so station difficulty under full
training (the *intrinsic* difficulty) can be contrasted with LOSO difficulty (how much
a station suffers when held out). The pooled test R2 of configurations without an
eval-1.1 row (the 9 Clustering_Backbone54_k2 grid points + the 12 NEW gating K-sweep
configs) is their temporal reference, merged into the LOSO summary by run_loso.py.

Uses the same parallel worker machinery as run_loso.py: the driver spawns n_parallel
`run_loso_worker.py --station full` subprocesses, one configuration per job.

Each configuration's pooled test R2 is validated against eval-1.1's tracked results
(`delta_grid_summary.csv` / `metrics_summary.csv`, i.e. `eval11_test_r2`) and must match
within ~1e-3, proving the baseline faithfully replicates 1.1.

Usage:
    python run_full_baseline.py                     # full run (68 configs)
    python run_full_baseline.py --max-configs 2     # smoke test
    python run_full_baseline.py --smoke --max-configs 1 --device cpu  # CPU smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from eval14.data import load_experiment_data
from run_loso import (
    build_config_frame,
    expected_version,
    fold_complete,
    load_configurations,
    load_job_meta,
    make_jobs,
    run_jobs,
    write_runtime,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.yaml"
PREDICTIONS_DIR = EXP_DIR / "predictions_full"


def aggregate_full(config_id: str, config: dict) -> dict | None:
    """One full-baseline config -> (pooled row, per-station rows); None if incomplete."""
    meta = load_job_meta(config_id, "full")
    if meta is None or meta.get("status") != "completed":
        return None
    if meta.get("data_version") != expected_version(config):
        return None
    pooled = meta.get("pooled", {})
    pooled_row = {
        "config_id": config_id,
        "strategy_name": meta.get("strategy_name", ""),
        "n_train_total": int(meta.get("n_train_total", 0)),
        "n_test": int(meta.get("n_test", 0)),
        "full_pooled_r2": pooled.get("r2", float("nan")),
        "full_pooled_rmse": pooled.get("rmse", float("nan")),
        "full_pooled_ubrmse": pooled.get("ubrmse", float("nan")),
        "full_pooled_bias": pooled.get("bias", float("nan")),
        "full_pooled_mae": pooled.get("mae", float("nan")),
        "train_time_s": float(meta.get("train_time_s", 0.0)),
    }
    station_rows = []
    for station, m in (meta.get("per_station") or {}).items():
        station_rows.append({
            "config_id": config_id,
            "station": station,
            "strategy_name": meta.get("strategy_name", ""),
            "n_train_total": int(meta.get("n_train_total", 0)),
            "n_test": int(m.get("n_test", 0)),
            "r2": m.get("r2", float("nan")),
            "rmse": m.get("rmse", float("nan")),
            "ubrmse": m.get("ubrmse", float("nan")),
            "bias": m.get("bias", float("nan")),
            "mae": m.get("mae", float("nan")),
            "pearson": m.get("pearson", float("nan")),
        })
    return pooled_row, station_rows


def merge_eval13_full_references(pooled_rows: list[dict], per_config_station: list[dict],
                                 configurations: list[dict], config: dict,
                                 cfg_frame: pd.DataFrame) -> tuple[list[dict], list[dict], set[str]]:
    """Merge eval-1.3's full-baseline rows for pinned configs not computed here.

    All 56 eval-1.1/eval-1.3 configs were already evaluated under the identical
    protocol in eval-1.3 (replicating eval-1.1 to 0.000000; 47 merged there from
    eval-1.2 as references, 9 Clustering_Backbone54_k2 grid points computed);
    re-computing them adds no information. Returns (pooled_rows,
    per_config_station, referenced_ids).
    """
    computed_ids = {r["config_id"] for r in pooled_rows}
    pinned_ids = {c["config_id"] for c in configurations}
    need_ref = sorted(pinned_ids - computed_ids)
    if not need_ref:
        return pooled_rows, per_config_station, set()
    src = PROJECT_ROOT / Path(config["loso"].get(
        "reference_eval13_dir", "notebooks/experiment/derived_8.4-eval-1.3"))
    full_path = src / "full_config_summary.csv"
    if not full_path.exists():
        print(f"[Refs] eval-1.3 full_config_summary.csv not found — {len(need_ref)} configs "
              f"missing (computed-only aggregation).", flush=True)
        return pooled_rows, per_config_station, set()

    fsum = pd.read_csv(full_path)
    fsum = fsum[fsum["config_id"].isin(need_ref)]
    strat_map = cfg_frame.set_index("config_id")["strategy_name"].to_dict()
    for _, row in fsum.iterrows():
        pooled_rows.append({
            "config_id": row["config_id"],
            "strategy_name": strat_map.get(row["config_id"], str(row.get("strategy_name_x", ""))),
            "n_train_total": int(row["n_train_total"]),
            "n_test": int(row["n_test"]),
            "full_pooled_r2": row["full_pooled_r2"],
            "full_pooled_rmse": row["full_pooled_rmse"],
            "full_pooled_ubrmse": row["full_pooled_ubrmse"],
            "full_pooled_bias": row["full_pooled_bias"],
            "full_pooled_mae": row["full_pooled_mae"],
            "train_time_s": row["train_time_s"],
            "is_reference": True,
        })
    fstat = pd.read_csv(src / "full_per_config_station.csv")
    fstat = fstat[fstat["config_id"].isin(need_ref)].copy()
    fstat["strategy_name"] = fstat["config_id"].map(strat_map)
    fstat["is_reference"] = True
    per_config_station.extend(fstat.to_dict(orient="records"))
    print(f"[Refs] Merged eval-1.3 full-baseline rows for {len(need_ref)} configurations "
          f"({len(fstat)} station rows).", flush=True)
    return pooled_rows, per_config_station, set(need_ref)


def _write_partial(per_config_station: list[dict], pooled_rows: list[dict]) -> None:
    """Crash-safety: flush accumulated rows to CSVs."""
    pd.DataFrame(per_config_station).to_csv(EXP_DIR / "full_per_config_station.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(EXP_DIR / "full_config_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs (smoke test).")
    parser.add_argument("--config-id", action="append", default=None, help="Only run these config ids (repeatable).")
    parser.add_argument("--n-parallel", type=int, default=None, help="Concurrent GPU workers.")
    parser.add_argument("--device", default=None, help="Override model device (e.g. cpu for login-node smoke).")
    parser.add_argument("--new-strategy-only", action="store_true",
                        help="Compute only the NEW gating K-sweep configurations (12 configs "
                             "per config.yaml gating_clustering_strategies); merge the other "
                             "56 configs' full-baseline results as eval-1.3 references.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial results.")
    parser.add_argument("--smoke", action="store_true", help="data_version -1 + n_estimators 100 (never reused).")
    args = parser.parse_args()

    import run_loso

    run_loso.SMOKE = args.smoke

    print("=" * 70, flush=True)
    print("Starting derived_8.4-eval-1.4 Full-Training Baseline (replicates eval-1.1)", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_experiment_data(PROJECT_ROOT, config)
    print(f"[Data] TrainVal={len(data.trainval)} samples, Test={len(data.test)} samples.", flush=True)

    all_configurations = load_configurations(data, config)

    # Pin ALL configurations (provenance / audit trail) before any run filter —
    # the workers read loso_configurations.json, and this script may run BEFORE
    # run_loso.py (slurm runs the full baseline first).
    with open(EXP_DIR / "loso_configurations.json", "w", encoding="utf-8") as f:
        json.dump(all_configurations, f, indent=2)

    if args.new_strategy_only:
        gating_strats = {
            str(e["strategy"]) for e in (config.get("gating_clustering_strategies") or {}).get("strategies", [])
        }
        configurations = [c for c in all_configurations if c["strategy_name"] in gating_strats]
        print(f"[Scope] Computing only the NEW gating K-sweep configs ({len(configurations)} "
              f"configs); the other {len(all_configurations) - len(configurations)} pinned "
              f"configs merge as eval-1.3 references.", flush=True)
    elif args.config_id:
        configurations = [c for c in all_configurations if c["config_id"] in args.config_id]
    else:
        configurations = list(all_configurations)
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    print(f"[Configs] Evaluating {len(configurations)} configs on full trainval (no LOSO).", flush=True)

    cfg_frame = build_config_frame(all_configurations)
    eval11_r2 = cfg_frame.set_index("config_id")["eval11_test_r2"].to_dict()

    n_parallel = args.n_parallel or int(config["loso"].get("n_parallel", 8))
    write_runtime(config, device=args.device, n_estimators=100 if args.smoke else None)

    jobs = make_jobs([c["config_id"] for c in configurations], ["full"], config)
    todo = [j for j in jobs if j[2] != "skip"]
    print(f"[Jobs] {len(jobs)} total ({len(jobs) - len(todo)} done, {len(todo)} to run) "
          f"with {n_parallel} parallel workers.", flush=True)
    run_jobs(todo, n_parallel, config)

    # ---- Aggregation ----
    done: set[str] = set()
    per_config_station: list[dict] = []
    pooled_rows: list[dict] = []
    if not args.no_resume and (EXP_DIR / "full_per_config_station.csv").exists():
        prev = pd.read_csv(EXP_DIR / "full_per_config_station.csv")
        version = expected_version(config)
        done = {
            cid for cid in prev["config_id"].unique()
            if fold_complete(cid, "full", version)  # data_version match (smoke never reused)
        }
        per_config_station = prev[prev["config_id"].isin(done)].to_dict(orient="records")
        for cid in done:
            meta = load_job_meta(cid, "full")
            if meta is not None and meta.get("status") == "completed":
                pooled_rows.append({
                    "config_id": cid,
                    "strategy_name": meta.get("strategy_name", ""),
                    "n_train_total": int(meta.get("n_train_total", 0)),
                    "n_test": int(meta.get("n_test", 0)),
                    "full_pooled_r2": meta["pooled"].get("r2", float("nan")),
                    "full_pooled_rmse": meta["pooled"].get("rmse", float("nan")),
                    "full_pooled_ubrmse": meta["pooled"].get("ubrmse", float("nan")),
                    "full_pooled_bias": meta["pooled"].get("bias", float("nan")),
                    "full_pooled_mae": meta["pooled"].get("mae", float("nan")),
                    "train_time_s": float(meta.get("train_time_s", 0.0)),
                })
        print(f"[Resume] Found {len(done)} completed configs (data_version match); skipping those.", flush=True)

    missing = 0
    for cfg in configurations:
        config_id = cfg["config_id"]
        if config_id in done:
            continue
        agg = aggregate_full(config_id, config)
        if agg is None:
            missing += 1
            continue
        pooled_row, station_rows = agg
        pooled_row["is_reference"] = False
        for r in station_rows:
            r["is_reference"] = False
        per_config_station.extend(station_rows)
        pooled_rows.append(pooled_row)
        _write_partial(per_config_station, pooled_rows)
        print(f"[Config] Finished {config_id} ({len(pooled_rows)}/{len(configurations)}), "
              f"pooled R2={pooled_row['full_pooled_r2']:.4f}", flush=True)
    if missing:
        print(f"[Warn] {missing} configs missing/incomplete — aggregation is partial.", flush=True)

    # Merge eval-1.3 full-baseline references for pinned configs not computed here
    # (all pinned configs, not just the run scope).
    pooled_rows, per_config_station, ref_ids = merge_eval13_full_references(
        pooled_rows, per_config_station, all_configurations, config, cfg_frame)

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
    summary["is_reference"] = summary["config_id"].isin(ref_ids)
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


if __name__ == "__main__":
    main()
