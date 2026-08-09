#!/usr/bin/env python3
"""Full-training baseline for derived_8.4-eval-2.0 (MLP, all 7 stations).

LOSO is an *addition* to the experiment, not a replacement: this script trains
every MLP configuration WITHOUT leave-one-station-out — router fit and
specialists trained on ALL 7 stations (train split), early-stopped on the val
split, evaluated on the full test set — exactly the mlp-1.3 / mlp-1.1 temporal
protocol. Per-station test metrics are collected so station difficulty under
full training (the *intrinsic* difficulty) can be contrasted with LOSO
difficulty (how much a station suffers when held out).

Replication check: because training is deterministic and the per-family seeds
match the source experiments (2-regime: {42, 7} from mlp-1.3; 1-regime: {42}
from mlp-1.1), each config's pooled test R2 must match the reference
metrics_summary.csv (|diff| ~ 0).

Usage:
    python run_full_baseline.py                     # full run (6 configs)
    python run_full_baseline.py --max-configs 1 --smoke   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
CONFIG_PATH = EXP_DIR / "config.yaml"
PREDICTIONS_DIR = EXP_DIR / "predictions_full"
sys.path.insert(0, str(EXP_DIR))

from eval20 import data as edata  # noqa: E402
from eval20.evaluator import compute_metrics  # noqa: E402
from run_loso import (  # noqa: E402
    build_config_frame,
    build_configurations,
    job_dir,
    make_jobs,
    run_jobs,
    seed_complete,
)

SMOKE = False
SMOKE_VERSION = -1


def aggregate_full(out: Path, family: str, cid: str, seeds: list[int], config: dict) -> dict | None:
    """Seed-mean the full-baseline fold -> predictions file + pooled metrics row."""
    version = SMOKE_VERSION if SMOKE else int(config["sweep"].get("data_version", 6))
    station = "full"
    metas = {}
    for seed in seeds:
        jdir = job_dir(out, family, cid, station, seed)
        meta_path = jdir / "meta.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        if m.get("status") == "completed" and m.get("config", {}).get("data_version") == version:
            metas[seed] = m
    if not metas:
        return None

    preds_list = [np.load(job_dir(out, family, cid, station, s) / "preds.npy") for s in sorted(metas)]
    mean_preds = np.mean(preds_list, axis=0)

    config_id = f"{family}__{cid}"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PREDICTIONS_DIR / f"{config_id}__full_preds.npy", mean_preds)

    test_meta = np.load(out / "artifacts" / "test_meta.npz", allow_pickle=True)
    y_test = test_meta["y_test"]
    pooled = compute_metrics(y_test, mean_preds)

    first = next(iter(metas.values()))
    return {
        "config_id": config_id,
        "strategy_name": f"MLP_{family}",
        "n_train_total": int(first.get("n_train_total", 0)),
        "n_test": int(len(y_test)),
        "full_pooled_r2": pooled["r2"],
        "full_pooled_rmse": pooled["rmse"],
        "full_pooled_ubrmse": pooled["ubrmse"],
        "full_pooled_bias": pooled["bias"],
        "full_pooled_mae": pooled["mae"],
        "full_pooled_pearson": pooled["pearson"],
        "train_time_s": float(np.mean([m.get("train_time_s", 0.0) for m in metas.values()])),
        "val_rmse": float(np.mean([m.get("val_rmse", float("nan")) for m in metas.values()])),
        "aux_rmse": float(np.mean([m.get("aux_rmse", float("nan")) for m in metas.values()])),
        "n_seeds": len(metas),
        "preds": mean_preds,
    }


def _torch_version() -> str:
    import torch

    return torch.__version__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs (smoke test).")
    parser.add_argument("--config-id", action="append", default=None, help="Only run these config ids (repeatable).")
    parser.add_argument("--n-parallel", type=int, default=None, help="Concurrent GPU workers.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial results.")
    parser.add_argument("--smoke", action="store_true", help="Cap all jobs at 3 epochs (data_version -1).")
    args = parser.parse_args()

    global SMOKE
    SMOKE = args.smoke

    print("=" * 70, flush=True)
    print("Starting derived_8.4-eval-2.0 Full-Training Baseline (MLP, all 7 stations)", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = edata.load_experiment_data(PROJECT_ROOT, config)
    print(f"[Data] Train={len(data.train)} Val={len(data.val)} Test={len(data.test)}.", flush=True)

    configurations = build_configurations(data, config)
    if args.config_id:
        configurations = [c for c in configurations if c["config_id"] in args.config_id
                          or c["config_id_short"] in args.config_id]
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    print(f"[Configs] Evaluating {len(configurations)} configs on full trainval (no LOSO).", flush=True)

    # cfg_frame carries each config's temporal reference R2 (set by
    # build_configurations from the mlp-1.3 / mlp-1.1 metrics_summary.csv).
    cfg_frame = build_config_frame(configurations)

    # Prebuild full (all-station) tensors.
    artifacts = EXP_DIR / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "logs").mkdir(parents=True, exist_ok=True)
    edata.build_all_full_tensors(data, config, artifacts)

    sweep_configs = edata.build_sweep_configs(config)
    if args.smoke:
        for cid, cfg in sweep_configs.items():
            cfg["max_epochs"] = 3
            cfg["patience"] = 2
            cfg["checkpoint_every"] = 1
            cfg["data_version"] = SMOKE_VERSION
    with open(artifacts / "sweep_configs.json", "w", encoding="utf-8") as f:
        json.dump(sweep_configs, f, indent=2)

    n_parallel = args.n_parallel or int(config["sweep"]["n_parallel"])
    jobs: list = []
    for family in sorted({c["family"] for c in configurations}):
        fam_seeds = edata.seeds_for_family(config, family)
        fam_configs = [c for c in configurations if c["family"] == family]
        fam_scope = [(c["family"], c["config_id_short"], "full") for c in fam_configs]
        jobs += make_jobs(EXP_DIR, fam_scope, fam_seeds, config)
    todo = [j for j in jobs if j[4] != "skip"]
    print(f"[Jobs] {len(jobs)} total ({len([j for j in jobs if j[4]=='skip'])} done, "
          f"{len([j for j in jobs if j[4]=='resume'])} resume, "
          f"{len([j for j in jobs if j[4]=='fresh'])} fresh)", flush=True)
    run_jobs(EXP_DIR, todo, n_parallel, config)

    # ---- Aggregate ----
    version = SMOKE_VERSION if SMOKE else int(config["sweep"].get("data_version", 6))
    done: set[str] = set()
    per_config_station: list[dict] = []
    pooled_rows: list[dict] = []
    preds_by_config: dict[str, np.ndarray] = {}
    if not args.no_resume and (EXP_DIR / "full_config_summary.csv").exists():
        prev = pd.read_csv(EXP_DIR / "full_config_summary.csv")

        def _config_valid(config_id: str) -> bool:
            if "__" not in config_id:
                return False
            family, cid = config_id.split("__", 1)
            seeds = edata.seeds_for_family(config, family)
            return all(
                seed_complete(job_dir(EXP_DIR, family, cid, "full", s), version)
                for s in seeds
            )

        valid_ids = [cid for cid in prev["config_id"] if _config_valid(cid)]
        done = set(valid_ids)
        prev = prev[prev["config_id"].isin(valid_ids)]
        # Drop metadata columns that the cfg_frame merge re-attaches (they may
        # carry _x/_y suffixes from the previous run's merge).
        drop_cols = [c for c in prev.columns if c in (
            "config_label", "strategy_name", "strategy_name_x", "strategy_name_y",
            "is_baseline", "is_winner", "mlp13_test_r2", "mlp13_test_r2_x",
            "mlp13_test_r2_y",
        )]
        if drop_cols:
            prev = prev.drop(columns=drop_cols)
        pooled_rows = prev.to_dict(orient="records")
        for cid in valid_ids:
            pred_file = PREDICTIONS_DIR / f"{cid}__full_preds.npy"
            if pred_file.exists():
                preds_by_config[cid] = np.load(pred_file)
        # Per-station rows for the valid configs (rebuilt for re-run configs).
        if (EXP_DIR / "full_per_config_station.csv").exists():
            prev_pcs = pd.read_csv(EXP_DIR / "full_per_config_station.csv")
            prev_pcs = prev_pcs[prev_pcs["config_id"].isin(valid_ids)]
            per_config_station = prev_pcs.to_dict(orient="records")
        print(f"[Resume] {len(done)} valid configs kept ({len(prev) - len(done)} stale dropped).", flush=True)

    test_meta = np.load(artifacts / "test_meta.npz", allow_pickle=True)
    y_test, stations_te = test_meta["y_test"], test_meta["station"]

    for c in configurations:
        family, cid = c["family"], c["config_id_short"]
        config_id = c["config_id"]
        if config_id in done:
            continue
        fam_seeds = edata.seeds_for_family(config, family)
        agg = aggregate_full(EXP_DIR, family, cid, fam_seeds, config)
        if agg is None:
            continue
        pooled_rows.append({k: v for k, v in agg.items() if k != "preds"})
        preds = agg["preds"]
        preds_by_config[config_id] = preds
        for station in sorted(set(stations_te.tolist())):
            mask = stations_te == station
            m = compute_metrics(y_test[mask], preds[mask])
            per_config_station.append({
                "config_id": config_id,
                "station": station,
                "strategy_name": f"MLP_{family}",
                "n_train_total": agg["n_train_total"],
                "n_test": int(mask.sum()),
                "r2": m["r2"], "rmse": m["rmse"], "ubrmse": m["ubrmse"],
                "bias": m["bias"], "mae": m["mae"], "pearson": m["pearson"],
            })
        _write_partial(per_config_station, pooled_rows)
        print(f"[Config] Finished {config_id}, pooled R2={agg['full_pooled_r2']:.4f}", flush=True)

    df_pcs = pd.DataFrame(per_config_station)
    df_pooled = pd.DataFrame(pooled_rows)

    # ---- Per-configuration summary ----
    summary = df_pooled.copy()
    station_agg = df_pcs.groupby("config_id").agg(
        n_stations=("station", "count"),
        full_station_mean_r2=("r2", "mean"),
        full_station_median_r2=("r2", "median"),
        full_station_min_r2=("r2", "min"),
        full_station_max_r2=("r2", "max"),
    ).reset_index()
    # Drop stale aggregation columns from resume-loaded rows (recomputed below).
    stale_agg = [c for c in summary.columns
                 if c == "n_stations" or c.startswith("n_stations_")
                 or c.startswith("full_station_")]
    if stale_agg:
        summary = summary.drop(columns=stale_agg)
    summary = summary.merge(station_agg, on="config_id", how="left")
    summary = summary.merge(
        cfg_frame[["config_id", "config_label", "strategy_name", "is_baseline", "is_winner",
                   "mlp13_test_r2"]],
        on="config_id", how="left",
    )
    summary["r2_diff"] = (summary["full_pooled_r2"] - summary["mlp13_test_r2"]).abs()
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
    print(f"[Artifacts] Wrote full_per_config_station.csv / full_config_summary.csv / "
          f"full_station_summary.csv", flush=True)

    # ---- Validation vs mlp-1.3 / mlp-1.1 (replication check) ----
    valid = summary.dropna(subset=["mlp13_test_r2"])
    if len(valid):
        two_regime = valid[valid["config_id"].str.startswith(("2regime_54", "2regime_96"))]
        one_regime = valid[valid["config_id"].str.startswith(("1regime_54", "1regime_96"))]
        max2 = float(two_regime["r2_diff"].max()) if len(two_regime) else float("nan")
        max1 = float(one_regime["r2_diff"].max()) if len(one_regime) else float("nan")
        print("\nVALIDATION vs mlp-1.3 / mlp-1.1 (pooled test R2):", flush=True)
        print(f"  configs compared: {len(valid)}", flush=True)
        print(f"  2-regime (vs mlp-1.3, same torch {_torch_version()}): "
              f"max |diff| = {max2:.6f} "
              f"{'PASS (bit-identical)' if max2 < 1e-6 else 'CHECK'}", flush=True)
        print(f"  1-regime (vs mlp-1.1, older torch): max |diff| = {max1:.6f} "
              f"(documented environment drift — mlp-1.1 ran on an earlier torch; "
              f"current torch re-runs the same configs/protocol)", flush=True)
        print(f"  overall max |diff| = {valid['r2_diff'].max():.6f}", flush=True)
    else:
        print("\nVALIDATION: no reference R2 available (nothing compared).", flush=True)

    print("\nSTATION DIFFICULTY UNDER FULL TRAINING (median R2 over configs; intrinsic difficulty)", flush=True)
    print(station_summary[["station", "median_r2", "mean_r2", "min_r2", "max_r2", "n_negative_r2"]].to_string(index=False), flush=True)


def _write_partial(per_config_station: list[dict], pooled_rows: list[dict]) -> None:
    pd.DataFrame(per_config_station).to_csv(EXP_DIR / "full_per_config_station.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(EXP_DIR / "full_config_summary.csv", index=False)


if __name__ == "__main__":
    main()
