#!/usr/bin/env python3
"""Main execution script for derived_8.4-eval-1.2 LOSO spatial generalization evaluation.

Runs leave-one-station-out evaluation for all 47 configurations inherited from
derived_8.4-eval-1.1 (2 baselines + 5 MoE strategies x 9 delta-grid points) across
the 7 WA stations of the derived_8.4 split. Collects per-configuration x per-station
metrics (pooled / per-year / per-regime), persists per-fold predictions (.npy) and
model weights (.json), and produces summary CSVs + LOSO figures.

Usage:
    python run_loso.py                      # full run (47 configs x 7 stations)
    python run_loso.py --max-configs 1 --max-stations 1 --skip-plots   # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eval12.data import load_experiment_data
from eval12.evaluator import LosoEvaluator, compute_metrics
from eval12.plots import (
    plot_config_station_heatmap,
    plot_config_summary_bar,
    plot_station_boxplot,
    plot_station_difficulty,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.yaml"
MODELS_DIR = EXP_DIR / "models"
PREDICTIONS_DIR = EXP_DIR / "predictions"

STRATEGY_ORDER = [
    "Global_Single",
    "Clustering_V0_Full_k2",
    "Clustering_Dynamic_k2",
    "Univariate_G_API_k2",
    "Seasonal_Binary_k2",
    "Trained_Gating_k2",
]
MOE_STRATEGIES = STRATEGY_ORDER[1:]


def parse_additions(value: object) -> list[str]:
    """Parse semicolon-joined cluster additions from eval-1.1 CSV cells."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [f for f in text.split(";") if f]


def load_configurations(data, config) -> list[dict]:
    """Build the 47 configurations from eval-1.1 artifacts (pinned for audit)."""
    src = PROJECT_ROOT / Path(config["loso"]["source_eval11_dir"])
    grid = pd.read_csv(src / "delta_grid_summary.csv")
    leaderboard = pd.read_csv(src / "metrics_summary.csv")

    # eval-1.1 test-set R2 by candidate_id (for LOSO vs temporal comparison).
    eval11_r2: dict[str, float] = {}
    for _, row in grid.iterrows():
        eval11_r2[str(row["candidate_id"])] = float(row["pooled_r2"])
    for _, row in leaderboard.iterrows():
        eval11_r2[str(row["candidate_id"])] = float(row["pooled_r2"])

    # eval-1.1 winning (c0, c1) per strategy, e.g. "Clustering_V0_Full_k2 (Winner c0=0, c1=10)".
    winner_by_strategy: dict[str, tuple[int, int]] = {}
    for _, row in leaderboard.iterrows():
        mname = str(row["model_name"])
        if "Winner" not in mname:
            continue
        strategy = str(row["strategy_name"])
        try:
            c0 = int(mname.split("c0=")[1].split(",")[0])
            c1 = int(mname.split("c1=")[1].split(")")[0])
        except (IndexError, ValueError):
            continue
        winner_by_strategy[strategy] = (c0, c1)

    configs: list[dict] = []

    def add_config(config_id, strategy, global_features, additions, c0=None, c1=None,
                   eval11_candidate_id=None, is_baseline=False):
        configs.append({
            "config_id": config_id,
            "strategy_name": strategy,
            "global_features": list(global_features),
            "cluster_additions": {str(k): list(v) for k, v in additions.items()},
            "is_baseline": bool(is_baseline),
            "cluster_0_count": c0,
            "cluster_1_count": c1,
            "eval11_candidate_id": eval11_candidate_id,
            "eval11_test_r2": eval11_r2.get(eval11_candidate_id, float("nan")),
            "is_winner": (
                not is_baseline and c0 is not None and c1 is not None
                and winner_by_strategy.get(strategy) == (c0, c1)
            ),
        })

    # 1. Baseline V0 (50 Feats) — Global_Single with V0 features.
    add_config(
        config_id="Baseline_V0_50",
        strategy="Global_Single",
        global_features=data.v0_features,
        additions={},
        eval11_candidate_id="Baseline_V0_50",
        is_baseline=True,
    )
    # 2. Global Single Model (54 Backbone).
    add_config(
        config_id="Global_Single_54",
        strategy="Global_Single",
        global_features=data.shared_backbone_54,
        additions={},
        eval11_candidate_id="Global_Single_backbone_0_0",
        is_baseline=True,
    )
    # 3. MoE strategies x 9 delta-grid points (fixed additions from eval-1.1).
    for strat in MOE_STRATEGIES:
        sub = grid[grid["strategy_name"] == strat].sort_values(
            ["cluster_0_count", "cluster_1_count"]
        )
        for _, row in sub.iterrows():
            c0 = int(row["cluster_0_count"])
            c1 = int(row["cluster_1_count"])
            add_config(
                config_id=f"{strat}_c0_{c0}_c1_{c1}",
                strategy=strat,
                global_features=data.shared_backbone_54,
                additions={
                    "0": parse_additions(row["cluster_0_additions"]),
                    "1": parse_additions(row["cluster_1_additions"]),
                },
                c0=c0,
                c1=c1,
                eval11_candidate_id=str(row["candidate_id"]),
            )
    return configs


def build_config_frame(configurations: list[dict]) -> pd.DataFrame:
    """DataFrame describing each configuration (used for plots and summary)."""
    rows = []
    for cfg in configurations:
        label = cfg["config_id"] if cfg["is_baseline"] else (
            f"{cfg['strategy_name']}  c0={cfg['cluster_0_count']}, c1={cfg['cluster_1_count']}"
        )
        rows.append({
            "config_id": cfg["config_id"],
            "config_label": label,
            "strategy_name": cfg["strategy_name"],
            "strategy_order": STRATEGY_ORDER.index(cfg["strategy_name"]),
            "is_baseline": cfg["is_baseline"],
            "is_winner": cfg["is_winner"],
            "cluster_0_count": cfg["cluster_0_count"],
            "cluster_1_count": cfg["cluster_1_count"],
            "eval11_test_r2": cfg["eval11_test_r2"],
            "n_global_features": len(cfg["global_features"]),
            "n_add0": len(cfg["cluster_additions"].get("0", [])),
            "n_add1": len(cfg["cluster_additions"].get("1", [])),
        })
    return pd.DataFrame(rows)


def compute_pooled_loso_metrics(df_pcs: pd.DataFrame, config, predictions_dir: Path) -> dict[str, dict[str, float]]:
    """Pooled LOSO metrics per config.

    Concatenates per-fold predictions across the 7 held-out stations (every test sample is
    predicted by a model trained without its own station) and computes sample-count-weighted
    R2/RMSE — directly comparable to eval-1.1's pooled test-set R2.
    """
    if predictions_dir is None or not predictions_dir.exists():
        return {}
    data_cfg = config["data"]
    test = pd.read_csv(PROJECT_ROOT / Path(data_cfg["splits"]["test"]))
    test = test.sort_values("station_id", kind="mergesort").reset_index(drop=True)
    target = str(data_cfg["target"])
    y_true = test[target].to_numpy(dtype=float)

    results: dict[str, dict[str, float]] = {}
    for config_id in df_pcs["config_id"].unique():
        stations = sorted(df_pcs.loc[df_pcs["config_id"] == config_id, "station"].unique())
        try:
            preds = np.concatenate([
                np.load(predictions_dir / f"{config_id}__{s}_preds.npy") for s in stations
            ])
        except FileNotFoundError:
            continue
        if len(preds) != len(y_true):
            raise ValueError(f"Pooled prediction length mismatch for {config_id}: {len(preds)} != {len(y_true)}")
        m = compute_metrics(y_true, preds)
        results[config_id] = {"loso_pooled_r2": m["r2"], "loso_pooled_rmse": m["rmse"]}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs (smoke test).")
    parser.add_argument("--max-stations", type=int, default=None, help="Limit number of stations (smoke test).")
    parser.add_argument("--config-id", action="append", default=None, help="Only run these config ids (repeatable).")
    parser.add_argument("--station", action="append", default=None, help="Only run these stations (repeatable).")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial results.")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("Starting derived_8.4-eval-1.2 LOSO Spatial Generalization Evaluation", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_experiment_data(PROJECT_ROOT, config)
    print(f"[Data] TrainVal={len(data.trainval)} samples, Test={len(data.test)} samples.", flush=True)
    print(f"[Backbone] Shared global backbone: {len(data.shared_backbone_54)} features.", flush=True)
    print(f"[V0] Baseline V0: {len(data.v0_features)} features.", flush=True)

    configurations = load_configurations(data, config)
    stations = sorted(data.test["station_id"].unique())
    print(f"[Configs] Loaded {len(configurations)} configurations from eval-1.1.", flush=True)
    print(f"[Stations] {len(stations)} held-out stations: {stations}", flush=True)

    if args.config_id:
        configurations = [c for c in configurations if c["config_id"] in args.config_id]
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    if args.station:
        stations = [s for s in stations if s in args.station]
    if args.max_stations:
        stations = stations[: args.max_stations]
    print(f"[Run] Evaluating {len(configurations)} configs x {len(stations)} stations.", flush=True)

    # Pin the exact configurations used (provenance / audit trail).
    pin_path = EXP_DIR / "loso_configurations.json"
    with open(pin_path, "w", encoding="utf-8") as f:
        json.dump(configurations, f, indent=2)
    print(f"[Artifacts] Pinned configurations to {pin_path.name}", flush=True)

    cfg_frame = build_config_frame(configurations)
    cfg_frame.to_csv(EXP_DIR / "loso_configs.csv", index=False)

    # Resume support: skip (config, station) pairs already completed and reload
    # their rows so partial CSVs are extended, not overwritten.
    done: set[tuple[str, str]] = set()
    per_config_station: list[dict] = []
    per_regime_rows: list[dict] = []
    per_year_rows: list[dict] = []
    if not args.no_resume and (EXP_DIR / "loso_per_config_station.csv").exists():
        prev = pd.read_csv(EXP_DIR / "loso_per_config_station.csv")
        done = set(zip(prev["config_id"], prev["station"]))
        per_config_station = prev.to_dict(orient="records")
        if (EXP_DIR / "loso_per_regime_metrics.csv").exists():
            per_regime_rows = pd.read_csv(EXP_DIR / "loso_per_regime_metrics.csv").to_dict(orient="records")
        if (EXP_DIR / "loso_per_year_metrics.csv").exists():
            per_year_rows = pd.read_csv(EXP_DIR / "loso_per_year_metrics.csv").to_dict(orient="records")
        print(f"[Resume] Found {len(done)} completed folds; skipping those.", flush=True)

    save_weights = bool(config["loso"]["save_weights"])
    save_predictions = bool(config["loso"]["save_predictions"])
    total = len(configurations) * len(stations)
    completed = 0

    for cfg in configurations:
        evaluator = LosoEvaluator(
            data,
            config,
            config_id=cfg["config_id"],
            strategy_name=cfg["strategy_name"],
            global_features=cfg["global_features"],
            cluster_additions=cfg["cluster_additions"],
            models_dir=MODELS_DIR,
            predictions_dir=PREDICTIONS_DIR,
            save_weights=save_weights,
            save_predictions=save_predictions,
        )
        for station in stations:
            if (cfg["config_id"], station) in done:
                completed += 1
                continue
            res = evaluator.evaluate_station(station)
            completed += 1

            per_config_station.append({
                "config_id": res.config_id,
                "station": res.station,
                "strategy_name": res.strategy_name,
                "n_train_total": res.n_train_total,
                "n_test": res.n_test,
                "r2": res.r2,
                "rmse": res.rmse,
                "ubrmse": res.ubrmse,
                "bias": res.bias,
                "mae": res.mae,
                "pearson": res.pearson,
                "train_time_s": res.train_time_s,
            })
            for cl, m in res.cluster_metrics.items():
                per_regime_rows.append({
                    "config_id": res.config_id,
                    "station": res.station,
                    "strategy_name": res.strategy_name,
                    "cluster": cl,
                    **m,
                })
            for year, m in res.yearly_metrics.items():
                per_year_rows.append({
                    "config_id": res.config_id,
                    "station": res.station,
                    "strategy_name": res.strategy_name,
                    "year": year,
                    **m,
                })
            if completed % 20 == 0:
                print(f"  [{completed}/{total}] folds done "
                      f"({res.config_id} @ {res.station}, r2={res.r2:.4f})", flush=True)
                _write_partial(per_config_station, per_regime_rows, per_year_rows)

        # Incremental save after each config (crash safety).
        _write_partial(per_config_station, per_regime_rows, per_year_rows)
        print(f"[Config] Finished {cfg['config_id']} "
              f"({len(stations)} stations). Progress {completed}/{total}.", flush=True)

    df_pcs = pd.DataFrame(per_config_station)
    df_regime = pd.DataFrame(per_regime_rows)
    df_year = pd.DataFrame(per_year_rows)

    # Merge configuration metadata for plotting / summary. When resuming, previously
    # written CSVs already carry these columns, so drop them before re-merging.
    right_cols = cfg_frame.drop(columns=["n_global_features", "n_add0", "n_add1"]).columns
    overlap = [c for c in right_cols if c in df_pcs.columns and c not in ("config_id", "strategy_name")]
    if overlap:
        df_pcs = df_pcs.drop(columns=overlap)
    df_pcs = df_pcs.merge(
        cfg_frame.drop(columns=["n_global_features", "n_add0", "n_add1"]),
        on=["config_id", "strategy_name"], how="left",
    )

    # ---- Per-configuration summary ----
    summary = df_pcs.groupby("config_id").agg(
        n_stations=("station", "count"),
        total_test_n=("n_test", "sum"),
        loso_mean_r2=("r2", "mean"),
        loso_median_r2=("r2", "median"),
        loso_std_r2=("r2", "std"),
        loso_min_r2=("r2", "min"),
        loso_max_r2=("r2", "max"),
        loso_mean_rmse=("rmse", "mean"),
        loso_mean_ubrmse=("ubrmse", "mean"),
        loso_mean_bias=("bias", "mean"),
        loso_mean_mae=("mae", "mean"),
    ).reset_index()
    summary = summary.merge(
        cfg_frame[["config_id", "config_label", "strategy_name", "is_baseline", "is_winner",
                   "cluster_0_count", "cluster_1_count", "eval11_test_r2"]],
        on="config_id", how="left",
    )
    # Best / worst held-out station per configuration.
    best = df_pcs.loc[df_pcs.groupby("config_id")["r2"].idxmax(), ["config_id", "station"]]
    worst = df_pcs.loc[df_pcs.groupby("config_id")["r2"].idxmin(), ["config_id", "station"]]
    summary = summary.merge(best.rename(columns={"station": "best_station"}), on="config_id")
    summary = summary.merge(worst.rename(columns={"station": "worst_station"}), on="config_id")
    summary["loso_minus_test_r2"] = summary["loso_mean_r2"] - summary["eval11_test_r2"]

    # Pooled LOSO R2/RMSE (concatenated folds) for direct comparison with eval-1.1.
    pooled = compute_pooled_loso_metrics(df_pcs, config, PREDICTIONS_DIR)
    if pooled:
        summary = summary.merge(
            pd.DataFrame(pooled).T.reset_index().rename(columns={"index": "config_id"}),
            on="config_id", how="left",
        )
        summary["pooled_loso_minus_test_r2"] = summary["loso_pooled_r2"] - summary["eval11_test_r2"]

    summary = summary.sort_values("loso_mean_r2", ascending=False).reset_index(drop=True)
    summary.to_csv(EXP_DIR / "loso_config_summary.csv", index=False)

    # ---- Per-station summary (station difficulty byproduct) ----
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
    station_summary.to_csv(EXP_DIR / "loso_station_summary.csv", index=False)

    df_pcs.to_csv(EXP_DIR / "loso_per_config_station.csv", index=False)
    df_regime.to_csv(EXP_DIR / "loso_per_regime_metrics.csv", index=False)
    df_year.to_csv(EXP_DIR / "loso_per_year_metrics.csv", index=False)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_per_config_station.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_config_summary.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_station_summary.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_per_regime_metrics.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_per_year_metrics.csv'}", flush=True)

    if not args.skip_plots:
        plot_config_station_heatmap(df_pcs, EXP_DIR)
        plot_config_summary_bar(summary, EXP_DIR)
        plot_station_difficulty(station_summary, EXP_DIR)
        plot_station_boxplot(df_pcs, EXP_DIR)
        print(f"[Plots] Generated LOSO figures in {EXP_DIR}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("LOSO CONFIGURATION LEADERBOARD (derived_8.4-eval-1.2)", flush=True)
    print("=" * 70, flush=True)
    cols = ["config_id", "strategy_name", "loso_mean_r2", "loso_std_r2", "loso_min_r2",
            "loso_max_r2", "loso_mean_rmse", "loso_mean_bias", "eval11_test_r2",
            "loso_minus_test_r2", "is_winner"]
    if "loso_pooled_r2" in summary.columns:
        cols = ["config_id", "strategy_name", "loso_mean_r2", "loso_pooled_r2", "loso_std_r2",
                "loso_min_r2", "loso_max_r2", "loso_mean_rmse", "loso_mean_bias",
                "eval11_test_r2", "loso_minus_test_r2", "is_winner"]
    print(summary[cols].to_string(index=False), flush=True)

    print("\nSTATION DIFFICULTY (median LOSO R2 across configs)", flush=True)
    print(station_summary[["station", "median_r2", "mean_r2", "min_r2", "max_r2", "n_negative_r2"]].to_string(index=False), flush=True)


def _write_partial(per_config_station, per_regime_rows, per_year_rows) -> None:
    """Crash-safety: flush accumulated rows to CSVs."""
    pd.DataFrame(per_config_station).to_csv(EXP_DIR / "loso_per_config_station.csv", index=False)
    pd.DataFrame(per_regime_rows).to_csv(EXP_DIR / "loso_per_regime_metrics.csv", index=False)
    pd.DataFrame(per_year_rows).to_csv(EXP_DIR / "loso_per_year_metrics.csv", index=False)


if __name__ == "__main__":
    main()
