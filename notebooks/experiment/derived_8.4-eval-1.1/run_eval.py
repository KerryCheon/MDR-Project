#!/usr/bin/env python3
"""Main execution script for derived_8.4-eval-1.1 MoE Routing Evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from xgboost import XGBRegressor

from eval11.data import load_experiment_data
from eval11.evaluator import StrategyEvaluator
from eval11.plots import (
    plot_diagnostics,
    plot_per_regime_diagnostics,
    plot_yearly_performance_linechart,
    plot_delta_grid_heatmap,
    plot_loss_curves,
)
from generate_station_year_table import generate_table_from_preds

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.yaml"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
MODELS_DIR = EXP_DIR / "models"


def load_gain_scores(data, config) -> dict[str, float]:
    candidate_pool_path = PROJECT_ROOT / Path(config["candidate_pool_file"])
    if candidate_pool_path.exists():
        df_pool = pd.read_csv(candidate_pool_path)
        if "feature" in df_pool.columns and "gain" in df_pool.columns:
            return dict(zip(df_pool["feature"], df_pool["gain"].astype(float)))

    print("[run_eval] Computing feature gain scores via XGBoost proxy model...", flush=True)
    params = dict(config["model"]["exact_params"])
    params["n_estimators"] = 500
    params["learning_rate"] = 0.01
    params["random_state"] = int(config["model"]["seed"])
    params["n_jobs"] = 1
    params["importance_type"] = "gain"

    model = XGBRegressor(**params)
    model.fit(
        data.trainval.loc[:, data.feature_columns],
        data.trainval[data.target].to_numpy(dtype=float),
        verbose=False,
    )
    return dict(zip(data.feature_columns, model.feature_importances_))


def main():
    print("=" * 70, flush=True)
    print("Starting derived_8.4-eval-1.1 MoE Routing Evaluation", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data(PROJECT_ROOT, config)
    print(f"[Data] Loaded TrainVal={len(data.trainval)} samples, Test={len(data.test)} samples.", flush=True)
    print(f"[Backbone] Shared global backbone count: {len(data.shared_backbone_54)} features.", flush=True)
    print(f"[V0] Baseline V0 feature count: {len(data.v0_features)} features.", flush=True)

    gain_scores = load_gain_scores(data, config)

    strategies = [
        "Global_Single",
        "Clustering_V0_Full_k2",
        "Clustering_Dynamic_k2",
        "Univariate_G_API_k2",
        "Seasonal_Binary_k2",
        "Trained_Gating_k2",
    ]

    addition_counts = config.get("delta_addition_counts", [0, 5, 10])

    summary_records = []
    per_regime_records = []
    grid_records = []
    loss_curves = {}

    y_test = data.test[data.target].to_numpy(dtype=float)

    # 1. Baseline V0 evaluation on Global_Single
    eval_global = StrategyEvaluator(data, config, "Global_Single", models_dir=MODELS_DIR)
    v0_res = eval_global.fit_and_evaluate(
        candidate_id="Baseline_V0_50",
        global_features=data.v0_features,
        include_predictions=True,
        save_weights=True,
    )
    rec = v0_res.as_record()
    rec["model_name"] = "Baseline V0 (50 Feats)"
    summary_records.append(rec)
    plot_diagnostics("Baseline V0 (50 Feats)", y_test, v0_res.predictions, EXP_DIR)
    loss_curves["Baseline V0 (50 Feats)"] = v0_res.rmse_curve

    # Loop over all strategies
    for strat in strategies:
        print(f"\n[{strat}] Evaluating routing strategy...", flush=True)
        evaluator = StrategyEvaluator(data, config, strat, models_dir=MODELS_DIR)

        # Backbone-only evaluation (0, 0)
        backbone_res = evaluator.fit_and_evaluate(
            candidate_id=f"{strat}_backbone_0_0",
            global_features=data.shared_backbone_54,
            cluster_additions={"0": [], "1": []},
            include_predictions=True,
            save_weights=True,
        )

        if strat == "Global_Single":
            rec = backbone_res.as_record()
            rec["model_name"] = "Global Single Model (54 Backbone)"
            summary_records.append(rec)
            plot_diagnostics("Global Single Model (54 Backbone)", y_test, backbone_res.predictions, EXP_DIR)
            loss_curves["Global Single Model (54 Backbone)"] = backbone_res.rmse_curve
            
            generate_table_from_preds(
                data.test, backbone_res.predictions, rec["model_name"], "station_year_metrics_global_single.png", " (Global Single)"
            )

            for cl, m in backbone_res.cluster_metrics.items():
                per_regime_records.append({
                    "strategy_name": strat,
                    "model_name": rec["model_name"],
                    "cluster": cl,
                    "n_train": m["n_train"],
                    "n_test": m["n_test"],
                    "r2": m["r2"],
                    "rmse": m["rmse"],
                    "ubrmse": m["ubrmse"],
                    "bias": m["bias"],
                    "mae": m["mae"],
                    "pearson": m["pearson"],
                })
            continue

        # Compute delta rankings for K=2 regimes
        delta_rankings = evaluator.compute_delta_rankings(
            global_features=data.shared_backbone_54,
            predictions=backbone_res.predictions,
            gain_scores=gain_scores,
            max_additions=max(addition_counts),
        )

        print(f"[{strat}] Top 5 Delta candidates:", flush=True)
        print(f"  Cluster 0: {delta_rankings.get('0', [])[:5]}", flush=True)
        print(f"  Cluster 1: {delta_rankings.get('1', [])[:5]}", flush=True)

        best_grid_res = None
        best_grid_r2 = -float("inf")
        winning_c0 = 0
        winning_c1 = 0

        for c0_count in addition_counts:
            for c1_count in addition_counts:
                c0_add = delta_rankings["0"][:c0_count]
                c1_add = delta_rankings["1"][:c1_count]

                cand_id = f"{strat}_c0_{c0_count}_c1_{c1_count}"
                save_this = (c0_count, c1_count) in [(0, 0), (0, 5), (0, 10), (5, 5), (10, 10)]
                grid_res = evaluator.fit_and_evaluate(
                    candidate_id=cand_id,
                    global_features=data.shared_backbone_54,
                    cluster_additions={"0": c0_add, "1": c1_add},
                    include_predictions=True,
                    save_weights=save_this,
                )

                g_rec = grid_res.as_record()
                g_rec["cluster_0_count"] = c0_count
                g_rec["cluster_1_count"] = c1_count
                grid_records.append(g_rec)

                if grid_res.pooled_r2 > best_grid_r2:
                    best_grid_r2 = grid_res.pooled_r2
                    best_grid_res = grid_res
                    winning_c0 = c0_count
                    winning_c1 = c1_count

        # Save winning configuration for this strategy
        rec_win = best_grid_res.as_record()
        win_mname = f"{strat} (Winner c0={winning_c0}, c1={winning_c1})"
        rec_win["model_name"] = win_mname
        summary_records.append(rec_win)

        # Make sure weights are saved for winner
        evaluator.fit_and_evaluate(
            candidate_id=f"{strat}_winner",
            global_features=data.shared_backbone_54,
            cluster_additions=best_grid_res.cluster_additions,
            include_predictions=True,
            save_weights=True,
        )

        loss_curves[win_mname] = best_grid_res.rmse_curve
        plot_diagnostics(win_mname, y_test, best_grid_res.predictions, EXP_DIR)
        plot_per_regime_diagnostics(win_mname, y_test, best_grid_res.predictions, evaluator.labels_test, EXP_DIR)

        # Generate Station x Year card for top strategies
        if strat in {"Trained_Gating_k2", "Clustering_V0_Full_k2", "Clustering_Dynamic_k2"}:
            san_strat = strat.lower().replace(" ", "_")
            generate_table_from_preds(
                data.test, best_grid_res.predictions, win_mname, f"station_year_metrics_{san_strat}.png", f" ({strat})"
            )

        # Also add backbone (0, 0) for comparison if winner is not (0, 0)
        if (winning_c0, winning_c1) != (0, 0):
            rec_bb = backbone_res.as_record()
            rec_bb["model_name"] = f"{strat} (Backbone c0=0, c1=0)"
            summary_records.append(rec_bb)

        # Record per-regime breakdown for winning model
        for cl, m in best_grid_res.cluster_metrics.items():
            per_regime_records.append({
                "strategy_name": strat,
                "model_name": rec_win["model_name"],
                "cluster": cl,
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "r2": m["r2"],
                "rmse": m["rmse"],
                "ubrmse": m["ubrmse"],
                "bias": m["bias"],
                "mae": m["mae"],
                "pearson": m["pearson"],
            })

    # Save summary tables to CSV
    df_summary = pd.DataFrame(summary_records).sort_values(by="pooled_r2", ascending=False)
    df_summary.to_csv(EXP_DIR / "metrics_summary.csv", index=False)
    print(f"\n[Artifacts] Wrote {EXP_DIR / 'metrics_summary.csv'}", flush=True)

    df_per_regime = pd.DataFrame(per_regime_records)
    df_per_regime.to_csv(EXP_DIR / "per_regime_metrics_summary.csv", index=False)
    print(f"[Artifacts] Wrote {EXP_DIR / 'per_regime_metrics_summary.csv'}", flush=True)

    df_grid = pd.DataFrame(grid_records)
    df_grid.to_csv(EXP_DIR / "delta_grid_summary.csv", index=False)
    print(f"[Artifacts] Wrote {EXP_DIR / 'delta_grid_summary.csv'}", flush=True)

    # Generate multi-model comparison plots
    plot_yearly_performance_linechart(df_summary, EXP_DIR)
    plot_delta_grid_heatmap(df_grid, EXP_DIR)
    plot_loss_curves(loss_curves, EXP_DIR)
    print(f"[Plots] Generated all diagnostic and comparative figures in {EXP_DIR}", flush=True)

    # Print summary leaderboard
    print("\n" + "=" * 70, flush=True)
    print("FINAL MODEL LEADERBOARD (derived_8.4-eval-1.1)", flush=True)
    print("=" * 70, flush=True)
    print(df_summary[["model_name", "strategy_name", "pooled_r2", "pooled_rmse", "year_2023_r2", "year_2024_r2", "year_2025_r2"]].to_string(index=False), flush=True)

    # Write selected_features.json
    selected_meta = {
        "shared_backbone_54": data.shared_backbone_54,
        "baseline_v0_50": data.v0_features,
        "leaderboard": df_summary.to_dict(orient="records"),
    }
    with open(EXP_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(selected_meta, f, indent=2)
    print(f"[Artifacts] Wrote {EXP_DIR / 'selected_features.json'}", flush=True)


if __name__ == "__main__":
    main()
