"""
Accelerated SHAP Feature Importance Analysis for Hybrid LSTM + XGBoost Models.
Uses C++/CUDA native XGBoost pred_contribs for maximum performance.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def compute_accelerated_shap(model: xgb.XGBRegressor, X_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values using XGBoost's native C++/CUDA pred_contribs.
    Returns:
        shap_values: np.ndarray of shape (N_samples, N_features)
        elapsed_sec: float execution time in seconds
    """
    t0 = time.perf_counter()
    dmat = xgb.DMatrix(X_df)
    booster = model.get_booster()
    import torch
    if torch.cuda.is_available():
        booster.set_param({"device": "cuda", "tree_method": "hist"})
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_values = contribs[:, :-1]  # Exclude last column (bias / base value)
    elapsed_sec = time.perf_counter() - t0
    return shap_values, elapsed_sec


def compute_shap_values_for_candidate(
    evaluator: Any,
    candidate_result: Any,
) -> Dict[str, Any]:
    """
    Computes SHAP values across test set for a fitted model strategy candidate.
    Supports both Global single models and Multi-cluster MoE models.
    """
    test_df = evaluator.data.test
    labels_test = evaluator.labels_test
    models = candidate_result.models
    global_features = candidate_result.global_features
    cluster_additions = candidate_result.cluster_additions

    # Gather union of all features across clusters
    all_features_set = set(global_features)
    for c_key, c_add in cluster_additions.items():
        all_features_set.update(c_add)
    all_features = sorted(list(all_features_set))

    N_test = len(test_df)
    shap_matrix = np.zeros((N_test, len(all_features)), dtype=np.float32)
    total_shap_time = 0.0

    unique_clusters = sorted(np.unique(labels_test))

    for c in unique_clusters:
        c_key = str(c)
        mask = (labels_test == c)
        if not np.any(mask):
            continue

        c_add = cluster_additions.get(c_key, [])
        features_c = list(dict.fromkeys(global_features + c_add))

        X_c = test_df.loc[mask, features_c]
        model_c = models[c_key]

        # pred_contribs SHAP (fast, exact)
        c_shap, c_time = compute_accelerated_shap(model_c, X_c)
        total_shap_time += c_time

        # Map back to full feature matrix columns
        feat_to_idx = {f: i for i, f in enumerate(all_features)}
        for loc_idx, f in enumerate(features_c):
            full_idx = feat_to_idx[f]
            shap_matrix[mask, full_idx] = c_shap[:, loc_idx]

    benchmark_info = {
        "xgboost_pred_contribs_time_s": total_shap_time,
    }

    mean_abs_shap = pd.Series(np.mean(np.abs(shap_matrix), axis=0), index=all_features)

    return {
        "shap_matrix": shap_matrix,
        "features": all_features,
        "mean_abs_shap": mean_abs_shap,
        "shap_calc_time_s": total_shap_time,
        "benchmark": benchmark_info,
    }


def run_full_shap_analysis(
    evaluator_map: Dict[str, Any],
    results_map: Dict[str, Any],
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """
    Runs accelerated SHAP analysis for all models and generates summary artifacts.
    Supports ctx_ / hh_ / hp_ feature prefixes (including PCA variants).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shap_results = {}

    print("\n" + "=" * 75, flush=True)
    print("Executing Accelerated SHAP Feature Importance Analysis", flush=True)
    print("=" * 75, flush=True)

    for name, res in results_map.items():
        evaluator = evaluator_map[name]
        print(f"[SHAP] Computing SHAP values for: {name}...", flush=True)
        shap_res = compute_shap_values_for_candidate(evaluator, res)
        shap_results[name] = shap_res
        bench = shap_res["benchmark"]
        print(f"  -> Done in {shap_res['shap_calc_time_s']:.3f}s (pred_contribs)", flush=True)

    # 1. Build unified SHAP importance summary dataframe
    all_all_features = set()
    for res in shap_results.values():
        all_all_features.update(res["features"])
    all_all_features = sorted(list(all_all_features))

    df_shap_summary = pd.DataFrame(index=all_all_features)
    for name, res in shap_results.items():
        df_shap_summary[name] = res["mean_abs_shap"].reindex(all_all_features, fill_value=0.0)

    df_shap_summary.to_csv(artifacts_dir / "shap_importance_summary.csv")
    print(f"[SHAP] Saved summary table to {artifacts_dir / 'shap_importance_summary.csv'}", flush=True)

    # 2. Build Top-20 features table per model
    top20_data = {}
    for name, res in shap_results.items():
        top_s = res["mean_abs_shap"].sort_values(ascending=False).head(20)
        top20_data[f"{name}_feature"] = top_s.index.tolist()
        top20_data[f"{name}_shap_val"] = top_s.values.tolist()

    df_top20 = pd.DataFrame(top20_data)
    df_top20.to_csv(artifacts_dir / "shap_top20_comparison.csv", index=False)

    # 3. Analyze LSTM feature contribution (ctx_, hh_) vs Tabular
    ctx_vs_tabular = {}
    for name, res in shap_results.items():
        s = res["mean_abs_shap"]
        lstm_mask = s.index.str.startswith("ctx_") | s.index.str.startswith("hh_") | s.index.str.startswith("hp_")
        lstm_sum = float(s[lstm_mask].sum())
        tab_sum = float(s[~lstm_mask].sum())
        total_sum = lstm_sum + tab_sum
        lstm_pct = float(lstm_sum / total_sum * 100) if total_sum > 0 else 0.0
        tab_pct = float(tab_sum / total_sum * 100) if total_sum > 0 else 0.0

        ctx_vs_tabular[name] = {
            "ctx_shap_sum": lstm_sum,
            "tabular_shap_sum": tab_sum,
            "total_shap_sum": total_sum,
            "ctx_pct": lstm_pct,
            "tabular_pct": tab_pct,
            "num_ctx_features": int(lstm_mask.sum()),
            "num_tabular_features": int((~lstm_mask).sum()),
        }

    with open(artifacts_dir / "shap_ctx_vs_tabular.json", "w") as f:
        json.dump(ctx_vs_tabular, f, indent=2)

    # 4. Generate SHAP Summary Visualization (2x3 grid for 6 models)
    n_models = len(shap_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    colors_tab = "#1f77b4"
    colors_ctx = "#ff7f0e"
    colors_hh = "#2ca02c"
    colors_hp = "#d62728"

    for idx, (name, res) in enumerate(shap_results.items()):
        ax = axes_flat[idx]
        top_series = res["mean_abs_shap"].sort_values(ascending=False).head(20)[::-1]

        def _color(feat):
            if feat.startswith("ctx_"):
                return colors_ctx
            if feat.startswith("hh_"):
                return colors_hh
            if feat.startswith("hp_"):
                return colors_hp
            return colors_tab

        colors = [_color(feat) for feat in top_series.index]

        ax.barh(top_series.index, top_series.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{name}\nTop 20 Features by Mean |SHAP|", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors_tab, edgecolor="black", label="Tabular"),
            Patch(facecolor=colors_ctx, edgecolor="black", label="CTX (160-dim)"),
            Patch(facecolor=colors_hh, edgecolor="black", label="Head Hidden (80-dim)"),
            Patch(facecolor=colors_hp, edgecolor="black", label="Head Pre-ReLU (80-dim)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(f"SHAP Feature Importance — derived_8.4-hybrid-lstm-1.4 ({n_models} XGBoost Models, V23 + PCA)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "shap_summary_plots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved visualization plot to {artifacts_dir / 'shap_summary_plots.png'}", flush=True)

    return {
        "df_summary": df_shap_summary,
        "df_top20": df_top20,
        "ctx_vs_tabular": ctx_vs_tabular,
        "shap_results": shap_results,
    }
