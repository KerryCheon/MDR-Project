"""Visualization and figure generator for derived_8.4-ece-additional-eval-1.0."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EXP_DIR = Path(__file__).resolve().parent
FIGURES_DIR = EXP_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CONFIG_COLORS = {
    "d80_no_weights": "#2b5c8f",     # Deep Steel Blue
    "d80_weighted": "#4682b4",       # Steel Blue
    "d84_no_weights": "#d95f02",     # Dark Orange/Rust
    "d84_weighted": "#f1884d",       # Warm Orange
}

CONFIG_LABELS = {
    "d80_no_weights": "D8.0 (5 st) — No Weights",
    "d80_weighted": "D8.0 (5 st) — Weighted (β=0.2)",
    "d84_no_weights": "D8.4 (7 st) — No Weights",
    "d84_weighted": "D8.4 (7 st) — Weighted (β=0.2)",
}


def plot_seed_boxplots(df_seed_ece: pd.DataFrame, df_seed_temp: pd.DataFrame):
    """Plot seed dispersion boxplots for ECE and Temporal R2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. ECE Spatial R2
    configs = ["d80_no_weights", "d80_weighted", "d84_no_weights", "d84_weighted"]
    data_ece = [df_seed_ece[df_seed_ece["config_id"] == c]["r2"].dropna().values for c in configs]
    labels = [CONFIG_LABELS.get(c, c) for c in configs]
    colors = [CONFIG_COLORS.get(c, "#888888") for c in configs]

    bp1 = ax1.boxplot(data_ece, patch_artist=True, tick_labels=labels, widths=0.55,
                      medianprops=dict(color="black", linewidth=1.5),
                      boxprops=dict(linewidth=1.2),
                      whiskerprops=dict(linewidth=1.2),
                      capprops=dict(linewidth=1.2))
    for patch, col in zip(bp1["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    # Add jittered points
    for i, pts in enumerate(data_ece):
        x = np.random.normal(i + 1, 0.04, size=len(pts))
        ax1.scatter(x, pts, color="black", alpha=0.75, s=25, zorder=4)

    ax1.set_title("In-Situ ECE Spatial Generalization ($R^2$ across Seeds)", fontweight="bold")
    ax1.set_ylabel("Coefficient of Determination ($R^2$)")
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.grid(True, alpha=0.3)

    # 2. Temporal In-Distribution R2
    data_temp = [df_seed_temp[df_seed_temp["config_id"] == c]["r2"].dropna().values for c in configs]
    bp2 = ax2.boxplot(data_temp, patch_artist=True, tick_labels=labels, widths=0.55,
                      medianprops=dict(color="black", linewidth=1.5),
                      boxprops=dict(linewidth=1.2),
                      whiskerprops=dict(linewidth=1.2),
                      capprops=dict(linewidth=1.2))
    for patch, col in zip(bp2["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    for i, pts in enumerate(data_temp):
        x = np.random.normal(i + 1, 0.04, size=len(pts))
        ax2.scatter(x, pts, color="black", alpha=0.75, s=25, zorder=4)

    ax2.set_title("In-Distribution Temporal Evaluation ($R^2$ across Seeds)", fontweight="bold")
    ax2.set_ylabel("Coefficient of Determination ($R^2$)")
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "seed_boxplot_ece_vs_temp_r2.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved: {out_path.name}")


def plot_transfer_gap(df_gap: pd.DataFrame):
    """Plot the transfer gap (Temporal R2 vs ECE R2) for all 4 models."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    configs = ["d80_no_weights", "d80_weighted", "d84_no_weights", "d84_weighted"]
    sub = df_gap.set_index("config_id").reindex(configs).reset_index()

    x = np.arange(len(configs))
    width = 0.35

    rects1 = ax.bar(x - width/2, sub["temp_r2_mean"], width, label="Temporal (In-Distribution)", color="#1b9e77", alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width/2, sub["r2_mean"], width, label="In-Situ ECE (Spatial Transfer)", color="#d95f02", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Mean $R^2$ (5 Seeds)")
    ax.set_title("Spatial Transfer Degradation Gap: In-Distribution vs In-Situ ECE Sensors", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], rotation=15, ha="right")
    ax.legend(loc="upper right", frameon=True)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    # Annotate values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, -12 if h < 0 else 3), textcoords="offset points", ha="center",
                    va="top" if h < 0 else "bottom", fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "temporal_vs_ece_transfer_gap.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved: {out_path.name}")


def plot_per_station_ece_bars(df_st_median: pd.DataFrame):
    """Plot per-station R2 comparison across the 5 ECE stations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    stations = sorted(df_st_median["station_id"].unique())
    configs = ["d80_no_weights", "d80_weighted", "d84_no_weights", "d84_weighted"]

    x = np.arange(len(stations))
    width = 0.18

    for i, cfg in enumerate(configs):
        sub = df_st_median[df_st_median["config_id"] == cfg].set_index("station_id").reindex(stations)
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, sub["r2"], width, label=CONFIG_LABELS.get(cfg, cfg),
                       color=CONFIG_COLORS.get(cfg, "#888888"), alpha=0.85, edgecolor="black")

    clean_station_names = [s.replace("ECE_", "").replace("_", " ") for s in stations]
    ax.set_ylabel("Median $R^2$ across 5 Seeds")
    ax.set_title("Per-Station Generalization across 5 In-Situ ECE Deployments", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(clean_station_names, rotation=15, ha="right")
    ax.legend(loc="lower left", frameon=True)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "per_station_ece_comparison_r2.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved: {out_path.name}")


def plot_ece_timeseries(preds_df: pd.DataFrame):
    """Plot observed vs predicted time series across the 5 ECE stations."""
    stations = sorted(preds_df["station_id"].unique())
    fig, axes = plt.subplots(len(stations), 1, figsize=(13, 2.5 * len(stations)), sharex=True)

    if len(stations) == 1:
        axes = [axes]

    for ax, st in zip(axes, stations):
        sub = preds_df[preds_df["station_id"] == st].sort_values("date").copy()
        sub["date_dt"] = pd.to_datetime(sub["date"])

        ax.plot(sub["date_dt"], sub["y_true"], color="black", linewidth=2.0, label="Observed In-Situ", marker="o", markersize=4)

        # Plot ensemble/mean predictions for D8.0 vs D8.4
        d80_nw_cols = [c for c in sub.columns if c.startswith("pred__d80_no_weights")]
        d84_nw_cols = [c for c in sub.columns if c.startswith("pred__d84_no_weights")]
        d80_w_cols = [c for c in sub.columns if c.startswith("pred__d80_weighted")]
        d84_w_cols = [c for c in sub.columns if c.startswith("pred__d84_weighted")]

        if d80_nw_cols:
            ax.plot(sub["date_dt"], sub[d80_nw_cols].mean(axis=1), color=CONFIG_COLORS["d80_no_weights"],
                    linestyle="--", linewidth=1.5, label="D8.0 No-Weights (Mean)")
        if d84_nw_cols:
            ax.plot(sub["date_dt"], sub[d84_nw_cols].mean(axis=1), color=CONFIG_COLORS["d84_no_weights"],
                    linestyle=":", linewidth=1.8, label="D8.4 No-Weights (Mean)")
        if d80_w_cols:
            ax.plot(sub["date_dt"], sub[d80_w_cols].mean(axis=1), color=CONFIG_COLORS["d80_weighted"],
                    linestyle="-.", linewidth=1.5, label="D8.0 Weighted (Mean)")
        if d84_w_cols:
            ax.plot(sub["date_dt"], sub[d84_w_cols].mean(axis=1), color=CONFIG_COLORS["d84_weighted"],
                    linestyle="-", linewidth=1.5, label="D8.4 Weighted (Mean)", alpha=0.7)

        clean_name = st.replace("ECE_", "").replace("_", " ")
        ax.set_title(f"Station: {clean_name}", fontweight="bold", fontsize=11)
        ax.set_ylabel("Soil Moisture ($m^3/m^3$)")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", frameon=True, ncol=3, fontsize=9)
    plt.xlabel("Date (2026)")
    plt.tight_layout()
    out_path = FIGURES_DIR / "ece_timeseries_predictions_overlay.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved: {out_path.name}")


def plot_feature_importance_comparison(df_fi: pd.DataFrame, top_k: int = 20):
    """Plot top K feature importances comparing D8.0 and D8.4 models."""
    top_df = df_fi.head(top_k).sort_values("mean_all", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top_df))

    height = 0.35
    ax.barh(y - height/2, top_df.get("d80_no_weights", 0), height=height, label="D8.0 (5 Stations)", color="#2b5c8f", alpha=0.85)
    ax.barh(y + height/2, top_df.get("d84_no_weights", 0), height=height, label="D8.4 (7 Stations)", color="#d95f02", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(top_df.index, fontsize=9)
    ax.set_xlabel("Normalized Feature Importance (Gini / Gain)")
    ax.set_title(f"Top {top_k} Feature Importances: Derived 8.0 vs Derived 8.4", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_importance_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved: {out_path.name}")


def plot_residual_distributions(preds_df: pd.DataFrame):
    """Plot residual distributions for the 4 models on the ECE dataset."""
    fig, ax = plt.subplots(figsize=(10, 5))

    configs = ["d80_no_weights", "d80_weighted", "d84_no_weights", "d84_weighted"]
    for cfg in configs:
        pred_cols = [c for c in preds_df.columns if c.startswith(f"pred__{cfg}")]
        if not pred_cols:
            continue
        mean_pred = preds_df[pred_cols].mean(axis=1)
        residuals = preds_df["y_true"] - mean_pred
        sns.kdeplot(residuals, ax=ax, label=CONFIG_LABELS.get(cfg, cfg), color=CONFIG_COLORS.get(cfg, "#888888"), linewidth=2.0)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Residual: True - Predicted ($m^3/m^3$)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Error Distributions on In-Situ ECE Sensor Deployments", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "residual_distribution_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved: {out_path.name}")


def main():
    print("[Plots] Loading CSV artifacts...")
    df_seed_ece = pd.read_csv(EXP_DIR / "seed_summary_ece.csv")
    df_seed_temp = pd.read_csv(EXP_DIR / "seed_summary_temporal.csv")
    df_gap = pd.read_csv(EXP_DIR / "transfer_gap_summary.csv")
    df_st_median = pd.read_csv(EXP_DIR / "station_median_summary_ece.csv")
    preds_df = pd.read_csv(EXP_DIR / "predictions_ece_df.csv")
    df_fi = pd.read_csv(EXP_DIR / "feature_importances.csv", index_col=0)

    print("[Plots] Generating publication figures...")
    plot_seed_boxplots(df_seed_ece, df_seed_temp)
    plot_transfer_gap(df_gap)
    plot_per_station_ece_bars(df_st_median)
    plot_ece_timeseries(preds_df)
    plot_feature_importance_comparison(df_fi)
    plot_residual_distributions(preds_df)
    print(f"[Plots] All figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
