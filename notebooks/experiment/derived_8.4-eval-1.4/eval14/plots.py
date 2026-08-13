"""LOSO visualization utilities for derived_8.4-eval-1.4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

STRATEGY_COLORS = {
    "Global_Single": "#7f7f7f",
    "Clustering_V0_Full_k2": "#1f77b4",
    "Clustering_V0_Full_k3": "#6495ed",   # NEW: K-sweep of the V0-Full router
    "Clustering_V0_Full_k4": "#87cefa",   # NEW
    "Clustering_Backbone54_k2": "#17becf",  # routes on the 54 backbone
    "Clustering_Backbone54_k3": "#20b2aa",  # NEW: K-sweep of the 54-backbone router
    "Clustering_Backbone54_k4": "#48d1cc",  # NEW
    "Clustering_Dynamic_k2": "#ff7f0e",
    "Clustering_Dynamic_k3": "#ffa54f",   # NEW: K-sweep of the 3-feature dynamic router
    "Clustering_Dynamic_k4": "#ffc125",   # NEW
    "Clustering_Static_k2": "#9370db",     # NEW: 58 static-attribute router (gating-analysis-1.0)
    "Clustering_Static_k3": "#8a2be2",     # NEW
    "Clustering_Static_k4": "#6a5acd",     # NEW
    "Clustering_Weather_k2": "#8b4513",    # NEW: 16 weather-driver router (gating-analysis-1.0)
    "Clustering_Weather_k3": "#a0522d",    # NEW
    "Clustering_Weather_k4": "#cd853f",    # NEW
    "Univariate_G_API_k2": "#2ca02c",
    "Seasonal_Binary_k2": "#d62728",
    "Trained_Gating_k2": "#9467bd",
}


def _station_order(df: pd.DataFrame) -> list[str]:
    # Sort stations by median LOSO R2 across configs (easiest first).
    med = df.groupby("station")["r2"].median()
    return list(med.sort_values(ascending=False).index)


def plot_config_station_heatmap(df_pcs: pd.DataFrame, out_dir: Path) -> Path:
    """R2 heatmap: rows = configurations (grouped by strategy), cols = held-out stations."""
    df = df_pcs.copy()
    if "config_label" not in df.columns:
        df["config_label"] = df["strategy_name"] + "  c0=" + df["cluster_0_count"].astype(str) + ", c1=" + df["cluster_1_count"].astype(str)
        df.loc[df["is_baseline"], "config_label"] = df.loc[df["is_baseline"], "config_id"]

    config_order = (
        df.sort_values(["strategy_order", "cluster_0_count", "cluster_1_count"])
        .drop_duplicates("config_label")["config_label"]
        .tolist()
    )
    stations = _station_order(df)

    piv = df.pivot_table(index="config_label", columns="station", values="r2")
    piv = piv.reindex(config_order)[stations]

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(stations) + 4), 0.55 * len(config_order) + 3))
    sns.heatmap(
        piv, annot=True, fmt=".3f", cmap="RdYlGn", center=0.0,
        vmin=-0.5, vmax=1.0, linewidths=0.5, ax=ax, cbar_kws={"label": "LOSO R²"},
    )
    ax.set_title("Leave-One-Station-Out R² — Configuration × Held-out Station", fontsize=13, weight="bold")
    ax.set_xlabel("Held-out Station (left = easier)")
    ax.set_ylabel("Configuration")
    plt.tight_layout()
    out_path = out_dir / "loso_r2_config_station_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_config_summary_bar(df_summary: pd.DataFrame, out_dir: Path) -> Path:
    """LOSO-mean R2 per configuration, colored by strategy, with min-max whiskers."""
    df = df_summary.sort_values("loso_mean_r2", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(12, max(8, 0.28 * len(df) + 3)))
    colors = [STRATEGY_COLORS.get(s, "#999999") for s in df["strategy_name"]]

    y = np.arange(len(df))
    ax.barh(y, df["loso_mean_r2"], color=colors, edgecolor="k", linewidth=0.4)
    ax.errorbar(
        df["loso_mean_r2"], y,
        xerr=[df["loso_mean_r2"] - df["loso_min_r2"], df["loso_max_r2"] - df["loso_mean_r2"]],
        fmt="none", ecolor="k", elinewidth=0.8, capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["config_label"], fontsize=8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("LOSO mean R² (across held-out stations)")
    ax.set_title("Leave-One-Station-Out Mean R² by Configuration\n(whiskers = min/max over stations)", fontsize=13, weight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=9, label=s)
        for s, c in STRATEGY_COLORS.items() if s in set(df["strategy_name"])
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Strategy")
    plt.tight_layout()
    out_path = out_dir / "loso_r2_config_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_station_difficulty(df_station: pd.DataFrame, out_dir: Path) -> Path:
    """Per-station LOSO R2 (median across configs) to identify difficult stations."""
    df = df_station.sort_values("median_r2", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(df) + 2), 5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["median_r2"], color="#1f77b4", edgecolor="k", linewidth=0.6)
    ax.errorbar(
        x, df["median_r2"],
        yerr=[df["median_r2"] - df["min_r2"], df["max_r2"] - df["median_r2"]],
        fmt="none", ecolor="k", elinewidth=0.8, capsize=3,
    )
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["station"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LOSO R² (median across configs)")
    ax.set_title("Station Difficulty — LOSO R² Across All Configurations\n(whiskers = min/max over configs)", fontsize=13, weight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, df["median_r2"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "loso_station_difficulty.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_station_boxplot(df_pcs: pd.DataFrame, out_dir: Path) -> Path:
    """Boxplot of per-config LOSO R2 for each held-out station."""
    df = df_pcs.copy()
    order = _station_order(df)
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(order) + 2), 5.5))
    data = [df.loc[df["station"] == s, "r2"].dropna().values for s in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel("LOSO R² (per configuration)")
    ax.set_xlabel("Held-out Station")
    ax.set_title("Distribution of LOSO R² Across Configurations by Held-out Station", fontsize=13, weight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = out_dir / "loso_r2_station_boxplot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Full-training baseline vs LOSO (intrinsic vs. generalization difficulty)
# ---------------------------------------------------------------------------


def plot_full_vs_loso_scatter(df_merge: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter of per-station median R2: full training (x) vs LOSO (y).

    ``df_merge`` has columns station / full_median_r2 / loso_median_r2. Points
    below the identity line are harder under LOSO than under full training
    (generalization-limited); points on the line are equally hard both ways
    (intrinsically hard stations).
    """
    df = df_merge.sort_values("loso_median_r2")
    fig, ax = plt.subplots(figsize=(8, 7))
    lims = [min(df["full_median_r2"].min(), df["loso_median_r2"].min()) - 0.05,
            max(df["full_median_r2"].max(), df["loso_median_r2"].max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=1.0, label="identity (equally hard both ways)")
    ax.scatter(df["full_median_r2"], df["loso_median_r2"], s=140,
               c=df["loso_median_r2"], cmap="RdYlGn", vmin=0.3, vmax=0.8,
               edgecolors="k", linewidths=0.8, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["station"].replace("_WA", "").replace("_", " "),
                    (row["full_median_r2"], row["loso_median_r2"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    rho, p = spearmanr(df["full_median_r2"], df["loso_median_r2"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Full-training median R² (trained on all stations)")
    ax.set_ylabel("LOSO median R² (station held out)")
    ax.set_title(f"Station difficulty: full training vs LOSO (Spearman ρ = {rho:+.2f}, p = {p:.2f})",
                 fontsize=12, weight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "full_vs_loso_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_full_vs_loso_bars(df_merge: pd.DataFrame, out_dir: Path) -> Path:
    """Paired bars: full-training median R2 vs LOSO median R2 per station.

    Sorted by LOSO difficulty (easiest left, hardest right); the gap between the
    two bars is the LOSO cost (full − LOSO).
    """
    df = df_merge.sort_values("loso_median_r2", ascending=False).copy()
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(df) + 2), 5.5))
    b1 = ax.bar(x - w / 2, df["full_median_r2"], w, label="Full training", color="#1f77b4", edgecolor="k", lw=0.5)
    b2 = ax.bar(x + w / 2, df["loso_median_r2"], w, label="LOSO (held out)", color="#ff7f0e", edgecolor="k", lw=0.5)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_WA", "") for s in df["station"]], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Median R² (over all configs)")
    ax.set_title("Station difficulty: full training vs LOSO\n(gap = LOSO cost of holding the station out)",
                 fontsize=12, weight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "full_vs_loso_station_bars.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_loso_gap_boxplot(df_full_pcs: pd.DataFrame, df_loso_pcs: pd.DataFrame, out_dir: Path) -> Path:
    """Per-station distribution of (full-training R2 − LOSO R2) over all configurations.

    Positive gap = the station suffers from being held out (generalization-limited);
    gap near 0 = equally hard both ways (intrinsically hard); negative = better
    under LOSO than full training (anomaly worth flagging).
    """
    merged = df_full_pcs[["config_id", "station", "r2"]].merge(
        df_loso_pcs[["config_id", "station", "r2"]],
        on=["config_id", "station"], suffixes=("_full", "_loso"),
    )
    merged["gap"] = merged["r2_full"] - merged["r2_loso"]
    order = (
        merged.groupby("station")["gap"].median()
        .sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(order) + 2), 5.5))
    data = [merged.loc[merged["station"] == s, "gap"].dropna().values for s in order]
    bp = ax.boxplot(data, labels=[s.replace("_WA", "") for s in order],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
    ax.axhline(0, color="k", lw=1.0, ls="--")
    ax.set_ylabel("Full-training R² − LOSO R² (per configuration)")
    ax.set_xlabel("Station (left = largest positive gap)")
    ax.set_title("LOSO cost per station — full-training R² minus LOSO R² over all configs",
                 fontsize=12, weight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "loso_gap_boxplot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
