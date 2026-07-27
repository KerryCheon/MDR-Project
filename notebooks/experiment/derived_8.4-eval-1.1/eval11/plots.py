"""Plotting and visualization utilities for derived_8.4-eval-1.1."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_diagnostics(mname: str, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.2, s=8, color="#1f77b4")
    m_val = max(np.max(y_true), np.max(y_pred)) if len(y_true) > 0 else 0.6
    ax.plot([0, m_val], [0, m_val], "r--", linewidth=1.5)
    ax.set_xlabel("Observed SM")
    ax.set_ylabel("Predicted SM")
    ax.set_title(f"Scatter: {mname}")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Residual histogram
    ax = axes[1]
    res = y_pred - y_true
    ax.hist(res, bins=50, color="#2ca02c", alpha=0.7, edgecolor="k")
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual (Pred - Obs)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals: {mname}")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    sanitized = mname.replace(":", "").replace(" ", "_").replace("=", "").replace("/", "_").replace("(", "").replace(")", "").replace("+", "plus")
    out_path = out_dir / f"diag_{sanitized}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def plot_per_regime_diagnostics(
    mname: str, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, out_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e"]
    for c in range(2):
        mask = labels == c
        if not mask.any():
            continue
        ax = axes[0]
        ax.scatter(
            y_true[mask],
            y_pred[mask],
            alpha=0.3,
            s=10,
            color=colors[c],
            label=f"Regime {c}",
        )

        ax = axes[1]
        res = (y_pred - y_true)[mask]
        ax.hist(res, bins=40, alpha=0.5, color=colors[c], label=f"Regime {c}", density=True)

    axes[0].plot([0, 0.6], [0, 0.6], "k--", alpha=0.7)
    axes[0].set_xlabel("Observed")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Per-Regime Scatter: {mname}")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].axvline(0, color="k", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Per-Regime Residual Density: {mname}")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    sanitized = mname.replace(":", "").replace(" ", "_").replace("=", "").replace("/", "_").replace("(", "").replace(")", "").replace("+", "plus")
    out_path = out_dir / f"per_regime_diag_{sanitized}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def plot_yearly_performance_linechart(summary_df: pd.DataFrame, out_dir: Path) -> Path:
    plt.figure(figsize=(10, 6))
    years = [2023, 2024, 2025]
    for idx, row in summary_df.iterrows():
        mname = row["model_name"]
        r2_vals = [row.get("year_2023_r2", np.nan), row.get("year_2024_r2", np.nan), row.get("year_2025_r2", np.nan)]
        plt.plot(years, r2_vals, marker="o", linewidth=2, label=mname)

    plt.xticks(years)
    plt.xlabel("Year")
    plt.ylabel("Test R²")
    plt.title("Yearly R² Stability Across Models (2023–2025)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = out_dir / "yearly_r2_linechart.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_delta_grid_heatmap(grid_df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    strategies = sorted(grid_df["strategy_name"].unique())

    for idx, strat in enumerate(strategies):
        if idx >= len(axes):
            break
        ax = axes[idx]
        sub = grid_df[grid_df["strategy_name"] == strat]
        piv = sub.pivot_table(index="cluster_0_count", columns="cluster_1_count", values="pooled_r2")
        sns.heatmap(piv, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax, cbar=False)
        ax.set_title(f"Strategy: {strat}")
        ax.set_xlabel("Cluster 1 Additions")
        ax.set_ylabel("Cluster 0 Additions")

    # Hide unused subplots if any
    for i in range(len(strategies), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Strategy × Add-Only Delta Grid Search (Pooled Test R²)", fontsize=14, weight="bold")
    plt.tight_layout()
    out_path = out_dir / "delta_grid_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_loss_curves(loss_curves: dict[str, list[float]], out_dir: Path) -> tuple[Path, Path]:
    # Consolidated loss curves
    plt.figure(figsize=(10, 6))
    for mname, curve in loss_curves.items():
        plt.plot(curve, label=mname, alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Test RMSE")
    plt.title("Consolidated Test Loss Curves (2,500 Iterations)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_cons = out_dir / "loss_curves_consolidated.png"
    plt.savefig(out_cons, dpi=150)
    plt.close()

    # Grouped loss curves
    plt.figure(figsize=(10, 6))
    for mname, curve in loss_curves.items():
        plt.plot(curve, label=mname, alpha=0.8, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Test RMSE")
    plt.title("Grouped Model Evaluation Loss Curves")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_group = out_dir / "loss_curves_grouped.png"
    plt.savefig(out_group, dpi=150)
    plt.close()

    return out_cons, out_group
