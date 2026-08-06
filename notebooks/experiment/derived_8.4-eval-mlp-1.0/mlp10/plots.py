"""Plotting helpers for derived_8.4-eval-mlp-1.0.

Re-exports the shared diagnostics from eval11.plots and adds MLP-specific
figures: per-epoch test-RMSE loss curves and the sweep summary plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval11.plots import (  # noqa: F401  (re-exported for the caller)
    plot_delta_grid_heatmap,
    plot_diagnostics,
    plot_loss_curves,
    plot_per_regime_diagnostics,
    plot_yearly_performance_linechart,
)


def plot_mlp_loss_curves(loss_curves: dict[str, list[float]], out_dir: Path, title_suffix: str = "") -> Path:
    """Consolidated per-epoch test-RMSE curves for the MLP models."""
    plt.figure(figsize=(11, 6.5))
    for mname, curve in loss_curves.items():
        plt.plot(curve, label=mname, alpha=0.85, linewidth=1.6)
    plt.xlabel("Epoch")
    plt.ylabel("Test RMSE")
    plt.title(f"MLP Test RMSE per Epoch (early stop on holdout){title_suffix}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = out_dir / "loss_curves_mlp.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_sweep_summary(sweep_df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter of test R² vs holdout RMSE per family (best configs labelled)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=False)
    for ax, family, fam_label in zip(
        axes, ["1regime", "2regime"], ["1-regime (Global 54)", "2-regime (Cluster c0=0,c1=10)"]
    ):
        sub = sweep_df[sweep_df["family"] == family]
        if sub.empty:
            ax.set_title(f"{fam_label} — no results")
            continue
        sc = ax.scatter(sub["holdout_rmse"], sub["test_r2"], s=60, c="#1f77b4", alpha=0.8)
        best = sub.iloc[0]
        ax.scatter([best["holdout_rmse"]], [best["test_r2"]], s=160, facecolors="none",
                   edgecolors="#d62728", linewidths=2, label="best (holdout)")
        for _, row in sub.head(5).iterrows():
            ax.annotate(row["config_id"], (row["holdout_rmse"], row["test_r2"]),
                        fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Holdout RMSE (selection)")
        ax.set_ylabel("Test R²")
        ax.set_title(f"Sweep: {fam_label}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    plt.suptitle("MLP Hyperparameter Sweep (27 configs per family)", fontsize=13, weight="bold")
    plt.tight_layout()
    out_path = out_dir / "sweep_summary.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
