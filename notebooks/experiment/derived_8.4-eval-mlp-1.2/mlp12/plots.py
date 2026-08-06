"""Plotting helpers for derived_8.4-eval-mlp-1.1.

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

FAMILY_LABELS = {
    "1regime_54": "1-regime Global (54 backbone)",
    "2regime_54": "2-regime Cluster (c0=54, c1=64)",
    "1regime_96": "1-regime Global (96 pool)",
    "2regime_96": "2-regime Cluster (96 pool)",
}


def plot_mlp_loss_curves(loss_curves: dict[str, list[float]], out_dir: Path, title_suffix: str = "") -> Path:
    """Consolidated per-epoch test-RMSE curves for the MLP models."""
    plt.figure(figsize=(11, 6.5))
    for mname, curve in loss_curves.items():
        plt.plot(curve, label=mname, alpha=0.85, linewidth=1.6)
    plt.xlabel("Epoch")
    plt.ylabel("Test RMSE")
    plt.title(f"Neural Tabular Test RMSE per Epoch (early stop on val){title_suffix}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = out_dir / "loss_curves_mlp.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_sweep_summary(sweep_df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter of test R² vs val RMSE per family (best configs labelled)."""
    families = [f for f in FAMILY_LABELS if f in set(sweep_df["family"])]
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = sweep_df[sweep_df["family"] == family]
        if sub.empty:
            ax.set_title(f"{FAMILY_LABELS[family]} — no results")
            continue
        xkey = "robust_score" if "robust_score" in sub.columns else "val_rmse"
        xlab = "Robust score (selection)" if xkey == "robust_score" else "Val RMSE (selection)"
        sc = ax.scatter(sub[xkey], sub["test_r2"], s=60, c="#1f77b4", alpha=0.8)
        best = sub.iloc[0]
        ax.scatter([best[xkey]], [best["test_r2"]], s=160, facecolors="none",
                   edgecolors="#d62728", linewidths=2, label="best (robust)")
        for _, row in sub.head(5).iterrows():
            ax.annotate(row["config_id"], (row[xkey], row["test_r2"]),
                        fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Test R²")
        ax.set_title(f"Sweep: {FAMILY_LABELS[family]}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    plt.suptitle("Neural Tabular Hyperparameter Sweep (48 configs, 2-regime families)", fontsize=13, weight="bold")
    plt.tight_layout()
    out_path = out_dir / "sweep_summary.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
