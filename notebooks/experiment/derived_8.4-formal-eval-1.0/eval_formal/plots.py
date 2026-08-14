"""Figures for derived_8.4-formal-eval-1.0 (generated only by the report notebook).

- temporal: per-config seed boxplots (R2), paired per-seed difference plots for
  headline comparisons, delta-source robustness bars per strategy.
- loso: per-station pair scatter (A vs B, identity line, "k of 7 stations" win
  annotation), per-station bars with seed error bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import seed_summary

HEADLINE_PAIRS = [
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Global_Single_54"),
    ("Clustering_Backbone54_k2_c0_10_c1_10", "Global_Single_54"),
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Baseline_V0_50"),
    ("Clustering_Backbone54_k2_c0_10_c1_10", "Baseline_V0_50"),
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Trained_Gating_k2_c0_5_c1_10"),
    ("Clustering_Backbone54_k2_c0_10_c1_10", "Trained_Gating_k2_c0_5_c1_10"),
]


def _label(config_id: str, cfg_frame: pd.DataFrame) -> str:
    row = cfg_frame[cfg_frame["config_id"] == config_id]
    if row.empty:
        return config_id
    label = str(row.iloc[0]["config_label"])
    return label if len(label) <= 42 else config_id


def plot_seed_boxplot(seed_df: pd.DataFrame, cfg_frame: pd.DataFrame, out_dir: Path,
                      metric: str = "r2", figsize=(16, 6)) -> Path:
    """Per-config boxplot of the per-seed metric (temporal)."""
    order = (
        seed_df.groupby("config_id")[metric].median()
        .sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=figsize)
    data = [seed_df.loc[seed_df["config_id"] == c, metric].dropna() for c in order]
    ax.boxplot(data, labels=[_label(c, cfg_frame) for c in order], showmeans=True)
    ax.set_title(f"Per-seed {metric.upper()} by configuration (temporal test set, 30 seeds)")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    path = out_dir / f"temporal_seed_boxplot_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_paired_differences(seed_df: pd.DataFrame, cfg_frame: pd.DataFrame, out_dir: Path,
                            pair: tuple[str, str], metric: str = "r2") -> Path:
    """Per-seed paired difference (A - B) with 95% t-CI for one headline pair."""
    a, b = pair
    sub = seed_df[seed_df["config_id"].isin([a, b])]
    pa = sub[sub["config_id"] == a].set_index("seed")[metric].sort_index()
    pb = sub[sub["config_id"] == b].set_index("seed")[metric].sort_index()
    common = pa.index.intersection(pb.index)
    diff = pa.loc[common].to_numpy() - pb.loc[common].to_numpy()
    s = seed_summary(pd.Series(diff))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(range(len(diff)), np.sort(diff), s=28, alpha=0.8)
    ax.axhline(0, color="grey", lw=1)
    ax.axhline(s["mean"], color="tab:red", lw=1.5, label=f"mean diff {s['mean']:.5f}")
    ax.axhline(s["ci_low"], color="tab:red", ls="--", lw=1, label=f"95% CI [{s['ci_low']:.5f}, {s['ci_high']:.5f}]")
    ax.axhline(s["ci_high"], color="tab:red", ls="--", lw=1)
    ax.set_title(f"Per-seed {metric.upper()} difference\n{_label(a, cfg_frame)} - {_label(b, cfg_frame)}")
    ax.set_xlabel("seed rank")
    ax.set_ylabel(f"$\\Delta$ {metric.upper()}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"paired_diff_{metric}_{a}_vs_{b}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_delta_robustness(seed_df: pd.DataFrame, cfg_frame: pd.DataFrame, out_dir: Path,
                          metric: str = "r2") -> Path:
    """Delta-source robustness: per strategy, temporal metric mean ± std over seeds
    for test-selected / val-selected / none (c0=c1=0) delta sources."""
    strategies = sorted(cfg_frame["strategy_name"].unique())
    strategies = [s for s in strategies if s not in ("Global_Single",)]
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.25
    for i, strategy in enumerate(strategies):
        rows = cfg_frame[cfg_frame["strategy_name"] == strategy]
        sources = []
        for source in ("test", "val", "none"):
            cid = rows[rows["delta_source"] == source]["config_id"]
            if not cid.empty:
                sources.append((source, cid.iloc[0]))
        for j, (source, cid) in enumerate(sources):
            vals = seed_df.loc[seed_df["config_id"] == cid, metric].dropna()
            if vals.empty:
                continue
            s = seed_summary(vals)
            x = i + (j - 1) * width
            ax.bar(x, s["mean"], width=width, yerr=s["std"] if s["std"] == s["std"] else 0,
                   label=source if i == 0 else None, capsize=3)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylabel(f"temporal {metric.upper()} (mean ± std over seeds)")
    ax.set_title(f"Delta-source robustness: test-selected vs val-selected vs no delta ({metric.upper()})")
    ax.legend(title="delta source")
    fig.tight_layout()
    path = out_dir / f"delta_robustness_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loso_pair(per_station_df: pd.DataFrame, cfg_frame: pd.DataFrame, out_dir: Path,
                   pair: tuple[str, str], metric: str = "r2") -> Path:
    """LOSO per-station pair scatter: A vs B (per-station median over seeds), with
    identity line and 'A wins k of 7 stations' annotation."""
    a, b = pair
    med = (
        per_station_df.groupby(["config_id", "station"])[metric]
        .median().reset_index()
    )
    ma = med[med["config_id"] == a].set_index("station")[metric]
    mb = med[med["config_id"] == b].set_index("station")[metric]
    common = ma.index.intersection(mb.index)
    x = mb.loc[common].to_numpy()
    y = ma.loc[common].to_numpy()
    wins = int(np.sum(y > x))
    n = len(common)

    fig, ax = plt.subplots(figsize=(7, 7))
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", lw=1, ls="--")
    ax.scatter(x, y, s=70)
    for i, st in enumerate(common):
        ax.annotate(st.replace("_WA", "").replace("_", " "), (x[i], y[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel(f"{_label(b, cfg_frame)} — {metric.upper()}")
    ax.set_ylabel(f"{_label(a, cfg_frame)} — {metric.upper()}")
    ax.set_title(f"LOSO per-station {metric.upper()}: {_label(a, cfg_frame)} wins {wins} of {n} stations")
    fig.tight_layout()
    path = out_dir / f"loso_pair_{metric}_{a}_vs_{b}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loso_station_bars(per_station_df: pd.DataFrame, cfg_frame: pd.DataFrame,
                           out_dir: Path, config_id: str, metric: str = "r2") -> Path:
    """Per-station LOSO metric with seed error bars for one config."""
    sub = per_station_df[per_station_df["config_id"] == config_id]
    stations = sorted(sub["station"].unique())
    fig, ax = plt.subplots(figsize=(12, 5))
    means, stds = [], []
    for st in stations:
        vals = sub.loc[sub["station"] == st, metric].dropna()
        s = seed_summary(vals)
        means.append(s["mean"])
        stds.append(s["std"] if s["std"] == s["std"] else 0.0)
    ax.bar(range(len(stations)), means, yerr=stds, capsize=4)
    ax.set_xticks(range(len(stations)))
    ax.set_xticklabels([s.replace("_WA", "") for s in stations], rotation=30, ha="right")
    ax.set_ylabel(f"LOSO {metric.upper()}")
    ax.set_title(f"LOSO per-station {metric.upper()} (mean ± std over seeds): {_label(config_id, cfg_frame)}")
    fig.tight_layout()
    path = out_dir / f"loso_station_bars_{metric}_{config_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
