"""Per-station prediction line charts for the v3 router salvage (1.1).

Every panel shows at most 5 lines (observed + 4 prediction series):
  - Family panels (10 files: 5 stations x V0 / Backbone):
    observed, <family> as_routed / c0_only / margin_fallback, paired baseline
    (Global_Single_50 on V0 panels, Global_Single_54 on Backbone panels).
  - Baseline-showdown overlay (1 file, 5 stacked station panels):
    observed, Global_Single_50, Global_Single_54, V0 c0_only, V0 as_routed.

All prediction lines are means over the 5 seeds in predictions_v3.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FIGURES_DIR = EXP_DIR / "figures"
MAX_LINES = 5

OBSERVED = ("Observed In-Situ", "black", "-", 2.0, True)
V0_SERIES = [
    ("Clustering_V0_Full_k2", "as_routed", "V0 as_routed", "#d62728", "--", 1.5, False),
    ("Clustering_V0_Full_k2", "c0_only", "V0 c0_only", "#1f77b4", "-", 1.5, False),
    ("Clustering_V0_Full_k2", "margin_fallback", "V0 margin_fallback", "#9467bd", "-.", 1.5, False),
    ("Global_Single_50", "direct", "Global-50 direct", "#2ca02c", "-", 1.5, False),
]
BACKBONE_SERIES = [
    ("Clustering_Backbone54_k2", "as_routed", "Backbone as_routed", "#d62728", "--", 1.5, False),
    ("Clustering_Backbone54_k2", "c0_only", "Backbone c0_only", "#1f77b4", "-", 1.5, False),
    ("Clustering_Backbone54_k2", "margin_fallback", "Backbone margin_fallback", "#9467bd", "-.", 1.5, False),
    ("Global_Single_54", "direct", "Global-54 direct", "#ff7f0e", "--", 1.5, False),
]
SHOWDOWN_SERIES = [
    ("Global_Single_50", "direct", "Global-50 direct", "#2ca02c", "-", 1.8, False),
    ("Global_Single_54", "direct", "Global-54 direct", "#ff7f0e", "--", 1.5, False),
    ("Clustering_V0_Full_k2", "c0_only", "V0 c0_only", "#1f77b4", "-", 1.5, False),
    ("Clustering_V0_Full_k2", "as_routed", "V0 as_routed", "#d62728", ":", 1.5, False),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def clean_name(station: str) -> str:
    return station.replace("ECE_", "").replace("_", " ")


def seed_mean_frame() -> pd.DataFrame:
    df = pd.read_csv(EXP_DIR / "predictions_v3.csv", low_memory=False)
    df["date_dt"] = pd.to_datetime(df["date"], errors="raise")
    grouped = df.groupby(["station_id", "date_dt", "family", "policy"],
                         sort=False)["y_pred"].mean().reset_index()
    truth = df.groupby(["station_id", "date_dt"], sort=False)["y_true"].mean().reset_index()
    return grouped, truth


def _draw_panel(ax, station: str, mean_df: pd.DataFrame, truth_df: pd.DataFrame,
                series: list[tuple]) -> None:
    label, color, style, width, markers = OBSERVED
    sub_t = truth_df[truth_df["station_id"] == station].sort_values("date_dt")
    ax.plot(sub_t["date_dt"], sub_t["y_true"], color=color, linestyle=style,
            linewidth=width, marker="o" if markers else None, markersize=4, label=label)
    for family, policy, leg, color, style, width, _ in series:
        sub = mean_df[(mean_df["station_id"] == station)
                      & (mean_df["family"] == family)
                      & (mean_df["policy"] == policy)].sort_values("date_dt")
        if sub.empty:
            raise ValueError(f"No predictions for {station} {family} {policy}")
        ax.plot(sub["date_dt"], sub["y_pred"], color=color, linestyle=style,
                linewidth=width, label=leg)
    n_lines = len(ax.get_lines())
    if n_lines > MAX_LINES:
        raise ValueError(f"{station}: {n_lines} lines exceeds MAX_LINES={MAX_LINES}")
    ax.set_title(f"Station: {clean_name(station)}", fontweight="bold", fontsize=11)
    ax.set_ylabel("Soil Moisture ($m^3/m^3$)")
    ax.grid(True, alpha=0.3)


def plot_family_panels(mean_df: pd.DataFrame, truth_df: pd.DataFrame,
                       stations: list[str]) -> list[str]:
    saved = []
    for suffix, series in (("v0", V0_SERIES), ("backbone", BACKBONE_SERIES)):
        for station in stations:
            fig, ax = plt.subplots(figsize=(11, 4))
            _draw_panel(ax, station, mean_df, truth_df, series)
            ax.legend(loc="best", frameon=True, fontsize=9)
            ax.set_xlabel("Date (2026)")
            fig.tight_layout()
            out = FIGURES_DIR / f"timeseries_v3_{station}_{suffix}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            saved.append(out.name)
    return saved


def plot_showdown_overlay(mean_df: pd.DataFrame, truth_df: pd.DataFrame,
                          stations: list[str]) -> str:
    fig, axes = plt.subplots(len(stations), 1, figsize=(12, 2.6 * len(stations)),
                             sharex=True)
    for ax, station in zip(axes, stations):
        _draw_panel(ax, station, mean_df, truth_df, SHOWDOWN_SERIES)
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)
    axes[-1].set_xlabel("Date (2026)")
    fig.tight_layout()
    out = FIGURES_DIR / "timeseries_v3_baselines_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    FIGURES_DIR.mkdir(exist_ok=True)
    mean_df, truth_df = seed_mean_frame()
    stations = sorted(mean_df["station_id"].unique())
    saved = plot_family_panels(mean_df, truth_df, stations)
    saved.append(plot_showdown_overlay(mean_df, truth_df, stations))
    print(f"Stations: {stations}")
    print(f"Saved {len(saved)} figures:")
    for name in saved:
        print(f"  figures/{name}")


if __name__ == "__main__":
    main()
