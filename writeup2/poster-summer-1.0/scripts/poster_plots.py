"""Poster figures called by the reproducible notebook, with dimensions in inches."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import numpy as np

from poster_data import ROOT, BOUNDARY, BLUE, OCHRE, RED, GRAY, DARK


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 17,
        "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK,
        "ytick.color": DARK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.spines.bottom": False, "pdf.fonttype": 42, "svg.fonttype": "path",
        "savefig.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, name: str) -> None:
    folder = ROOT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(folder / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"Saved figures/{name}.png + vector PDF/SVG")


def station_map(data: dict) -> None:
    style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.16)
    boundary = json.loads(BOUNDARY.read_text())
    # The cached Natural Earth file contains world-wide admin-1 features.
    features = [feature for feature in boundary["features"]
                if feature["properties"].get("iso_3166_2") == "US-WA"]
    assert len(features) == 1, "Expected exactly one Washington state boundary"
    print("Map boundary: Natural Earth admin-1, iso_3166_2=US-WA")
    for feature in features:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            ax.add_patch(Polygon(polygon[0], facecolor="#F0F3F4", edgecolor="#A6B0B6", linewidth=1.0))
    offsets = {
        "Beaver Pass": (-30, 8), "Darrington": (-40, -20),
        "Quinault": (-23, -23), "Cayuse Pass": (28, 10),
        "Paradise": (30, -23), "Spokane": (-12, 15),
        "Sourdough Gulch": (-20, -23),
    }
    for row in data["stations"].itertuples():
        color = BLUE if row.dominant_cluster == 0 else OCHRE
        marker = "o" if row.dominant_cluster == 0 else "D"
        ax.scatter(row.longitude, row.latitude, s=80, color=color, marker=marker,
                   edgecolor="white", linewidth=1.1, zorder=5)
        dx, dy = offsets[row.name]
        label = row.name.replace("Sourdough Gulch", "Sourdough\nGulch")
        ax.annotate(label, (row.longitude, row.latitude), xytext=(dx, dy),
                    textcoords="offset points", fontsize=14, fontweight="bold",
                    ha="center", va="center", color=color,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8}, zorder=6)
    ax.set_xlim(-125.05, -116.20)
    ax.set_ylim(45.45, 49.55)
    ax.set_aspect(1 / np.cos(np.radians(47.5)))
    ax.axis("off")
    handles = [
        Line2D([], [], color=BLUE, marker="o", lw=0, markersize=8, label="Wet / maritime inputs"),
        Line2D([], [], color=OCHRE, marker="D", lw=0, markersize=8, label="Drier / seasonal inputs"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=1, fontsize=13.5, frameon=False, handletextpad=0.4, labelspacing=0.35)
    save(fig, "washington_stations")


def performance(data: dict) -> None:
    style()
    m = data["metrics"]
    fig, ax = plt.subplots(figsize=(10.9, 3.0))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.85, bottom=0.24)
    x = np.array([0, 1.2])
    width = 0.30
    values = [("Global", [m["temporal_global"], m["loso_global"]], GRAY, -width / 2),
              ("Two-regime", [m["temporal_two"], m["loso_two"]], RED, width / 2)]
    for label, scores, color, offset in values:
        bars = ax.bar(x + offset, scores, width * 0.92, label=label, color=color)
        for bar, value in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.4f}",
                    ha="center", fontsize=19, fontweight="bold", color=color)
    ax.set_ylim(0, 1.01)
    ax.set_xlim(-0.55, 1.75)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_ylabel("R²", rotation=0, labelpad=13, fontsize=19)
    ax.set_xticks(x, ["Later years\nknown stations", "Held-out\nstations"], fontsize=18)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="y", alpha=0.14)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.31), ncol=2,
              fontsize=17, frameon=False)
    save(fig, "model_comparison")


def station_effects(data: dict) -> None:
    style()
    table = data["station_effects"]
    fig, ax = plt.subplots(figsize=(10.9, 3.55))
    fig.subplots_adjust(left=0.28, right=0.91, top=0.99, bottom=0.22)
    y = np.arange(len(table))
    colors = [RED if value > 0 else GRAY for value in table.difference]
    ax.barh(y, table.difference, height=0.61, color=colors)
    for pos, value in enumerate(table.difference):
        # The short loss bars get labels to the right of zero so station names remain clear.
        x = value + 0.008 if value >= 0 else 0.010
        ax.text(x, pos, f"{value:+.4f}", va="center", ha="left",
                fontsize=16, fontweight="bold", color=colors[pos])
    ax.axvline(0, color="#56626B", lw=1)
    ax.set_xlim(-0.085, 0.415)
    ax.set_yticks(y, table.name, fontsize=17)
    ax.invert_yaxis()
    ax.set_xticks([0, 0.1, 0.2, 0.3], ["0", "+0.1", "+0.2", "+0.3"], fontsize=15)
    ax.set_xlabel("Change in held-out-station R²", fontsize=17, labelpad=5)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    save(fig, "held_out_station_effects")


def ece(data: dict) -> None:
    style()
    m = data["metrics"]
    labels = ["Standard routing", "Precipitation fallback", "Global model"]
    values = [m["ece_standard"], m["ece_fallback"], m["ece_global"]]
    fig, ax = plt.subplots(figsize=(10.9, 2.15))
    fig.subplots_adjust(left=0.32, right=0.98, top=0.96, bottom=0.32)
    colors = [OCHRE, RED, GRAY]
    ax.barh(range(3), values, height=0.58, color=colors)
    for pos, (value, color) in enumerate(zip(values, colors)):
        ax.text(value + 0.004, pos, f"{value:.4f}", va="center", fontsize=19,
                fontweight="bold", color=color)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.21)
    ax.set_yticks(range(3), labels, fontsize=17)
    ax.set_xticks([0, 0.1, 0.2], ["0", "0.1", "0.2"], fontsize=15)
    ax.set_xlabel("RMSE (m³/m³) · lower is better", fontsize=17, labelpad=5)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    save(fig, "ece_routing")


def generate_all(data: dict) -> None:
    station_map(data)
    performance(data)
    station_effects(data)
    ece(data)
