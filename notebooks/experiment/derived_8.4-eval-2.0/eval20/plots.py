"""LOSO figures for derived_8.4-eval-2.0 — reuse eval-1.2's plots with MLP colors.

All plot functions are the eval-1.2 implementations (eval12.plots); the only
change is the strategy color map, extended in place with the MLP families and
the XGBoost reference tag so the bar chart groups them visually.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
EVAL12_DIR = EXP_DIR.parent / "derived_8.4-eval-1.2"
if str(EVAL12_DIR) not in sys.path:
    sys.path.append(str(EVAL12_DIR))

from eval12 import plots as _p  # noqa: E402

# Extend the eval-1.2 strategy color map with the MLP families + XGBoost refs.
_p.STRATEGY_COLORS.update({
    "MLP_1regime_54": "#8c8c8c",
    "MLP_1regime_96": "#c5b0d5",
    "MLP_2regime_54": "#1f77b4",
    "MLP_2regime_96": "#2ca02c",
    "XGBoost_Reference": "#d62728",
})

# Re-export the eval-1.2 plot functions (identical implementations).
plot_config_station_heatmap = _p.plot_config_station_heatmap
plot_config_summary_bar = _p.plot_config_summary_bar
plot_station_difficulty = _p.plot_station_difficulty
plot_station_boxplot = _p.plot_station_boxplot
plot_full_vs_loso_scatter = _p.plot_full_vs_loso_scatter
plot_full_vs_loso_bars = _p.plot_full_vs_loso_bars
plot_loso_gap_boxplot = _p.plot_loso_gap_boxplot
