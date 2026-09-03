"""Router-only salvage 1.1 (v3 + global baselines): config and pure-function checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_11 = Path("notebooks/experiment/derived_8.4-ece-router-salvage-1.1").resolve()
RUNNER = EXP_11 / "run_salvage.py"


def _module():
    spec = importlib.util.spec_from_file_location("router_salvage_v3", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_uses_v3_and_lists_baselines():
    config = yaml.safe_load((EXP_11 / "config.yaml").read_text(encoding="utf-8"))
    assert config["datasets"]["ece_v3"] == "data/splits/derived_8.4_ece_v3"
    assert "ece_zero_filled" not in config["datasets"]
    assert "ece_native_missing" not in config["datasets"]
    assert config["baselines"] == ["Global_Single_54", "Global_Single_50"]
    assert config["families"] == ["Clustering_V0_Full_k2", "Clustering_Backbone54_k2"]
    assert "direct" in config["policies"]
    assert config["margin_fallback"]["fallback_reference"] == "Global_Single_54"
    assert config["seeds"] == [42, 7, 13, 101, 123]
    assert config["experiment"]["constraint"].startswith("ECE strictly unseen")


def test_v3_split_schema_guards_hold():
    df = pd.read_csv(EXP_11.parents[2] / "data/splits/derived_8.4_ece_v3/test.csv",
                     low_memory=False)
    assert len(df) == 150
    assert len(df.columns) == 499
    assert (df["station_id"].value_counts() == 30).all()
    smap_val = [c for c in df.columns if "SMAP" in c and not c.endswith("_mask")]
    assert len(smap_val) == 82
    assert bool((df[smap_val] != 0.0).all().all())


def test_compute_metrics_shared_with_1_0():
    module = _module()
    y_true = np.array([0.05, 0.06, 0.07])
    y_pred = np.array([0.06, 0.06, 0.06])
    metrics = module.compute_metrics(y_true, y_pred)
    assert metrics["bias"] == float(np.mean(y_pred - y_true))
    assert metrics["rmse"] > 0
    assert metrics["ubrmse"] <= metrics["rmse"] + 1e-12


def test_predictions_v3_covers_chart_series():
    preds = pd.read_csv(EXP_11 / "predictions_v3.csv", low_memory=False)
    assert set(preds["seed"].unique()) == {42, 7, 13, 101, 123}
    assert (preds.groupby(["seed", "station_id"]).size() == 30).all()
    expected = {
        ("Clustering_V0_Full_k2", "as_routed"),
        ("Clustering_V0_Full_k2", "c0_only"),
        ("Clustering_V0_Full_k2", "margin_fallback"),
        ("Clustering_Backbone54_k2", "as_routed"),
        ("Clustering_Backbone54_k2", "c0_only"),
        ("Clustering_Backbone54_k2", "margin_fallback"),
        ("Global_Single_54", "direct"),
        ("Global_Single_50", "direct"),
    }
    assert set(zip(preds["family"], preds["policy"])) == expected
    # Spot-check: seed-mean RMSE of one cell matches station_metrics.csv.
    station = pd.read_csv(EXP_11 / "station_metrics.csv", low_memory=False)
    cell = preds[(preds["seed"] == 42)
                 & (preds["station_id"] == "ECE_Renton_Home")
                 & (preds["family"] == "Global_Single_50")]
    rmse = float((((cell["y_pred"] - cell["y_true"]) ** 2).mean()) ** 0.5)
    ref = station[(station["seed"] == 42)
                  & (station["station"] == "ECE_Renton_Home")
                  & (station["family"] == "Global_Single_50")]
    assert abs(rmse - float(ref["rmse"].iloc[0])) < 1e-9


def test_timeseries_figures_exist_within_line_budget():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    module = _load_plots()
    stations = ["ECE_BBG_Lost_Meadow", "ECE_Renton_Garden_North"]
    mean_df, truth_df = module.seed_mean_frame()
    fig, ax = plt.subplots()
    module._draw_panel(ax, stations[0], mean_df, truth_df, module.V0_SERIES)
    assert len(ax.get_lines()) == 5
    plt.close(fig)
    overlay = EXP_11 / "figures/timeseries_v3_baselines_overlay.png"
    assert overlay.exists(), "overlay figure must exist after notebook execution"
    station_figs = list((EXP_11 / "figures").glob("timeseries_v3_*_*.png"))
    assert len(station_figs) == 10, f"expected 10 station panels, got {len(station_figs)}"


def _load_plots():
    spec = importlib.util.spec_from_file_location(
        "salvage_plots", EXP_11 / "plot_timeseries.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
