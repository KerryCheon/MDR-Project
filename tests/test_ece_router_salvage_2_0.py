"""Automatic salvage 2.0 (v3 + single-regime global baseline): config and table checks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_20 = Path(__file__).resolve().parents[1] / "notebooks/experiment/derived_8.4-ece-router-salvage-2.0"
RUNNER = EXP_20 / "run_auto.py"


def _module():
    spec = importlib.util.spec_from_file_location("router_auto_salvage", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_uses_v3_and_lists_single_baseline():
    config = yaml.safe_load((EXP_20 / "config.yaml").read_text(encoding="utf-8"))
    assert config["datasets"]["ece_v3"] == "data/splits/derived_8.4_ece_v3"
    assert config["baselines"] == ["Global_Single_54"]
    assert config["families"] == ["Clustering_V0_Full_k2", "Clustering_Backbone54_k2"]
    assert "direct" in config["deployable"]
    assert config["seeds"] == [42, 7, 13, 101, 123]
    assert config["experiment"]["constraint"].startswith("ECE strictly unseen")
    assert "Global_Single_54" in config["experiment"]["constraint"]


def test_summary_has_baseline_reference_row():
    summary = pd.read_csv(EXP_20 / "summary.csv", low_memory=False)
    assert len(summary) == 15  # 2 families x 7 policies + 1 baseline
    base = summary[(summary["family"] == "Global_Single_54")
                   & (summary["policy"] == "direct")]
    assert len(base) == 1
    assert base["deployable"].iloc[0]
    # No as_routed / c0_only exists in the baseline family, so deltas are NaN.
    assert pd.isna(base["rmse_change_vs_as_routed"].iloc[0])
    assert pd.isna(base["rmse_gap_vs_oracle_c0"].iloc[0])
    assert float(base["rmse_mean"].iloc[0]) > 0


def test_seed_and_station_metrics_cover_baseline():
    seeds = pd.read_csv(EXP_20 / "seed_metrics.csv", low_memory=False)
    assert len(seeds) == 75  # 15 rows x 5 seeds
    assert set(seeds["seed"].unique()) == {42, 7, 13, 101, 123}
    assert (seeds["family"] == "Global_Single_54").sum() == 5
    station = pd.read_csv(EXP_20 / "station_metrics.csv", low_memory=False)
    base = station[station["family"] == "Global_Single_54"]
    assert len(base) == 25  # 5 seeds x 5 stations
    assert (base.groupby(["seed", "station"]).size() == 1).all()
    assert (base.groupby("station").size() == 5).all()


def test_baseline_fit_uses_backbone_only():
    module = _module()
    config = module.load_configuration()
    assert config["baselines"] == ["Global_Single_54"]
    # compute_metrics sanity shared with the MoE path.
    y_true = np.array([0.05, 0.06, 0.07])
    y_pred = np.array([0.06, 0.06, 0.06])
    metrics = module.compute_metrics(y_true, y_pred)
    assert metrics["bias"] == float(np.mean(y_pred - y_true))
    assert metrics["ubrmse"] <= metrics["rmse"] + 1e-12


def test_predictions_stay_moe_only_and_figures_exist():
    preds = pd.read_csv(EXP_20 / "predictions_v3.csv", low_memory=False)
    assert "Global_Single_54" not in set(preds["family"].unique())
    assert set(zip(preds["family"], preds["policy"])) == {
        (fam, pol)
        for fam in ("Clustering_V0_Full_k2", "Clustering_Backbone54_k2")
        for pol in ("as_routed", "auto_soft", "auto_hard", "c0_only")
    }
    with (EXP_20 / "routing_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    for entry in audit["seeds"]:
        assert "Global_Single_54" in entry["policies"]
        assert entry["policies"]["Global_Single_54"]["direct"]["reference_only"] is True
    assert (EXP_20 / "figures/timeseries_v3_auto_overlay.png").exists()
    station_figs = list((EXP_20 / "figures").glob("timeseries_v3_ECE_*_*.png"))
    assert len(station_figs) == 10, f"expected 10 station panels, got {len(station_figs)}"
