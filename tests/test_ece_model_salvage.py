"""Tests for the no-SMAP model-salvage experiment."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.4-ece-model-salvage-1.0"
RUNNER_PATH = EXP_DIR / "run_model_salvage.py"


def _runner():
    spec = importlib.util.spec_from_file_location("ece_model_salvage_test_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_has_all_requested_model_families_and_seeds():
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert config["seeds"] == [42, 7, 13, 101, 123]
    assert len(config["models"]) == 7
    assert {model["id"] for model in config["models"]} == {
        "Clustering_V0_Full_k2_no_smap",
        "Clustering_Backbone54_k2_no_smap",
        "Trained_Gating_k2_no_smap",
        "Univariate_G_API_k2_no_smap",
        "Clustering_Dynamic_k2_no_smap",
        "Seasonal_Binary_k2_no_smap",
        "Global_Single_54_no_smap",
    }


def test_all_models_have_one_original_reference_mapping():
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text(encoding="utf-8"))
    mappings = {model["id"]: model["parent_reference"] for model in config["models"]}
    assert len(mappings) == 7
    assert all(mappings.values())
    assert len(set(mappings.values())) == 7


def test_drop_smap_removes_every_case_variant():
    runner = _runner()
    assert runner.drop_smap(["G_API", "SMAP_x", "smap_lag", "LST_modis"]) == ["G_API", "LST_modis"]


def test_metric_includes_level_and_first_difference_trend_scores():
    runner = _runner()
    frame = pd.DataFrame({
        "station_id": ["a"] * 4,
        "date": pd.to_datetime(["2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23"]),
        "target": [0.10, 0.09, 0.075, 0.07],
        "prediction": [0.11, 0.10, 0.085, 0.08],
    })
    metrics = runner.compute_metrics(frame)
    assert metrics["rmse"] > 0
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["diff_pearson"] == pytest.approx(1.0)
    assert metrics["prediction_std"] == pytest.approx(metrics["target_std"])


def test_requested_router_feature_sets_are_smap_free():
    runner = _runner()
    config = runner.load_configuration()
    dynamic = runner.drop_smap(config["features"]["dynamic_parent"])
    assert dynamic == ["G_API", "LST_modis"]
    assert all("smap" not in feature.lower() for feature in dynamic)


def test_canonical_feature_counts_match_strict_smap_removal():
    data_dir = PROJECT_ROOT / "data/splits/derived_8.4"
    if not (data_dir / "train.csv").exists():
        pytest.skip("Repository data are not hydrated.")
    runner = _runner()
    data = runner.load_data(runner.load_configuration())
    assert len(data.v0_parent) == 50
    assert len(data.v0_features) == 42
    assert len(data.backbone_parent) == 54
    assert len(data.backbone_features) == 42


def test_special_gates_are_deterministic_and_two_regime():
    runner = _runner()
    from eval_formal.routers import SeasonalBinaryRouter, UnivariateGAPIRouter

    frame = pd.DataFrame({"G_API": [0.1, 0.9], "month": [7, 12]})
    gapi = UnivariateGAPIRouter("G_API").fit(frame)
    seasonal = SeasonalBinaryRouter().fit(frame)
    assert np.array_equal(gapi.predict(frame), np.array([0, 1]))
    assert np.array_equal(seasonal.predict(frame), np.array([0, 1]))


def _synthetic_reference_comparison() -> pd.DataFrame:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text(encoding="utf-8"))
    rows = []
    for model in config["models"]:
        for seed in config["seeds"]:
            for dataset, window, new_rmse in (
                ("ece_spatial", "spatial_ece_v3_full", 0.8),
                ("wa_temporal", "temporal_full", 1.2),
            ):
                row = {
                    "model_id": model["id"],
                    "seed": seed,
                    "dataset": dataset,
                    "window": window,
                    "scope": "__pooled__",
                    "reference_config_id": model["parent_reference"],
                    "reference_only": True,
                    "rmse_original": 1.0,
                    "rmse_no_smap": new_rmse,
                    "diff_pearson": 0.2,
                    "diff_pearson_original": 0.1,
                }
                for metric in ["mae", "bias", "ubrmse", "r2", "pearson"]:
                    row[f"{metric}_original"] = 0.5
                    row[f"{metric}_no_smap"] = 0.6
                rows.append(row)
    return pd.DataFrame(rows)


def test_old_vs_new_effects_have_complete_paired_seed_coverage():
    runner = _runner()
    config = runner.load_configuration()
    comparison = _synthetic_reference_comparison()
    seed_effects, summary = runner.build_old_vs_new_effects(
        comparison,
        expected_seeds=config["seeds"],
        expected_model_ids=[model["id"] for model in config["models"]],
    )
    assert len(seed_effects) == 7 * 5 * 2
    assert len(summary) == 7 * 2
    assert not seed_effects.duplicated(["model_id", "seed", "split"]).any()
    assert set(seed_effects.groupby(["model_id", "split"])["seed"].nunique()) == {5}


def test_old_vs_new_effect_signs_percentages_and_trend_changes():
    runner = _runner()
    comparison = _synthetic_reference_comparison()
    seed_effects, summary = runner.build_old_vs_new_effects(
        comparison,
        expected_seeds=[42, 7, 13, 101, 123],
    )
    ece = seed_effects[seed_effects["split"] == "ECE spatial"].iloc[0]
    wa = seed_effects[seed_effects["split"] == "WA temporal"].iloc[0]
    assert ece["rmse_change_no_smap_minus_original"] == pytest.approx(-0.2)
    assert ece["effect_rmse"] == pytest.approx(0.2)
    assert ece["rmse_change_pct_no_smap_vs_original"] == pytest.approx(-20.0)
    assert ece["effect_rmse_pct"] == pytest.approx(20.0)
    assert ece["improved"] and not ece["worsened"]
    assert wa["rmse_change_no_smap_minus_original"] == pytest.approx(0.2)
    assert wa["effect_rmse"] == pytest.approx(0.2)
    assert wa["rmse_change_pct_no_smap_vs_original"] == pytest.approx(20.0)
    assert wa["effect_rmse_pct"] == pytest.approx(20.0)
    assert wa["worsened"] and not wa["improved"]
    assert set(summary.loc[summary["split"] == "ECE spatial", "improved_seeds"]) == {5}
    assert set(summary.loc[summary["split"] == "ECE spatial", "worsened_seeds"]) == {0}
    assert set(summary.loc[summary["split"] == "WA temporal", "improved_seeds"]) == {0}
    assert set(summary.loc[summary["split"] == "WA temporal", "worsened_seeds"]) == {5}
    assert set(summary["pearson_change_mean"].round(8)) == {0.1}
    assert set(summary["diff_pearson_change_mean"].round(8)) == {0.1}


def test_old_vs_new_effects_reject_duplicate_pairs():
    runner = _runner()
    comparison = _synthetic_reference_comparison()
    duplicate = pd.concat([comparison, comparison.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate paired reference rows"):
        runner.build_old_vs_new_effects(duplicate)


def test_generated_outputs_have_complete_coverage_when_present():
    summary_path = EXP_DIR / "summary.csv"
    seed_metrics_path = EXP_DIR / "seed_metrics.csv"
    effect_seed_path = EXP_DIR / "old_vs_new_effect_seed.csv"
    effect_summary_path = EXP_DIR / "old_vs_new_effect_summary.csv"
    if not summary_path.exists() or not seed_metrics_path.exists() or not effect_seed_path.exists() or not effect_summary_path.exists():
        pytest.skip("Full experiment outputs are generated by the GPU job.")
    summary = pd.read_csv(summary_path, low_memory=False)
    seed_metrics = pd.read_csv(seed_metrics_path, low_memory=False)
    effect_seed = pd.read_csv(effect_seed_path, low_memory=False)
    effect_summary = pd.read_csv(effect_summary_path, low_memory=False)
    expected_models = {
        "Clustering_V0_Full_k2_no_smap",
        "Clustering_Backbone54_k2_no_smap",
        "Trained_Gating_k2_no_smap",
        "Univariate_G_API_k2_no_smap",
        "Clustering_Dynamic_k2_no_smap",
        "Seasonal_Binary_k2_no_smap",
        "Global_Single_54_no_smap",
    }
    if set(seed_metrics["seed"]) != {42, 7, 13, 101, 123}:
        pytest.skip("Only smoke outputs are present; full five-seed job has not completed.")
    assert set(seed_metrics["model_id"]) == expected_models
    pooled = summary[summary["scope"] == "__pooled__"]
    assert set(pooled["model_id"]) == expected_models
    assert set(pooled["window"]) == {"temporal_full", "temporal_summer_2025", "spatial_ece_v3_full"}
    assert len(effect_seed) == 7 * 5 * 2
    assert len(effect_summary) == 7 * 2
    assert not effect_seed.duplicated(["model_id", "seed", "split"]).any()
    assert set(effect_seed.groupby(["model_id", "split"])["seed"].nunique()) == {5}
    assert set(effect_summary["model_id"]) == expected_models
    assert set(effect_summary["split"]) == {"ECE spatial", "WA temporal"}
    assert effect_summary["pearson_change_mean"].notna().all()


def test_smap_invariance_output_is_zero_when_present():
    path = EXP_DIR / "smap_invariance.csv"
    if not path.exists():
        pytest.skip("Full experiment outputs are generated by the GPU job.")
    invariance = pd.read_csv(path)
    assert len(invariance) == 7
    assert (invariance["max_abs_prediction_difference"] <= 1e-12).all()
    assert (invariance["changed_regime_labels"] == 0).all()


def test_readme_figure_links_include_figure_directory_when_present():
    readme_path = EXP_DIR / "README.md"
    if not readme_path.exists():
        pytest.skip("Report README is generated after notebook execution.")
    readme = readme_path.read_text(encoding="utf-8")
    image_targets = re.findall(r"!\[[^]]+\]\(([^)]+)\)", readme)
    figure_targets = [target for target in image_targets if target.endswith(".png")]
    assert figure_targets
    assert all(target.startswith("figures/") for target in figure_targets)
    assert all((EXP_DIR / target).exists() for target in figure_targets)


def test_readme_contains_old_vs_new_effect_section_when_report_is_present():
    readme_path = EXP_DIR / "README.md"
    figure_path = EXP_DIR / "figures/old_vs_new_rmse_effect.png"
    if not readme_path.exists() or not figure_path.exists():
        pytest.skip("The comparison report is generated after notebook execution.")
    readme = readme_path.read_text(encoding="utf-8")
    assert "## Effect of Removing SMAP: ECE Benefit vs WA Degradation" in readme
    assert "(figures/old_vs_new_rmse_effect.png)" in readme
    assert "effect_rmse_mean" in readme
