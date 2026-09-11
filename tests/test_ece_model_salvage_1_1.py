"""Focused contracts for the variable-size derived_8.4 salvage 1.1 run."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.4-ece-model-salvage-1.1"
RUNNER_PATH = EXP_DIR / "run_model_salvage.py"
CONFIG_PATH = EXP_DIR / "config.yaml"
SELECTION_CONFIG_PATH = EXP_DIR / "feature_selection_config.yaml"
SELECTION_BY_SIZE = EXP_DIR / "feature_selection_artifacts/selected_features_by_size.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("ece_model_salvage_11_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


MODEL_SIZES = {
    "Clustering_V0_Full_k2_no_smap_fs60": 60,
    "Clustering_Backbone_k2_no_smap_fs60": 60,
    "Trained_Gating_k2_no_smap_fs60": 60,
    "Univariate_G_API_k2_no_smap_fs60": 60,
    "Clustering_Dynamic_k2_no_smap_fs60": 60,
    "Seasonal_Binary_k2_no_smap_fs60": 60,
    "Global_Single_40_no_smap_fs40": 40,
    "Global_Single_50_no_smap_fs50": 50,
    "Global_Single_60_no_smap_fs60": 60,
    "Global_Single_69_no_smap_fs69": 69,
}
MODEL_IDS = list(MODEL_SIZES)
SEEDS = [42, 7, 13]
FEATURE_SIZES = [40, 50, 60, 69]
WINDOWS = {
    ("wa_temporal", "temporal_full"),
    ("wa_temporal", "temporal_summer_2025"),
    ("ece_spatial", "spatial_ece_v3_full"),
}


def test_local_selector_is_present_and_has_no_runtime_2_0_import():
    local_files = sorted((EXP_DIR / "fs20").glob("*.py"))
    assert local_files
    for path in local_files + [EXP_DIR / "run_feature_selection.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any("derived_8.4-feature-selection-2.0" in name for name in imported)


def test_config_has_expected_variable_size_model_matrix():
    config = _yaml(CONFIG_PATH)
    assert config["seeds"] == SEEDS
    assert [spec["id"] for spec in config["models"]] == MODEL_IDS
    assert [int(spec["feature_size"]) for spec in config["models"]] == list(MODEL_SIZES.values())
    assert all(spec["two_regime"] for spec in config["models"][:6])
    assert all(not spec["two_regime"] for spec in config["models"][6:])
    assert config["router_seed"] == 42
    assert config["selected_feature_sizes"] == FEATURE_SIZES
    assert all("fs54" not in model_id for model_id in MODEL_IDS)

    selection = _yaml(SELECTION_CONFIG_PATH)["search"]
    assert selection["global_feature_min"] == 40
    assert selection["global_feature_max"] == 69
    assert selection["normalization_sizes"] == FEATURE_SIZES
    assert selection["delta_addition_counts"] == [0]


def test_local_selection_manifests_are_nested_hashed_and_non_smap():
    if not SELECTION_BY_SIZE.exists():
        pytest.skip("variable-size feature-selection stage has not produced manifests")
    payload = json.loads(SELECTION_BY_SIZE.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["feature_sizes"]
    previous = None
    for size in FEATURE_SIZES:
        record = payload["feature_sizes"][str(size)]
        features = record["features"]
        assert len(features) == size
        assert not any("smap" in feature.lower() for feature in features)
        assert record["feature_hash"] == hashlib.sha256(
            "\n".join(features).encode("utf-8")
        ).hexdigest()[:16]
        manifest = json.loads(
            (EXP_DIR / f"feature_selection_artifacts/selected_features_{size}.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["status"] == "complete"
        assert manifest["global_features"] == features
        assert manifest["selected_feature_count"] == size
        assert not any(manifest.get("cluster_additions", {}).values())
        if previous is not None:
            assert set(previous).issubset(features)
        previous = features


def test_effect_sign_conventions_and_percentage_denominator():
    runner = _load_runner()
    rows = []
    for model_id in MODEL_IDS:
        for dataset, window in WINDOWS - {("wa_temporal", "temporal_summer_2025")}:
            for seed in SEEDS:
                rows.append({
                    "model_id": model_id,
                    "seed": seed,
                    "dataset": dataset,
                    "window": window,
                    "scope": "__pooled__",
                    "reference_config_id": "original",
                    "rmse_original": 1.0,
                    "rmse_no_smap": 0.8 if dataset == "ece_spatial" else 1.2,
                    "mae_original": 1.0,
                    "mae_no_smap": 0.9,
                    "bias_original": 0.0,
                    "bias_no_smap": 0.1,
                    "ubrmse_original": 1.0,
                    "ubrmse_no_smap": 0.9,
                    "r2_original": 0.5,
                    "r2_no_smap": 0.4,
                    "pearson_original": 0.5,
                    "pearson_no_smap": 0.4,
                })
    seed_effects, summary = runner.build_old_vs_new_effects(
        pd.DataFrame(rows), expected_seeds=SEEDS, expected_model_ids=MODEL_IDS
    )
    ece = seed_effects[seed_effects["split"] == "ECE spatial"].iloc[0]
    wa = seed_effects[seed_effects["split"] == "WA temporal"].iloc[0]
    assert ece["effect_rmse"] == pytest.approx(0.2)
    assert ece["effect_rmse_pct"] == pytest.approx(20.0)
    assert wa["effect_rmse"] == pytest.approx(0.2)
    assert wa["effect_rmse_pct"] == pytest.approx(20.0)
    assert len(summary) == len(MODEL_IDS) * 2
    assert "diff_pearson_change_mean" in summary.columns


def test_feature_selection_round_comparison_is_global_only_and_interpretable():
    runner = _load_runner()
    config = runner.load_configuration()
    global_ids = [
        spec["id"] for spec in config["models"] if spec["router"] == "global"
    ]
    rows = []
    for model_id in global_ids:
        size = int(next(spec["feature_size"] for spec in config["models"] if spec["id"] == model_id))
        for seed in SEEDS:
            for dataset, window in (
                ("ece_spatial", "spatial_ece_v3_full"),
                ("wa_temporal", "temporal_full"),
            ):
                before = 1.0
                after = {40: 0.8, 50: 0.9, 60: 1.1, 69: 1.2}[size]
                if dataset == "wa_temporal":
                    after = {40: 1.1, 50: 1.2, 60: 1.3, 69: 1.4}[size]
                rows.append({
                    "model_id": model_id,
                    "old_model_id": "Global_Single_54_no_smap",
                    "seed": seed,
                    "dataset": dataset,
                    "window": window,
                    "old_rmse": before,
                    "new_rmse": after,
                    "old_pearson": 0.5,
                    "new_pearson": 0.6,
                    "change_pearson": 0.1,
                    "old_diff_pearson": 0.2,
                    "new_diff_pearson": 0.25,
                    "change_diff_pearson": 0.05,
                })
    seed_frame, summary = runner.build_feature_selection_round_comparison(
        pd.DataFrame(rows), config, expected_seeds=SEEDS
    )
    assert len(seed_frame) == 4 * 3 * 2
    assert len(summary) == 4 * 2
    assert set(seed_frame["model_id"]) == set(global_ids)
    assert not seed_frame.duplicated(["model_id", "seed", "split"]).any()
    ece_40 = summary[
        (summary["feature_size"] == 40) & (summary["split"] == "ECE spatial")
    ].iloc[0]
    wa_40 = summary[
        (summary["feature_size"] == 40) & (summary["split"] == "WA temporal")
    ].iloc[0]
    assert ece_40["rmse_effect_mean"] == pytest.approx(0.2)
    assert ece_40["rmse_effect_pct_mean"] == pytest.approx(20.0)
    assert wa_40["rmse_effect_mean"] == pytest.approx(0.1)
    assert wa_40["rmse_effect_pct_mean"] == pytest.approx(10.0)
    assert ece_40["pearson_change_mean"] == pytest.approx(0.1)
    assert ece_40["diff_pearson_change_mean"] == pytest.approx(0.05)
    interpretation = runner.interpret_feature_selection_round(summary, SEEDS)
    assert "40-feature model" in interpretation
    assert "Pooled WA degradation ranges" in interpretation


def test_completed_feature_selection_round_artifacts_are_complete():
    summary_path = EXP_DIR / "feature_selection_round_effect_summary.csv"
    seed_path = EXP_DIR / "feature_selection_round_effect_seed.csv"
    if not summary_path.exists() or not seed_path.exists():
        pytest.skip("feature-selection round comparison has not been generated")
    seed_frame = pd.read_csv(seed_path)
    summary = pd.read_csv(summary_path)
    global_ids = set(MODEL_IDS[-4:])
    assert set(seed_frame["model_id"]) == global_ids
    assert len(seed_frame) == 4 * len(SEEDS) * 2
    assert not seed_frame.duplicated(["model_id", "seed", "split"]).any()
    assert set(seed_frame["seed"]) == set(SEEDS)
    assert set(summary["feature_size"]) == set(FEATURE_SIZES)
    assert len(summary) == 4 * 2
    assert set(summary["n_seeds"]) == {len(SEEDS)}
    assert summary[["pearson_change_mean", "diff_pearson_change_mean"]].notna().all().all()


def test_router_inputs_are_non_smap_and_sized_by_model():
    runner = _load_runner()
    if not SELECTION_BY_SIZE.exists():
        pytest.skip("variable-size feature-selection stage has not produced manifests")
    config = runner.load_configuration()
    data = runner.load_data(config)
    assert sorted(data.selected_feature_sets) == FEATURE_SIZES
    assert {size: len(features) for size, features in data.selected_feature_sets.items()} == {
        size: size for size in FEATURE_SIZES
    }
    assert all(
        not any("smap" in feature.lower() for feature in features)
        for features in data.selected_feature_sets.values()
    )
    assert not any("smap" in feature.lower() for feature in data.v0_features)
    assert data.dynamic_features == ["G_API", "LST_modis"]
    for spec in config["models"]:
        assert len(runner._model_features(spec, data)) == int(spec["feature_size"])
    assert runner._router_feature_names(config["models"][1], data) == data.selected_feature_sets[60]
    assert runner._router_feature_names(config["models"][4], data) == data.dynamic_features


def test_gapi_and_seasonal_routes_are_deterministic():
    runner = _load_runner()
    if not SELECTION_BY_SIZE.exists():
        pytest.skip("variable-size feature-selection stage has not produced manifests")
    config = runner.load_configuration()
    data = runner.load_data(config)
    params = runner._base_params(config, smoke=True, device_override="cpu")
    for spec in (config["models"][3], config["models"][5]):
        router = runner.fit_router(spec, data, config, params)
        first = runner._labels(router, data.ece)
        second = runner._labels(router, data.ece)
        np.testing.assert_array_equal(first, second)


def test_completed_prediction_coverage_has_no_duplicates_and_complete_metrics():
    prediction_path = EXP_DIR / "predictions.csv"
    if not prediction_path.exists():
        pytest.skip("full variable-size model stage has not completed")
    predictions = pd.read_csv(prediction_path, low_memory=False)
    assert not predictions["model_id"].astype(str).str.contains("fs54", case=False).any()
    expected_pairs = {(model_id, seed) for model_id in MODEL_IDS for seed in SEEDS}
    if set(zip(predictions.model_id, predictions.seed)) != expected_pairs:
        pytest.skip("prediction table is not yet the complete 10-variant run")
    key_columns = ["model_id", "seed", "dataset", "window", "station_id", "date"]
    assert not predictions.duplicated(key_columns).any()
    for model_id, seed in expected_pairs:
        subset = predictions[(predictions.model_id == model_id) & (predictions.seed == seed)]
        assert set(map(tuple, subset[["dataset", "window"]].drop_duplicates().to_numpy())) == WINDOWS
    metrics = pd.read_csv(EXP_DIR / "seed_metrics.csv", low_memory=False)
    pooled = metrics[metrics.scope == "__pooled__"]
    assert len(pooled) == len(MODEL_IDS) * len(SEEDS) * len(WINDOWS)
    assert pooled["diff_pearson"].notna().all()


def test_no_smap_predictions_are_native_missing_invariant():
    path = EXP_DIR / "smap_invariance.csv"
    if not path.exists():
        pytest.skip("SMAP-invariance artifact has not been generated")
    invariance = pd.read_csv(path)
    assert set(invariance.model_id) == set(MODEL_IDS)
    assert (invariance["max_abs_prediction_difference"] == 0).all()
    assert (invariance["changed_regime_labels"] == 0).all()


def test_global_version_inputs_and_generated_charts():
    runner = _load_runner()
    if not (EXP_DIR / "predictions.csv").exists() or not SELECTION_BY_SIZE.exists():
        pytest.skip("full variable-size model stage has not completed")
    config = runner.load_configuration()
    data = runner.load_data(config)
    status = runner.global_version_source_status(data, config, SEEDS)
    if not status["ready"]:
        pytest.skip("old and 1.0 common-seed chart inputs are not complete")
    paths = runner.make_global_version_charts(data, config, EXP_DIR / "figures", SEEDS)
    assert len(paths) == 5
    chart_data = pd.read_csv(EXP_DIR / "global_version_predictions.csv")
    assert len(chart_data) == 150
    assert chart_data.station_id.nunique() == 5
    assert chart_data.groupby("station_id").date.nunique().eq(30).all()
    provenance = json.loads((EXP_DIR / "global_version_provenance.json").read_text())
    assert provenance["line_count"] == 7
    assert len(provenance["line_labels"]) == 7
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_readme_has_variable_size_section_and_valid_figure_links():
    readme = EXP_DIR / "README.md"
    prediction_path = EXP_DIR / "predictions.csv"
    if not prediction_path.exists() or not SELECTION_BY_SIZE.exists():
        pytest.skip("README is awaiting full execution")
    predictions = pd.read_csv(prediction_path, usecols=["model_id", "seed"])
    if set(zip(predictions.model_id, predictions.seed)) != {
        (model_id, seed) for model_id in MODEL_IDS for seed in SEEDS
    }:
        pytest.skip("README is awaiting the complete run")
    text = readme.read_text(encoding="utf-8")
    assert "40/50/60/69" in text
    assert "## Before vs after the new feature-selection round" in text
    assert "ECE benefit" in text
    assert "Pooled WA degradation" in text
    assert "## Global model version comparison" in text
    links = [Path(match) for match in re.findall(r"\]\((figures/[^)]+\.png)\)", text)]
    assert links
    assert all((EXP_DIR / link).exists() for link in links)
    assert len([link for link in links if "global_model_versions" in link.name]) == 5


def test_readme_tables_fences_and_provenance_are_well_formed():
    readme = EXP_DIR / "README.md"
    if not readme.exists():
        pytest.skip("README is generated after notebook execution")
    text = readme.read_text(encoding="utf-8")
    assert "REPORT_BEGIN::" not in text
    assert "REPORT_END::" not in text
    assert "INTERPRETATION_BEGIN" not in text
    assert "INTERPRETATION_END" not in text
    assert text.count("```") % 2 == 0

    blocks = []
    current = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("|") and line.rstrip().endswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    assert len(blocks) >= 8
    for block in blocks:
        assert len(block) >= 2
        pipe_count = block[0].count("|")
        assert pipe_count >= 3
        assert all(line.count("|") == pipe_count for line in block)
        separator = [cell.strip() for cell in block[1].strip("|").split("|")]
        assert all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)

    start = text.index("## Feature-selection provenance")
    end = text.index("## Input audit", start)
    section = text[start:end]
    table_lines = [
        line for line in section.splitlines()
        if line.startswith("|") and line.rstrip().endswith("|")
    ]
    assert len(table_lines) == 6
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    assert headers == [
        "status", "selected_features", "smap_features", "candidate_pool",
        "delta_additions", "selection_period", "fit_scope",
    ]
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in table_lines[2:]
    ]
    assert [int(row[1]) for row in rows] == [40, 50, 60, 69]
    assert all(row[4] == "none" for row in rows)
    manifests = re.findall(
        r"### Selected feature manifest \((\d+) features\)\n\n```text\n(.*?)\n```",
        section,
        flags=re.DOTALL,
    )
    assert [int(size) for size, _ in manifests] == [40, 50, 60, 69]
    assert all(
        len([line for line in body.splitlines() if line.strip()]) == int(size)
        and not any("smap" in line.lower() for line in body.splitlines())
        for size, body in manifests
    )
