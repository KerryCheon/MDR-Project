import importlib.util
import json
from pathlib import Path
import stat
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "notebooks/experiment/derived_8.2-feature-selection-2.2/generate_results.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "feature_selection_generate_results",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_PATH.parent))
    return module


def _metric_row(
    *,
    artifact_set="nested",
    dataset="derived_8.2",
    model="2.2_global",
    beta=0.0,
    r2=0.5,
    rmse=0.1,
    mae=0.08,
    bias=-0.02,
):
    return {
        "artifact_set": artifact_set,
        "dataset": dataset,
        "model": model,
        "beta": beta,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
    }


def _split(station_rows):
    return pd.DataFrame(
        station_rows,
        columns=["station_id", "date", "soil_moisture_5cm"],
    )


def test_selected_arm_rows_are_unique_and_finite():
    module = _module()
    table = module.build_selected_arm_error_table()
    keys = [
        "dataset",
        "artifact_arm",
        "beta",
        "baseline_model",
        "evaluation_period",
    ]
    assert not table.duplicated(keys).any()
    assert np.isfinite(table.select_dtypes(include="number")).all().all()
    assert not table["unbiased_sota_eligible"].any()


def test_source_row_validation_rejects_duplicate_and_nonfinite_rows():
    module = _module()
    duplicate = pd.DataFrame({"id": [1, 1], "score": [0.2, 0.3]})
    with pytest.raises(ValueError, match="duplicate source rows"):
        module.validate_unique_finite_rows(
            duplicate,
            source="fixture",
            unique_by=["id"],
        )

    nonfinite = pd.DataFrame({"id": [1], "score": [np.nan]})
    with pytest.raises(ValueError, match="non-finite"):
        module.validate_unique_finite_rows(nonfinite, source="fixture")

    invalid_nullable = pd.DataFrame({"id": [1], "score": ["not-a-number"]})
    with pytest.raises(ValueError, match="non-numeric"):
        module.validate_unique_finite_rows(
            invalid_nullable,
            source="fixture",
            nullable_numeric_columns=("score",),
        )


def test_historical_baseline_mapping_is_dataset_specific():
    module = _module()
    assert module.historical_baselines("derived_8.0") == ("hand_mdr_v25",)
    assert module.historical_baselines("derived_8.2") == ("V3", "2.1_c1")
    with pytest.raises(ValueError, match="no historical-baseline mapping"):
        module.historical_baselines("unknown")


def test_candidate_source_uses_ordered_tuple_with_duplicate_counts():
    module = _module()
    station_paths = [["a", "b"], ["c", "d"]]
    forward_paths = [["b", "a"], ["c", "d"]]
    assert module.map_candidate_source(["a", "b"], station_paths, forward_paths) == ("station_time")
    assert module.map_candidate_source(["b", "a"], station_paths, forward_paths) == ("forward_time")
    assert module.map_candidate_source(["c", "d"], station_paths, forward_paths) == ("both")
    with pytest.raises(ValueError, match="absent"):
        module.map_candidate_source(["missing"], station_paths, forward_paths)


def test_candidate_ranks_keep_same_count_paths_distinct():
    module = _module()
    frame = pd.DataFrame(
        [
            {
                "artifact_set": "crossed",
                "dataset": "derived_8.2",
                "beta": 0.0,
                "candidate_index": 0,
                "n_features": 50,
                "selector_combined_ucb": 0.8,
                "held_station_ucb": 0.8,
                "held_station_pooled_R2": 0.7,
                "held_station_pooled_RMSE": 0.1,
                "full_train_aggregate_R2": 0.7,
                "full_train_aggregate_RMSE": 0.1,
                "retrospective_test_R2": 0.4,
                "retrospective_test_RMSE": 0.2,
            },
            {
                "artifact_set": "crossed",
                "dataset": "derived_8.2",
                "beta": 0.0,
                "candidate_index": 1,
                "n_features": 50,
                "selector_combined_ucb": 0.7,
                "held_station_ucb": 0.7,
                "held_station_pooled_R2": 0.6,
                "held_station_pooled_RMSE": 0.11,
                "full_train_aggregate_R2": 0.6,
                "full_train_aggregate_RMSE": 0.11,
                "retrospective_test_R2": 0.8,
                "retrospective_test_RMSE": 0.08,
            },
            {
                "artifact_set": "crossed",
                "dataset": "derived_8.2",
                "beta": 0.0,
                "candidate_index": 2,
                "n_features": 40,
                "selector_combined_ucb": 0.9,
                "held_station_ucb": 0.9,
                "held_station_pooled_R2": 0.9,
                "held_station_pooled_RMSE": 0.07,
                "full_train_aggregate_R2": 0.9,
                "full_train_aggregate_RMSE": 0.07,
                "retrospective_test_R2": 0.6,
                "retrospective_test_RMSE": 0.12,
            },
        ]
    )
    ranked = module.add_candidate_ranks(frame).set_index("candidate_index")
    assert ranked.loc[1, "selector_combined_ucb_rank"] == 1
    assert ranked.loc[2, "held_station_pooled_rank"] == 1
    assert ranked.loc[2, "full_train_aggregate_rank"] == 1
    assert ranked.loc[1, "retrospective_rank"] == 1
    assert ranked.loc[1, "retrospective_ceiling"]
    assert ranked.loc[2, "held_station_pooled_winner"]
    assert ranked.loc[2, "full_train_aggregate_winner"]


def test_candidate_diagnostics_require_candidate_beta_cartesian_product():
    module = _module()
    complete = pd.DataFrame(
        [{"candidate_index": candidate, "beta": beta} for candidate in (0, 1) for beta in (0.0, 0.2)]
    )
    module.validate_candidate_diagnostic_grid(
        complete,
        candidate_count=2,
        expected_betas=(0.0, 0.2),
        source="fixture",
    )

    with pytest.raises(ValueError, match="candidate/beta grid mismatch"):
        module.validate_candidate_diagnostic_grid(
            complete.loc[complete["beta"].eq(0.0)],
            candidate_count=2,
            expected_betas=(0.0, 0.2),
            source="fixture",
        )


def test_matched_outer_metrics_preserve_saved_fold_geometry():
    module = _module()
    validation = _split(
        [
            ("a", "2021-01-01", 0.0),
            ("a", "2021-01-02", 1.0),
            ("b", "2021-01-01", 2.0),
            ("b", "2021-01-02", 3.0),
        ]
    )
    saved_folds = [
        {
            "fold_id": "outer_year_2021_stations_0",
            "validation_year": 2021,
            "held_out_stations": ["a"],
            "n_validation": 2,
        },
        {
            "fold_id": "outer_year_2021_stations_1",
            "validation_year": 2021,
            "held_out_stations": ["b"],
            "n_validation": 2,
        },
    ]
    rows = []
    for fold in saved_folds:
        for beta, rmse in ((0.0, 0.1), (0.2, 0.2)):
            rows.append(
                {
                    "fold_id": f"{fold['fold_id']}_beta_{beta:g}",
                    "validation_year": 2021,
                    "held_out_stations": fold["held_out_stations"],
                    "beta": beta,
                    "n_validation": 2,
                    "rmse": rmse,
                    "nrmse": 0.5 if fold["held_out_stations"] == ["a"] else 0.7,
                }
            )
    metrics = module.compute_matched_outer_metrics(
        rows,
        validation=validation,
        saved_folds=saved_folds,
        expected_betas=(0.0, 0.2),
        confidence_z=1.0,
    )
    assert metrics[0.0]["held_station_row_count"] == 4
    assert metrics[0.0]["held_station_pooled_RMSE"] == pytest.approx(0.1)
    assert metrics[0.0]["held_station_pooled_R2"] == pytest.approx(0.992)
    assert metrics[0.0]["held_station_mean_nrmse"] == pytest.approx(0.6)
    assert metrics[0.0]["held_station_ucb"] == pytest.approx(0.7)


def test_saved_selector_and_matched_geometry_winners_are_distinguished():
    module = _module()
    transfer = module.build_candidate_transfer_table()
    alignment = module.build_candidate_winner_alignment_table(transfer)
    assert len(alignment) == 10
    assert alignment["selector_matches_held_station_pooled"].sum() == 9
    assert alignment["held_station_ucb_matches_pooled"].sum() == 8
    assert alignment.loc[
        alignment["artifact_set"].eq("nested"),
        "selector_matches_held_station_pooled",
    ].all()
    assert not alignment.loc[
        alignment["artifact_set"].eq("nested"),
        "selector_matches_full_train_aggregate",
    ].any()


def test_station_coverage_reports_absence_return_gap_and_low_variance():
    module = _module()
    frames = {}
    for dataset in ("derived_8.0", "derived_8.2"):
        frames[(dataset, "train")] = _split(
            [
                ("returning", "2020-01-01", 0.2),
                ("returning", "2020-01-02", 0.2),
                ("regular", "2020-01-01", 0.1),
                ("development_only", "2020-01-01", 0.4),
            ]
        )
        frames[(dataset, "val")] = _split([("regular", "2021-01-01", 0.2)])
        frames[(dataset, "test")] = _split(
            [
                ("returning", "2024-01-01", 0.3),
                ("returning", "2024-01-02", 0.3),
                ("regular", "2023-01-01", 0.25),
                ("test_only", "2023-02-01", 0.5),
            ]
        )
    table = module.build_station_coverage_table(frames)
    row = table.loc[(table["dataset"] == "derived_8.2") & (table["station"] == "returning")].iloc[0]
    assert row["val_row_count"] == 0
    assert row["train_row_count"] == 2
    assert row["test_row_count"] == 2
    assert row["return_gap_days"] == 1460
    assert row["test_target_standard_deviation"] == 0.0

    development_only = table.loc[
        (table["dataset"] == "derived_8.2")
        & (table["station"] == "development_only")
    ].iloc[0]
    assert development_only["test_row_count"] == 0
    assert pd.isna(development_only["return_gap_days"])
    assert pd.isna(development_only["test_target_mean"])
    assert pd.isna(development_only["test_target_standard_deviation"])

    test_only = table.loc[
        (table["dataset"] == "derived_8.2") & (table["station"] == "test_only")
    ].iloc[0]
    assert test_only["train_row_count"] == 0
    assert test_only["val_row_count"] == 0
    assert test_only["test_row_count"] == 1
    assert test_only["last_development_observation"] == ""
    assert pd.isna(test_only["return_gap_days"])
    assert test_only["test_target_mean"] == pytest.approx(0.5)


def test_station_metrics_require_every_test_station_and_model():
    module = _module()
    coverage = pd.DataFrame(
        [
            {"dataset": dataset, "station": station, "test_row_count": 2}
            for dataset in ("derived_8.0", "derived_8.2")
            for station in ("a", "b")
        ]
    )
    models = {
        "derived_8.0": ("2.2_global", "hand_mdr_v25"),
        "derived_8.2": ("2.2_global", "V3", "2.1_c1"),
    }
    metrics = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": model,
                "beta": 0.0,
                "station_id": station,
            }
            for dataset, dataset_models in models.items()
            for model in dataset_models
            for station in ("a", "b")
        ]
    )
    selected = module.validate_station_metric_coverage(
        metrics,
        coverage=coverage,
        artifact_set="final",
        beta=0.0,
    )
    assert len(selected) == 10

    missing_station = metrics.loc[
        ~(metrics["dataset"].eq("derived_8.2") & metrics["model"].eq("V3") & metrics["station_id"].eq("b"))
    ]
    with pytest.raises(ValueError, match="station metric coverage mismatch"):
        module.validate_station_metric_coverage(
            missing_station,
            coverage=coverage,
            artifact_set="final",
            beta=0.0,
        )

    missing_model = metrics.loc[~metrics["model"].eq("hand_mdr_v25")]
    with pytest.raises(ValueError, match="station metric coverage mismatch"):
        module.validate_station_metric_coverage(
            missing_model,
            coverage=coverage,
            artifact_set="final",
            beta=0.0,
        )


def test_outer_fold_coverage_distinguishes_assigned_from_observed():
    module = _module()
    train = _split(
        [
            ("absent", "2020-01-01", 0.1),
            ("observed", "2020-01-01", 0.2),
        ]
    )
    validation = _split(
        [
            ("observed", "2021-01-01", 0.2),
            ("observed", "2021-01-02", 0.3),
        ]
    )
    saved_folds = [
        {
            "fold_id": "outer_year_2021_stations_0",
            "validation_year": 2021,
            "held_out_stations": ["absent", "observed"],
        }
    ]
    table = module.compute_outer_fold_coverage(
        dataset="derived_8.2",
        train=train,
        validation=validation,
        saved_folds=saved_folds,
        station_groups=[["absent", "observed"]],
    ).set_index("assigned_held_out_station")
    assert table.loc["absent", "listed_in_saved_fold"]
    assert table.loc["absent", "listed_but_not_scored"]
    assert not table.loc["absent", "truly_scored"]
    assert table.loc["absent", "development_station_never_in_outer_period"]
    assert table.loc["observed", "truly_scored"]


def test_outer_fold_coverage_uses_only_configured_trailing_years():
    module = _module()
    train = _split(
        [
            ("a", "2019-01-01", 0.1),
            ("b", "2019-01-01", 0.2),
        ]
    )
    validation = _split(
        [
            (station, f"{year}-01-01", value)
            for year in (2020, 2021, 2022)
            for station, value in (("a", 0.2), ("b", 0.3))
        ]
    )
    saved_folds = [
        {
            "fold_id": f"outer_year_{year}_stations_0",
            "validation_year": year,
            "held_out_stations": ["a", "b"],
        }
        for year in (2021, 2022)
    ]
    table = module.compute_outer_fold_coverage(
        dataset="fixture",
        train=train,
        validation=validation,
        saved_folds=saved_folds,
        station_groups=[["a", "b"]],
        max_validation_years=2,
    )
    assert set(table["validation_year"]) == {2021, 2022}
    assert table["listed_in_saved_fold"].all()


def test_live_split_hash_check_rejects_same_row_count_edits(tmp_path):
    module = _module()
    split_dir = tmp_path / "data/splits/fixture"
    split_dir.mkdir(parents=True)
    split_path = split_dir / "train.csv"
    split_path.write_text("station_id,target\na,1\nb,2\n", encoding="utf-8")
    expected = {"fixture": {"train": module.sha256_file(split_path)}}
    module.PROJECT_ROOT = tmp_path
    module.DATASETS = ("fixture",)
    module.SPLITS = ("train",)
    assert module.verify_live_split_hashes(expected) == expected

    split_path.write_text("station_id,target\na,1\nb,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live split hash mismatch"):
        module.verify_live_split_hashes(expected)


def test_recorded_split_hashes_must_reach_consensus():
    module = _module()
    first = "0" * 64
    second = "1" * 64
    records = {("fixture", "train"): [("selection", first), ("evaluation", first)]}
    assert module.resolve_split_hash_consensus(
        records,
        datasets=("fixture",),
        splits=("train",),
    ) == {"fixture": {"train": first}}

    records[("fixture", "train")].append(("diagnostics", second))
    with pytest.raises(ValueError, match="split hash disagreement"):
        module.resolve_split_hash_consensus(
            records,
            datasets=("fixture",),
            splits=("train",),
        )


def test_generated_summary_builders_derive_counts_and_examples():
    module = _module()
    alignment = pd.DataFrame(
        [
            {
                "artifact_set": artifact_set,
                "dataset": "derived_8.2",
                "beta": beta,
                "selector_matches_held_station_pooled": selector_match,
                "held_station_ucb_matches_pooled": ucb_match,
                "selector_matches_full_train_aggregate": False,
            }
            for artifact_set, beta, selector_match, ucb_match in (
                ("nested", 0.0, True, True),
                ("nested", 0.2, True, False),
                ("crossed", 0.0, False, True),
            )
        ]
    )
    summary = module.build_candidate_alignment_summary_table(alignment).set_index(
        "scope"
    )
    assert summary.loc["all", "comparison_count"] == 3
    assert summary.loc["all", "selector_pooled_match_count"] == 2
    assert summary.loc["all", "ucb_pooled_match_count"] == 2
    assert "crossed/derived_8.2/beta=0" in summary.loc[
        "all", "selector_pooled_mismatches"
    ]

    station_errors = pd.DataFrame(
        [
            {
                **_metric_row(r2=r2, rmse=rmse),
                "artifact_arm": "nested",
                "station": station,
            }
            for station, r2, rmse in (
                ("negative", -0.1, 0.2),
                ("largest", 0.3, 0.4),
                ("ordinary", 0.5, 0.1),
            )
        ]
    )
    selected = module.build_station_diagnostic_selection_table(station_errors)
    assert set(selected["station"]) == {"negative", "largest"}

    router = pd.DataFrame(
        {
            "router_feature": ["a", "b", "c"],
            "in_nested_global": [True, False, False],
            "in_regime_0_expert": [True, True, False],
            "in_regime_1_expert": [True, False, True],
        }
    )
    router_summary = module.build_router_feature_summary_table(router).set_index(
        "feature_set"
    )
    assert router_summary.loc["nested_global", "unavailable_router_feature_count"] == 2
    assert router_summary.loc["nested_global", "unavailable_router_features"] == "b; c"


def test_beta_effect_difference_uses_beta_02_minus_beta_00():
    module = _module()
    metrics = pd.DataFrame(
        [
            _metric_row(beta=0.0, r2=0.5, bias=-0.1),
            _metric_row(beta=0.2, r2=0.6, bias=-0.12),
        ]
    )
    effects = module.build_beta_effect_table(metrics).set_index("metric")
    assert effects.loc["R2", "difference_beta_0_2_minus_0_0"] == pytest.approx(0.1)
    assert effects.loc["Bias", "difference_beta_0_2_minus_0_0"] == pytest.approx(-0.02)


def test_moe_differences_are_against_shared_and_global():
    module = _module()
    model_values = {
        "2.2_global": 0.7,
        "2.2_clustering_frozen_k2_shared_only": 0.5,
        "2.2_clustering_dynamic_k2_shared_plus_delta": 0.6,
        "2.2_clustering_refit_k2_shared_plus_delta": 0.58,
    }
    rows = []
    for beta in (0.0, 0.2):
        for model, r2 in model_values.items():
            rows.append(_metric_row(model=model, beta=beta, r2=r2))
    table = module.build_moe_error_table(pd.DataFrame(rows))
    r2 = table.loc[(table["beta"] == 0.0) & (table["metric"] == "R2")].iloc[0]
    assert r2["frozen_delta_minus_global"] == pytest.approx(-0.1)
    assert r2["frozen_delta_minus_shared_only"] == pytest.approx(0.1)
    assert r2["refit_delta_minus_global"] == pytest.approx(-0.12)


def test_router_feature_coverage_exposes_router_only_inputs():
    module = _module()
    coverage = module.build_router_feature_coverage_table().set_index("router_feature")
    assert coverage.loc["G_API", "in_nested_global"]
    for feature in ("SMAP_sm_pm_interp_lag1", "LST_modis"):
        assert not coverage.loc[feature, "in_nested_global"]
        assert not coverage.loc[feature, "in_regime_0_expert"]
        assert not coverage.loc[feature, "in_regime_1_expert"]


def test_feature_intersection_and_jaccard():
    module = _module()
    table = module.compute_pairwise_feature_overlap(
        {"a": ["x", "y"], "b": ["y", "z"]},
        dataset="fixture",
    )
    row = table.iloc[0]
    assert row["intersection"] == 1
    assert row["union"] == 3
    assert row["jaccard"] == pytest.approx(1 / 3)


def test_retrospective_manifest_must_reject_unbiased_eligibility(tmp_path):
    module = _module()
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "retrospective_test": True,
                "unbiased_sota_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unbiased_sota_eligible=false"):
        module.validate_retrospective_manifest(path)

    path.write_text(
        json.dumps(
            {
                "retrospective_test": True,
                "unbiased_sota_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    assert module.validate_retrospective_manifest(path)["unbiased_sota_eligible"] is False


@pytest.mark.parametrize(
    "document, message",
    [
        ("no markers", "exactly one"),
        (
            "<!-- BEGIN GENERATED ERROR ANALYSIS -->\n"
            "<!-- BEGIN GENERATED ERROR ANALYSIS -->\n"
            "<!-- END GENERATED ERROR ANALYSIS -->",
            "exactly one",
        ),
        (
            "<!-- END GENERATED ERROR ANALYSIS -->\n<!-- BEGIN GENERATED ERROR ANALYSIS -->",
            "reversed",
        ),
    ],
)
def test_generated_block_rejects_missing_duplicate_or_reversed_markers(
    document,
    message,
):
    module = _module()
    with pytest.raises(ValueError, match=message):
        module.replace_generated_block(document, "new")


def test_generated_block_preserves_all_prose_outside_markers():
    module = _module()
    prefix = "prefix bytes\n" + module.BEGIN_ERROR_ANALYSIS
    suffix = module.END_ERROR_ANALYSIS + "\nsuffix bytes\n"
    document = prefix + "\nold generated text\n" + suffix
    replaced = module.replace_generated_block(document, "new generated text\n")
    assert replaced[: len(prefix)] == prefix
    assert replaced[-len(suffix) :] == suffix
    assert "old generated text" not in replaced
    assert "new generated text" in replaced


def test_report_manifest_and_completion_cover_generated_evidence():
    module = _module()
    report_dir = module.REPORT_DIR
    assert module.artifact_is_complete(report_dir, [])
    manifest = json.loads((report_dir / "report_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["markdown_outputs"]) == {"RESULTS.md", "CONTINUATION.md"}
    for name, entry in manifest["markdown_outputs"].items():
        path = module.EXP_DIR / entry["path"]
        assert path.name == name
        assert module.sha256_file(path) == entry["sha256"]
        assert stat.S_IMODE(path.stat().st_mode) & stat.S_IRGRP
    evidence_names = {Path(entry["path"]).name for entry in manifest["generated_evidence"]}
    assert evidence_names == set(module.EVIDENCE_FILES.values())
    for entry in manifest["generated_evidence"]:
        path = module.EXP_DIR / entry["path"]
        assert entry["builder"] == module.EVIDENCE_SPECS[entry["key"]][1]
        assert module.sha256_file(path) == entry["sha256"]
        assert stat.S_IMODE(path.stat().st_mode) & stat.S_IRGRP
    assert manifest["build"]["command"] == module.REPORT_COMMAND
    assert manifest["build"]["check_command"] == f"{module.REPORT_COMMAND} --check"
    assert {
        (entry["dataset"], entry["split"])
        for entry in manifest["split_inputs"]
    } == {
        (dataset, split)
        for dataset in module.DATASETS
        for split in module.SPLITS
    }
    marker = json.loads((report_dir / "completion.json").read_text(encoding="utf-8"))
    assert "../../RESULTS.md" in marker["files"]
    assert "../../CONTINUATION.md" in marker["files"]
    assert any(name.endswith("/data/splits/derived_8.2/test.csv") for name in marker["files"])
    assert stat.S_IMODE((report_dir / "report_manifest.json").stat().st_mode) & (stat.S_IRGRP)
    assert stat.S_IMODE((report_dir / "completion.json").stat().st_mode) & (stat.S_IRGRP)


def test_report_registry_names_tracked_builder_functions():
    module = _module()
    assert set(module.EVIDENCE_FILES) == set(module.EVIDENCE_SPECS)
    for filename, builder in module.EVIDENCE_SPECS.values():
        assert filename.endswith(".csv")
        assert callable(getattr(module, builder))


def test_generated_report_check_passes_without_writing():
    module = _module()
    before = {
        path: path.stat().st_mtime_ns
        for path in (
            module.RESULTS_PATH,
            module.CONTINUATION_PATH,
            module.REPORT_DIR / "report_manifest.json",
            module.REPORT_DIR / "completion.json",
        )
    }
    module.main(["--check"])
    assert {path: path.stat().st_mtime_ns for path in before} == before
