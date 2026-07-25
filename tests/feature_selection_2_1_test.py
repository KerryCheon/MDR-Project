"""Focused contract tests for derived_8.3 feature-selection 2.1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = (
    PROJECT_ROOT
    / "notebooks"
    / "experiment"
    / "derived_8.3-feature-selection-2.1"
)
sys.path.insert(0, str(EXP_DIR))

from fs21.artifacts import (  # noqa: E402
    atomic_write_json,
    completion_is_valid,
    sha256_file,
    stable_json_hash,
    write_completion,
)
from fs21.benchmark import (  # noqa: E402
    benchmark_paired_bootstrap,
    verify_historical_alignment,
)
from fs21.bootstrap import paired_hierarchical_bootstrap  # noqa: E402
from fs21.constants import EXACT_LEARNER_PARAMS, LEDGER_COLUMNS  # noqa: E402
from fs21.data import (  # noqa: E402
    _guard_development_path,
    ordered_feature_hash,
    read_yaml,
)
from fs21.decision import (  # noqa: E402
    benchmark_sota_verdict,
    choose_beta,
    choose_global_candidate,
    choose_moe,
)
from fs21.folds import (  # noqa: E402
    FoldTask,
    assert_train_before_origin,
    balanced_station_partition,
    build_inner_folds,
    build_outer_tasks,
    station_repeat_pairs,
)
from fs21.ledger import (  # noqa: E402
    assert_candidate_coverage,
    collapse_primary_repeats,
    collapse_secondary_repeats,
    validate_ledger,
)
from fs21.metrics import metric_record, primary_risk  # noqa: E402
from fs21.modeling import (  # noqa: E402
    fit_model,
    learner_parameters,
    temporal_weights,
    validate_exact_learner,
)
from fs21.moe import regime_coverage  # noqa: E402
from fs21.ranking import (  # noqa: E402
    complete_order_for_endpoint,
    consensus_features,
    progressive_bridge_sizes,
    rank_features,
)
from fs21.router import (  # noqa: E402
    _best_centroid_mapping,
    fit_router,
)
from fs21.state import RunJournal  # noqa: E402
from run_all import RUN_ALL_RELATIVE_PATH, _migrate_worker_resume  # noqa: E402


def _config() -> dict:
    return read_yaml(EXP_DIR / "global_config.yaml")


def _moe_config() -> dict:
    return read_yaml(EXP_DIR / "moe_config.yaml")


def _small_fold_config() -> dict:
    config = _config()
    config["folds"] = dict(config["folds"])
    config["folds"].update(
        {
            "minimum_train_rows": 1,
            "minimum_validation_rows": 1,
            "station_partitions": 2,
            "partition_seeds": [42, 43],
            "station_time_learner_seeds": [42, 43],
            "forward_time_learner_seeds": [42, 43],
        }
    )
    return config


def _synthetic_context(*, missing_last_station: bool = False) -> pd.DataFrame:
    rows = []
    stations = [f"station_{index}" for index in range(5)]
    for year in range(2017, 2023):
        for station_index, station in enumerate(stations):
            if missing_last_station and year == 2022 and station == stations[-1]:
                continue
            rows.append(
                {
                    "station_id": station,
                    "date": f"{year}-01-{station_index + 1:02d}",
                    "_year": year,
                    "_month": 1,
                    "_row_key": f"{station}\x1f{year}-01-{station_index + 1:02d}",
                    "soil_moisture_5cm": float(station_index + year / 10000),
                }
            )
    return pd.DataFrame(rows)


def _ledger_row(
    *,
    candidate: str,
    family: str,
    origin: int,
    station: str,
    date: str,
    truth: float,
    prediction: float,
    repeat: int,
    beta: float = 0.0,
) -> dict:
    residual = truth - prediction
    return {
        "model": "fixture",
        "candidate": candidate,
        "path_source": "fixture",
        "endpoint": 2,
        "actual_count": 2,
        "ordered_feature_hash": "a" * 64,
        "fold_family": family,
        "outer_origin": origin,
        "fold_id": f"{family}_{origin}_r{repeat}",
        "station_partition_seed": 42 + repeat if family == "station_time" else np.nan,
        "learner_seed": 42 + repeat,
        "station": station,
        "date": date,
        "year": origin,
        "month": int(date[5:7]),
        "truth": truth,
        "prediction": prediction,
        "residual": residual,
        "absolute_error": abs(residual),
        "squared_error": residual**2,
        "beta": beta,
        "model_config_id": f"{candidate}_{family}_{origin}_{repeat}",
        "router_regime": np.nan,
        "route_distance": np.nan,
    }


def _paired_ledger() -> pd.DataFrame:
    rows = []
    for candidate, error in (("V0", 0.20), ("compact", 0.10)):
        for family in ("forward_time", "station_time"):
            repeats = 3 if family == "forward_time" else 7
            for origin in (2020, 2021, 2022):
                for station_index, station in enumerate(("a", "b", "c")):
                    for day in (1, 2):
                        truth = 1.0 + station_index * 0.1
                        for repeat in range(repeats):
                            rows.append(
                                _ledger_row(
                                    candidate=candidate,
                                    family=family,
                                    origin=origin,
                                    station=station,
                                    date=f"{origin}-{day:02d}-01",
                                    truth=truth,
                                    prediction=truth - error - repeat * 0.001,
                                    repeat=repeat,
                                )
                            )
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def _eligible_summary(
    candidate: str,
    *,
    risk: float,
    count: int,
    source: str,
    form: str = "selected_k",
    stability: float = 0.5,
) -> dict:
    comparisons = {
        "combined_primary_rmse": {
            "delta": -0.01,
            "ci_upper": -0.001,
            "bootstrap_standard_error": 0.002,
        },
        "forward_time_rmse": {
            "delta": -0.005,
            "ci_upper": 0.0,
            "bootstrap_standard_error": 0.002,
        },
        "station_time_rmse": {
            "delta": -0.005,
            "ci_upper": 0.0,
            "bootstrap_standard_error": 0.002,
        },
        "p90_station_year_rmse": {
            "delta": 0.0,
            "ci_upper": 0.0,
            "bootstrap_standard_error": 0.002,
        },
        "worst_station_rmse": {
            "delta": 0.0,
            "ci_upper": 0.0,
            "bootstrap_standard_error": 0.002,
        },
        "p90_month_rmse": {
            "delta": 0.0,
            "ci_upper": 0.0,
            "bootstrap_standard_error": 0.002,
        },
    }
    return {
        "candidate": candidate,
        "combined_primary_rmse": risk,
        "actual_count": count,
        "path_source": source,
        "list_form": form,
        "selection_stability": stability,
        "coverage_matches_v0": True,
        "promotable": True,
        "bootstrap": {
            "comparisons": comparisons,
            "candidate_primary_bootstrap_standard_error": 0.02,
        },
    }


def test_train_before_origin_is_strict():
    frame = _synthetic_context()
    valid = FoldTask(
        "forward_time",
        2020,
        "valid",
        None,
        42,
        (),
        tuple(frame.index[frame["_year"] < 2020]),
        tuple(frame.index[frame["_year"] == 2020]),
    )
    assert_train_before_origin(frame, valid)
    invalid = FoldTask(
        "forward_time",
        2020,
        "invalid",
        None,
        42,
        (),
        tuple(frame.index[frame["_year"] <= 2020]),
        tuple(frame.index[frame["_year"] == 2020]),
    )
    with pytest.raises(AssertionError, match="training leakage"):
        assert_train_before_origin(frame, invalid)


def test_held_stations_are_excluded_from_outer_and_candidate_training():
    frame = _synthetic_context()
    config = _small_fold_config()
    tasks, _, _ = build_outer_tasks(frame, config)
    outer = next(
        task
        for task in tasks
        if task.family == "station_time" and task.origin == 2020
    )
    outer_training = frame.iloc[list(outer.train_index)].reset_index(drop=True)
    assert not set(outer.held_stations).intersection(outer_training["station_id"])
    inner = build_inner_folds(
        outer_training,
        config,
        family="station_time",
        partition_seed=42,
    )
    for fold in inner:
        train_stations = set(
            outer_training.iloc[list(fold.train_index)]["station_id"].astype(str)
        )
        assert not train_stations.intersection(fold.held_stations)


def test_station_partitions_are_deterministic_balanced_and_seeded():
    counts = {f"s{index}": 100 + index * 13 for index in range(9)}
    first = balanced_station_partition(counts, n_partitions=5, seed=42)
    assert first == balanced_station_partition(counts, n_partitions=5, seed=42)
    alternatives = [
        balanced_station_partition(counts, n_partitions=5, seed=seed)
        for seed in range(43, 47)
    ]
    assert any(mapping != first for mapping in alternatives)
    loads = {
        group: sum(counts[station] for station, value in first.items() if value == group)
        for group in range(5)
    }
    assert max(loads.values()) - min(loads.values()) <= max(counts.values())
    assert set(first.values()) == set(range(5))


def test_repeat_geometry_deduplicates_base_pair():
    pairs = station_repeat_pairs(_config())
    assert len(pairs) == 7
    assert pairs.count((42, 42)) == 1


def test_zero_observation_outer_fold_is_rejected():
    frame = _synthetic_context(missing_last_station=True)
    with pytest.raises(ValueError, match="zero-observation"):
        build_outer_tasks(frame, _small_fold_config())


def test_development_guard_rejects_test_csv():
    with pytest.raises(PermissionError, match="may not access"):
        _guard_development_path(Path("test.csv"))
    _guard_development_path(Path("train.csv"))


def test_development_source_audit_has_no_test_read_path():
    import preflight

    audited = preflight._audit_no_test_csv_read()
    assert "run_benchmark.py" not in audited
    assert "run_all.py" in audited


def test_exact_learner_and_native_missing_handling():
    config = _config()
    validate_exact_learner(config)
    params = learner_parameters(config, seed=43, device="cpu", smoke=False)
    assert {key: params[key] for key in EXACT_LEARNER_PARAMS} == EXACT_LEARNER_PARAMS
    assert params["random_state"] == 43
    assert "missing" not in params
    assert params["n_jobs"] == 1
    broken = _config()
    broken["learner"]["max_depth"] = 6
    with pytest.raises(ValueError, match="not exact"):
        validate_exact_learner(broken)


def test_xgboost_accepts_nan_without_global_imputation():
    config = _config()
    X = pd.DataFrame({"a": [0.0, np.nan, 1.0, 2.0], "b": [1.0, 2.0, np.nan, 4.0]})
    model = fit_model(
        X,
        np.asarray([0.0, 0.5, 1.0, 1.5]),
        train_years=np.asarray([2017, 2018, 2019, 2020]),
        beta=0.0,
        config=config,
        seed=42,
        device="cpu",
        smoke=True,
    )
    assert np.isfinite(model.predict(X)).all()


def test_beta_weights_are_normalized_and_zero_has_no_weights():
    years = np.asarray([2017, 2018, 2019, 2020])
    assert temporal_weights(years, 0.0) is None
    weights = temporal_weights(years, 0.2)
    assert np.mean(weights) == pytest.approx(1.0)
    assert np.all(np.diff(weights) > 0)
    expected = np.exp(0.2 * (years - 2020))
    assert np.allclose(weights, expected / expected.mean())


def test_router_uses_fit_frame_means_and_target_free_alignment():
    columns = [f"f{index}" for index in range(50)]
    frame = pd.DataFrame(
        {
            column: np.linspace(index, index + 1, 20)
            for index, column in enumerate(columns)
        }
    )
    frame.loc[0, "f0"] = np.nan
    router = fit_router(frame, _moe_config()["router"], columns)
    assert router.means[0] == pytest.approx(frame["f0"].mean())
    validation = frame.iloc[:2].copy()
    validation.loc[:, "f0"] = np.nan
    transformed = router.transform(validation)
    expected = (router.means[0] - router.scaler_mean[0]) / router.scaler_scale[0]
    assert np.allclose(transformed[:, 0], expected)
    mapping = _best_centroid_mapping(
        np.asarray([[10.0, 10.0], [0.0, 0.0]]),
        np.asarray([[0.0, 0.0], [10.0, 10.0]]),
    )
    assert mapping == {0: 1, 1: 0}


def test_ledger_residual_sign_and_raw_alignment():
    row = _ledger_row(
        candidate="V0",
        family="forward_time",
        origin=2020,
        station="a",
        date="2020-01-01",
        truth=0.5,
        prediction=0.2,
        repeat=0,
    )
    ledger = validate_ledger(pd.DataFrame([row], columns=LEDGER_COLUMNS))
    assert ledger.iloc[0]["residual"] == pytest.approx(0.3)
    broken = ledger.copy()
    broken.loc[0, "residual"] = -0.3
    with pytest.raises(ValueError, match="truth - prediction"):
        validate_ledger(broken)


def test_repeat_collapses_do_not_inflate_effective_rows():
    rows = [
        _ledger_row(
            candidate="V0",
            family="forward_time",
            origin=2020,
            station="a",
            date="2020-01-01",
            truth=1.0,
            prediction=prediction,
            repeat=index,
        )
        for index, prediction in enumerate((0.0, 0.5, 1.0))
    ]
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    primary = collapse_primary_repeats(ledger)
    secondary = collapse_secondary_repeats(ledger)
    assert len(primary) == len(secondary) == 1
    assert primary.iloc[0]["mean_squared_error"] == pytest.approx((1.0 + 0.25) / 3)
    assert secondary.iloc[0]["prediction"] == pytest.approx(0.5)
    assert primary.iloc[0]["repeat_count"] == 3
    assert secondary.iloc[0]["repeat_count"] == 3


def test_beta_arms_remain_separate_repeat_groups():
    rows = []
    for beta in (0.0, 0.2):
        rows.append(
            _ledger_row(
                candidate="V0",
                family="forward_time",
                origin=2020,
                station="a",
                date="2020-01-01",
                truth=1.0,
                prediction=0.8,
                repeat=0,
                beta=beta,
            )
        )
    collapsed = collapse_primary_repeats(pd.DataFrame(rows, columns=LEDGER_COLUMNS))
    assert len(collapsed) == 2
    assert set(collapsed["beta"]) == {0.0, 0.2}


def test_candidate_coverage_checks_rows_and_repeats():
    ledger = _paired_ledger()
    assert_candidate_coverage(ledger, "compact", reference="V0")
    broken = ledger.drop(ledger.index[-1])
    with pytest.raises(ValueError, match="coverage"):
        assert_candidate_coverage(broken, "compact", reference="V0")


def test_ordered_feature_hash_is_stable_and_order_sensitive():
    assert ordered_feature_hash(["a", "b"]) == ordered_feature_hash(["a", "b"])
    assert ordered_feature_hash(["a", "b"]) != ordered_feature_hash(["b", "a"])


def test_station_year_macro_primary_risk():
    ledger = _paired_ledger()
    risk = primary_risk(ledger, "compact", beta=0.0)
    assert risk["combined_primary_rmse"] > 0
    expected_forward = risk["station_year_blocks"].loc[
        risk["station_year_blocks"]["fold_family"] == "forward_time",
        "station_year_rmse",
    ].mean()
    assert risk["forward_time_rmse"] == pytest.approx(expected_forward)
    assert risk["combined_primary_rmse"] == pytest.approx(
        0.5 * risk["forward_time_rmse"] + 0.5 * risk["station_time_rmse"]
    )


def test_hierarchical_bootstrap_is_deterministic():
    ledger = _paired_ledger()
    first = paired_hierarchical_bootstrap(
        ledger, "compact", "V0", replicates=50, seed=42
    )
    second = paired_hierarchical_bootstrap(
        ledger, "compact", "V0", replicates=50, seed=42
    )
    assert first == second
    assert first["comparisons"]["combined_primary_rmse"]["ci_upper"] < 0


def test_hierarchical_bootstrap_compares_distinct_beta_arms():
    ledger = _paired_ledger()
    compact = ledger.loc[ledger["candidate"] == "compact"].copy()
    compact["candidate"] = "compact_beta_0_2"
    compact["beta"] = 0.2
    ledger = pd.concat([ledger, compact], ignore_index=True)
    result = paired_hierarchical_bootstrap(
        ledger,
        "compact_beta_0_2",
        "compact",
        candidate_beta=0.2,
        reference_beta=0.0,
        replicates=20,
        seed=42,
    )
    assert result["candidate_beta"] == 0.2
    assert result["reference_beta"] == 0.0
    assert result["beta"] is None
    assert result["comparisons"]["combined_primary_rmse"]["delta"] == pytest.approx(0.0)


def test_progressive_bridge_generation_matches_protocol():
    rows = progressive_bridge_sizes(496, [150, 125, 100, 80, 65, 50, 40])
    sizes = [row["size"] for row in rows]
    assert sizes[:3] == [346, 196, 150]
    assert [
        row["size"] for row in rows if row["endpoint"]
    ] == [150, 125, 100, 80, 65, 50, 40]


def test_complete_order_reconstructs_eliminated_cohorts():
    path = {
        "start_size": 6,
        "endpoints": {"2": ["a", "b"]},
        "steps": [
            {
                "size": 4,
                "features": ["a", "b", "c", "d"],
                "ranking": [
                    {"feature": feature}
                    for feature in ["a", "b", "c", "d", "e", "f"]
                ],
            },
            {
                "size": 2,
                "features": ["a", "b"],
                "ranking": [
                    {"feature": feature} for feature in ["a", "b", "c", "d"]
                ],
            },
        ],
    }
    assert complete_order_for_endpoint(path, 2) == ["a", "b", "c", "d", "e", "f"]


class _SumModel:
    def predict(self, X):
        return np.nan_to_num(X.to_numpy(dtype=float), nan=0.0).sum(axis=1)


def test_ranking_is_name_agnostic_and_ties_use_original_position(monkeypatch):
    import fs21.ranking as ranking_module

    monkeypatch.setattr(ranking_module, "fit_model", lambda *args, **kwargs: _SumModel())
    frame = pd.DataFrame(
        {
            "station_id": ["a", "b", "a", "b"],
            "_year": [2017, 2017, 2019, 2019],
            "soil_moisture_5cm": [0.0, 0.0, 2.0, 2.0],
            "x": [0.0, 0.0, 1.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    fold = FoldTask(
        "forward_time", 2019, "f", None, 42, (), (0, 1), (2, 3)
    )
    ordered, _ = rank_features(
        frame,
        ["x", "y"],
        [fold],
        config=_config(),
        learner_seed=42,
        device="cpu",
        permutation_repeats=1,
        smoke=True,
    )
    renamed = frame.rename(columns={"x": "alpha", "y": "omega"})
    renamed_order, _ = rank_features(
        renamed,
        ["alpha", "omega"],
        [fold],
        config=_config(),
        learner_seed=42,
        device="cpu",
        permutation_repeats=1,
        smoke=True,
    )
    assert ordered == ["x", "y"]
    assert renamed_order == ["alpha", "omega"]


def test_one_standard_error_prefers_smallest_count():
    best = _eligible_summary(
        "selected_k__forward_time__k100",
        risk=0.10,
        count=100,
        source="forward_time",
    )
    smaller = _eligible_summary(
        "selected_k__station_time__k50",
        risk=0.115,
        count=50,
        source="station_time",
    )
    decision = choose_global_candidate([best, smaller])
    assert decision["winner"] == smaller["candidate"]


def test_path_source_tie_break_is_deterministic():
    forward = _eligible_summary(
        "selected_k__forward_time__k50",
        risk=0.10,
        count=50,
        source="forward_time",
    )
    station = _eligible_summary(
        "selected_k__station_time__k50",
        risk=0.10,
        count=50,
        source="station_time",
    )
    assert choose_global_candidate([forward, station])["winner"] == station["candidate"]


def test_global_decision_automatically_falls_back_to_v0():
    failed = _eligible_summary(
        "selected_k__station_time__k50",
        risk=0.1,
        count=50,
        source="station_time",
    )
    failed["bootstrap"]["comparisons"]["combined_primary_rmse"]["ci_upper"] = 0.001
    decision = choose_global_candidate([failed])
    assert decision["global_gate_passed"] is False
    assert decision["winner"] == "V0"
    assert decision["best_failed_candidate"] == failed["candidate"]


def test_consensus_orders_frequency_rank_and_position():
    universe = ["a", "b", "c", "d"]
    rankings = [
        {"year": 2020, "selected": ["a", "b"], "ordered": ["a", "b", "c", "d"]},
        {"year": 2021, "selected": ["b", "c"], "ordered": ["b", "c", "a", "d"]},
        {"year": 2022, "selected": ["b", "a"], "ordered": ["b", "a", "d", "c"]},
    ]
    selected, table = consensus_features(rankings, count=2, universe=universe)
    assert selected == ["b", "a"]
    assert table.iloc[0]["selection_frequency"] == 3


def test_zero_target_variance_r2_is_nan_with_reason():
    metrics = metric_record([1.0, 1.0], [1.0, 0.9])
    assert np.isnan(metrics["R2"])
    assert metrics["R2_reason"] == "zero_target_variance"


def test_regime_coverage_reports_origins_stations_and_dispersion():
    ledger = _paired_ledger().copy()
    ledger["router_regime"] = np.where(ledger["station"] == "a", 0, 1)
    coverage = regime_coverage(ledger)
    assert set(coverage["router_regime"]) == {0, 1}
    assert (coverage["row_count"] > 0).all()
    assert (coverage["station_count"] >= 1).all()


def test_moe_promotion_is_prohibited_after_failed_global_gate():
    decision = choose_moe(
        global_gate_passed=False,
        single_global_id="V0",
        moe_summary=_eligible_summary(
            "moe", risk=0.08, count=50, source="station_time"
        ),
    )
    assert decision["moe_promoted"] is False
    assert decision["benchmark_sota_eligible"] is False
    assert decision["reason"] == "global_gate_failed"


def test_beta_selection_requires_significance_and_no_family_regression():
    summary = _eligible_summary(
        "beta", risk=0.1, count=50, source="station_time"
    )["bootstrap"]
    assert choose_beta(summary)["selected_beta"] == 0.2
    summary["comparisons"]["forward_time_rmse"]["delta"] = 0.001
    assert choose_beta(summary)["selected_beta"] == 0.0


def test_benchmark_sota_margin_and_unbiased_fields_are_independent():
    verdict = benchmark_sota_verdict(
        global_gate_passed=True,
        challenger_development_eligible=True,
        r2=0.6648718115185884,
        rmse=0.060,
        historical_r2=0.6618718115185884,
        historical_rmse=0.06042772002760553,
        r2_margin=0.003,
        paired_primary_ci_upper=-0.0001,
        worst_station_delta=0.0,
        worst_station_standard_error=0.001,
        p90_month_delta=0.0,
        p90_month_standard_error=0.001,
        alignment_verified=True,
    )
    assert verdict["benchmark_sota_eligible"] is True
    assert verdict["unbiased_sota_eligible"] is False
    assert verdict["unbiased_generalization_claim_eligible"] is False
    below_margin = dict(verdict)
    below = benchmark_sota_verdict(
        global_gate_passed=True,
        challenger_development_eligible=True,
        r2=0.6648,
        rmse=0.060,
        historical_r2=0.6618718115185884,
        historical_rmse=0.06042772002760553,
        r2_margin=0.003,
        paired_primary_ci_upper=-0.0001,
        worst_station_delta=0.0,
        worst_station_standard_error=0.001,
        p90_month_delta=0.0,
        p90_month_standard_error=0.001,
        alignment_verified=True,
    )
    assert below["benchmark_sota_eligible"] is False
    assert below["checks"]["r2_margin_passed"] is False


def test_benchmark_bootstrap_is_paired_and_deterministic():
    rows = []
    for candidate, error in (("historical", 0.2), ("challenger", 0.1)):
        for year in (2023, 2024, 2025):
            for station in ("a", "b"):
                for day in (1, 2):
                    row = _ledger_row(
                        candidate=candidate,
                        family="benchmark",
                        origin=year,
                        station=station,
                        date=f"{year}-01-{day:02d}",
                        truth=1.0,
                        prediction=1.0 - error,
                        repeat=0,
                    )
                    row["fold_id"] = "benchmark"
                    row["model_config_id"] = candidate
                    rows.append(row)
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    first = benchmark_paired_bootstrap(
        ledger, "challenger", "historical", replicates=30, seed=42
    )
    second = benchmark_paired_bootstrap(
        ledger, "challenger", "historical", replicates=30, seed=42
    )
    assert first == second
    assert first["comparisons"]["station_year_macro_rmse"]["ci_upper"] < 0


def test_historical_prediction_label_alignment(monkeypatch, tmp_path):
    labels_path = tmp_path / "labels.npy"
    predictions_path = tmp_path / "predictions.npy"
    metadata_path = tmp_path / "metadata.json"
    metrics_path = tmp_path / "metrics.csv"
    target = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    np.save(labels_path, target)
    np.save(predictions_path, np.asarray([0.11, 0.19, 0.31]))
    metadata_path.write_text(
        json.dumps(
            {
                "config_id": 16,
                "config_crc32": "5fa48398",
                "arm": "global_v0",
                "strat": "Clustering_V0_Full_k2",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "Model ID": 16,
                "Arm": "global_v0",
                "Strategy": "Clustering_V0_Full_k2",
                "R2": 0.66,
                "RMSE": 0.06,
            }
        ]
    ).to_csv(metrics_path, index=False)
    paths = {
        "labels": labels_path,
        "predictions": predictions_path,
        "metadata": metadata_path,
        "metrics": metrics_path,
    }
    import fs21.benchmark as benchmark_module

    monkeypatch.setattr(
        benchmark_module,
        "resolve_repo_path",
        lambda value: paths[value],
    )
    benchmark = pd.DataFrame(
        {
            "soil_moisture_5cm": target.astype(float),
            "_row_key": ["a", "b", "c"],
        }
    )
    registry = {
        "historical_best": {
            "labels_source": "labels",
            "predictions_source": "predictions",
            "metadata_source": "metadata",
            "metrics_source": "metrics",
            "expected_label_count": 3,
            "model_id": 16,
            "r2": 0.66,
            "rmse": 0.06,
            "labels_sha256": sha256_file(labels_path),
            "predictions_sha256": sha256_file(predictions_path),
            "metadata_sha256": sha256_file(metadata_path),
            "metrics_sha256": sha256_file(metrics_path),
        }
    }
    predictions, result = verify_historical_alignment(benchmark, registry)
    assert len(predictions) == 3
    assert result["alignment_verified"] is True
    benchmark.loc[1, "soil_moisture_5cm"] = 0.25
    with pytest.raises(RuntimeError, match="could not be proven"):
        verify_historical_alignment(benchmark, registry)


def test_benchmark_requires_explicit_confirmation(monkeypatch):
    import run_benchmark

    called = []
    monkeypatch.setattr(run_benchmark, "run_benchmark", lambda **kwargs: called.append(kwargs))
    with pytest.raises(SystemExit):
        run_benchmark.main([])
    assert called == []


def test_completion_marker_detects_corruption(tmp_path):
    output = tmp_path / "result.json"
    atomic_write_json(output, {"value": 1})
    write_completion(tmp_path, ["result.json"])
    assert completion_is_valid(tmp_path, ["result.json"])
    atomic_write_json(output, {"value": 2})
    assert not completion_is_valid(tmp_path, ["result.json"])


def test_interrupted_run_journal_resumes_only_same_fingerprint(tmp_path):
    fingerprint = {"fingerprint_sha256": "abc"}
    environment = {"device": "cpu", "workers": 1}
    path = tmp_path / "run_state.json"
    journal = RunJournal(
        path,
        command=["run"],
        fingerprint=fingerprint,
        environment=environment,
    )
    journal.stage_started("stage", ["stage"])
    journal.stage_failed("stage", RuntimeError("interrupted"))
    resumed = RunJournal(
        path,
        command=["run", "resume"],
        fingerprint=fingerprint,
        environment=environment,
    )
    assert resumed.stage_status("stage") == "failed"
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        RunJournal(
            path,
            command=["run"],
            fingerprint={"fingerprint_sha256": "changed"},
            environment=environment,
        )


def test_worker_only_resume_migration_preserves_valid_checkpoints(tmp_path):
    artifact_root = tmp_path / "development"
    stage = artifact_root / "stages" / "01_preflight"
    atomic_write_json(stage / "preflight.json", {"status": "ok"})
    write_completion(stage, ["preflight.json"])
    rank_unit = (
        artifact_root
        / "stages"
        / "05_robust_candidate_generation"
        / "rank_units"
        / "unit_a"
    )
    atomic_write_json(rank_unit / "path.json", {"features": ["a"]})
    write_completion(rank_unit, ["path.json"])
    common = {
        "git_revision": "abc",
        "split_hashes": {"train.csv": "same"},
        "device": "cuda",
        "smoke": False,
    }
    old_fingerprint = {
        **common,
        "workers": 4,
        "runtime_inputs": {
            RUN_ALL_RELATIVE_PATH: "old-runner",
            "global_config.yaml": "same-config",
        },
        "fingerprint_sha256": "old-fingerprint",
    }
    new_fingerprint = {
        **common,
        "workers": 6,
        "runtime_inputs": {
            RUN_ALL_RELATIVE_PATH: "new-runner",
            "global_config.yaml": "same-config",
        },
        "fingerprint_sha256": "new-fingerprint",
    }
    state_path = artifact_root / "run_state.json"
    atomic_write_json(
        state_path,
        {
            "fingerprint": old_fingerprint,
            "environment": {"device": "cuda", "workers": 4},
            "status": "failed",
            "failure": {"type": "KeyboardInterrupt", "message": ""},
            "completed": None,
            "stages": [
                {"name": "01_preflight", "status": "complete"},
                {
                    "name": "05_robust_candidate_generation",
                    "status": "failed",
                },
            ],
        },
    )
    record = _migrate_worker_resume(
        state_path=state_path,
        artifact_root=artifact_root,
        new_fingerprint=new_fingerprint,
        new_environment={"device": "cuda", "workers": 6},
        configured_workers=4,
        requested_workers=6,
    )
    migrated = json.loads(state_path.read_text(encoding="utf-8"))
    assert record["scope"] == "execution_concurrency_only"
    assert record["reused_completed_stage5_rank_units"] == 1
    assert migrated["fingerprint"] == new_fingerprint
    assert migrated["environment"]["workers"] == 6
    assert migrated["status"] == "running"
    assert (artifact_root / "worker_resume_override.json").is_file()
