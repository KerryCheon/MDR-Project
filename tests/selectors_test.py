# Jakob Balkovec
# selectors_test.py
# Unit tests for feature selection and selectors

import pytest
import numpy as np
import pandas as pd

from Modeling.Src.soilmoist_fl.Selectors.correlation import select_correlation
from Modeling.Src.soilmoist_fl.Selectors.rf_importance import select_rf_importance
from Modeling.Src.soilmoist_fl.Selectors.mi import select_mi
from Modeling.Src.soilmoist_fl.Selectors.elasticnet import select_elasticnet
from Modeling.Src.soilmoist_fl.Selectors.xgb_importance import select_xgb_importance
from Modeling.Src.soilmoist_fl.Selectors.family_coverage import (
    enforce_min_family_coverage,
    infer_coverage_family,
)
from Modeling.Src.soilmoist_fl.Selectors.stability import stability_bootstrap
from Modeling.Src.soilmoist_fl.Selectors import grouped_oof as grouped_oof_module
from Modeling.Src.soilmoist_fl.Selectors.grouped_oof import (
    build_forward_time_folds,
    build_station_time_folds,
    evaluate_forward_station_time_candidates,
    select_grouped_oof,
)
from Modeling.Src.soilmoist_fl import select_features


def test_select_correlation():
    # Construct a dataset where col1 and col2 are highly correlated
    # col1 has higher correlation with y than col2
    np.random.seed(42)
    x1 = np.random.randn(100)
    x2 = x1 + np.random.randn(100) * 0.01  # extremely high correlation
    x3 = np.random.randn(100)  # independent
    
    # y correlates strongly with x1
    y = x1 * 2.0 + np.random.randn(100) * 0.1
    
    df = pd.DataFrame({
        "col1": x1,
        "col2": x2,
        "col3": x3
    })
    
    out = select_correlation(df, y, threshold=0.9)
    selected = out["selected"]
    dropped = out["dropped"]
    
    # We expect one of col1 or col2 to be dropped, and col3 to be kept
    assert "col3" in selected
    assert ("col1" in selected and "col2" in dropped) or ("col2" in selected and "col1" in dropped)
    # Since col1 is more correlated with y, it should be the one kept
    assert "col1" in selected
    assert "col2" in dropped


def test_select_rf_importance():
    np.random.seed(42)
    X = pd.DataFrame({
        "col1": np.random.randn(100),
        "col2": np.random.randn(100),
        "col3": np.random.randn(100),
        "empty_col": [np.nan] * 100  # all NaN
    })
    y = X["col1"] * 2.0 + np.random.randn(100) * 0.1
    
    out = select_rf_importance(X, y, k=2, n_estimators=10, random_state=42)
    selected = out["selected"]
    ranked = out["ranked"]
    scores = out["scores"]
    
    assert len(selected) == 2
    assert "empty_col" in ranked
    # empty_col should get 0.0 score and be ranked at the bottom
    assert scores["empty_col"] == 0.0
    assert ranked[-1] == "empty_col"
    # col1 should be the top-ranked feature
    assert ranked[0] == "col1"


def test_select_mi_alignment():
    # Test that mi selector doesn't scramble feature names when an empty column is dropped
    X = pd.DataFrame({
        "empty_col": [np.nan] * 100,
        "col1": np.random.randn(100),
        "col2": np.random.randn(100)
    })
    # y correlates only with col1
    y = X["col1"] * 5.0 + np.random.randn(100) * 0.1
    
    out = select_mi(X, y, k=2, random_state=42)
    scores = out["scores"]
    ranked = out["ranked"]
    
    # empty_col should have 0.0 score
    assert scores["empty_col"] == 0.0
    # col1 should have a high MI score
    assert scores["col1"] > scores["col2"]
    assert ranked[0] == "col1"
    assert ranked[-1] == "empty_col"


def test_select_elasticnet_alignment():
    # Test that elasticnet selector aligns features correctly when an empty column is dropped,
    # and that standard ElasticNet works with a fixed alpha parameter.
    X = pd.DataFrame({
        "empty_col": [np.nan] * 100,
        "col1": np.random.randn(100),
        "col2": np.random.randn(100)
    })
    y = X["col1"] * 3.0 + np.random.randn(100) * 0.1
    
    # Test ElasticNetCV
    out_cv = select_elasticnet(X, y, k=2, random_state=42)
    assert out_cv["scores"]["empty_col"] == 0.0
    assert out_cv["ranked"][0] == "col1"
    assert out_cv["ranked"][-1] == "empty_col"
    
    # Test standard ElasticNet with fixed alpha
    opt_alpha = out_cv["alpha"]
    opt_l1_ratio = out_cv["l1_ratio"]
    
    out_fixed = select_elasticnet(
        X, y, k=2, random_state=42, alpha=opt_alpha, l1_ratio=opt_l1_ratio
    )
    assert out_fixed["alpha"] == opt_alpha
    assert out_fixed["l1_ratio"] == opt_l1_ratio
    assert out_fixed["scores"]["empty_col"] == 0.0
    assert out_fixed["ranked"][0] == "col1"
    assert out_fixed["ranked"][-1] == "empty_col"


def test_select_features():
    np.random.seed(42)
    X = pd.DataFrame({
        "J_spatial": np.random.randn(100),
        "ts_feat_1": np.random.randn(100),
        "ts_feat_2": np.random.randn(100),
    })
    y = X["J_spatial"] * 2.0 + X["ts_feat_1"] * 1.5 + np.random.randn(100) * 0.1
    
    stages = [
        {"kind": "correlation", "threshold": 0.95},
        {"kind": "elasticnet", "k": 2}
    ]
    
    config = {
        "selection": {
            "top_k": 2,
            "stages": stages,
            "bypass": {"enabled": False},
        },
        "logging": {
            "level": "INFO",
            "console": True,
            "log_to_file": False
        }
    }
    
    res = select_features(
        X_train=X,
        y_train=y,
        config=config,
        verbose=False
    )
    
    selected = res["selected_features"]
    assert len(selected) > 0
    assert "ts_feat_1" in selected or "J_spatial" in selected


def test_select_xgb_importance_alignment():
    np.random.seed(42)
    X = pd.DataFrame({
        "empty_col": [np.nan] * 100,
        "col1": np.random.randn(100),
        "col2": np.random.randn(100),
    })
    y = X["col1"] * 3.0 + np.random.randn(100) * 0.1

    out = select_xgb_importance(
        X, y, k=2, random_state=42,
        params={"n_estimators": 20, "max_depth": 3},
    )
    assert out["scores"]["empty_col"] == 0.0
    assert out["ranked"][-1] == "empty_col"
    assert out["ranked"][0] == "col1"
    assert len(out["selected"]) == 2


def test_infer_coverage_family():
    assert infer_coverage_family("SMAP_sm_pm_interp_rollrange30") == "satellite"
    assert infer_coverage_family("s2_b8") == "satellite"
    assert infer_coverage_family("G_API") == "hydro"
    assert infer_coverage_family("V_rollmin_G_API_kobs30") == "hydro"
    assert infer_coverage_family("elev") == "static"
    assert infer_coverage_family("J_clay_wfrac_b0") == "static"
    assert infer_coverage_family("D_sin_DOY") == "calendar"
    assert infer_coverage_family("sin_year") == "calendar"
    assert infer_coverage_family("A_d_E_SAR_diff_kobs14") == "satellite"
    assert infer_coverage_family("V_rollmean_LST_modis_kobs30") == "satellite"


def test_family_coverage_promotes_satellite():
    selected = ["elev", "slope", "G_API"]  # static + hydro, no satellite
    available = selected + ["SMAP_sm_pm_interp", "s2_b8", "noise"]
    scores = {
        "elev": 0.5,
        "slope": 0.4,
        "G_API": 0.3,
        "SMAP_sm_pm_interp": 0.9,
        "s2_b8": 0.2,
        "noise": 0.1,
    }
    out = enforce_min_family_coverage(
        selected=selected,
        ranked_scores=scores,
        available=available,
        min_per_family=1,
        families=["satellite", "hydro", "static"],
    )
    assert "SMAP_sm_pm_interp" in out["selected"]
    assert any(p["family"] == "satellite" for p in out["promoted"])
    assert out["family_counts_after"]["satellite"] >= 1


def test_stability_bootstrap_xgb():
    np.random.seed(0)
    n = 80
    X = pd.DataFrame({
        "good": np.random.randn(n),
        "noise_a": np.random.randn(n),
        "noise_b": np.random.randn(n),
    })
    y = X["good"] * 2.0 + np.random.randn(n) * 0.05

    out = stability_bootstrap(
        X, y,
        base="xgb",
        n_boot=5,
        sample_frac=0.8,
        min_freq=0.4,
        top_k=2,
        random_state=0,
        base_k=2,
        base_kwargs={"params": {"n_estimators": 15, "max_depth": 3}},
    )
    assert "good" in out["selected"] or out["ranked"][0] == "good"
    assert len(out["ranked"]) >= 1


def test_select_features_xgb_path():
    np.random.seed(1)
    n = 120
    X = pd.DataFrame({
        "SMAP_a": np.random.randn(n),
        "G_API": np.random.randn(n),
        "elev": np.random.randn(n),
        "noise": np.random.randn(n),
        "D_sin_DOY": np.sin(np.linspace(0, 6.28, n)),
    })
    y = (
        X["SMAP_a"] * 1.5
        + X["G_API"] * 1.0
        + X["elev"] * 0.5
        + X["D_sin_DOY"] * 0.3
        + np.random.randn(n) * 0.1
    )
    config = {
        "selection": {
            "top_k": 4,
            "stability_n_boot": 4,
            "bypass": {"enabled": False},
            "stages": [
                {"kind": "xgb_importance", "k": 4, "params": {"n_estimators": 20, "max_depth": 3}},
                {"kind": "family_coverage", "min_per_family": 1},
                {
                    "kind": "stability",
                    "base": "xgb",
                    "k": 4,
                    "min_freq": 0.25,
                    "stability_n_boot": 4,
                    "params": {"n_estimators": 15, "max_depth": 3},
                },
            ],
        },
        "models": [],
        "logging": {"level": "WARNING", "console": False, "log_to_file": False},
    }
    res = select_features(X_train=X, y_train=y, config=config, verbose=False)
    assert len(res["selected_features"]) > 0


def _grouped_oof_fixture(seed=42, station_effect=False):
    rng = np.random.default_rng(seed)
    stations = [f"station_{idx}" for idx in range(8)]
    station_offsets = dict(
        zip(stations, rng.permutation(np.linspace(-4.0, 4.0, len(stations))))
    )
    rows = []
    for year in range(2017, 2023):
        for station_index, station in enumerate(stations):
            for day in range(12):
                main = rng.normal()
                interaction_a = rng.normal()
                interaction_b = rng.normal()
                offset = station_offsets[station] if station_effect else 0.0
                target = (
                    1.8 * main
                    + 1.5 * interaction_a * interaction_b
                    + offset
                    + rng.normal(scale=0.08)
                )
                rows.append(
                    {
                        "station_id": station,
                        "date": f"{year}-01-{day + 1:02d}",
                        "main": main,
                        "interaction_a": interaction_a,
                        "interaction_b": interaction_b,
                        "station_proxy": float(station_index),
                        "noise": rng.normal(),
                        "constant": 1.0,
                        "target": target,
                    }
                )
    return pd.DataFrame(rows)


def _fast_grouped_config(candidate_sizes):
    return {
        "candidate_sizes": candidate_sizes,
        "n_station_folds": 4,
        "min_train_years": 2,
        "max_validation_years": 2,
        "min_train_rows": 40,
        "min_validation_rows": 10,
        "confidence_z": 1.0,
        "permutation_repeats": 1,
        "train_weight_betas": [0.0],
        "parallel_workers": 2,
        "random_state": 7,
        "model_params": {
            "n_estimators": 35,
            "max_depth": 3,
            "learning_rate": 0.08,
            "n_jobs": 1,
        },
    }


def test_grouped_oof_folds_exclude_station_and_future_time():
    df = _grouped_oof_fixture()
    context = df[["station_id", "date"]]
    folds = build_station_time_folds(
        context,
        n_station_folds=4,
        min_train_years=2,
        max_validation_years=2,
        min_train_rows=40,
        min_validation_rows=10,
    )
    dates = pd.to_datetime(context["date"])
    for fold in folds:
        train_stations = set(context.iloc[fold.train_index]["station_id"])
        validation_stations = set(context.iloc[fold.validation_index]["station_id"])
        assert train_stations.isdisjoint(validation_stations)
        assert dates.iloc[fold.train_index].dt.year.max() < fold.validation_year
        assert set(dates.iloc[fold.validation_index].dt.year) == {
            fold.validation_year
        }


def test_forward_time_folds_keep_stations_and_exclude_future_rows():
    df = _grouped_oof_fixture()
    context = df[["station_id", "date"]]
    folds = build_forward_time_folds(
        context,
        min_train_years=2,
        max_validation_years=2,
        min_train_rows=40,
        min_validation_rows=10,
    )
    dates = pd.to_datetime(context["date"])
    expected_stations = set(context["station_id"])
    for fold in folds:
        assert fold.fold_family == "forward_time"
        assert not fold.held_out_stations
        assert set(context.iloc[fold.train_index]["station_id"]) == expected_stations
        assert set(context.iloc[fold.validation_index]["station_id"]) == expected_stations
        assert dates.iloc[fold.train_index].max() < dates.iloc[fold.validation_index].min()


def test_grouped_oof_is_invariant_to_feature_renaming():
    df = _grouped_oof_fixture()
    features = ["main", "interaction_a", "interaction_b", "noise"]
    config = _fast_grouped_config([3])
    original = select_grouped_oof(
        df[features],
        df["target"],
        df[["station_id", "date"]],
        config=config,
    )
    renamed_columns = ["feature_0", "feature_1", "feature_2", "feature_3"]
    renamed = df[features].copy()
    renamed.columns = renamed_columns
    renamed_out = select_grouped_oof(
        renamed,
        df["target"],
        df[["station_id", "date"]],
        config=config,
    )
    selected_positions = [features.index(feature) for feature in original["selected"]]
    renamed_positions = [
        renamed_columns.index(feature) for feature in renamed_out["selected"]
    ]
    assert selected_positions == renamed_positions


def test_grouped_oof_tied_importance_is_invariant_to_feature_renaming():
    df = _grouped_oof_fixture()
    features = ["z_constant", "a_constant", "y_constant", "b_constant"]
    tied = pd.DataFrame(
        np.ones((len(df), len(features))),
        columns=features,
    )
    config = _fast_grouped_config([2])
    original = select_grouped_oof(
        tied,
        df["target"],
        df[["station_id", "date"]],
        config=config,
    )

    renamed_columns = ["a", "z", "b", "y"]
    renamed = tied.copy()
    renamed.columns = renamed_columns
    renamed_out = select_grouped_oof(
        renamed,
        df["target"],
        df[["station_id", "date"]],
        config=config,
    )

    selected_positions = [features.index(feature) for feature in original["selected"]]
    renamed_positions = [
        renamed_columns.index(feature) for feature in renamed_out["selected"]
    ]
    assert selected_positions == [0, 1]
    assert renamed_positions == selected_positions


def test_grouped_oof_rejects_non_generalizing_station_proxy():
    df = _grouped_oof_fixture(station_effect=True)
    features = ["main", "station_proxy", "noise"]

    pooled = select_xgb_importance(
        df[features],
        df["target"],
        k=3,
        random_state=7,
        params={"n_estimators": 80, "max_depth": 4},
    )
    grouped = select_grouped_oof(
        df[features],
        df["target"],
        df[["station_id", "date"]],
        config=_fast_grouped_config([1]),
    )

    assert pooled["scores"]["station_proxy"] > pooled["scores"]["noise"]
    assert grouped["selected"] == ["main"]


def test_grouped_oof_preserves_interaction_pair_without_univariate_filter():
    df = _grouped_oof_fixture()
    features = ["interaction_a", "interaction_b", "noise"]
    out = select_grouped_oof(
        df[features],
        df["target"] - 1.8 * df["main"],
        df[["station_id", "date"]],
        config=_fast_grouped_config([2]),
    )
    assert set(out["selected"]) == {"interaction_a", "interaction_b"}


def test_grouped_oof_regime_delta_requires_positive_paired_lcb():
    df = _grouped_oof_fixture()
    features = ["main", "constant"]
    out = select_grouped_oof(
        df[features],
        1.8 * df["main"],
        df[["station_id", "date"]],
        config=_fast_grouped_config([2]),
        required_features=["main"],
    )
    assert out["selected"] == ["main"]
    assert out["stopping_reason"] == "no_regime_delta_with_positive_paired_lcb"


def test_grouped_oof_progressive_elimination_reaches_requested_size():
    df = _grouped_oof_fixture()
    features = ["main", "interaction_a", "interaction_b", "noise", "constant"]
    config = _fast_grouped_config([2])
    config["progressive_elimination"] = True
    out = select_grouped_oof(
        df[features],
        df["target"],
        df[["station_id", "date"]],
        config=config,
    )
    assert len(out["selected"]) == 2
    assert out["selection_path"][0]["n_features"] == 2
    assert out["config"]["progressive_elimination"] is True


def test_grouped_oof_returns_importance_for_winning_checkpoint(monkeypatch):
    df = _grouped_oof_fixture()
    features = ["main", "interaction_a", "interaction_b", "noise"]

    def fake_fold_rows(
        X,
        y,
        years,
        active_features,
        folds,
        config,
        *,
        collect_importance,
        feature_positions,
    ):
        del X, y, years, folds, config, feature_positions
        nrmse = {4: 0.4, 3: 0.1, 2: 0.3}[len(active_features)]
        rows = [
            {"fold_id": "fold_0", "nrmse": nrmse},
            {"fold_id": "fold_1", "nrmse": nrmse},
        ]
        deltas = {
            feature: [float(len(active_features) - index)] * 2
            for index, feature in enumerate(active_features)
        }
        if not collect_importance:
            deltas = {feature: [] for feature in active_features}
        return rows, deltas

    monkeypatch.setattr(grouped_oof_module, "_fold_rows", fake_fold_rows)
    out = select_grouped_oof(
        df[features],
        df["target"],
        df[["station_id", "date"]],
        config=_fast_grouped_config([3, 2]),
    )

    assert len(out["selected"]) == 3
    assert set(out["ranked"]) == set(out["selected"])
    assert set(out["scores"]) == set(out["selected"])
    assert set(out["importance_detail"]) == set(out["selected"])
    assert all(out["scores"][feature] != 0.0 for feature in out["selected"])


def test_select_features_grouped_oof_stage():
    df = _grouped_oof_fixture()
    features = ["main", "interaction_a", "interaction_b", "noise"]
    config = {
        "selection": {
            "stages": [
                {
                    "kind": "grouped_oof",
                    **_fast_grouped_config([3]),
                }
            ],
            "bypass": {"enabled": False},
        },
        "models": [],
        "logging": {
            "level": "WARNING",
            "console": False,
            "log_to_file": False,
        },
    }
    result = select_features(
        X_train=df[features],
        y_train=df["target"],
        config=config,
        selection_context=df[["station_id", "date"]],
        verbose=False,
    )
    assert len(result["selected_features"]) == 3
    assert result["selection_diagnostics"]["folds"]


def test_outer_selection_uses_strict_future_and_held_out_stations():
    rng = np.random.default_rng(12)
    rows = []
    for year in range(2017, 2023):
        for station in ["a", "b", "c", "d"]:
            for day in range(15):
                stable = rng.normal()
                spurious = rng.normal()
                if year <= 2020:
                    target = stable + 4.0 * spurious + rng.normal(scale=0.05)
                else:
                    target = stable + rng.normal(scale=0.05)
                rows.append(
                    {
                        "station_id": station,
                        "date": f"{year}-02-{day + 1:02d}",
                        "stable": stable,
                        "spurious": spurious,
                        "target": target,
                    }
                )
    frame = pd.DataFrame(rows)
    inner = frame[pd.to_datetime(frame["date"]).dt.year <= 2020].reset_index(
        drop=True
    )
    outer = frame[pd.to_datetime(frame["date"]).dt.year >= 2021].reset_index(
        drop=True
    )
    config = _fast_grouped_config([2, 1])
    result = evaluate_forward_station_time_candidates(
        inner[["stable", "spurious"]],
        inner["target"],
        inner[["station_id", "date"]],
        outer[["stable", "spurious"]],
        outer["target"],
        outer[["station_id", "date"]],
        [["stable", "spurious"], ["stable"]],
        config=config,
    )
    assert result["selected"] == ["stable"]
    assert all(fold["validation_year"] >= 2021 for fold in result["folds"])
    for fold in result["folds"]:
        assert fold["held_out_stations"]


@pytest.mark.parametrize(
    "outer_station_names",
    [("outer_c",), ("outer_c", "outer_d")],
)
def test_outer_selection_scores_stations_absent_from_training(
    outer_station_names,
):
    rng = np.random.default_rng(19)
    inner_rows = []
    outer_rows = []
    for year in (2019, 2020):
        for station in ("train_a", "train_b"):
            for day in range(12):
                signal = rng.normal()
                inner_rows.append(
                    {
                        "station_id": station,
                        "date": f"{year}-03-{day + 1:02d}",
                        "signal": signal,
                        "target": 2.0 * signal,
                    }
                )
    for year in (2021, 2022):
        for station in outer_station_names:
            for day in range(12):
                signal = rng.normal()
                outer_rows.append(
                    {
                        "station_id": station,
                        "date": f"{year}-03-{day + 1:02d}",
                        "signal": signal,
                        "target": 2.0 * signal,
                    }
                )
    inner = pd.DataFrame(inner_rows)
    outer = pd.DataFrame(outer_rows)

    result = evaluate_forward_station_time_candidates(
        inner[["signal"]],
        inner["target"],
        inner[["station_id", "date"]],
        outer[["signal"]],
        outer["target"],
        outer[["station_id", "date"]],
        [["signal"]],
        config=_fast_grouped_config([1]),
    )

    held_out = {
        station
        for fold in result["folds"]
        for station in fold["held_out_stations"]
    }
    assert held_out == set(outer_station_names)
    assert sum(fold["n_validation"] for fold in result["folds"]) == len(outer)


def test_outer_regime_delta_can_remain_empty():
    df = _grouped_oof_fixture()
    years = pd.to_datetime(df["date"]).dt.year
    inner = df[years <= 2020].reset_index(drop=True)
    outer = df[years >= 2021].reset_index(drop=True)
    config = _fast_grouped_config([2, 1])
    result = evaluate_forward_station_time_candidates(
        inner[["main", "constant"]],
        1.8 * inner["main"],
        inner[["station_id", "date"]],
        outer[["main", "constant"]],
        1.8 * outer["main"],
        outer[["station_id", "date"]],
        [["main", "constant"], ["main"]],
        config=config,
        required_features=["main"],
    )
    assert result["selected"] == ["main"]
    assert result["stopping_reason"] == "no_outer_delta_with_positive_paired_lcb"
