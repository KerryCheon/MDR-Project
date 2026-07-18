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

