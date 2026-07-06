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
