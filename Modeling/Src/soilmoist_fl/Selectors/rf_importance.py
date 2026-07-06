# Jakob Balkovec
# Random Forest Importance Selector

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Selectors.base import _basic_xy_checks, _get_feature_cols, _top_k, log_top


def select_rf_importance(
    X,
    y,
    k=60,
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
):
    log = get_logger("selectors.rf_importance")

    y = _basic_xy_checks(X, y)
    feature_cols = _get_feature_cols(X)

    y_num = pd.to_numeric(y, errors="coerce").to_numpy()
    if np.isnan(y_num).any():
        raise ValueError("select_rf_importance: y contains NaNs after numeric coercion")

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # SimpleImputer may drop all-NaN features
    kept_features = imp.get_feature_names_out(feature_cols)

    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )

    rf.fit(X_imp, y_num)
    importances = rf.feature_importances_

    # Map kept features to importances
    score_map = {f: float(imp_val) for f, imp_val in zip(kept_features, importances)}

    # Fill dropped features with 0.0
    for f in feature_cols:
        if f not in score_map:
            score_map[f] = 0.0

    # Sort and rank all features
    pairs = list(score_map.items())
    pairs.sort(key=lambda t: -t[1])
    ranked = [p[0] for p in pairs]
    selected = _top_k(ranked, k)

    log.info("select_rf_importance: fitted RF for %d features, selected k=%d", len(feature_cols), len(selected))
    log_top(log, "RF|importance", ranked, score_map=score_map, n=15)

    return {
        "kind": "rf_importance",
        "ranked": ranked,
        "scores": score_map,
        "selected": selected,
        "k": int(k),
    }
