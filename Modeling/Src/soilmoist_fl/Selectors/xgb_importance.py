# XGBoost gain-importance feature selector (tree-aligned for final models)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Selectors.base import (
    _basic_xy_checks,
    _get_feature_cols,
    _top_k,
    log_top,
)

# Fast defaults for selection-time importance (final training uses 1.3-lite params)
DEFAULT_XGB_IMPORTANCE_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "min_child_weight": 5,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "n_jobs": 1,
    "verbosity": 0,
}


def select_xgb_importance(
    X,
    y,
    k=60,
    params=None,
    importance_type="gain",
    random_state=42,
    n_jobs=1,
):
    """Rank features by XGBoost importance (default: gain).

    Uses median imputation and maps scores via get_feature_names_out so
    all-NaN columns receive score 0.0 rather than scrambling alignments.
    """
    log = get_logger("selectors.xgb_importance")

    y = _basic_xy_checks(X, y)
    feature_cols = _get_feature_cols(X)

    y_num = pd.to_numeric(y, errors="coerce").to_numpy()
    if np.isnan(y_num).any():
        raise ValueError("select_xgb_importance: y contains NaNs after numeric coercion")

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    kept_features = list(imp.get_feature_names_out(feature_cols))

    model_params = dict(DEFAULT_XGB_IMPORTANCE_PARAMS)
    if params:
        model_params.update(params)
    model_params["random_state"] = int(random_state)
    model_params["n_jobs"] = int(n_jobs)
    # Ensure non-interactive / quiet
    model_params.setdefault("verbosity", 0)

    model = XGBRegressor(**model_params)
    model.fit(X_imp, y_num)

    # Prefer booster gain when available; fall back to feature_importances_
    try:
        booster = model.get_booster()
        score_raw = booster.get_score(importance_type=importance_type)
        # XGBoost names features f0, f1, ... matching column order of fit matrix
        importances = np.zeros(len(kept_features), dtype=float)
        for i in range(len(kept_features)):
            importances[i] = float(score_raw.get(f"f{i}", 0.0))
    except Exception:
        importances = np.asarray(model.feature_importances_, dtype=float)

    score_map = {f: float(v) for f, v in zip(kept_features, importances)}
    for f in feature_cols:
        if f not in score_map:
            score_map[f] = 0.0

    pairs = list(score_map.items())
    pairs.sort(key=lambda t: -t[1])
    ranked = [p[0] for p in pairs]
    selected = _top_k(ranked, k)

    log.info(
        "select_xgb_importance: fitted XGB for %d features, selected k=%d (importance=%s)",
        len(feature_cols),
        len(selected),
        importance_type,
    )
    log_top(log, "XGB|importance", ranked, score_map=score_map, n=15)

    return {
        "kind": "xgb_importance",
        "ranked": ranked,
        "scores": score_map,
        "selected": selected,
        "k": int(k) if k is not None else None,
        "importance_type": importance_type,
    }
