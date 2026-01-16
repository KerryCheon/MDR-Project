# Jakob Balkovec
# Mutual Information Selector

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Selectors.base import _basic_xy_checks, _get_feature_cols, _rank_dict_from_scores, _top_k, log_top


def select_mi(X, y, k=120, random_state=42, n_neighbors=3):
    log = get_logger("selectors.mi")

    y = _basic_xy_checks(X, y)
    feature_cols = _get_feature_cols(X)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # mutual_info_regression expects numeric matrix + 1d y
    y_num = pd.to_numeric(y, errors="coerce").to_numpy()
    if np.isnan(y_num).any():
        raise ValueError("select_mi: y contains NaNs after numeric coercion")

    scores = mutual_info_regression(
        X_imp,
        y_num,
        random_state=int(random_state),
        n_neighbors=int(n_neighbors),
    )

    ranked, score_map = _rank_dict_from_scores(feature_cols, scores)
    selected = _top_k(ranked, k)

    log.info("select_mi: computed MI for %d features, selected k=%d", len(feature_cols), len(selected))
    log_top(log, "MI", ranked, score_map=score_map, n=15)

    return {
        "kind": "mi",
        "ranked": ranked,
        "scores": score_map,
        "selected": selected,
        "k": int(k),
    }
