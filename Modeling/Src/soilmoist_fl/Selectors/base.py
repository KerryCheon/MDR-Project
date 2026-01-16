# Jakob Balkovec
# Selector Base

import numpy as np
import pandas as pd

from Modeling.Utils.logging import get_logger


def _ensure_1d(y):
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be 1D (Series) or single-column DataFrame")
        return y.iloc[:, 0]
    return y


def _basic_xy_checks(X, y):
    if X is None or y is None:
        raise ValueError("X or y is None")
    if X.shape[0] == 0:
        raise ValueError("X is empty")
    y = _ensure_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch X={X.shape[0]} y={y.shape[0]}")
    return y


def _get_feature_cols(X):
    return list(X.columns)


def _rank_dict_from_scores(feature_cols, scores):
    pairs = list(zip(feature_cols, scores))
    pairs.sort(key=lambda t: (-(t[1] if t[1] is not None else -np.inf)))
    ranked = [p[0] for p in pairs]
    score_map = {p[0]: float(p[1]) if p[1] is not None else float("nan") for p in pairs}
    return ranked, score_map


def _top_k(ranked, k):
    if k is None:
        return ranked
    k = int(k)
    if k <= 0:
        return []
    return ranked[:k]


def log_top(log, name, ranked, score_map=None, n=15):
    head = ranked[:n]
    if score_map is None:
        log.info("%s top %d: %s", name, len(head), head)
    else:
        preview = [(f, round(score_map.get(f, float("nan")), 6)) for f in head]
        log.info("%s top %d: %s", name, len(head), preview)
