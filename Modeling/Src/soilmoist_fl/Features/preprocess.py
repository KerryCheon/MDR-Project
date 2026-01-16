# Jakob Balkovec
# Preprocessing

import numpy as np
import pandas as pd

from Modeling.Utils.logging import get_logger


def split_xy(df, target, drop_cols=None):
    log = get_logger("features.preprocess")

    if df is None or df.shape[0] == 0:
        raise ValueError("split_xy: df is empty")

    if target not in df.columns:
        raise ValueError(f"split_xy: target column not found: {target}")

    drop_cols = drop_cols or []
    drop_set = set(drop_cols)
    drop_set.add(target)

    feature_cols = [c for c in df.columns if c not in drop_set]

    X = df[feature_cols].copy()
    y = df[target].copy()

    log.info("split_xy: X=%s y=%s (target=%s dropped_cols=%d)",
             X.shape, y.shape, target, len(drop_cols))

    return X, y, feature_cols


def coerce_numeric(X, na_value=np.nan, drop_non_numeric=False):
    # note: this will turn station_id and date into a NaN, but that's
    # not that relevant here since they'll get dropped later

    log = get_logger("features.preprocess")

    if X is None or X.shape[0] == 0:
        raise ValueError("coerce_numeric: X is empty")

    bad_cols = []
    out = X.copy()

    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue

        coerced = pd.to_numeric(out[c], errors="coerce")
        # if coercion creates *all* NaN, it's basically non-numeric
        if coerced.isna().all():
            bad_cols.append(c)
        out[c] = coerced

    if bad_cols:
        msg = f"coerce_numeric: non-numeric columns detected: {bad_cols[:20]}"
        if drop_non_numeric:
            out = out.drop(columns=bad_cols)
            log.warning("%s (dropped %d cols)", msg, len(bad_cols))
        else:
            log.warning("%s (kept as NaN after coercion)", msg)

    if na_value is not np.nan:
        out = out.fillna(na_value)

    return out, bad_cols


def basic_sanity_checks(X, y):
    log = get_logger("features.preprocess")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"basic_sanity_checks: row mismatch X={X.shape[0]} y={y.shape[0]}")

    if y.isna().any():
        raise ValueError(f"basic_sanity_checks: target contains NaNs (count={int(y.isna().sum())})")

    y_num = pd.to_numeric(y, errors="coerce")
    if np.isfinite(y_num.to_numpy()).all() is False:
        raise ValueError("basic_sanity_checks: target contains non-finite values")

    log.info("basic_sanity_checks: OK (rows=%d, cols=%d)", X.shape[0], X.shape[1])


def preprocess_split(df, target, drop_cols=None, drop_non_numeric=False):
    X, y, feature_cols = split_xy(df, target, drop_cols=drop_cols)

    X_num, bad_cols = coerce_numeric(X, drop_non_numeric=drop_non_numeric)

    basic_sanity_checks(X_num, y)

    return X_num, y, feature_cols, bad_cols
