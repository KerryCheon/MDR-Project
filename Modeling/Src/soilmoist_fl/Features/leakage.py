# Jakob Balkovec
# Leakage Checks

import pandas as pd

from Modeling.Utils.logging import get_logger


def check_no_future_leakage(df, time_col="date", strict_order=False):
    log = get_logger("features.leakage")

    if time_col is None:
        log.info("check_no_future_leakage: time_col=None, skipping time checks")
        return True, []

    if time_col not in df.columns:
        log.warning("check_no_future_leakage: time_col '%s' not found, skipping time checks", time_col)
        return True, []

    issues = []

    t = pd.to_datetime(df[time_col], errors="coerce")

    if t.isna().any():
        issues.append(f"time_col '{time_col}' has unparseable values (NaT count={int(t.isna().sum())})")

    if strict_order:
        # only meaningful if df is supposed to be time-sorted
        if t.notna().any():
            diffs = t.diff()
            if (diffs.dt.total_seconds() < 0).any():
                issues.append(f"time_col '{time_col}' is not non-decreasing (data not sorted by time)")

    if issues:
        for msg in issues:
            log.warning("check_no_future_leakage: %s", msg)
        return False, issues

    log.info("check_no_future_leakage: OK")
    return True, []


def check_feature_name_leakage(feature_cols, forbidden=None):
    log = get_logger("features.leakage")

    forbidden = forbidden or []
    bad = []

    for f in feature_cols:
        name = str(f).lower()
        for token in forbidden:
            if token and token.lower() in name:
                bad.append(f)
                break

    if bad:
        log.error("check_feature_name_leakage: forbidden tokens found in features: %s", bad[:25])
        return False, bad

    log.info("check_feature_name_leakage: OK")
    return True, []


def check_fold_boundary_gap(train_df, val_df, time_col="date", gap_days=0):
    log = get_logger("features.leakage")

    if not time_col or time_col not in train_df.columns or time_col not in val_df.columns:
        log.info("check_fold_boundary_gap: missing time_col, skipping boundary check")
        return True, []

    issues = []

    t_train = pd.to_datetime(train_df[time_col], errors="coerce")
    t_val = pd.to_datetime(val_df[time_col], errors="coerce")

    if t_train.isna().all() or t_val.isna().all():
        log.info("check_fold_boundary_gap: time parsing failed, skipping boundary check")
        return True, []

    train_max = t_train.max()
    val_min = t_val.min()

    if pd.isna(train_max) or pd.isna(val_min):
        log.info("check_fold_boundary_gap: missing extrema, skipping boundary check")
        return True, []

    delta_days = (val_min - train_max).days

    if gap_days and delta_days < gap_days:
        issues.append(f"gap_days violated: required={gap_days} actual={delta_days} (val_min={val_min.date()} train_max={train_max.date()})")

    if val_min <= train_max:
        issues.append(f"temporal overlap or inversion: val_min={val_min.date()} train_max={train_max.date()}")

    if issues:
        for msg in issues:
            log.warning("check_fold_boundary_gap: %s", msg)
        return False, issues

    log.info("check_fold_boundary_gap: OK (delta_days=%d)", delta_days)
    return True, []
