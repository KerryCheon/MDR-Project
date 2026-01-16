# Jakob Balkovec
# Validator (Splits)
import numpy as np
import pandas as pd

from Modeling.Utils.logging import get_logger


class ValidationIssue:
    def __init__(self, fold, level, message):
        self.fold = fold
        self.level = level  # "ERROR" or "WARN"
        self.message = message


class ValidationReport:
    def __init__(self, ok, issues, meta):
        self.ok = ok
        self.issues = issues
        self.meta = meta


def _is_finite_series(s):
    # handles numeric and object types safely
    s_num = pd.to_numeric(s, errors="coerce")
    arr = s_num.to_numpy()
    return np.isfinite(arr).all()


def _colset(df):
    return set(map(str, df.columns.tolist()))


def _check_same_columns(fold, train_cols, val_cols, test_cols, allow_missing_in_val_test=None):
    issues = []
    allow = allow_missing_in_val_test or set()

    # train is the reference here
    missing_val = (train_cols - val_cols) - allow
    missing_test = (train_cols - test_cols) - allow

    extra_val = val_cols - train_cols
    extra_test = test_cols - train_cols

    if missing_val:
        issues.append(ValidationIssue(fold, "ERROR", f"Val missing cols vs train: {sorted(missing_val)[:20]}"))
    if missing_test:
        issues.append(ValidationIssue(fold, "ERROR", f"Test missing cols vs train: {sorted(missing_test)[:20]}"))

    if extra_val:
        issues.append(ValidationIssue(fold, "ERROR", f"Val has extra cols vs train: {sorted(extra_val)[:20]}"))
    if extra_test:
        issues.append(ValidationIssue(fold, "ERROR", f"Test has extra cols vs train: {sorted(extra_test)[:20]}"))

    return issues


def validate_loaded_data(loaded, config):

    log = get_logger("data.validate")

    data_cfg = (config or {}).get("data", {})
    target = data_cfg.get("target")
    if not target:
        raise ValueError("Missing required config: data.target")

    time_col = data_cfg.get("time_col")
    id_cols = list(data_cfg.get("id_cols", []) or [])
    allow_missing = set(data_cfg.get("allow_missing_in_val_test", []) or [])

    log.info(
        "Validating data: folds=%d target=%s time_col=%s id_cols=%s",
        len(loaded.folds), target, time_col, id_cols,
    )

    issues = []

    for fold in loaded.folds:
        issues.extend(_validate_fold(fold, target, time_col, id_cols, allow_missing))

    ok = not any(i.level == "ERROR" for i in issues)

    if ok:
        log.info("Validation passed for %d fold(s).", len(loaded.folds))
    else:
        err_count = sum(1 for i in issues if i.level == "ERROR")
        warn_count = sum(1 for i in issues if i.level == "WARN")
        log.error("Validation failed. errors=%d warns=%d", err_count, warn_count)

    # Log issues (trim spam a bit)
    for i in issues[:50]:
        if i.level == "ERROR":
            log.error("[%s] %s", i.fold, i.message)
        else:
            log.warning("[%s] %s", i.fold, i.message)

    meta = {
        "n_folds": loaded.meta.get("n_folds"),
        "fold_names": loaded.meta.get("fold_names"),
        "target": target,
        "time_col": time_col,
        "id_cols": id_cols,
    }

    return ValidationReport(ok=ok, issues=issues, meta=meta)


def _validate_fold(fold, target, time_col, id_cols, allow_missing):
    log = get_logger("data.validate")
    issues = []
    name = fold.name

    log.debug(
        "Fold %s shapes: train=%s val=%s test=%s",
        name, fold.train.shape, fold.val.shape, fold.test.shape,
    )

    # non-empty checks
    if fold.train is None or fold.train.shape[0] == 0:
        issues.append(ValidationIssue(name, "ERROR", "Train split is empty."))
        return issues
    if fold.val is None or fold.val.shape[0] == 0:
        issues.append(ValidationIssue(name, "ERROR", "Val split is empty."))
    if fold.test is None or fold.test.shape[0] == 0:
        issues.append(ValidationIssue(name, "ERROR", "Test split is empty."))

    # duplicate column names
    for split_name, df in [("train", fold.train), ("val", fold.val), ("test", fold.test)]:
        cols = list(map(str, df.columns.tolist()))
        if len(cols) != len(set(cols)):
            issues.append(ValidationIssue(name, "ERROR", f"{split_name} has duplicate column names."))

    # required columns
    required = [target] + ([time_col] if time_col else []) + id_cols
    for split_name, df in [("train", fold.train), ("val", fold.val), ("test", fold.test)]:
        for col in required:
            if col and col not in df.columns:
                issues.append(ValidationIssue(name, "ERROR", f"{split_name} missing required column: {col}"))

    # target sanity
    for split_name, df in [("train", fold.train), ("val", fold.val), ("test", fold.test)]:
        if target in df.columns:
            y = df[target]
            if y.isna().any():
                issues.append(ValidationIssue(name, "ERROR", f"{split_name} target has NaNs (count={int(y.isna().sum())})."))
            if not _is_finite_series(y):
                issues.append(ValidationIssue(name, "ERROR", f"{split_name} target has inf or non-numeric values."))

    # column consistency across splits
    train_cols = _colset(fold.train)
    val_cols = _colset(fold.val)
    test_cols = _colset(fold.test)
    issues.extend(_check_same_columns(name, train_cols, val_cols, test_cols, allow_missing_in_val_test=allow_missing))

    # optional time sanity
    if time_col and time_col in fold.train.columns:
        t = fold.train[time_col]
        if t.isna().all():
            issues.append(ValidationIssue(name, "WARN", "time_col is all-NaN in train."))
        else:
            # this is not enforcing ordering, just checking it can be parsed/sorted
            try:
                _ = pd.to_datetime(t, errors="raise")
            except Exception:
                issues.append(ValidationIssue(name, "WARN", "time_col cannot be parsed as datetime."))

    return issues
