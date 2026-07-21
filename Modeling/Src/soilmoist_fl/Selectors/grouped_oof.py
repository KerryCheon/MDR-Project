"""Generalization-aligned feature selection with grouped out-of-fold utility.

This selector deliberately does not inspect feature names.  Station identifiers
and timestamps are accepted only as fold metadata and are never passed to the
predictive model.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from Modeling.Src.soilmoist_fl.Selectors.base import _basic_xy_checks
from Modeling.Utils.logging import get_logger


DEFAULT_MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "min_child_weight": 5,
    "n_estimators": 160,
    "learning_rate": 0.04,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.5,
    "reg_alpha": 0.03,
    "tree_method": "hist",
    "n_jobs": 1,
    "verbosity": 0,
}


@dataclass(frozen=True)
class StationTimeFold:
    """One future-year, held-out-station fold."""

    fold_id: str
    train_index: np.ndarray
    validation_index: np.ndarray
    validation_year: int
    held_out_stations: tuple[str, ...]
    fold_family: str = "station_time"


def _numeric_target(y) -> np.ndarray:
    values = pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("grouped_oof: target contains non-finite values")
    return values.astype(float, copy=False)


def _standard_error(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _balanced_station_groups(stations: pd.Series, n_splits: int) -> list[tuple[str, ...]]:
    """Assign stations to deterministic row-balanced folds."""
    station_text = stations.astype(str)
    counts = station_text.value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("grouped_oof requires at least two stations")

    n_groups = min(max(2, int(n_splits)), len(counts))
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    loads = [0 for _ in range(n_groups)]

    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    for station, count in ordered:
        group_idx = min(range(n_groups), key=lambda idx: (loads[idx], idx))
        groups[group_idx].append(str(station))
        loads[group_idx] += int(count)

    return [tuple(sorted(group)) for group in groups if group]


def build_station_time_folds(
    context: pd.DataFrame,
    *,
    station_col: str = "station_id",
    time_col: str = "date",
    n_station_folds: int = 4,
    min_train_years: int = 2,
    max_validation_years: int = 4,
    min_train_rows: int = 100,
    min_validation_rows: int = 20,
) -> list[StationTimeFold]:
    """Build folds that hold out both future time and groups of stations.

    For validation year ``t``, training uses years strictly before ``t`` and
    excludes every station assigned to that validation station group.
    """
    if station_col not in context.columns:
        raise ValueError(f"grouped_oof context is missing station column: {station_col}")
    if time_col not in context.columns:
        raise ValueError(f"grouped_oof context is missing time column: {time_col}")

    stations = context[station_col].astype(str).reset_index(drop=True)
    dates = pd.to_datetime(context[time_col], errors="coerce").reset_index(drop=True)
    if dates.isna().any():
        raise ValueError("grouped_oof context contains unparseable dates")

    years = dates.dt.year.astype(int)
    unique_years = sorted(int(year) for year in years.unique())
    if len(unique_years) <= int(min_train_years):
        raise ValueError(
            "grouped_oof has too few years for rolling validation: "
            f"{unique_years}"
        )

    validation_years = unique_years[int(min_train_years):]
    if max_validation_years:
        validation_years = validation_years[-int(max_validation_years):]

    station_groups = _balanced_station_groups(stations, n_station_folds)
    folds: list[StationTimeFold] = []
    station_values = stations.to_numpy()
    year_values = years.to_numpy()

    for year in validation_years:
        for group_idx, held_out in enumerate(station_groups):
            held_mask = np.isin(station_values, np.asarray(held_out, dtype=object))
            train_mask = (year_values < year) & ~held_mask
            validation_mask = (year_values == year) & held_mask
            train_index = np.flatnonzero(train_mask)
            validation_index = np.flatnonzero(validation_mask)
            if len(train_index) < int(min_train_rows):
                continue
            if len(validation_index) < int(min_validation_rows):
                continue
            folds.append(
                StationTimeFold(
                    fold_id=f"year_{year}_stations_{group_idx}",
                    train_index=train_index,
                    validation_index=validation_index,
                    validation_year=int(year),
                    held_out_stations=held_out,
                )
            )

    if len(folds) < 2:
        raise ValueError(
            "grouped_oof produced fewer than two usable folds; "
            "reduce minimum row requirements or provide more station-year coverage"
        )
    return folds


def build_forward_time_folds(
    context: pd.DataFrame,
    *,
    station_col: str = "station_id",
    time_col: str = "date",
    min_train_years: int = 2,
    max_validation_years: int = 4,
    min_train_rows: int = 100,
    min_validation_rows: int = 20,
) -> list[StationTimeFold]:
    """Build pure forward-time folds that retain all available stations."""
    if station_col not in context.columns:
        raise ValueError(f"grouped_oof context is missing station column: {station_col}")
    if time_col not in context.columns:
        raise ValueError(f"grouped_oof context is missing time column: {time_col}")

    dates = pd.to_datetime(context[time_col], errors="coerce").reset_index(drop=True)
    if dates.isna().any():
        raise ValueError("grouped_oof context contains unparseable dates")
    years = dates.dt.year.astype(int)
    unique_years = sorted(int(year) for year in years.unique())
    if len(unique_years) <= int(min_train_years):
        raise ValueError(
            "grouped_oof has too few years for rolling validation: "
            f"{unique_years}"
        )
    validation_years = unique_years[int(min_train_years):]
    if max_validation_years:
        validation_years = validation_years[-int(max_validation_years):]

    year_values = years.to_numpy()
    folds = []
    for year in validation_years:
        train_index = np.flatnonzero(year_values < year)
        validation_index = np.flatnonzero(year_values == year)
        if len(train_index) < int(min_train_rows):
            continue
        if len(validation_index) < int(min_validation_rows):
            continue
        folds.append(
            StationTimeFold(
                fold_id=f"year_{year}_all_stations",
                train_index=train_index,
                validation_index=validation_index,
                validation_year=int(year),
                held_out_stations=(),
                fold_family="forward_time",
            )
        )
    if len(folds) < 2:
        raise ValueError(
            "grouped_oof produced fewer than two usable forward-time folds; "
            "reduce minimum row requirements or provide more years"
        )
    return folds


def _temporal_weights(years: np.ndarray, beta: float) -> np.ndarray | None:
    if float(beta) == 0.0:
        return None
    years = np.asarray(years, dtype=float)
    weights = np.exp(float(beta) * (years - float(np.max(years))))
    return weights / float(np.mean(weights))


def _normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    scale = float(np.std(y_true))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = 1.0
    return rmse, rmse / scale


def _model_params(config: dict, random_state: int) -> dict:
    params = dict(DEFAULT_MODEL_PARAMS)
    params.update(config.get("model_params", {}) or {})
    params["random_state"] = int(random_state)
    params["n_jobs"] = int(params.get("n_jobs", 1))
    return params


def _fold_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    years: np.ndarray,
    features: list[str],
    folds: list[StationTimeFold],
    config: dict,
    *,
    collect_importance: bool,
    feature_positions: dict[str, int],
) -> tuple[list[dict], dict[str, list[float]]]:
    """Evaluate a set and optionally compute paired permutation deltas."""
    betas = [float(beta) for beta in config.get("train_weight_betas", [0.0])]
    random_state = int(config.get("random_state", 42))
    repeats = int(config.get("permutation_repeats", 1))
    permutation_batch_size = max(1, int(config.get("permutation_batch_size", 16)))
    parallel_workers = max(1, int(config.get("parallel_workers", 16)))
    metric_rows: list[dict] = []
    deltas: dict[str, list[float]] = {feature: [] for feature in features}

    def _run_task(task):
        fold_number, fold, beta_number, beta = task
        X_train = X.iloc[fold.train_index][features]
        X_validation = X.iloc[fold.validation_index][features]
        y_train = y[fold.train_index]
        y_validation = y[fold.validation_index]

        seed = random_state + fold_number * 1009 + beta_number * 9176
        model = XGBRegressor(**_model_params(config, seed))
        sample_weight = _temporal_weights(years[fold.train_index], beta)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        prediction = np.asarray(model.predict(X_validation)).ravel()
        rmse, nrmse = _normalized_rmse(y_validation, prediction)
        metric_row = {
            "fold_id": f"{fold.fold_id}_beta_{beta:g}",
            "fold_family": fold.fold_family,
            "validation_year": fold.validation_year,
            "held_out_stations": list(fold.held_out_stations),
            "beta": beta,
            "n_train": int(len(fold.train_index)),
            "n_validation": int(len(fold.validation_index)),
            "rmse": rmse,
            "nrmse": nrmse,
        }

        if not collect_importance:
            return metric_row, None

        repeat_deltas = {feature: [] for feature in features}
        for repeat in range(repeats):
            for start in range(0, len(features), permutation_batch_size):
                batch = features[start:start + permutation_batch_size]
                permuted_frames = []
                for feature in batch:
                    position = int(feature_positions[feature])
                    perm_seed = (
                        random_state
                        + fold_number * 1_000_003
                        + beta_number * 100_003
                        + position * 101
                        + repeat
                    )
                    rng = np.random.default_rng(perm_seed)
                    X_permuted = X_validation.copy()
                    values = X_permuted[feature].to_numpy(copy=True)
                    X_permuted[feature] = values[rng.permutation(len(values))]
                    permuted_frames.append(X_permuted)

                batch_frame = pd.concat(permuted_frames, ignore_index=True)
                batch_prediction = np.asarray(model.predict(batch_frame)).reshape(
                    len(batch),
                    len(X_validation),
                )
                for batch_index, feature in enumerate(batch):
                    _, perm_nrmse = _normalized_rmse(
                        y_validation,
                        batch_prediction[batch_index],
                    )
                    repeat_deltas[feature].append(float(perm_nrmse - nrmse))

        task_deltas = {
            feature: float(np.mean(repeat_deltas[feature]))
            for feature in features
        }
        return metric_row, task_deltas

    tasks = [
        (fold_number, fold, beta_number, beta)
        for fold_number, fold in enumerate(folds)
        for beta_number, beta in enumerate(betas)
    ]
    if parallel_workers == 1:
        results = map(_run_task, tasks)
    else:
        executor = ThreadPoolExecutor(max_workers=parallel_workers)
        results = executor.map(_run_task, tasks)

    try:
        for metric_row, task_deltas in results:
            metric_rows.append(metric_row)
            if task_deltas is not None:
                for feature in features:
                    deltas[feature].append(task_deltas[feature])
    finally:
        if parallel_workers != 1:
            executor.shutdown(wait=True)

    return metric_rows, deltas


def _summarize_importance(
    deltas: dict[str, list[float]],
    *,
    confidence_z: float,
    feature_positions: dict[str, int],
) -> tuple[list[str], dict[str, float], dict[str, dict]]:
    detail = {}
    for feature, values in deltas.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        mean = float(np.mean(finite)) if len(finite) else float("-inf")
        se = _standard_error(finite)
        lcb = mean - float(confidence_z) * se
        detail[feature] = {
            "mean_delta_nrmse": mean,
            "standard_error": se,
            "lower_confidence_bound": lcb,
            "n_folds": int(len(finite)),
        }
    ranked = sorted(
        detail,
        key=lambda feature: (
            -detail[feature]["lower_confidence_bound"],
            -detail[feature]["mean_delta_nrmse"],
            feature_positions[feature],
        ),
    )
    scores = {
        feature: float(detail[feature]["lower_confidence_bound"])
        for feature in ranked
    }
    return ranked, scores, detail


def _summarize_feature_set(
    features: list[str],
    fold_rows: list[dict],
    confidence_z: float,
) -> dict:
    values = np.asarray([row["nrmse"] for row in fold_rows], dtype=float)
    mean = float(np.mean(values))
    se = _standard_error(values)
    return {
        "n_features": int(len(features)),
        "features": list(features),
        "mean_nrmse": mean,
        "standard_error": se,
        "upper_confidence_bound": mean + float(confidence_z) * se,
        "fold_metrics": fold_rows,
    }


def _paired_improvement(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    confidence_z: float,
) -> dict:
    baseline = {row["fold_id"]: float(row["nrmse"]) for row in baseline_rows}
    candidate = {row["fold_id"]: float(row["nrmse"]) for row in candidate_rows}
    common = sorted(set(baseline) & set(candidate))
    improvements = np.asarray(
        [baseline[key] - candidate[key] for key in common],
        dtype=float,
    )
    mean = float(np.mean(improvements)) if len(improvements) else float("-inf")
    se = _standard_error(improvements)
    return {
        "mean_delta_nrmse": mean,
        "standard_error": se,
        "lower_confidence_bound": mean - float(confidence_z) * se,
        "n_pairs": int(len(improvements)),
    }


def select_grouped_oof(
    X: pd.DataFrame,
    y,
    context: pd.DataFrame,
    *,
    config: dict | None = None,
    required_features: Iterable[str] | None = None,
) -> dict:
    """Select features by configurable future-time generalization utility.

    The algorithm starts with the full candidate set, repeatedly ranks features
    by out-of-fold permutation utility, and refits after each reduction.  This
    avoids irreversible univariate prefilters and lets correlated substitutes
    become visible after competing features are removed.
    """
    log = get_logger("selectors.grouped_oof")
    config = dict(config or {})
    _basic_xy_checks(X, y)
    y_values = _numeric_target(y)

    if len(context) != len(X):
        raise ValueError(
            "grouped_oof context row count must match X: "
            f"{len(context)} != {len(X)}"
        )

    X_work = X.reset_index(drop=True).copy()
    context_work = context.reset_index(drop=True).copy()
    feature_positions = {
        feature: position for position, feature in enumerate(X_work.columns)
    }
    required = list(dict.fromkeys(required_features or []))
    missing_required = [feature for feature in required if feature not in X_work.columns]
    if missing_required:
        raise ValueError(f"grouped_oof missing required features: {missing_required[:10]}")

    station_col = str(config.get("station_col", "station_id"))
    time_col = str(config.get("time_col", "date"))
    fold_strategy = str(config.get("fold_strategy", "station_time"))
    fold_kwargs = {
        "station_col": station_col,
        "time_col": time_col,
        "min_train_years": int(config.get("min_train_years", 2)),
        "max_validation_years": int(config.get("max_validation_years", 4)),
        "min_train_rows": int(config.get("min_train_rows", 100)),
        "min_validation_rows": int(config.get("min_validation_rows", 20)),
    }
    if fold_strategy == "station_time":
        folds = build_station_time_folds(
            context_work,
            n_station_folds=int(config.get("n_station_folds", 4)),
            **fold_kwargs,
        )
    elif fold_strategy == "forward_time":
        folds = build_forward_time_folds(context_work, **fold_kwargs)
    else:
        raise ValueError(
            "grouped_oof fold_strategy must be 'station_time' or 'forward_time', "
            f"got {fold_strategy!r}"
        )
    years = pd.to_datetime(context_work[time_col]).dt.year.to_numpy(dtype=int)
    confidence_z = float(config.get("confidence_z", 1.0))

    configured_sizes = [
        int(size) for size in config.get("candidate_sizes", [80, 65, 50, 40])
    ]
    minimum_size = len(required)
    candidate_sizes = sorted(
        {
            min(len(X_work.columns), max(minimum_size, size))
            for size in configured_sizes
            if size > 0
        },
        reverse=True,
    )
    if minimum_size and minimum_size not in candidate_sizes:
        candidate_sizes.append(minimum_size)
    if not candidate_sizes:
        candidate_sizes = [len(X_work.columns)]

    current = list(X_work.columns)
    path = []
    current_fold_rows = None
    current_ranked = None
    current_scores = None
    current_detail = None
    checkpoint_importance = {}

    for target_size in candidate_sizes:
        target_size = min(int(target_size), len(current))
        while target_size < len(current):
            if bool(config.get("progressive_elimination", False)):
                # Do not discard more features in one refit than the requested
                # retained checkpoint. This creates data-derived bridge sizes
                # (for example 500 -> 350 -> 200 -> 150) without another
                # fractional pruning hyperparameter.
                next_size = max(target_size, len(current) - target_size)
            else:
                next_size = target_size
            if current_ranked is None:
                current_fold_rows, deltas = _fold_rows(
                    X_work,
                    y_values,
                    years,
                    current,
                    folds,
                    config,
                    collect_importance=True,
                    feature_positions=feature_positions,
                )
                current_ranked, current_scores, current_detail = (
                    _summarize_importance(
                        deltas,
                        confidence_z=confidence_z,
                        feature_positions=feature_positions,
                    )
                )
            removable_ranked = [
                feature for feature in current_ranked if feature not in required
            ]
            keep_removable = max(0, next_size - len(required))
            current = list(dict.fromkeys(required + removable_ranked[:keep_removable]))
            current_fold_rows = None
            current_ranked = None
            current_scores = None
            current_detail = None

        if current_fold_rows is None:
            current_fold_rows, deltas = _fold_rows(
                X_work,
                y_values,
                years,
                current,
                folds,
                config,
                collect_importance=True,
                feature_positions=feature_positions,
            )
            current_ranked, current_scores, current_detail = _summarize_importance(
                deltas,
                confidence_z=confidence_z,
                feature_positions=feature_positions,
            )
        summary = _summarize_feature_set(current, current_fold_rows, confidence_z)
        path.append(summary)
        checkpoint_importance[tuple(current)] = {
            "ranked": list(current_ranked),
            "scores": dict(current_scores),
            "detail": dict(current_detail),
        }
        log.info(
            "grouped_oof candidate: n=%d mean_nrmse=%.6f ucb=%.6f",
            len(current),
            summary["mean_nrmse"],
            summary["upper_confidence_bound"],
        )

    baseline_summary = None
    if required:
        baseline_summary = next(
            summary for summary in path if summary["features"] == required
        )
        for summary in path:
            summary["paired_improvement_vs_required"] = _paired_improvement(
                baseline_summary["fold_metrics"],
                summary["fold_metrics"],
                confidence_z,
            )

        admissible = [
            summary
            for summary in path
            if summary["n_features"] > len(required)
            and summary["paired_improvement_vs_required"]["lower_confidence_bound"] > 0.0
        ]
        if admissible:
            winner = min(
                admissible,
                key=lambda item: (
                    item["upper_confidence_bound"],
                    item["n_features"],
                ),
            )
            stopping_reason = "positive_paired_lcb"
        else:
            winner = baseline_summary
            stopping_reason = "no_regime_delta_with_positive_paired_lcb"
    else:
        winner = min(
            path,
            key=lambda item: (
                item["upper_confidence_bound"],
                item["n_features"],
            ),
        )
        stopping_reason = "minimum_grouped_oof_upper_confidence_bound"

    fold_metadata = [
        {
            "fold_id": fold.fold_id,
            "fold_family": fold.fold_family,
            "validation_year": fold.validation_year,
            "held_out_stations": list(fold.held_out_stations),
            "n_train": int(len(fold.train_index)),
            "n_validation": int(len(fold.validation_index)),
        }
        for fold in folds
    ]
    selected = list(winner["features"])
    winner_importance = checkpoint_importance[tuple(selected)]
    return {
        "kind": "grouped_oof",
        "selected": selected,
        "ranked": winner_importance["ranked"],
        "scores": winner_importance["scores"],
        "importance_detail": winner_importance["detail"],
        "selection_path": path,
        "baseline_required": baseline_summary,
        "folds": fold_metadata,
        "stopping_reason": stopping_reason,
        "required_features": required,
        "config": config,
    }


def evaluate_forward_station_time_candidates(
    X_train: pd.DataFrame,
    y_train,
    context_train: pd.DataFrame,
    X_outer: pd.DataFrame,
    y_outer,
    context_outer: pd.DataFrame,
    candidates: Iterable[Iterable[str]],
    *,
    config: dict | None = None,
    required_features: Iterable[str] | None = None,
) -> dict:
    """Choose among inner-selected candidates on an outer time holdout.

    Candidate feature lists must be created without using ``X_outer`` or
    ``y_outer``.  Every outer fold validates a future year on stations excluded
    from the corresponding fit, so feature ranking and feature-count selection
    occur on disjoint observations.
    """
    log = get_logger("selectors.grouped_oof.outer")
    config = dict(config or {})
    _basic_xy_checks(X_train, y_train)
    _basic_xy_checks(X_outer, y_outer)
    y_train_values = _numeric_target(y_train)
    y_outer_values = _numeric_target(y_outer)

    if len(context_train) != len(X_train):
        raise ValueError("outer selection train context row count must match X_train")
    if len(context_outer) != len(X_outer):
        raise ValueError("outer selection context row count must match X_outer")
    if list(X_train.columns) != list(X_outer.columns):
        raise ValueError("outer selection feature columns must match inner columns")

    station_col = str(config.get("station_col", "station_id"))
    time_col = str(config.get("time_col", "date"))
    for label, context in (("train", context_train), ("outer", context_outer)):
        if station_col not in context.columns or time_col not in context.columns:
            raise ValueError(
                f"outer selection {label} context requires {station_col} and {time_col}"
            )

    X_train_work = X_train.reset_index(drop=True)
    X_outer_work = X_outer.reset_index(drop=True)
    train_context = context_train.reset_index(drop=True).copy()
    outer_context = context_outer.reset_index(drop=True).copy()
    train_dates = pd.to_datetime(train_context[time_col], errors="coerce")
    outer_dates = pd.to_datetime(outer_context[time_col], errors="coerce")
    if train_dates.isna().any() or outer_dates.isna().any():
        raise ValueError("outer selection context contains unparseable dates")
    if int(train_dates.dt.year.max()) >= int(outer_dates.dt.year.min()):
        raise ValueError(
            "outer selection requires a strict forward-time boundary between "
            "inner training and outer validation"
        )

    candidate_lists = []
    seen = set()
    for candidate in candidates:
        features = list(dict.fromkeys(candidate))
        missing = [feature for feature in features if feature not in X_train_work.columns]
        if missing:
            raise ValueError(f"outer candidate contains missing features: {missing[:10]}")
        key = tuple(features)
        if key not in seen:
            candidate_lists.append(features)
            seen.add(key)
    if not candidate_lists:
        raise ValueError("outer selection requires at least one candidate feature list")

    required = list(dict.fromkeys(required_features or []))
    if required and tuple(required) not in seen:
        candidate_lists.append(required)

    train_stations = train_context[station_col].astype(str)
    outer_stations = outer_context[station_col].astype(str)
    # Preserve training-population balancing while extending the grouping pool
    # with every genuinely new validation station. A matching inner station is
    # excluded below; an outer-only station is absent from the fit by design.
    outer_only_stations = sorted(set(outer_stations) - set(train_stations))
    station_population = pd.concat(
        [train_stations, pd.Series(outer_only_stations, dtype="object")],
        ignore_index=True,
    )
    station_groups = _balanced_station_groups(
        station_population,
        int(config.get("n_station_folds", 4)),
    )
    outer_years = outer_dates.dt.year.to_numpy(dtype=int)
    train_years = train_dates.dt.year.to_numpy(dtype=int)
    train_station_values = train_stations.to_numpy()
    outer_station_values = outer_stations.to_numpy()
    validation_years = sorted(int(year) for year in np.unique(outer_years))
    max_years = int(config.get("max_validation_years", len(validation_years)))
    if max_years:
        validation_years = validation_years[-max_years:]

    betas = [float(beta) for beta in config.get("train_weight_betas", [0.0])]
    random_state = int(config.get("random_state", 42))
    min_train_rows = int(config.get("min_train_rows", 100))
    min_validation_rows = int(config.get("min_validation_rows", 20))
    confidence_z = float(config.get("confidence_z", 1.0))
    parallel_workers = max(1, int(config.get("parallel_workers", 16)))

    fold_specs = []
    for year in validation_years:
        for group_index, held_out in enumerate(station_groups):
            train_mask = ~np.isin(
                train_station_values,
                np.asarray(held_out, dtype=object),
            )
            validation_mask = (
                (outer_years == year)
                & np.isin(outer_station_values, np.asarray(held_out, dtype=object))
            )
            train_index = np.flatnonzero(train_mask)
            validation_index = np.flatnonzero(validation_mask)
            if len(train_index) < min_train_rows or len(validation_index) < min_validation_rows:
                continue
            fold_specs.append(
                {
                    "fold_id": f"outer_year_{year}_stations_{group_index}",
                    "validation_year": year,
                    "held_out_stations": list(held_out),
                    "train_index": train_index,
                    "validation_index": validation_index,
                }
            )
    if len(fold_specs) < 2:
        raise ValueError("outer selection produced fewer than two usable folds")

    summaries = []
    for features in candidate_lists:
        def _run_outer_task(task):
            fold_index, fold, beta_index, beta = task
            X_fold_train = X_train_work.iloc[fold["train_index"]][features]
            X_fold_outer = X_outer_work.iloc[fold["validation_index"]][features]
            y_fold_train = y_train_values[fold["train_index"]]
            y_fold_outer = y_outer_values[fold["validation_index"]]
            # Keep the seed paired across candidate sets so feature-list
            # comparisons are not confounded by a different tree sample.
            seed = random_state + fold_index * 1009 + beta_index * 9176
            model = XGBRegressor(**_model_params(config, seed))
            model.fit(
                X_fold_train,
                y_fold_train,
                sample_weight=_temporal_weights(
                    train_years[fold["train_index"]],
                    beta,
                ),
            )
            prediction = np.asarray(model.predict(X_fold_outer)).ravel()
            rmse, nrmse = _normalized_rmse(y_fold_outer, prediction)
            return {
                "fold_id": f"{fold['fold_id']}_beta_{beta:g}",
                "validation_year": int(fold["validation_year"]),
                "held_out_stations": list(fold["held_out_stations"]),
                "beta": beta,
                "n_train": int(len(fold["train_index"])),
                "n_validation": int(len(fold["validation_index"])),
                "rmse": rmse,
                "nrmse": nrmse,
            }

        tasks = [
            (fold_index, fold, beta_index, beta)
            for fold_index, fold in enumerate(fold_specs)
            for beta_index, beta in enumerate(betas)
        ]
        if parallel_workers == 1:
            rows = list(map(_run_outer_task, tasks))
        else:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                rows = list(executor.map(_run_outer_task, tasks))
        summary = _summarize_feature_set(features, rows, confidence_z)
        summaries.append(summary)
        log.info(
            "outer candidate: n=%d mean_nrmse=%.6f ucb=%.6f",
            len(features),
            summary["mean_nrmse"],
            summary["upper_confidence_bound"],
        )

    baseline_summary = None
    if required:
        baseline_summary = next(
            summary for summary in summaries if summary["features"] == required
        )
        for summary in summaries:
            summary["paired_improvement_vs_required"] = _paired_improvement(
                baseline_summary["fold_metrics"],
                summary["fold_metrics"],
                confidence_z,
            )
        admissible = [
            summary
            for summary in summaries
            if summary["n_features"] > len(required)
            and summary["paired_improvement_vs_required"]["lower_confidence_bound"]
            > 0.0
        ]
        if admissible:
            winner = min(
                admissible,
                key=lambda item: (
                    item["upper_confidence_bound"],
                    item["n_features"],
                ),
            )
            stopping_reason = "outer_positive_paired_lcb"
        else:
            winner = baseline_summary
            stopping_reason = "no_outer_delta_with_positive_paired_lcb"
    else:
        winner = min(
            summaries,
            key=lambda item: (
                item["upper_confidence_bound"],
                item["n_features"],
            ),
        )
        stopping_reason = "minimum_outer_upper_confidence_bound"

    return {
        "kind": "forward_station_time_outer_selection",
        "selected": list(winner["features"]),
        "candidate_summaries": summaries,
        "baseline_required": baseline_summary,
        "folds": [
            {
                key: value
                for key, value in fold.items()
                if key not in {"train_index", "validation_index"}
            }
            | {
                "n_train": int(len(fold["train_index"])),
                "n_validation": int(len(fold["validation_index"])),
            }
            for fold in fold_specs
        ],
        "stopping_reason": stopping_reason,
        "required_features": required,
        "config": config,
    }
