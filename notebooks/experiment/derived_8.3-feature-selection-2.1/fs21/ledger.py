"""Raw OOF prediction ledger construction and repeat-safe collapsing."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .constants import LEDGER_COLUMNS
from .data import numeric_frame, ordered_feature_hash
from .folds import FoldTask, assert_train_before_origin
from .modeling import fit_model, model_configuration_id
from .artifacts import stable_json_hash


PRIMARY_REPEAT_KEYS = [
    "candidate",
    "fold_family",
    "outer_origin",
    "station",
    "date",
    "beta",
]


def prediction_rows(
    frame: pd.DataFrame,
    task: FoldTask,
    features: list[str],
    *,
    candidate: str,
    path_source: str,
    endpoint: int | None,
    beta: float,
    config: Mapping[str, object],
    device: str,
    smoke: bool = False,
    model_name: str = "1.3-lite",
) -> pd.DataFrame:
    assert_train_before_origin(frame, task)
    if not features or len(features) != len(set(features)):
        raise ValueError(f"candidate {candidate} has an empty or duplicate feature list")
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"candidate {candidate} is missing features: {missing[:10]}")
    train = frame.iloc[list(task.train_index)]
    validation = frame.iloc[list(task.validation_index)]
    data = dict(config["data"])
    target = str(data["target"])
    station_col = str(data["station_col"])
    time_col = str(data["time_col"])
    feature_hash = ordered_feature_hash(features)
    model = fit_model(
        numeric_frame(train, features),
        train[target].to_numpy(dtype=float),
        train_years=train["_year"].to_numpy(dtype=int),
        beta=beta,
        config=config,
        seed=task.learner_seed,
        device=device,
        smoke=smoke,
    )
    truth = validation[target].to_numpy(dtype=float)
    prediction = np.asarray(
        model.predict(numeric_frame(validation, features)), dtype=float
    ).ravel()
    residual = truth - prediction
    configuration_id = model_configuration_id(
        candidate=candidate,
        feature_hash=feature_hash,
        beta=beta,
        learner_seed=task.learner_seed,
        device=device,
    )
    output = pd.DataFrame(
        {
            "model": model_name,
            "candidate": candidate,
            "path_source": path_source,
            "endpoint": np.nan if endpoint is None else int(endpoint),
            "actual_count": len(features),
            "ordered_feature_hash": feature_hash,
            "fold_family": task.family,
            "outer_origin": task.origin,
            "fold_id": task.fold_id,
            "station_partition_seed": task.partition_seed,
            "learner_seed": task.learner_seed,
            "station": validation[station_col].astype(str).to_numpy(),
            "date": validation[time_col].astype(str).to_numpy(),
            "year": validation["_year"].to_numpy(dtype=int),
            "month": validation["_month"].to_numpy(dtype=int),
            "truth": truth,
            "prediction": prediction,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "squared_error": np.square(residual),
            "beta": float(beta),
            "model_config_id": configuration_id,
            "router_regime": np.nan,
            "route_distance": np.nan,
        }
    )
    return validate_ledger(output)


def validate_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(LEDGER_COLUMNS).difference(ledger.columns))
    if missing:
        raise ValueError(f"prediction ledger is missing columns: {missing}")
    output = ledger.loc[:, list(LEDGER_COLUMNS)].copy()
    truth = output["truth"].to_numpy(dtype=float)
    prediction = output["prediction"].to_numpy(dtype=float)
    residual = output["residual"].to_numpy(dtype=float)
    if not np.allclose(residual, truth - prediction, rtol=0.0, atol=1e-12):
        raise ValueError("ledger residual must be truth - prediction")
    if not np.allclose(
        output["absolute_error"].to_numpy(dtype=float),
        np.abs(residual),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("ledger absolute errors are inconsistent")
    if not np.allclose(
        output["squared_error"].to_numpy(dtype=float),
        np.square(residual),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("ledger squared errors are inconsistent")
    key_columns = [
        "candidate",
        "fold_id",
        "station",
        "date",
        "beta",
        "model_config_id",
    ]
    if output.duplicated(key_columns).any():
        raise ValueError("raw ledger contains duplicate model/row predictions")
    if not np.isfinite(output[["truth", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError("ledger truth or prediction is non-finite")
    return output


def collapse_primary_repeats(ledger: pd.DataFrame) -> pd.DataFrame:
    """Average squared error, never validation rows, across seed repeats."""
    frame = validate_ledger(ledger)
    invariant = ["year", "month", "truth"]
    grouped = frame.groupby(PRIMARY_REPEAT_KEYS, sort=True, dropna=False)
    for column in invariant:
        counts = grouped[column].nunique(dropna=False)
        if (counts != 1).any():
            raise ValueError(f"repeat rows disagree on {column}")
    collapsed = grouped.agg(
        year=("year", "first"),
        month=("month", "first"),
        truth=("truth", "first"),
        mean_squared_error=("squared_error", "mean"),
        repeat_count=("squared_error", "size"),
    ).reset_index()
    return collapsed


def collapse_secondary_repeats(ledger: pd.DataFrame) -> pd.DataFrame:
    """Average predictions first and derive one residual per validation row."""
    frame = validate_ledger(ledger)
    grouped = frame.groupby(PRIMARY_REPEAT_KEYS, sort=True, dropna=False)
    for column in ("year", "month", "truth"):
        if (grouped[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"repeat rows disagree on {column}")
    collapsed = grouped.agg(
        year=("year", "first"),
        month=("month", "first"),
        truth=("truth", "first"),
        prediction=("prediction", "mean"),
        repeat_count=("prediction", "size"),
    ).reset_index()
    collapsed["residual"] = collapsed["truth"] - collapsed["prediction"]
    collapsed["absolute_error"] = collapsed["residual"].abs()
    collapsed["squared_error"] = collapsed["residual"].pow(2)
    return collapsed


def coverage_signature(ledger: pd.DataFrame, candidate: str) -> pd.DataFrame:
    subset = collapse_primary_repeats(ledger.loc[ledger["candidate"] == candidate])
    return subset[
        [
            "fold_family",
            "outer_origin",
            "station",
            "date",
            "beta",
            "repeat_count",
        ]
    ].sort_values(
        ["fold_family", "outer_origin", "station", "date", "beta"]
    ).reset_index(drop=True)


def assert_candidate_coverage(
    ledger: pd.DataFrame,
    candidate: str,
    *,
    reference: str = "V0",
) -> None:
    candidate_coverage = coverage_signature(ledger, candidate)
    reference_coverage = coverage_signature(ledger, reference)
    comparison_columns = [
        "fold_family",
        "outer_origin",
        "station",
        "date",
    ]
    candidate_rows = candidate_coverage[comparison_columns].sort_values(
        comparison_columns
    ).reset_index(drop=True)
    reference_rows = reference_coverage[comparison_columns].sort_values(
        comparison_columns
    ).reset_index(drop=True)
    if not candidate_rows.equals(reference_rows):
        raise ValueError(f"OOF row coverage for {candidate} differs from {reference}")
    candidate_counts = (
        candidate_coverage.set_index(comparison_columns)["repeat_count"].sort_index()
    )
    reference_counts = (
        reference_coverage.set_index(comparison_columns)["repeat_count"].sort_index()
    )
    if not candidate_counts.equals(reference_counts):
        raise ValueError(f"repeat coverage for {candidate} differs from {reference}")


def relabel_candidate(ledger: pd.DataFrame, candidate: str) -> pd.DataFrame:
    """Relabel cached predictions while retaining a traceable configuration ID."""
    output = validate_ledger(ledger).copy()
    old_candidate = output["candidate"].astype(str)
    old_model_id = output["model_config_id"].astype(str)
    output["candidate"] = str(candidate)
    output["model_config_id"] = [
        stable_json_hash(
            {
                "candidate": str(candidate),
                "source_candidate": source_candidate,
                "source_model_config_id": source_model_id,
            }
        )
        for source_candidate, source_model_id in zip(old_candidate, old_model_id)
    ]
    return validate_ledger(output)
