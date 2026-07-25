"""Deterministic paired hierarchical bootstrap over stations and origins."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .ledger import collapse_primary_repeats


BOOTSTRAP_METRICS = (
    "combined_primary_rmse",
    "forward_time_rmse",
    "station_time_rmse",
    "p90_station_year_rmse",
    "worst_station_rmse",
    "p90_month_rmse",
)


def _aligned_primary_rows(
    ledger: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    candidate_beta: float,
    reference_beta: float,
) -> pd.DataFrame:
    candidate_rows = collapse_primary_repeats(
        ledger.loc[
            (ledger["candidate"] == candidate)
            & (ledger["beta"] == float(candidate_beta))
        ]
    )
    reference_rows = collapse_primary_repeats(
        ledger.loc[
            (ledger["candidate"] == reference)
            & (ledger["beta"] == float(reference_beta))
        ]
    )
    keys = [
        "fold_family",
        "outer_origin",
        "station",
        "date",
        "year",
        "month",
        "truth",
    ]
    merged = candidate_rows.merge(
        reference_rows,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
        indicator=True,
    )
    if merged.empty or not (merged["_merge"] == "both").all():
        raise ValueError(f"bootstrap rows for {candidate} and {reference} are unpaired")
    if not np.array_equal(
        merged["repeat_count_candidate"].to_numpy(),
        merged["repeat_count_reference"].to_numpy(),
    ):
        raise ValueError("bootstrap repeat coverage differs from reference")
    return merged.drop(columns="_merge")


def _risk_metrics(rows: pd.DataFrame, error_column: str) -> dict[str, float]:
    block = (
        rows.groupby(["fold_family", "station", "outer_origin"], sort=False)[
            error_column
        ]
        .mean()
        .pow(0.5)
        .rename("rmse")
        .reset_index()
    )
    family = block.groupby("fold_family", sort=False)["rmse"].mean()
    if not {"forward_time", "station_time"}.issubset(family.index):
        raise ValueError("bootstrap sample lost a fold family")
    station = (
        rows.groupby("station", sort=False)[error_column].mean().pow(0.5)
    )
    month = rows.groupby("month", sort=False)[error_column].mean().pow(0.5)
    return {
        "combined_primary_rmse": float(
            0.5 * family["forward_time"] + 0.5 * family["station_time"]
        ),
        "forward_time_rmse": float(family["forward_time"]),
        "station_time_rmse": float(family["station_time"]),
        "p90_station_year_rmse": float(block["rmse"].quantile(0.9)),
        "worst_station_rmse": float(station.max()),
        "p90_month_rmse": float(month.quantile(0.9)),
    }


def _sample_hierarchy(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    stations = np.asarray(sorted(rows["station"].unique()), dtype=object)
    sampled_stations = rng.choice(stations, size=len(stations), replace=True)
    pieces = []
    for sampled_position, station in enumerate(sampled_stations):
        station_rows = rows.loc[rows["station"] == station]
        origins = np.asarray(sorted(station_rows["outer_origin"].unique()), dtype=int)
        sampled_origins = rng.choice(origins, size=len(origins), replace=True)
        for origin_position, origin in enumerate(sampled_origins):
            piece = station_rows.loc[station_rows["outer_origin"] == int(origin)].copy()
            piece["station"] = f"s{sampled_position}:{station}"
            piece["outer_origin"] = origin_position
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def paired_hierarchical_bootstrap(
    ledger: pd.DataFrame,
    candidate: str,
    reference: str = "V0",
    *,
    beta: float = 0.0,
    candidate_beta: float | None = None,
    reference_beta: float | None = None,
    replicates: int = 2000,
    seed: int = 42,
) -> dict:
    if replicates < 2:
        raise ValueError("bootstrap needs at least two replicates")
    candidate_beta = float(beta if candidate_beta is None else candidate_beta)
    reference_beta = float(beta if reference_beta is None else reference_beta)
    aligned = _aligned_primary_rows(
        ledger,
        candidate,
        reference,
        candidate_beta=candidate_beta,
        reference_beta=reference_beta,
    )
    candidate_point = _risk_metrics(aligned, "mean_squared_error_candidate")
    reference_point = _risk_metrics(aligned, "mean_squared_error_reference")
    rng = np.random.default_rng(int(seed))
    distributions = {metric: [] for metric in BOOTSTRAP_METRICS}
    candidate_risk = []
    for _ in range(int(replicates)):
        sample = _sample_hierarchy(aligned, rng)
        candidate_metrics = _risk_metrics(sample, "mean_squared_error_candidate")
        reference_metrics = _risk_metrics(sample, "mean_squared_error_reference")
        candidate_risk.append(candidate_metrics["combined_primary_rmse"])
        for metric in BOOTSTRAP_METRICS:
            distributions[metric].append(
                candidate_metrics[metric] - reference_metrics[metric]
            )
    comparisons = {}
    for metric, values in distributions.items():
        array = np.asarray(values, dtype=float)
        comparisons[metric] = {
            "candidate": candidate_point[metric],
            "reference": reference_point[metric],
            "delta": candidate_point[metric] - reference_point[metric],
            "ci_lower": float(np.quantile(array, 0.025)),
            "ci_upper": float(np.quantile(array, 0.975)),
            "bootstrap_standard_error": float(np.std(array, ddof=1)),
        }
    return {
        "candidate": candidate,
        "reference": reference,
        "beta": (
            float(candidate_beta)
            if candidate_beta == reference_beta
            else None
        ),
        "candidate_beta": float(candidate_beta),
        "reference_beta": float(reference_beta),
        "replicates": int(replicates),
        "seed": int(seed),
        "comparisons": comparisons,
        "candidate_primary_bootstrap_standard_error": float(
            np.std(np.asarray(candidate_risk), ddof=1)
        ),
    }


def paired_array_interval(
    candidate,
    reference,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    replicates: int = 2000,
    seed: int = 42,
) -> dict:
    """Small paired helper used for station-only and method-screen diagnostics."""
    left = np.asarray(candidate, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired bootstrap arrays must be aligned and nonempty")
    rng = np.random.default_rng(int(seed))
    deltas = []
    for _ in range(int(replicates)):
        indices = rng.integers(0, len(left), size=len(left))
        deltas.append(float(statistic(left[indices]) - statistic(right[indices])))
    values = np.asarray(deltas)
    return {
        "delta": float(statistic(left) - statistic(right)),
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "bootstrap_standard_error": float(np.std(values, ddof=1)),
        "replicates": int(replicates),
        "seed": int(seed),
    }
