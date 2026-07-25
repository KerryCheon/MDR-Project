"""Primary station-year risk and variance-aware secondary diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .ledger import collapse_primary_repeats, collapse_secondary_repeats


def _pearson(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.std(truth) <= np.finfo(float).eps:
        return float("nan")
    if np.std(prediction) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(truth, prediction)[0, 1])


def metric_record(
    truth: Iterable[float],
    prediction: Iterable[float],
    *,
    variance_epsilon: float = 1e-15,
) -> dict:
    y = np.asarray(list(truth), dtype=float)
    pred = np.asarray(list(prediction), dtype=float)
    if len(y) != len(pred) or not len(y):
        raise ValueError("metrics require aligned nonempty truth and prediction")
    residual = y - pred
    target_variance = float(np.var(y))
    target_std = float(np.std(y))
    target_range = float(np.max(y) - np.min(y))
    q25, q75 = np.quantile(y, [0.25, 0.75])
    if target_variance <= float(variance_epsilon):
        r2 = float("nan")
        r2_reason = "zero_target_variance"
    else:
        r2 = float(1.0 - np.sum(np.square(residual)) / np.sum(np.square(y - y.mean())))
        r2_reason = "defined"
    return {
        "target_count": int(len(y)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "target_range": target_range,
        "target_variance": target_variance,
        "target_standard_deviation": target_std,
        "target_iqr": float(q75 - q25),
        "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "R2": r2,
        "R2_reason": r2_reason,
        "MAE": float(np.mean(np.abs(residual))),
        "Pearson": _pearson(y, pred),
        "Bias": float(np.mean(residual)),
    }


def primary_risk(ledger: pd.DataFrame, candidate: str, *, beta: float) -> dict:
    collapsed = collapse_primary_repeats(
        ledger.loc[
            (ledger["candidate"] == candidate) & (ledger["beta"] == float(beta))
        ]
    )
    if collapsed.empty:
        raise ValueError(f"no OOF rows for {candidate}, beta={beta}")
    blocks = (
        collapsed.groupby(
            ["fold_family", "station", "outer_origin"], sort=True
        )["mean_squared_error"]
        .mean()
        .pow(0.5)
        .rename("station_year_rmse")
        .reset_index()
    )
    families = {
        family: float(group["station_year_rmse"].mean())
        for family, group in blocks.groupby("fold_family", sort=True)
    }
    required = {"forward_time", "station_time"}
    if set(families) != required:
        raise ValueError(f"primary risk requires both fold families; got {families}")
    combined = 0.5 * families["forward_time"] + 0.5 * families["station_time"]
    return {
        "candidate": candidate,
        "beta": float(beta),
        "combined_primary_rmse": float(combined),
        "forward_time_rmse": families["forward_time"],
        "station_time_rmse": families["station_time"],
        "station_year_blocks": blocks,
        "p90_station_year_rmse": float(blocks["station_year_rmse"].quantile(0.9)),
    }


def secondary_metric_tables(
    ledger: pd.DataFrame,
    candidate: str,
    *,
    beta: float,
    variance_epsilon: float = 1e-15,
) -> dict[str, pd.DataFrame]:
    collapsed = collapse_secondary_repeats(
        ledger.loc[
            (ledger["candidate"] == candidate) & (ledger["beta"] == float(beta))
        ]
    )
    if collapsed.empty:
        raise ValueError(f"no secondary rows for {candidate}, beta={beta}")

    def summarize(group_columns: list[str]) -> pd.DataFrame:
        rows = []
        grouped = [((), collapsed)] if not group_columns else collapsed.groupby(
            group_columns, sort=True, dropna=False
        )
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_columns, keys))
            row.update(
                metric_record(
                    group["truth"],
                    group["prediction"],
                    variance_epsilon=variance_epsilon,
                )
            )
            rows.append(row)
        return pd.DataFrame(rows)

    overall = summarize([])
    station = summarize(["station"])
    month = summarize(["month"])
    year = summarize(["outer_origin"])
    station_year = summarize(["station", "outer_origin"])
    station_macro = float(station["RMSE"].mean())
    worst_station = float(station["RMSE"].max())
    p90_month = float(month["RMSE"].quantile(0.9))
    overall["station_macro_RMSE"] = station_macro
    overall["worst_station_RMSE"] = worst_station
    overall["p90_station_year_RMSE"] = float(
        station_year["RMSE"].quantile(0.9)
    )
    overall["p90_month_RMSE"] = p90_month
    return {
        "overall": overall,
        "station": station,
        "month": month,
        "year": year,
        "station_year": station_year,
    }

