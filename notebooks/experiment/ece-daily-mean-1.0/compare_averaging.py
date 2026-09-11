"""Daily-mean definition comparison for ECE data.

Compares four daily estimators computed from the same sub-hourly samples:
simple_mean (current ece_pipe.py), hourly_weighted_mean (mean of hourly means,
mirrors USCRN equal-hour weighting), median, and midnight_snapshot
(closest sample to 00:00 local). All values in fraction m3/m3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_estimators(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns parsed_time, sm_pct, date, hour."""
    work = df.copy()
    work["sm_frac"] = work["sm_pct"] / 100.0
    rows = []
    for date, sub in work.groupby("date"):
        sub = sub.sort_values("parsed_time")
        simple_mean = sub["sm_frac"].mean()
        hourly_means = sub.groupby("hour")["sm_frac"].mean()
        hourly_weighted = hourly_means.mean()
        median = sub["sm_frac"].median()
        midnight = pd.to_datetime(date)
        snap_idx = (sub["parsed_time"] - midnight).abs().idxmin()
        snapshot = float(sub.loc[snap_idx, "sm_frac"])
        rows.append(
            {
                "date": pd.to_datetime(date),
                "n": int(len(sub)),
                "n_hours": int(sub["hour"].nunique()),
                "simple_mean": float(simple_mean),
                "hourly_weighted_mean": float(hourly_weighted),
                "median": float(median),
                "midnight_snapshot": float(snapshot),
                "diurnal_range": float(sub.groupby("hour")["sm_frac"].mean().max()
                                       - sub.groupby("hour")["sm_frac"].mean().min())
                if sub["hour"].nunique() > 1 else 0.0,
            }
        )
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["simple_minus_hourly"] = out["simple_mean"] - out["hourly_weighted_mean"]
    out["simple_minus_median"] = out["simple_mean"] - out["median"]
    out["simple_minus_snapshot"] = out["simple_mean"] - out["midnight_snapshot"]
    return out


def summarize_deltas(est: pd.DataFrame) -> pd.DataFrame:
    """Max/mean absolute deltas between simple mean and alternatives."""
    return pd.DataFrame(
        [
            {
                "comparison": "simple vs hourly_weighted",
                "mean_abs_delta": float(est["simple_minus_hourly"].abs().mean()),
                "max_abs_delta": float(est["simple_minus_hourly"].abs().max()),
            },
            {
                "comparison": "simple vs median",
                "mean_abs_delta": float(est["simple_minus_median"].abs().mean()),
                "max_abs_delta": float(est["simple_minus_median"].abs().max()),
            },
            {
                "comparison": "simple vs midnight_snapshot",
                "mean_abs_delta": float(est["simple_minus_snapshot"].abs().mean()),
                "max_abs_delta": float(est["simple_minus_snapshot"].abs().max()),
            },
        ]
    )
