from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    return a / (b + eps)


def compute_ndmi(df: pd.DataFrame, nir_col: str = "s2_b8", swir_col: str = "s2_b11", eps: float = 1e-6) -> pd.Series:
    # NDMI = (NIR - SWIR) / (NIR + SWIR)

    nir = df[nir_col].astype(float)
    swir = df[swir_col].astype(float)
    return (nir - swir) / (nir + swir + eps)


def compute_sar_ratio(df: pd.DataFrame, vv_col: str = "s1_vv", vh_col: str = "s1_vh", eps: float = 1e-6) -> pd.Series:
    vv = df[vv_col].astype(float)
    vh = df[vh_col].astype(float)
    return safe_divide(vv, vh, eps=eps)


def compute_api(
    df: pd.DataFrame,
    precip_col: str = "precip_mm",
    decay: float = 0.90,
    group_col: str = "station_id",
    date_col: str = "date",
) -> pd.Series:
    # API_t = P_t + decay * API_{t-1}  (computed per station, time-ordered)

    out = pd.Series(index=df.index, dtype=float)

    for sid, g in df.groupby(group_col, sort=False):
        g = g.sort_values(date_col)
        p = g[precip_col].astype(float).fillna(0.0).values

        api = np.zeros_like(p, dtype=float)
        acc = 0.0
        for i in range(len(p)):
            acc = p[i] + decay * acc
            api[i] = acc

        out.loc[g.index] = api

    return out


def compute_days_since_last_rain(
    df: pd.DataFrame,
    precip_col: str = "precip_mm",
    threshold_mm: float = 0.5,
    group_col: str = "station_id",
    date_col: str = "date", ) -> pd.Series:

    # DSLR_t = number of days since last day where precip >= threshold_mm (per station).
    # Assumes daily-ish data. Uses date differences to count days robustly.

    out = pd.Series(index=df.index, dtype=float)

    for sid, g in df.groupby(group_col, sort=False):
        g = g.sort_values(date_col)
        dates = pd.to_datetime(g[date_col]).values
        p = g[precip_col].astype(float).fillna(0.0).values

        dslr = np.zeros(len(g), dtype=float)
        last_rain_date = None

        for i in range(len(g)):
            if p[i] >= threshold_mm:
                last_rain_date = dates[i]
                dslr[i] = 0.0
            else:
                if last_rain_date is None:
                    # never rained yet in this series
                    dslr[i] = np.nan
                else:
                    dslr[i] = float((dates[i] - last_rain_date) / np.timedelta64(1, "D"))

        out.loc[g.index] = dslr

    # If you hate NaN at the very beginning, you can fill with a big number:
    # out = out.fillna(out.max())
    return out


def rolling_std(
    df: pd.DataFrame,
    col: str,
    window: int = 7,
    group_col: str = "station_id",
    date_col: str = "date",
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = window

    out = pd.Series(index=df.index, dtype=float)
    for sid, g in df.groupby(group_col, sort=False):
        g = g.sort_values(date_col)
        out.loc[g.index] = g[col].astype(float).rolling(window=window, min_periods=min_periods).std().values
    return out


def temporal_gradient(
    df: pd.DataFrame,
    col: str,
    k: int = 7,
    group_col: str = "station_id",
    date_col: str = "date",
) -> pd.Series:
    # grad_t(k) = (x_t - x_{t-k}) / k  (per station)

    out = pd.Series(index=df.index, dtype=float)
    for sid, g in df.groupby(group_col, sort=False):
        g = g.sort_values(date_col)
        x = g[col].astype(float)
        out.loc[g.index] = (x - x.shift(k)) / float(k)
    return out


def train_only_monthly_anomaly(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    col: str,
    date_col: str = "date",
) -> pd.Series:
    # SA_t = x_t - mean_train(x | month)
    # Returns anomaly for full_df using monthly means computed on train_df only.

    train_dates = pd.to_datetime(train_df[date_col])
    full_dates = pd.to_datetime(full_df[date_col])

    train_month = train_dates.dt.month
    full_month = full_dates.dt.month

    train_means = train_df.groupby(train_month)[col].mean()

    # fallback mean if a month doesn't exist in train (rare)
    fallback = float(train_df[col].mean())

    means_for_full = full_month.map(lambda m: train_means.get(m, fallback)).astype(float)
    return full_df[col].astype(float) - means_for_full.values
