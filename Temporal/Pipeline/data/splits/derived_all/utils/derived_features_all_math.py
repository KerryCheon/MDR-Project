# derived_features_all_math.py

from __future__ import annotations

import numpy as np
import pandas as pd

def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    return a / (b + eps)

def _group_sorted(df: pd.DataFrame, group_col: str, date_col: str):
    for sid, g in df.groupby(group_col, sort=False):
        yield sid, g.sort_values(date_col).copy()

def compute_ndvi(df: pd.DataFrame, nir_col: str = "s2_b8", red_col: str = "s2_b4", eps: float = 1e-6) -> pd.Series:
    nir = df[nir_col].astype(float)
    red = df[red_col].astype(float)
    return (nir - red) / (nir + red + eps)

def compute_ndmi(df: pd.DataFrame, nir_col: str = "s2_b8", swir_col: str = "s2_b11", eps: float = 1e-6) -> pd.Series:
    nir = df[nir_col].astype(float)
    swir = df[swir_col].astype(float)
    return (nir - swir) / (nir + swir + eps)

def compute_msi(df: pd.DataFrame, swir_col: str = "s2_b11", nir_col: str = "s2_b8", eps: float = 1e-6) -> pd.Series:
    swir = df[swir_col].astype(float)
    nir = df[nir_col].astype(float)
    return safe_divide(swir, nir, eps=eps)

def compute_sar_ratio(df: pd.DataFrame, vv_col: str = "s1_vv", vh_col: str = "s1_vh", eps: float = 1e-6) -> pd.Series:
    vv = df[vv_col].astype(float)
    vh = df[vh_col].astype(float)
    return safe_divide(vv, vh, eps=eps)

def compute_sar_diff(df: pd.DataFrame, vv_col: str = "s1_vv", vh_col: str = "s1_vh") -> pd.Series:
    vv = df[vv_col].astype(float)
    vh = df[vh_col].astype(float)
    return vv - vh

def compute_api(
    df: pd.DataFrame,
    precip_col: str = "precip_mm",
    decay: float = 0.90,
    group_col: str = "station_id",
    date_col: str = "date",
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        p = np.nan_to_num(g[precip_col].astype(float).values, nan=0.0, posinf=0.0, neginf=0.0)
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
    date_col: str = "date",
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        dates = pd.to_datetime(g[date_col]).values
        p = np.nan_to_num(g[precip_col].astype(float).values, nan=0.0, posinf=0.0, neginf=0.0)
        dslr = np.zeros(len(g), dtype=float)
        last_rain_date = None
        for i in range(len(g)):
            if p[i] >= threshold_mm:
                last_rain_date = dates[i]
                dslr[i] = 0.0
            else:
                if last_rain_date is None:
                    dslr[i] = np.nan
                else:
                    dslr[i] = float((dates[i] - last_rain_date) / np.timedelta64(1, "D"))
        out.loc[g.index] = dslr
    return out

def compute_rain_sums_days(
    df: pd.DataFrame,
    precip_col: str = "precip_mm",
    window_days: int = 7,
    group_col: str = "station_id",
    date_col: str = "date",
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        g2 = g[[date_col, precip_col]].copy()
        g2[date_col] = pd.to_datetime(g2[date_col])
        g2[precip_col] = np.nan_to_num(g2[precip_col].astype(float).values, nan=0.0, posinf=0.0, neginf=0.0)
        sums = np.zeros(len(g2), dtype=float)
        dates = g2[date_col].values
        p = g2[precip_col].values
        for i in range(len(g2)):
            start = dates[i] - np.timedelta64(window_days, "D")
            j = np.searchsorted(dates, start, side="left")
            sums[i] = float(np.sum(p[j : i + 1]))
        out.loc[g.index] = sums
    return out

def series_lags(df: pd.DataFrame, col: str, lag_kobs: int, group_col: str = "station_id", date_col: str = "date") -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        x = g[col].astype(float)
        out.loc[g.index] = x.shift(lag_kobs).values
    return out

def series_diffs(df: pd.DataFrame, col: str, kobs: int = 1, group_col: str = "station_id", date_col: str = "date") -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        x = g[col].astype(float)
        out.loc[g.index] = (x - x.shift(kobs)).values
    return out

def series_pct_change(
    df: pd.DataFrame,
    col: str,
    group_col: str = "station_id",
    date_col: str = "date",
    eps: float = 1e-6,
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        x = g[col].astype(float)
        prev = x.shift(1)
        out.loc[g.index] = safe_divide(x - prev, prev, eps=eps).values
    return out

def series_gradient_kobs(
    df: pd.DataFrame,
    col: str,
    kobs: int = 7,
    group_col: str = "station_id",
    date_col: str = "date",
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        x = g[col].astype(float)
        out.loc[g.index] = ((x - x.shift(kobs)) / float(kobs)).values
    return out

def rolling_mean(
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
    for _, g in _group_sorted(df, group_col, date_col):
        out.loc[g.index] = g[col].astype(float).rolling(window=window, min_periods=min_periods).mean().values
    return out

def rolling_std(
    df: pd.DataFrame,
    col: str,
    window: int = 7,
    group_col: str = "station_id",
    date_col: str = "date",
    ddof: int = 0,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = window
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        out.loc[g.index] = g[col].astype(float).rolling(window=window, min_periods=min_periods).std(ddof=ddof).values
    return out

def rolling_min(
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
    for _, g in _group_sorted(df, group_col, date_col):
        out.loc[g.index] = g[col].astype(float).rolling(window=window, min_periods=min_periods).min().values
    return out

def rolling_max(
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
    for _, g in _group_sorted(df, group_col, date_col):
        out.loc[g.index] = g[col].astype(float).rolling(window=window, min_periods=min_periods).max().values
    return out

def rolling_range(
    df: pd.DataFrame,
    col: str,
    window: int = 7,
    group_col: str = "station_id",
    date_col: str = "date",
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = window
    mn = rolling_min(df, col=col, window=window, group_col=group_col, date_col=date_col, min_periods=min_periods)
    mx = rolling_max(df, col=col, window=window, group_col=group_col, date_col=date_col, min_periods=min_periods)
    return mx - mn

def rolling_cv(
    df: pd.DataFrame,
    col: str,
    window: int = 7,
    group_col: str = "station_id",
    date_col: str = "date",
    eps: float = 1e-6,
    ddof: int = 0,
    min_periods: int | None = None,
) -> pd.Series:
    mu = rolling_mean(df, col=col, window=window, group_col=group_col, date_col=date_col, min_periods=min_periods)
    sd = rolling_std(df, col=col, window=window, group_col=group_col, date_col=date_col, ddof=ddof, min_periods=min_periods)
    return safe_divide(sd, mu, eps=eps)

def ema(
    df: pd.DataFrame,
    col: str,
    alpha: float = 0.2,
    group_col: str = "station_id",
    date_col: str = "date",
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        x = g[col].astype(float)
        out.loc[g.index] = x.ewm(alpha=alpha, adjust=False).mean().values
    return out

def smm_index(
    df: pd.DataFrame,
    col: str,
    alpha: float = 0.85,
    n_lags: int = 5,
    group_col: str = "station_id",
    date_col: str = "date",
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        x = g[col].astype(float).values
        smm = np.zeros(len(g), dtype=float)
        for i in range(len(g)):
            acc = 0.0
            for j in range(1, n_lags + 1):
                if i - j < 0:
                    continue
                acc += (alpha ** j) * float(x[i - j])
            smm[i] = acc
        out.loc[g.index] = smm
    return out

def train_only_monthly_anomaly_global(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    col: str,
    date_col: str = "date",
) -> pd.Series:
    train_dates = pd.to_datetime(train_df[date_col])
    full_dates = pd.to_datetime(full_df[date_col])
    train_month = train_dates.dt.month
    full_month = full_dates.dt.month
    train_means = train_df.groupby(train_month)[col].mean()
    fallback = float(train_df[col].mean())
    means_for_full = full_month.map(lambda m: train_means.get(m, fallback)).astype(float)
    return full_df[col].astype(float) - means_for_full.values

def train_only_monthly_zscore_global(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    col: str,
    date_col: str = "date",
    eps: float = 1e-6,
) -> pd.Series:
    train_dates = pd.to_datetime(train_df[date_col])
    full_dates = pd.to_datetime(full_df[date_col])
    train_month = train_dates.dt.month
    full_month = full_dates.dt.month
    mu = train_df.groupby(train_month)[col].mean()
    sd = train_df.groupby(train_month)[col].std(ddof=0)
    mu_fallback = float(train_df[col].mean())
    sd_fallback = float(train_df[col].std(ddof=0))
    if not np.isfinite(sd_fallback) or sd_fallback < eps:
        sd_fallback = 1.0
    mu_for_full = full_month.map(lambda m: mu.get(m, mu_fallback)).astype(float)
    sd_for_full = full_month.map(lambda m: sd.get(m, sd_fallback)).astype(float).clip(lower=eps)
    return (full_df[col].astype(float) - mu_for_full.values) / sd_for_full.values

def compute_time_since_last_spike(
    df: pd.DataFrame,
    diff_col: str,
    zthr: float = 2.0,
    group_col: str = "station_id",
    date_col: str = "date",
) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in _group_sorted(df, group_col, date_col):
        d = g[diff_col].astype(float).values
        dates = pd.to_datetime(g[date_col]).values
        mu = np.nanmean(d)
        sd = np.nanstd(d)
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        z = (d - mu) / sd
        ts = np.zeros(len(g), dtype=float)
        last = None
        for i in range(len(g)):
            if np.isfinite(z[i]) and z[i] >= zthr:
                last = dates[i]
                ts[i] = 0.0
            else:
                if last is None:
                    ts[i] = np.nan
                else:
                    ts[i] = float((dates[i] - last) / np.timedelta64(1, "D"))
        out.loc[g.index] = ts
    return out
