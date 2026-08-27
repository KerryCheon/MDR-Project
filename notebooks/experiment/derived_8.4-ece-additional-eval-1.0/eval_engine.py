"""Core evaluation engine and statistics for derived_8.4-ece-additional-eval-1.0."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_data_root(config: Dict[str, Any], exp_dir: Path) -> Path:
    """Resolve data root directory."""
    raw_root = Path(config["datasets"]["root"])
    if raw_root.is_absolute():
        return raw_root
    return (exp_dir / raw_root).resolve()


def compute_sample_weights(df: pd.DataFrame, beta: float = 0.2) -> np.ndarray:
    """Compute exponential year weights matching MDR-v25."""
    df_date = pd.to_datetime(df["date"], errors="coerce")
    years = df_date.dt.year.astype(float)
    max_year = years.max()
    weights = np.exp(beta * (years - max_year))
    weights = weights / weights.mean()
    return weights.to_numpy()


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    prefix: str = "",
) -> Dict[str, float]:
    """Compute comprehensive regression metrics."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # Filter out NaNs if present
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    n = len(y_true)
    if n == 0:
        return {
            f"{prefix}n": 0,
            f"{prefix}r2": np.nan,
            f"{prefix}rmse": np.nan,
            f"{prefix}mae": np.nan,
            f"{prefix}ubrmse": np.nan,
            f"{prefix}bias": np.nan,
            f"{prefix}pearson_r": np.nan,
            f"{prefix}med_ae": np.nan,
            f"{prefix}p90_ae": np.nan,
        }

    err = y_true - y_pred
    ae = np.abs(err)

    r2 = float(r2_score(y_true, y_pred)) if n >= 2 and np.var(y_true) > 1e-12 else np.nan
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(err))
    ubrmse = float(np.std(err))

    if n >= 2 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        pr, _ = stats.pearsonr(y_true, y_pred)
        pearson_r = float(pr)
    else:
        pearson_r = np.nan

    return {
        f"{prefix}n": int(n),
        f"{prefix}r2": r2,
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}ubrmse": ubrmse,
        f"{prefix}bias": bias,
        f"{prefix}pearson_r": pearson_r,
        f"{prefix}med_ae": float(np.median(ae)),
        f"{prefix}p90_ae": float(np.quantile(ae, 0.90)),
    }


def compute_per_station_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target_col: str = "soil_moisture_5cm",
    station_col: str = "station_id",
) -> pd.DataFrame:
    """Compute metrics individually for each station."""
    rows = []
    stations = sorted(df[station_col].unique())
    for st in stations:
        mask = (df[station_col] == st).to_numpy()
        sub_true = df.loc[mask, target_col].to_numpy()
        sub_pred = y_pred[mask]
        m = compute_metrics(sub_true, sub_pred)
        m["station_id"] = st
        rows.append(m)
    return pd.DataFrame(rows)


def seed_summary(series: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Compute descriptive statistics and 95% t-confidence intervals over random seeds."""
    arr = np.asarray(series, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med = float(np.median(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if n > 1 and std > 1e-12:
        se = std / np.sqrt(n)
        tc = stats.t.ppf(0.975, df=n - 1)
        ci_low = float(mean - tc * se)
        ci_high = float(mean + tc * se)
    else:
        ci_low, ci_high = mean, mean
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": med,
        "min": mn,
        "max": mx,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def paired_hypothesis_test(
    a_values: Union[pd.Series, np.ndarray],
    b_values: Union[pd.Series, np.ndarray],
    metric_name: str = "r2",
) -> Dict[str, Any]:
    """Compute paired t-test, Wilcoxon test, sign test, and effect size."""
    a = np.asarray(a_values, dtype=float).ravel()
    b = np.asarray(b_values, dtype=float).ravel()
    valid = ~(np.isnan(a) | np.isnan(b))
    a = a[valid]
    b = b[valid]
    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff)) if n > 0 else np.nan
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    if n > 1 and std_diff > 1e-12:
        se = std_diff / np.sqrt(n)
        tc = stats.t.ppf(0.975, df=n - 1)
        ci_low = float(mean_diff - tc * se)
        ci_high = float(mean_diff + tc * se)
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except Exception:
            w_p = np.nan
    else:
        ci_low, ci_high = mean_diff, mean_diff
        t_p, w_p = np.nan, np.nan

    wins = int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))
    effective_n = n - ties
    if effective_n > 0:
        sign_p = float(stats.binomtest(wins, effective_n, 0.5).pvalue)
    else:
        sign_p = 1.0

    cohen_d = float(mean_diff / std_diff) if std_diff > 1e-12 else np.nan

    return {
        "n": n,
        "metric": metric_name,
        "mean_A": float(np.mean(a)) if n > 0 else np.nan,
        "mean_B": float(np.mean(b)) if n > 0 else np.nan,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_p": float(t_p) if not np.isnan(t_p) else np.nan,
        "wilcoxon_p": float(w_p) if not np.isnan(w_p) else np.nan,
        "sign_p": sign_p,
        "wins": wins,
        "pct_A_better": float(wins / n * 100.0) if n > 0 else np.nan,
        "cohen_d": cohen_d,
    }
