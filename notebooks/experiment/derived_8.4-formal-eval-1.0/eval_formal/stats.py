"""Statistical machinery for derived_8.4-formal-eval-1.0.

Three layers of inference (see README.md for the full write-up):

1. Seed-level (fitting stochasticity under the frozen temporal split):
   - per (config, metric): mean, std, median, min/max, 95% t-CI over seeds;
   - pairwise A vs B: mean difference, paired t-test, Wilcoxon signed-rank,
     % seeds where A beats B.
2. Sample-level (test-set sampling variability): paired cluster bootstrap over
   (station, month) blocks — resample blocks with replacement, recompute pooled
   metrics from per-block sufficient statistics, percentile 95% CI, two-sided
   bootstrap p-value for pairwise differences. Paired across models (same blocks).
   i.i.d. bootstrap is invalid on autocorrelated daily soil-moisture series; month
   blocks respect within-block dependence (sensitivity: (station, year) blocks).
3. Multiplicity: Benjamini-Hochberg FDR over the reported comparison family.

LOSO spatial: win counts ("A beats B on k of 7 stations", per-station median over
seeds), two-sided sign test, and paired t-test / Wilcoxon on the 7 per-station
medians (n = 7 — low power; 6/7 wins is NOT significant at 0.05, 7/7 is p ~ 0.016).

Self-tests (run_self_tests) verify each function against known answers on synthetic
data before the expensive GPU run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

METRIC_KEYS = ("r2", "rmse", "ubrmse", "bias", "mae", "pearson")

# ---------------------------------------------------------------------------
# Seed-level statistics
# ---------------------------------------------------------------------------


def seed_summary(series: pd.Series, alpha: float = 0.05) -> dict[str, float]:
    """Mean / std / median / min / max / 95% t-CI of a per-seed metric series."""
    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan")}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    if n > 1 and std == std:
        tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        half = tcrit * std / np.sqrt(n)
    else:
        half = float("nan")
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "ci_low": mean - half if half == half else float("nan"),
        "ci_high": mean + half if half == half else float("nan"),
    }


def paired_test(a: pd.Series, b: pd.Series, alpha: float = 0.05) -> dict[str, float]:
    """Paired tests on per-seed metric differences (A - B), higher-is-better metrics
    are sign-flipped by the caller for RMSE/MAE/bias (see compare_metrics)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return {"n": float(n), "mean_diff": float("nan"), "std_diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "t_p": float("nan"),
                "wilcoxon_p": float("nan"), "pct_a_better": float("nan")}
    diff = x - y
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    half = tcrit * std_diff / np.sqrt(n)
    if std_diff > 0:
        t_stat, t_p = stats.ttest_rel(x, y)
        try:
            _, w_p = stats.wilcoxon(x, y)
        except ValueError:
            w_p = float("nan")
    elif mean_diff != 0:
        # Constant nonzero difference across all seeds: perfectly consistent.
        t_p = 0.0
        w_p = 0.0
    else:
        t_p = 1.0
        w_p = 1.0
    return {
        "n": float(n),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_low": mean_diff - half,
        "ci_high": mean_diff + half,
        "t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "pct_a_better": float(np.mean(diff > 0) * 100.0),
    }


# ---------------------------------------------------------------------------
# Sample-level: paired cluster bootstrap over (station, month) blocks
# ---------------------------------------------------------------------------


def _block_aggregates(y_true: np.ndarray, y_pred: np.ndarray,
                      block_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-block sufficient statistics.

    Returns (block_ids_sorted, stats) where stats columns are
    [n, sum_y, sum_y2, sum_pred, sum_pred2, sum_yp, sum_ad, sum_d]
    (ad = |pred - y|, d = pred - y). Enables O(blocks) metric recomputation per
    bootstrap resample instead of O(samples).
    """
    uniq = np.unique(block_ids)
    stats_out = np.zeros((len(uniq), 8), dtype=float)
    for i, blk in enumerate(uniq):
        mask = block_ids == blk
        y = y_true[mask]
        p = y_pred[mask]
        d = p - y
        stats_out[i, 0] = mask.sum()
        stats_out[i, 1] = y.sum()
        stats_out[i, 2] = (y * y).sum()
        stats_out[i, 3] = p.sum()
        stats_out[i, 4] = (p * p).sum()
        stats_out[i, 5] = (y * p).sum()
        stats_out[i, 6] = np.abs(d).sum()
        stats_out[i, 7] = d.sum()
    return uniq, stats_out


def _metrics_from_agg(agg: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized metric computation from aggregated block stats (shape (B, 8))."""
    n = agg[:, 0]
    sum_y, sum_y2 = agg[:, 1], agg[:, 2]
    sum_p, sum_p2 = agg[:, 3], agg[:, 4]
    sum_yp, sum_ad, sum_d = agg[:, 5], agg[:, 6], agg[:, 7]
    ss_res = sum_y2 - 2 * sum_yp + sum_p2
    ss_tot = sum_y2 - (sum_y * sum_y) / n
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    rmse = np.sqrt(ss_res / n)
    mae = sum_ad / n
    bias = sum_d / n
    return {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias}


def cluster_bootstrap(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray | None,
                      block_ids: np.ndarray, n_resamples: int = 2000, seed: int = 42,
                      alpha: float = 0.05) -> dict:
    """Paired cluster bootstrap over blocks.

    Returns per metric (r2, rmse, mae, bias):
      - model A: {ci_low, ci_high, mean}
      - difference A - B (if y_pred_b given): {ci_low, ci_high, mean, p}
    where p = 2 * min(P(d* <= 0), P(d* >= 0)) (percentile bootstrap p-value).
    """
    rng = np.random.default_rng(seed)
    uniq, agg_a = _block_aggregates(y_true, y_pred_a, block_ids)
    n_blocks = len(uniq)
    idx = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    ma = _metrics_from_agg(agg_a[idx].sum(axis=1))

    def _model_summary(metric: str, vals: np.ndarray) -> dict[str, float]:
        return {
            "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "mean": float(np.mean(vals)),
        }

    result: dict = {}
    for metric in ("r2", "rmse", "mae", "bias"):
        result[metric] = {"A": _model_summary(metric, ma[metric])}
    if y_pred_b is not None:
        _, agg_b = _block_aggregates(y_true, y_pred_b, block_ids)
        mb = _metrics_from_agg(agg_b[idx].sum(axis=1))
        for metric in ("r2", "rmse", "mae", "bias"):
            result[metric]["B"] = _model_summary(metric, mb[metric])
            d = ma[metric] - mb[metric]
            p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
            result[metric]["diff"] = {
                "ci_low": float(np.percentile(d, 100 * alpha / 2)),
                "ci_high": float(np.percentile(d, 100 * (1 - alpha / 2))),
                "mean": float(np.mean(d)),
                "p": float(np.clip(p, 1.0 / n_resamples, 1.0)),
            }
    return result


# ---------------------------------------------------------------------------
# Multiplicity: Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (two-sided)."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.full(n, float("nan"))
    q_prev = 1.0
    for i in range(n - 1, -1, -1):
        q_i = ranked[i] * n / (i + 1)
        q_i = min(q_i, q_prev)
        q_prev = q_i
        q[i] = q_i
    q[order] = q
    return q


# ---------------------------------------------------------------------------
# LOSO: wins, sign test, per-station paired tests
# ---------------------------------------------------------------------------


def sign_test(k: int, n: int) -> float:
    """Two-sided binomial sign test p-value for k wins out of n stations."""
    if n <= 0 or k < 0 or k > n:
        raise ValueError("k and n must satisfy 0 <= k <= n, n > 0")
    # Two-sided: p = 2 * min(P(X <= k), P(X >= k)), capped at 1.
    p_lo = stats.binom.cdf(k, n, 0.5)
    p_hi = stats.binom.sf(k - 1, n, 0.5)
    return float(min(1.0, 2.0 * min(p_lo, p_hi)))


def station_pair_test(med_a: pd.Series, med_b: pd.Series, alpha: float = 0.05) -> dict[str, float]:
    """Paired tests on per-station medians (n = 7 stations, low power — descriptive)."""
    x = np.asarray(med_a, dtype=float)
    y = np.asarray(med_b, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return {"n": float(n), "mean_diff": float("nan"), "t_p": float("nan"),
                "wilcoxon_p": float("nan"), "wins": int(0), "sign_p": float("nan")}
    diff = x - y
    wins = int(np.sum(diff > 0))
    if np.std(diff, ddof=1) > 0:
        t_p = float(stats.ttest_rel(x, y).pvalue)
        try:
            w_p = float(stats.wilcoxon(x, y).pvalue)
        except ValueError:
            w_p = float("nan")
    else:
        t_p = 1.0
        w_p = 1.0
    return {
        "n": float(n),
        "mean_diff": float(np.mean(diff)),
        "t_p": t_p,
        "wilcoxon_p": w_p,
        "wins": wins,
        "sign_p": sign_test(wins, n),
    }


# ---------------------------------------------------------------------------
# Self-tests (run before the expensive GPU run; see README)
# ---------------------------------------------------------------------------


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"self-test failed: {name}")
    print(f"  [ok] {name}", flush=True)


def run_self_tests(verbose: bool = True) -> None:
    """Verify stats functions on synthetic data with known answers."""
    if verbose:
        print("eval_formal.stats self-tests", flush=True)

    rng = np.random.default_rng(7)

    # 1. seed_summary: known mean/std.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = seed_summary(pd.Series(x))
    _check("seed_summary mean", abs(s["mean"] - 3.0) < 1e-12)
    _check("seed_summary std (ddof=1)", abs(s["std"] - np.std(x, ddof=1)) < 1e-12)
    _check("seed_summary median", abs(s["median"] - 3.0) < 1e-12)
    # t-CI: mean ± t_{0.975,4} * std/sqrt(5); t=2.776.
    half = 2.7764451051977987 * np.std(x, ddof=1) / np.sqrt(5)
    _check("seed_summary t-CI", abs(s["ci_low"] - (3.0 - half)) < 1e-9)

    # 2. paired_test on a known difference.
    a = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    b = a - 2.0
    pt = paired_test(pd.Series(a), pd.Series(b))
    _check("paired_test mean_diff", abs(pt["mean_diff"] - 2.0) < 1e-12)
    _check("paired_test t_p (constant diff)", pt["t_p"] < 1e-12)
    _check("paired_test pct_a_better", abs(pt["pct_a_better"] - 100.0) < 1e-9)
    same = paired_test(pd.Series(a), pd.Series(a))
    _check("paired_test identical -> p=1", abs(same["t_p"] - 1.0) < 1e-12)

    # 3. sign_test: 7/7 -> 2*(1/2)^7 = 0.015625; 6/7 -> 0.125.
    _check("sign_test 7/7", abs(sign_test(7, 7) - 0.015625) < 1e-12)
    _check("sign_test 6/7", abs(sign_test(6, 7) - 0.125) < 1e-12)
    _check("sign_test 0/7", abs(sign_test(0, 7) - 0.015625) < 1e-12)

    # 4. bh_fdr: monotone q-values; smallest p gets p*n/1 = 0.001*5 = 0.005.
    pvals = np.array([0.001, 0.02, 0.03, 0.5, 0.8])
    q = bh_fdr(pvals)
    _check("bh_fdr monotone", np.all(np.diff(q[np.argsort(pvals)]) >= -1e-12))
    _check("bh_fdr min p", abs(q[pvals == 0.001] - 0.001 * 5) < 1e-12)

    # 5. cluster_bootstrap: with one block per sample (i.i.d. case) and huge n,
    #    the bootstrap CI must contain the true R2; identical models -> diff p = 1.
    n = 4000
    y = rng.normal(0, 1, n)
    pred = 0.5 * y + rng.normal(0, 1, n) * 0.5
    block_ids = np.arange(n)  # each sample its own block (i.i.d. reference)
    boot = cluster_bootstrap(y, pred, pred, block_ids, n_resamples=2000, seed=42)
    true_r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    _check("bootstrap CI contains true R2",
           boot["r2"]["A"]["ci_low"] <= true_r2 <= boot["r2"]["A"]["ci_high"])
    _check("bootstrap identical models -> diff p=1", abs(boot["r2"]["diff"]["p"] - 1.0) < 1e-6)
    _check("bootstrap diff mean ~ 0",
           abs(boot["r2"]["diff"]["mean"]) < 1e-9)

    # 6. cluster_bootstrap with grouped blocks: adding a constant offset to one
    #    model's predictions must shift bias by that constant.
    pred_b = pred + 0.05
    boot2 = cluster_bootstrap(y, pred, pred_b, block_ids, n_resamples=1000, seed=42)
    _check("bootstrap bias diff", abs(boot2["bias"]["diff"]["mean"] - (-0.05)) < 1e-6)

    # 7. station_pair_test: known wins.
    ma = np.array([0.6, 0.7, 0.8, 0.9, 0.5, 0.4, 0.3])
    mb = np.array([0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2])
    st = station_pair_test(pd.Series(ma), pd.Series(mb))
    _check("station_pair_test wins=7", st["wins"] == 7)
    _check("station_pair_test sign_p 7/7", abs(st["sign_p"] - 0.015625) < 1e-12)

    if verbose:
        print("  all self-tests passed", flush=True)


if __name__ == "__main__":
    run_self_tests()
