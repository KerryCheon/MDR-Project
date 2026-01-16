# Jakob Balkovec
# Scoring

import numpy as np
import pandas as pd

from Modeling.Utils.logging import get_logger


def _get(d, key, default=float("nan")):
    if d is None:
        return default
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def compute_score(metric_rows, weights=None, k=None, prefer_split="val", metric="r2"):
    log = get_logger("ranking.score")

    weights = weights or {}
    w_mean = float(weights.get("mean_r2", 1.0))
    w_std = float(weights.get("std_r2", -0.5))
    w_gap = float(weights.get("gap", -0.2))
    w_k = float(weights.get("k_penalty", -0.001))

    df = pd.DataFrame(metric_rows).copy()
    if df.empty:
        raise ValueError("compute_score: metric_rows is empty")

    if "split" not in df.columns or metric not in df.columns:
        raise ValueError(f"compute_score: rows must include 'split' and '{metric}'")

    # Use prefer_split first; fallback to any split present
    use_df = df[df["split"] == prefer_split] if prefer_split in set(df["split"]) else df

    vals = use_df[metric].to_numpy()
    vals = vals[np.isfinite(vals)]

    mean_r2 = float(np.mean(vals)) if len(vals) else float("nan")
    std_r2 = float(np.std(vals)) if len(vals) else float("nan")

    # Try to compute train->val gap if both exist
    gap = float("nan")
    if ("train" in set(df["split"])) and (prefer_split in set(df["split"])):
        tr_vals = df[df["split"] == "train"][metric].to_numpy()
        va_vals = df[df["split"] == prefer_split][metric].to_numpy()
        tr_vals = tr_vals[np.isfinite(tr_vals)]
        va_vals = va_vals[np.isfinite(va_vals)]
        if len(tr_vals) and len(va_vals):
            gap = float(np.mean(tr_vals) - np.mean(va_vals))

    k_val = float(k) if k is not None else float("nan")

    score = float("nan")
    if np.isfinite(mean_r2):
        score = w_mean * mean_r2
        if np.isfinite(std_r2):
            score += w_std * std_r2
        if np.isfinite(gap):
            score += w_gap * gap
        if np.isfinite(k_val):
            score += w_k * k_val

    out = {
        "score": score,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "gap": gap,
        "k": k_val,
        "weights": {"mean_r2": w_mean, "std_r2": w_std, "gap": w_gap, "k_penalty": w_k},
        "prefer_split": prefer_split,
        "metric": metric,
    }

    log.info("compute_score: %s", out)
    return out


def rank_experiments(experiments, key="score", descending=True):
    log = get_logger("ranking.score")

    if experiments is None:
        return []

    def _key_fn(x):
        v = x.get(key, float("nan"))
        try:
            v = float(v)
        except Exception:
            v = float("nan")
        if not np.isfinite(v):
            return -np.inf if descending else np.inf
        return v

    ranked = sorted(experiments, key=_key_fn, reverse=descending)
    log.info("rank_experiments: ranked=%d by=%s", len(ranked), key)
    return ranked
