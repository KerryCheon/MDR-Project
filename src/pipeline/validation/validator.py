# Jakob Balkovec
# Validation Runner for Imputation Engines
# Nov 22nd 2025
# validator.py

# Implements a validation framework that masks real values and evaluates imputation accuracy. This way we can
# quantitatively assess how well our imputation methods perform on known data

import numpy as np
import pandas as pd
from collections import Counter

from pipeline.utils.config import load_config
from pipeline.utils.logger import get_logger

class ValidationRunner:
    # desc: Runs validation of imputation by masking real values

    def __init__(self, config=None):
        # pre: p_mask is fraction of real values to hide (0 < p_mask <= 1)
        # post: ValidationRunner stores config for repeatable evaluations
        self.cfg = (config or load_config()).get("validator", {})
        self.log_cfg = self.cfg.get("logging", {})

        seed = self.cfg.get("random_seed", 42)
        p_mask = self.cfg.get("mask_fraction", 0.1)

        self.seed = int(seed)
        self.p_mask = float(p_mask)
        self.logger = get_logger("validator")

    def _mask_values(self, df, col):
        # pre: df[col] exists
        # post: returns df_masked, masked_index
        # desc: masks out a percentage of real observations for validation

        s = df[col]
        real_idx = s.notna()

        real_positions = df.index[real_idx]
        n_real = len(real_positions)

        if n_real == 0:
            self.logger.warning(f"No real observations for column '{col}' — cannot validate.")
            return df.copy(), []

        n_mask = max(1, int(n_real * self.p_mask))

        np.random.seed(self.seed)
        masked_points = np.random.choice(real_positions, size=n_mask, replace=False)

        df_masked = df.copy()
        df_masked.loc[masked_points, col] = np.nan

        self.logger.info(
            f"Validation: masking {n_mask}/{n_real} ({self.p_mask*100:.1f}%) real points for feature '{col}'."
        )

        return df_masked, masked_points

    def _compute_metrics(self, true_vals, pred_vals):
        # pre: arrays of true and predicted values
        # post: returns metrics dict with safe handling for edge cases

        diff = pred_vals - true_vals

        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae  = float(np.mean(np.abs(diff)))

        if len(true_vals) == 0 or np.any(true_vals == 0):
            mape = float("nan")
        else:
            mape = float(np.mean(np.abs(diff / true_vals)))

        bias = float(np.mean(diff))

        if len(true_vals) < 2:
            corr = float("nan")
        else:
            try:
                corr = float(np.corrcoef(pred_vals, true_vals)[0, 1])
            except Exception:
                corr = float("nan")

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "bias": bias,
            "corr": corr,
        }

    def _compute_gap_lengths(self, df, col, masked_points):
        # pre: df has 'date' and col with real values except masked_points
        # post: returns dict mapping masked_index -> gap_length_days

        s = df[col]
        dates = pd.to_datetime(df["date"])

        real_mask = s.notna()
        real_dates = dates[real_mask]

        gap_lengths = {}

        for idx in masked_points:
            t = dates.loc[idx]

            # nearest real before
            before = real_dates[real_dates < t]
            after = real_dates[real_dates > t]

            if len(before) == 0 or len(after) == 0:
                # edge case
                gap_lengths[idx] = 999
                continue

            nearest_before = before.max()
            nearest_after = after.min()

            gap = min(
                (t - nearest_before).days,
                (nearest_after - t).days
            )
            gap_lengths[idx] = int(gap)

        return gap_lengths


    def evaluate(self, df, col, ensemble_fn):
        # pre: df DataFrame, col str, ensemble_fn callable
        # post: returns validation results dict
        # desc: runs validation for a single feature column

        logger = self.logger.getChild(col)
        logger.info(f"Starting validation run for feature '{col}'")

        df_masked, masked_points = self._mask_values(df, col)

        if len(masked_points) == 0:
            logger.warning("No masked points; skipping evaluation.")
            return {"valid": False}

        df_out = ensemble_fn(df_masked, col)

        pred = df_out.loc[masked_points, col + "_interp"].astype(float)
        true = df.loc[masked_points, col].astype(float)

        metrics = self._compute_metrics(true.values, pred.values)

        gap_lengths = self._compute_gap_lengths(df, col, masked_points)

        # aggregate gap--lentghs instead of adding them one by one...no duplicates
        gap_lengths = {idx: gap_lengths[idx] for idx in masked_points}

        # buckets
        buckets = {
            "short_1_3":  lambda g: 1 <= g <= 3,
            "medium_4_7": lambda g: 4 <= g <= 7,
            "long_8_14":  lambda g: 8 <= g <= 14,
            "huge_15+":   lambda g: g > 14,
        }

        stratified = {}

        bucket_defs = {
            "short_1_3":  (1, 3),
            "medium_4_7": (4, 7),
            "long_8_14":  (8, 14),
            "huge_15+":   (15, None),
        }

        for name, (lo, hi) in bucket_defs.items():
            pts = []
            for idx, gap in gap_lengths.items():
                if hi is None:
                    ok = gap >= lo
                else:
                    ok = lo <= gap <= hi
                if ok:
                    pts.append(idx)

            if not pts:
                stratified[name] = {
                    "range": (lo, hi),
                    "n": 0,
                    "percentage": 0.0,
                    "metrics": None,
                }
                continue

            pred_bucket = df_out.loc[pts, col + "_interp"].astype(float)
            true_bucket = df.loc[pts, col].astype(float)

            stratified[name] = {
                "range": (lo, hi),
                "n": len(pts),
                "percentage": len(pts) / len(masked_points),
                "metrics": self._compute_metrics(true_bucket.values, pred_bucket.values),
            }

        logger.info(
            f"Validation complete for '{col}': RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, Bias={metrics['bias']:.4f}"
        )

        return {
            "feature": col,
            "n_masked": int(len(masked_points)),
            "masked_points": [str(x) for x in masked_points],
            "metrics": metrics,
            "stratified": stratified,
            "valid": True,
        }


def compute_all_gap_lengths(df, col):
    # pre: df has 'date' and col
    # post: returns list of gap lengths in days (ints)
    # desc: computes all natural gaps in the dataset for a given column

    s = df[col]
    dates = pd.to_datetime(df["date"])

    real_mask = s.notna()
    real_dates = dates[real_mask].to_numpy(dtype="datetime64[ns]")

    if real_dates.size < 2:
        return []

    gaps = []
    for i in range(1, len(real_dates)):
        # timedelta64 difference
        td = real_dates[i] - real_dates[i - 1]

        # convert to numeric day count
        days = td / np.timedelta64(1, "D")

        gaps.append(int(days))

    return gaps


def bucket_gap_statistics(gaps):
    # pre: gaps is list of gap lengths in days
    # post: returns dict of bucketed gap statistics
    # desc: buckets gap lengths into predefined categories and computes stats

    total = len(gaps)

    buckets = {
        "short_1_3":  {"range": (1, 3),  "cond": lambda g: 1 <= g <= 3},
        "medium_4_7": {"range": (4, 7),  "cond": lambda g: 4 <= g <= 7},
        "long_8_14":  {"range": (8, 14), "cond": lambda g: 8 <= g <= 14},
        "huge_15+":   {"range": (15, None), "cond": lambda g: g > 14},
    }

    bucket_stats = {}

    for name, cfg in buckets.items():
        cond = cfg["cond"]
        filtered = [g for g in gaps if cond(g)]

        counts = dict(Counter(filtered))

        bucket_stats[name] = {
            "range": cfg["range"],
            "n": len(filtered),
            "percentage": (len(filtered) / total) if total > 0 else None,
            "counts": counts,
            "min": min(filtered) if filtered else None,
            "max": max(filtered) if filtered else None,
            "mean": float(np.mean(filtered)) if filtered else None,
            "std": float(np.std(filtered)) if filtered else None,
        }

    return bucket_stats


def compute_confidence_vs_gap(df, col):
    # pre: df has col + '_gap_length' and col + '_conf'
    # post: returns dict mapping gap_length -> mean confidence
    # desc: computes average confidence per gap length

    if (col + "_gap_length") not in df.columns:
        return {}

    gap = df[col + "_gap_length"].astype(float)
    conf = df[col + "_conf"].astype(float)

    out = {}
    for g in sorted(gap.unique()):
        pts = conf[gap == g]
        if len(pts) > 0:
            out[int(g)] = float(pts.mean())  # keep int keys

    return out


def attach_gap_metadata(df, col):
    # pre: df has 'date' and col
    # post: df with new column col + '_gap_length'
    # desc: attaches gap length metadata to each row

    s = df[col]
    dates = pd.to_datetime(df["date"])

    real_mask = s.notna()
    real_dates = dates[real_mask].values

    gap_lengths = []

    for t in dates.values:
        diffs = np.abs(real_dates - t)
        nearest = diffs.min()

        days = int(nearest / np.timedelta64(1, "D")) # timedelta64
        gap_lengths.append(days)

    df[col + "_gap_length"] = gap_lengths
    return df
