# Jakob Balkovec
# Robustness

import numpy as np
import pandas as pd

from Utils.logging import get_logger


def train_val_gap(train_metrics, val_metrics, metric="r2"):
    tr = float(train_metrics.get(metric, float("nan")))
    va = float(val_metrics.get(metric, float("nan")))
    if not np.isfinite(tr) or not np.isfinite(va):
        return float("nan")
    return float(tr - va)


def stability_summary(metric_rows, metric="r2"):
    log = get_logger("evaluation.robustness")

    df = pd.DataFrame(metric_rows)
    if "split" not in df.columns or metric not in df.columns:
        raise ValueError("stability_summary: metric_rows must contain 'split' and metric columns")

    out = {}

    for split in df["split"].unique():
        vals = df[df["split"] == split][metric].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f"{split}_{metric}_mean"] = float("nan")
            out[f"{split}_{metric}_std"] = float("nan")
        else:
            out[f"{split}_{metric}_mean"] = float(np.mean(vals))
            out[f"{split}_{metric}_std"] = float(np.std(vals))

    log.info("stability_summary(%s): %s", metric, out)
    return out


def robustness_block(train_metrics, val_metrics, test_metrics=None):
    log = get_logger("evaluation.robustness")

    out = {
        "gap_r2": train_val_gap(train_metrics, val_metrics, metric="r2"),
        "gap_rmse": train_val_gap(train_metrics, val_metrics, metric="rmse"),
        "gap_rel_rmse": train_val_gap(train_metrics, val_metrics, metric="rel_rmse"),
    }

    if test_metrics is not None:
        out["val_minus_test_r2"] = float(val_metrics.get("r2", float("nan"))) - float(test_metrics.get("r2", float("nan")))

    log.info("robustness_block: %s", out)
    return out
