# Jakob Balkovec
# Metrics

import numpy as np
import pandas as pd

from Utils.logging import get_logger


def _to_numpy(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def _finite_mask(y_true, y_pred):
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    m = np.isfinite(yt) & np.isfinite(yp)
    return yt[m], yp[m], m


def r2_score(y_true, y_pred):
    yt, yp, _ = _finite_mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    denom = np.sum((yt - np.mean(yt)) ** 2)
    if denom == 0:
        return float("nan")
    num = np.sum((yt - yp) ** 2)
    return float(1.0 - (num / denom))


def mse(y_true, y_pred):
    yt, yp, _ = _finite_mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred):
    v = mse(y_true, y_pred)
    if not np.isfinite(v):
        return float("nan")
    return float(np.sqrt(v))


def mae(y_true, y_pred):
    yt, yp, _ = _finite_mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def bias_me(y_true, y_pred):
    yt, yp, _ = _finite_mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(yp - yt))


def rel_rmse(y_true, y_pred, eps=1e-12):
    yt, yp, _ = _finite_mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    denom = np.mean(np.abs(yt)) + float(eps)
    return float(np.sqrt(np.mean((yt - yp) ** 2)) / denom)


def metrics_block(split_name, y_true, y_pred):
    yt, yp, m = _finite_mask(y_true, y_pred)
    dropped = int((~m).sum())

    out = {
        "split": split_name,
        "n": int(yt.size),
        "dropped_nonfinite": dropped,
        "r2": r2_score(yt, yp),
        "rmse": rmse(yt, yp),
        "rel_rmse": rel_rmse(yt, yp),
        "mae": mae(yt, yp),
        "bias_me": bias_me(yt, yp),
    }
    return out


def log_metrics_table(rows, title="Metrics"):
    log = get_logger("evaluation.metrics")

    df = pd.DataFrame(rows)
    cols = [c for c in ["split", "n", "r2", "rmse", "rel_rmse", "mae", "bias_me", "dropped_nonfinite"] if c in df.columns]
    df = df[cols]

    log.info("%s\n%s", title, df.to_string(index=False))
    return df
