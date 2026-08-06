"""Data preprocessing and tensor preparation for derived_8.4-eval-mlp-1.0.

Preprocessing follows the repo LSTM convention (Models/Temporal/lstm/train_v9.py):
  _clean_inf -> SimpleImputer(median) -> StandardScaler -> clip(-5, 5),
fit ONLY on the trainval portion (per cluster subset for the 2-regime
specialists) so there is no test leakage. The target is left in original
units (RMSE-consistent with the XGBoost baselines).

Holdout: a TEMPORAL 10% carve-out of trainval (each station's last 10% of rows
by date), fixed across all sweep configs of a given feature set (used for
early stopping / LR scheduling / config ranking). Temporal, because trainval
(2017-2022) and test (2023-2025) are temporally disjoint and a random holdout
would leak temporally-adjacent rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def _clean_inf(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return X


def make_holdout_split(
    stations: np.ndarray,
    dates: np.ndarray,
    frac: float = 0.10,
    seed: int = 42,
    salt: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal per-station holdout indices.

    Returns (train_idx, hold_idx) into the passed row order. The holdout is
    the LAST `frac` of each station's rows sorted by date — mirroring how the
    derived_8.4 test split (2023-2025) is temporally disjoint from trainval
    (2017-2022). A random/stratified holdout would leak temporally-adjacent
    rows into early stopping and flatter the selection metric.
    """
    del seed, salt  # deterministic by construction (date order); kept for API parity
    stations_arr = np.asarray(stations)
    dates_arr = pd.to_datetime(np.asarray(dates))
    train_idx: list[int] = []
    hold_idx: list[int] = []
    for st in np.unique(stations_arr):
        idx = np.where(stations_arr == st)[0]
        order = np.argsort(dates_arr[idx], kind="stable")
        idx_sorted = idx[order]
        n_hold = max(1, int(round(len(idx_sorted) * frac)))
        hold_idx.extend(idx_sorted[-n_hold:].tolist())
        train_idx.extend(idx_sorted[:-n_hold].tolist())
    train_idx = np.asarray(sorted(train_idx), dtype=np.int64)
    hold_idx = np.asarray(sorted(hold_idx), dtype=np.int64)
    return train_idx, hold_idx


def build_feature_set(
    trainval: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    station_col: str = "station_id",
    *,
    holdout_frac: float = 0.10,
    seed: int = 42,
    salt: str = "",
    test_positions: np.ndarray | None = None,
    date_col: str = "date",
) -> dict[str, Any]:
    """Preprocess one feature set (rows = all trainval rows or a cluster subset).

    Returns a dict with the arrays needed by the trainer:
      X_train, y_train, X_hold, y_hold, X_test, y_test,
      train_idx, hold_idx, test_idx, feature_names, n_features
    test_idx are positions into the ORIGINAL test frame so per-cluster
    predictions can be scattered back for pooled metrics.
    """
    feats = list(feature_cols)
    X_tv = _clean_inf(trainval[feats].to_numpy(dtype=np.float64))
    X_te = _clean_inf(test[feats].to_numpy(dtype=np.float64))
    y_tv = trainval[target].to_numpy(dtype=np.float64)
    y_te = test[target].to_numpy(dtype=np.float64)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tv = imputer.fit_transform(X_tv)
    X_te = imputer.transform(X_te)
    X_tv = scaler.fit_transform(X_tv)
    X_te = scaler.transform(X_te)
    X_tv = np.clip(X_tv, -5.0, 5.0)
    X_te = np.clip(X_te, -5.0, 5.0)

    train_idx, hold_idx = make_holdout_split(
        trainval[station_col].to_numpy(),
        trainval[date_col].to_numpy(),
        frac=holdout_frac,
        seed=seed,
        salt=salt,
    )
    if test_positions is None:
        test_idx = np.arange(len(test), dtype=np.int64)
    else:
        test_idx = np.asarray(test_positions, dtype=np.int64)

    return {
        "X_train": X_tv[train_idx].astype(np.float32),
        "y_train": y_tv[train_idx].astype(np.float32),
        "X_hold": X_tv[hold_idx].astype(np.float32),
        "y_hold": y_tv[hold_idx].astype(np.float32),
        "X_test": X_te.astype(np.float32),
        "y_test": y_te.astype(np.float32),
        "train_idx": train_idx,
        "hold_idx": hold_idx,
        "test_idx": test_idx,
        "feature_names": feats,
        "n_features": len(feats),
        "imputer_median_": imputer.statistics_.tolist(),
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
    }


def save_feature_set(path: Path, fs: dict[str, Any]) -> None:
    """Persist a feature set dict to an .npz (plus a sibling .json for scalars)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_train=fs["X_train"],
        y_train=fs["y_train"],
        X_hold=fs["X_hold"],
        y_hold=fs["y_hold"],
        X_test=fs["X_test"],
        y_test=fs["y_test"],
        train_idx=fs["train_idx"],
        hold_idx=fs["hold_idx"],
        test_idx=fs["test_idx"],
        feature_names=np.asarray(fs["feature_names"], dtype=object),
    )
    meta = {
        "n_features": fs["n_features"],
        "imputer_median_": fs["imputer_median_"],
        "scaler_mean_": fs["scaler_mean_"],
        "scaler_scale_": fs["scaler_scale_"],
        "feature_names": fs["feature_names"],
    }
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_feature_set(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_hold": data["X_hold"],
        "y_hold": data["y_hold"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "train_idx": data["train_idx"],
        "hold_idx": data["hold_idx"],
        "test_idx": data["test_idx"],
        "feature_names": list(data["feature_names"]),
        "n_features": int(data["X_train"].shape[1]),
    }
