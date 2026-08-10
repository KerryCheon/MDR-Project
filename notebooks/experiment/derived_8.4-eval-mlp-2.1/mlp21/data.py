"""Data preprocessing and tensor preparation for derived_8.4-eval-mlp-2.0.

Protocol (data_version 6) — train on the OFFICIAL train split (2017-2020),
early-stop / select configs on the OFFICIAL val split (2021-2022), evaluate on
the untouched test split (2023-2025). Identical preprocessing to mlp-1.3
(data_version 5): an AUXILIARY 2020 holdout (the last year of train, n=2,519)
is preprocessed with the same train-fitted imputer/scaler (no leakage — it is
a subset of train) and evaluated every epoch at the best-val epoch. The
aux2020 RMSE is a DIAGNOSTIC ONLY: mlp-1.2 documented it measures train fit,
not generalization, so it is never used for selection in 2.0.

Preprocessing follows the repo LSTM convention:
  _clean_inf -> SimpleImputer(median) -> StandardScaler -> clip(-5, 5),
fit ONLY on the train portion (per cluster subset for 2-regime specialists).
The target stays in original units (RMSE-consistent with XGBoost baselines).
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


def build_feature_set(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    *,
    test_positions: np.ndarray | None = None,
    aux: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Preprocess one feature set using the official train/val/test split.

    Returns the arrays needed by the trainer:
      X_train, y_train, X_val, y_val, X_test, y_test,
      train_idx, val_idx, test_idx, feature_names, n_features,
      X_aux, y_aux (optional 2020 holdout, same imputer/scaler as train)
    test_idx are positions into the ORIGINAL test frame so per-cluster
    predictions can be scattered back for pooled metrics.
    """
    feats = [f for f in feature_cols if f in train.columns]
    if len(feats) != len(set(feature_cols)):
        missing = sorted(set(feature_cols) - set(feats))
        raise ValueError(f"Missing feature columns: {missing}")

    X_tr = _clean_inf(train[feats].to_numpy(dtype=np.float64))
    X_va = _clean_inf(val[feats].to_numpy(dtype=np.float64))
    X_te = _clean_inf(test[feats].to_numpy(dtype=np.float64))
    y_tr = train[target].to_numpy(dtype=np.float64)
    y_va = val[target].to_numpy(dtype=np.float64)
    y_te = test[target].to_numpy(dtype=np.float64)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr = imputer.fit_transform(X_tr)
    X_va = imputer.transform(X_va) if X_va.shape[0] else X_va
    X_te = imputer.transform(X_te) if X_te.shape[0] else X_te
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va) if X_va.shape[0] else X_va
    X_te = scaler.transform(X_te) if X_te.shape[0] else X_te
    X_tr = np.clip(X_tr, -5.0, 5.0)
    X_va = np.clip(X_va, -5.0, 5.0) if X_va.shape[0] else X_va
    X_te = np.clip(X_te, -5.0, 5.0) if X_te.shape[0] else X_te

    if test_positions is None:
        test_idx = np.arange(len(test), dtype=np.int64)
    else:
        test_idx = np.asarray(test_positions, dtype=np.int64)

    out = {
        "X_train": X_tr.astype(np.float32),
        "y_train": y_tr.astype(np.float32),
        "X_val": X_va.astype(np.float32),
        "y_val": y_va.astype(np.float32),
        "X_test": X_te.astype(np.float32),
        "y_test": y_te.astype(np.float32),
        "train_idx": np.arange(len(train), dtype=np.int64),
        "val_idx": np.arange(len(val), dtype=np.int64),
        "test_idx": test_idx,
        "feature_names": feats,
        "n_features": len(feats),
        "imputer_median_": imputer.statistics_.tolist(),
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
    }

    if aux is not None and len(aux) > 0:
        X_aux = _clean_inf(aux[feats].to_numpy(dtype=np.float64))
        X_aux = imputer.transform(X_aux)
        X_aux = scaler.transform(X_aux)
        X_aux = np.clip(X_aux, -5.0, 5.0)
        out["X_aux"] = X_aux.astype(np.float32)
        out["y_aux"] = aux[target].to_numpy(dtype=np.float64).astype(np.float32)
    else:
        out["X_aux"] = np.empty((0, len(feats)), dtype=np.float32)
        out["y_aux"] = np.empty(0, dtype=np.float32)
    return out


def save_feature_set(path: Path, fs: dict[str, Any]) -> None:
    """Persist a feature set dict to an .npz (plus a sibling .json for scalars)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_train=fs["X_train"],
        y_train=fs["y_train"],
        X_val=fs["X_val"],
        y_val=fs["y_val"],
        X_test=fs["X_test"],
        y_test=fs["y_test"],
        X_aux=fs["X_aux"],
        y_aux=fs["y_aux"],
        train_idx=fs["train_idx"],
        val_idx=fs["val_idx"],
        test_idx=fs["test_idx"],
        feature_names=np.asarray(fs["feature_names"], dtype=object),
    )
    meta = {
        "n_features": fs["n_features"],
        "imputer_median_": fs["imputer_median_"],
        "scaler_mean_": fs["scaler_mean_"],
        "scaler_scale_": fs["scaler_scale_"],
        "feature_names": fs["feature_names"],
        "n_aux": int(len(fs["y_aux"])),
    }
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_feature_set(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "X_aux": data["X_aux"],
        "y_aux": data["y_aux"],
        "train_idx": data["train_idx"],
        "val_idx": data["val_idx"],
        "test_idx": data["test_idx"],
        "feature_names": list(data["feature_names"]),
        "n_features": int(data["X_train"].shape[1]),
    }
