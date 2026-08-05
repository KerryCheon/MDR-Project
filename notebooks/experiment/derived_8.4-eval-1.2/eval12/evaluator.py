"""LOSO fold-aware model evaluation for derived_8.4-eval-1.2.

For every (configuration, held-out station) pair, the router is refitted on the
fold's trainval (rows from train + val excluding the held-out station) and the
experts are trained per regime cluster on that fold trainval, then evaluated on
all test rows of the held-out station. Metrics (pooled / per-year / per-regime),
per-fold predictions (.npy) and model weights (.json, same format as
derived_8.4-eval-1.1) are persisted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from .data import ExperimentData
from .routers import TrainedGatingRouter, get_router


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Pooled regression metrics; mirrors derived_8.4-eval-1.1."""
    if len(y_true) == 0:
        return {
            "r2": float("nan"),
            "rmse": float("nan"),
            "ubrmse": float("nan"),
            "bias": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
        }
    bias = float(np.mean(y_pred - y_true))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    ubrmse_val = rmse ** 2 - bias ** 2
    ubrmse = float(np.sqrt(max(0.0, ubrmse_val)))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    if len(y_true) > 1 and np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        pr, _ = pearsonr(y_true, y_pred)
        pearson = float(pr)
    else:
        pearson = float("nan")
    return {
        "r2": r2,
        "rmse": rmse,
        "ubrmse": ubrmse,
        "bias": bias,
        "mae": mae,
        "pearson": pearson,
    }


@dataclass
class LosoFoldResult:
    """Metrics and artifacts for one (configuration, held-out station) fold."""

    config_id: str
    strategy_name: str
    station: str
    n_train_total: int
    n_test: int
    r2: float
    rmse: float
    ubrmse: float
    bias: float
    mae: float
    pearson: float
    yearly_metrics: dict[str, dict[str, float]]
    cluster_metrics: dict[str, dict[str, float]]
    train_time_s: float
    predictions: np.ndarray
    cluster_labels_test: np.ndarray


@dataclass
class FullEvalResult:
    """Metrics for one configuration trained on ALL stations (no LOSO).

    This is the full-training baseline that replicates derived_8.4-eval-1.1:
    the router is fit and the experts are trained on the entire trainval (all 7
    stations), then evaluated on the full test set. ``station_metrics`` breaks
    the pooled metrics down per test station, so per-station difficulty under
    full training can be compared against LOSO difficulty.
    """

    config_id: str
    strategy_name: str
    n_train_total: int
    n_test: int
    pooled: dict[str, float]
    station_metrics: dict[str, dict[str, float]]
    yearly_metrics: dict[str, dict[str, float]]
    cluster_metrics: dict[str, dict[str, float]]
    train_time_s: float
    predictions: np.ndarray
    cluster_labels_test: np.ndarray


class LosoEvaluator:
    """Evaluates a single fixed configuration under leave-one-station-out."""

    def __init__(
        self,
        data: ExperimentData,
        config: dict[str, Any],
        *,
        config_id: str,
        strategy_name: str,
        global_features: Iterable[str],
        cluster_additions: dict[int | str, Iterable[str]] | None = None,
        models_dir: Path | None = None,
        predictions_dir: Path | None = None,
        save_weights: bool = False,
        save_predictions: bool = False,
    ) -> None:
        self.data = data
        self.config = config
        self.config_id = config_id
        self.strategy_name = strategy_name
        self.global_features = self.validate_features(global_features)
        raw = cluster_additions or {}
        self.cluster_additions = {
            str(k): self.validate_features(feats) for k, feats in raw.items()
        }
        self.cluster_additions.setdefault("0", [])
        self.cluster_additions.setdefault("1", [])
        self.seed = int(config["model"]["seed"])
        self.models_dir = models_dir
        self.predictions_dir = predictions_dir
        self.save_weights = save_weights
        self.save_predictions = save_predictions

    def validate_features(self, features: Iterable[str]) -> list[str]:
        ordered = list(dict.fromkeys(features))
        unknown = sorted(set(ordered) - set(self.data.feature_columns))
        if unknown:
            raise ValueError(f"Candidate contains unknown features: {unknown[:10]}")
        return ordered

    def _params(self) -> dict[str, Any]:
        params = dict(self.config["model"]["exact_params"])
        params["random_state"] = self.seed
        params["n_jobs"] = 1
        return params

    def evaluate_station(self, station: str) -> LosoFoldResult:
        """Run one LOSO fold: train on all stations but `station`, test on `station`."""
        trainval = self.data.trainval
        test = self.data.test

        fold_trainval = trainval[trainval["station_id"] != station].reset_index(drop=True)
        fold_test = test[test["station_id"] == station].reset_index(drop=True)
        if len(fold_test) == 0:
            raise ValueError(f"No test rows for held-out station {station}.")

        # Router is refitted on the fold's trainval only: the held-out station's
        # rows never influence routing (no leakage into the routing decision).
        router = get_router(self.strategy_name, self.data.v0_features, seed=self.seed)
        router.fit(fold_trainval)
        labels_trainval = np.asarray(router.predict(fold_trainval)).ravel().astype(int)
        labels_test = np.asarray(router.predict(fold_test)).ravel().astype(int)

        y_trainval = fold_trainval[self.data.target].to_numpy(dtype=float)
        y_test = fold_test[self.data.target].to_numpy(dtype=float)

        predictions = np.zeros(len(fold_test), dtype=float)
        cluster_metrics: dict[str, dict[str, float]] = {}
        params = self._params()
        started = perf_counter()
        fitted_models: dict[str, Any] = {}

        clusters_to_eval = (0,) if self.strategy_name == "Global_Single" else (0, 1)

        for cluster in clusters_to_eval:
            cluster_key = str(cluster)
            features = self.validate_features(
                [*self.global_features, *self.cluster_additions[cluster_key]]
            )
            train_mask = labels_trainval == cluster
            test_mask = labels_test == cluster

            if not train_mask.any():
                # Fallback: predict the fold trainval mean (mirrors eval-1.1).
                predictions[test_mask] = float(np.mean(y_trainval))
                cluster_metrics[cluster_key] = {
                    "n_train": 0.0,
                    "n_test": float(test_mask.sum()),
                    **compute_metrics(y_test[test_mask], predictions[test_mask]),
                }
                continue

            expert = XGBRegressor(**params)
            expert.fit(
                fold_trainval.loc[train_mask, features],
                y_trainval[train_mask],
                verbose=False,
            )
            fitted_models[cluster_key] = expert

            if test_mask.any():
                pred = np.asarray(
                    expert.predict(fold_test.loc[test_mask, features])
                ).ravel()
                predictions[test_mask] = pred
                cluster_metrics[cluster_key] = {
                    "n_train": float(train_mask.sum()),
                    "n_test": float(test_mask.sum()),
                    **compute_metrics(y_test[test_mask], pred),
                }
            else:
                cluster_metrics[cluster_key] = {
                    "n_train": float(train_mask.sum()),
                    "n_test": 0.0,
                    **compute_metrics(np.array([]), np.array([])),
                }

        yearly_metrics: dict[str, dict[str, float]] = {}
        for year in sorted(fold_test["year"].unique()):
            mask = fold_test["year"].to_numpy() == year
            yearly_metrics[str(int(year))] = compute_metrics(y_test[mask], predictions[mask])

        pooled = compute_metrics(y_test, predictions)
        train_time = perf_counter() - started

        # Persist predictions (.npy) and model weights (.json, eval-1.1 format).
        if self.save_predictions and self.predictions_dir is not None:
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.predictions_dir / f"{self.config_id}__{station}_preds.npy", predictions)
            np.save(self.predictions_dir / f"{self.config_id}__{station}_labels_te.npy", labels_test)

        if self.save_weights and self.models_dir is not None:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "config_id": self.config_id,
                "strategy_name": self.strategy_name,
                "station": station,
                "train_time_s": train_time,
                "n_train_total": int(len(fold_trainval)),
                "n_test": int(len(fold_test)),
                "pooled_r2": pooled["r2"],
                "pooled_rmse": pooled["rmse"],
            }
            with open(self.models_dir / f"{self.config_id}__{station}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            if isinstance(router, TrainedGatingRouter) and router.clf is not None:
                router.clf.save_model(self.models_dir / f"{self.config_id}__{station}_gating.json")

            for cl_key, exp_model in fitted_models.items():
                if self.strategy_name == "Global_Single":
                    exp_path = self.models_dir / f"{self.config_id}__{station}_reg.json"
                else:
                    exp_path = self.models_dir / f"{self.config_id}__{station}_spec_{cl_key}.json"
                exp_model.save_model(exp_path)

        return LosoFoldResult(
            config_id=self.config_id,
            strategy_name=self.strategy_name,
            station=station,
            n_train_total=int(len(fold_trainval)),
            n_test=int(len(fold_test)),
            r2=pooled["r2"],
            rmse=pooled["rmse"],
            ubrmse=pooled["ubrmse"],
            bias=pooled["bias"],
            mae=pooled["mae"],
            pearson=pooled["pearson"],
            yearly_metrics=yearly_metrics,
            cluster_metrics=cluster_metrics,
            train_time_s=train_time,
            predictions=predictions,
            cluster_labels_test=labels_test,
        )

    def evaluate_full(self) -> FullEvalResult:
        """Train on ALL stations (no LOSO) and evaluate on the full test set.

        Replicates derived_8.4-eval-1.1: router fit on the entire trainval (all 7
        stations) and experts trained per regime cluster on the entire trainval.
        Returns pooled / per-station / per-year / per-regime metrics; the
        per-station breakdown is the *intrinsic* difficulty measure used to
        contrast with LOSO difficulty (a station can only be hard to *fit* when
        its own rows are in the training set).
        """
        trainval = self.data.trainval
        test = self.data.test
        y_trainval = trainval[self.data.target].to_numpy(dtype=float)
        y_test = test[self.data.target].to_numpy(dtype=float)

        router = get_router(self.strategy_name, self.data.v0_features, seed=self.seed)
        router.fit(trainval)
        labels_trainval = np.asarray(router.predict(trainval)).ravel().astype(int)
        labels_test = np.asarray(router.predict(test)).ravel().astype(int)

        predictions = np.zeros(len(test), dtype=float)
        cluster_metrics: dict[str, dict[str, float]] = {}
        params = self._params()
        started = perf_counter()

        clusters_to_eval = (0,) if self.strategy_name == "Global_Single" else (0, 1)

        for cluster in clusters_to_eval:
            cluster_key = str(cluster)
            features = self.validate_features(
                [*self.global_features, *self.cluster_additions[cluster_key]]
            )
            train_mask = labels_trainval == cluster
            test_mask = labels_test == cluster

            if not train_mask.any():
                # Fallback: predict the trainval mean (mirrors eval-1.1).
                predictions[test_mask] = float(np.mean(y_trainval))
                cluster_metrics[cluster_key] = {
                    "n_train": 0.0,
                    "n_test": float(test_mask.sum()),
                    **compute_metrics(y_test[test_mask], predictions[test_mask]),
                }
                continue

            expert = XGBRegressor(**params)
            expert.fit(
                trainval.loc[train_mask, features],
                y_trainval[train_mask],
                verbose=False,
            )
            if test_mask.any():
                pred = np.asarray(
                    expert.predict(test.loc[test_mask, features])
                ).ravel()
                predictions[test_mask] = pred
                cluster_metrics[cluster_key] = {
                    "n_train": float(train_mask.sum()),
                    "n_test": float(test_mask.sum()),
                    **compute_metrics(y_test[test_mask], pred),
                }
            else:
                cluster_metrics[cluster_key] = {
                    "n_train": float(train_mask.sum()),
                    "n_test": 0.0,
                    **compute_metrics(np.array([]), np.array([])),
                }

        pooled = compute_metrics(y_test, predictions)
        pooled["n"] = int(len(y_test))

        station_ids = test["station_id"].to_numpy()
        station_metrics: dict[str, dict[str, float]] = {}
        for s in sorted(test["station_id"].unique()):
            mask = station_ids == s
            m = compute_metrics(y_test[mask], predictions[mask])
            m["n"] = int(mask.sum())
            station_metrics[str(s)] = m

        yearly_metrics: dict[str, dict[str, float]] = {}
        for year in sorted(test["year"].unique()):
            mask = test["year"].to_numpy() == year
            yearly_metrics[str(int(year))] = compute_metrics(y_test[mask], predictions[mask])

        train_time = perf_counter() - started

        if self.save_predictions and self.predictions_dir is not None:
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{self.config_id}__full"
            np.save(self.predictions_dir / f"{stem}_preds.npy", predictions)
            np.save(self.predictions_dir / f"{stem}_labels_te.npy", labels_test)
            meta = {
                "config_id": self.config_id,
                "strategy_name": self.strategy_name,
                "train_time_s": train_time,
                "n_train_total": int(len(trainval)),
                "n_test": int(len(test)),
                "pooled_r2": pooled["r2"],
                "pooled_rmse": pooled["rmse"],
                "pooled_ubrmse": pooled["ubrmse"],
                "pooled_bias": pooled["bias"],
                "pooled_mae": pooled["mae"],
            }
            with open(self.predictions_dir / f"{stem}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return FullEvalResult(
            config_id=self.config_id,
            strategy_name=self.strategy_name,
            n_train_total=int(len(trainval)),
            n_test=int(len(test)),
            pooled=pooled,
            station_metrics=station_metrics,
            yearly_metrics=yearly_metrics,
            cluster_metrics=cluster_metrics,
            train_time_s=train_time,
            predictions=predictions,
            cluster_labels_test=labels_test,
        )
