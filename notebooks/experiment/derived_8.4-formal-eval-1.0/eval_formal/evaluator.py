"""Seed-aware fold/model evaluation for derived_8.4-formal-eval-1.0.

Adapted from derived_8.4-eval-1.3/eval13/evaluator.py with one protocol change:
the XGBoost expert regressors use a per-job ``seed`` (random_state), while the router /
gating classifier stays at the fixed ``router_seed`` (config ``model.router_seed``) so
the per-regime delta additions stay attached to the same regime labels across seeds
(see config.yaml). Artifact naming embeds the seed (``<config_id>__s<seed>__<station>``)
so per-seed weights and predictions never collide and job completion can be verified by
file presence.
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
    """Pooled regression metrics; mirrors derived_8.4-eval-1.1/-1.3."""
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
    config_id: str
    strategy_name: str
    seed: int
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
    config_id: str
    strategy_name: str
    seed: int
    n_train_total: int
    n_test: int
    pooled: dict[str, float]
    station_metrics: dict[str, dict[str, float]]
    yearly_metrics: dict[str, dict[str, float]]
    cluster_metrics: dict[str, dict[str, float]]
    train_time_s: float
    predictions: np.ndarray
    cluster_labels_test: np.ndarray


class FormalEvaluator:
    """Evaluates one fixed configuration under full (temporal) or LOSO evaluation."""

    def __init__(
        self,
        data: ExperimentData,
        config: dict[str, Any],
        *,
        config_id: str,
        strategy_name: str,
        seed: int,
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
        self.seed = int(seed)
        self.router_seed = int(config["model"].get("router_seed", 42))
        self.global_features = self.validate_features(global_features)
        raw = cluster_additions or {}
        self.cluster_additions = {
            str(k): self.validate_features(feats) for k, feats in raw.items()
        }
        self.cluster_additions.setdefault("0", [])
        self.cluster_additions.setdefault("1", [])
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

    def _stem(self, station: str) -> str:
        return f"{self.config_id}__s{self.seed}__{station}"

    def _router(self):
        return get_router(
            self.strategy_name,
            self.data.v0_features,
            backbone_54=self.data.shared_backbone_54,
            seed=self.router_seed,
            device=self.config["model"]["exact_params"].get("device", "cuda"),
        )

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
        router = self._router()
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

        stem = self._stem(station)
        self._persist(stem, predictions, labels_test, pooled, train_time,
                      n_train_total=int(len(fold_trainval)),
                      router=router, fitted_models=fitted_models)

        return LosoFoldResult(
            config_id=self.config_id,
            strategy_name=self.strategy_name,
            seed=self.seed,
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
        """Temporal evaluation: train experts on ALL stations (trainval), test on the full
        test set. Replicates derived_8.4-eval-1.1's protocol (router fit on trainval)."""
        trainval = self.data.trainval
        test = self.data.test
        y_trainval = trainval[self.data.target].to_numpy(dtype=float)
        y_test = test[self.data.target].to_numpy(dtype=float)

        router = self._router()
        router.fit(trainval)
        labels_trainval = np.asarray(router.predict(trainval)).ravel().astype(int)
        labels_test = np.asarray(router.predict(test)).ravel().astype(int)

        predictions = np.zeros(len(test), dtype=float)
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
            fitted_models[cluster_key] = expert

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

        stem = self._stem("full")
        self._persist(stem, predictions, labels_test, pooled, train_time,
                      n_train_total=int(len(trainval)),
                      router=router, fitted_models=fitted_models)

        return FullEvalResult(
            config_id=self.config_id,
            strategy_name=self.strategy_name,
            seed=self.seed,
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

    def _persist(self, stem: str, predictions: np.ndarray, labels_test: np.ndarray,
                 pooled: dict[str, float], train_time: float, *, n_train_total: int,
                 router, fitted_models: dict[str, Any]) -> None:
        """Save predictions, labels, model weights and a per-fold meta.json under `stem`.
        `stem` embeds config_id + seed (+ station), so per-seed artifacts never collide."""
        if self.save_predictions and self.predictions_dir is not None:
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.predictions_dir / f"{stem}_preds.npy", predictions)
            np.save(self.predictions_dir / f"{stem}_labels_te.npy", labels_test)

        if self.save_weights and self.models_dir is not None:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(router, TrainedGatingRouter) and router.clf is not None:
                router.clf.save_model(self.models_dir / f"{stem}_gating.json")
            for cl_key, exp_model in fitted_models.items():
                if self.strategy_name == "Global_Single":
                    exp_path = self.models_dir / f"{stem}_reg.json"
                else:
                    exp_path = self.models_dir / f"{stem}_spec_{cl_key}.json"
                exp_model.save_model(exp_path)

        meta = {
            "config_id": self.config_id,
            "strategy_name": self.strategy_name,
            "seed": self.seed,
            "train_time_s": train_time,
            "n_train_total": int(n_train_total),
            "n_test": int(len(predictions)),
            "pooled_r2": pooled["r2"],
            "pooled_rmse": pooled["rmse"],
        }
        meta_path = self.models_dir / f"{stem}_meta.json" if self.models_dir is not None \
            else self.predictions_dir / f"{stem}_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
