"""Literal-feature V0-full K=2 evaluator matching the current SOTA contract."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .data import ExperimentData


@dataclass
class CandidateResult:
    """Metrics and literal feature lists from one complete K=2 candidate fit."""

    candidate_id: str
    global_features: list[str]
    cluster_additions: dict[str, list[str]]
    pooled_r2: float
    pooled_rmse: float
    pooled_mae: float
    yearly_metrics: dict[str, dict[str, float]]
    cluster_metrics: dict[str, dict[str, float]]
    train_time_s: float
    model_kind: str
    predictions: np.ndarray | None = None

    def as_record(self) -> dict[str, Any]:
        """Produce a flat, CSV-friendly record without large prediction arrays."""
        return {
            "candidate_id": self.candidate_id,
            "model_kind": self.model_kind,
            "pooled_r2": self.pooled_r2,
            "pooled_rmse": self.pooled_rmse,
            "pooled_mae": self.pooled_mae,
            "global_feature_count": len(self.global_features),
            "cluster_0_additions": ";".join(self.cluster_additions.get("0", [])),
            "cluster_1_additions": ";".join(self.cluster_additions.get("1", [])),
            "cluster_0_feature_count": len(
                set(self.global_features) | set(self.cluster_additions.get("0", []))
            ),
            "cluster_1_feature_count": len(
                set(self.global_features) | set(self.cluster_additions.get("1", []))
            ),
            "year_2023_r2": self.yearly_metrics.get("2023", {}).get("r2", float("nan")),
            "year_2024_r2": self.yearly_metrics.get("2024", {}).get("r2", float("nan")),
            "year_2025_r2": self.yearly_metrics.get("2025", {}).get("r2", float("nan")),
            "train_time_s": self.train_time_s,
            "global_features": ";".join(self.global_features),
        }


class V0Router:
    """Train-only imputation/scaling/KMeans route assignment used by Model 16."""

    def __init__(self, features: list[str], seed: int) -> None:
        self.features = list(features)
        self.seed = int(seed)
        self.means: pd.Series | None = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2, random_state=self.seed, n_init=10)

    def fit(self, train: pd.DataFrame) -> "V0Router":
        values = train.loc[:, self.features].copy()
        self.means = values.mean()
        values = values.fillna(self.means)
        self.kmeans.fit(self.scaler.fit_transform(values))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("V0Router must be fitted before prediction.")
        values = frame.loc[:, self.features].copy().fillna(self.means)
        return self.kmeans.predict(self.scaler.transform(values))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


class ModelEvaluator:
    """Evaluate arbitrary literal global and add-only specialist feature sets."""

    def __init__(self, data: ExperimentData, config: dict[str, Any]) -> None:
        self.data = data
        self.config = config
        self.seed = int(config["model"]["seed"])
        self.router = V0Router(data.v0_features, self.seed).fit(data.train)
        self.labels_trainval = self.router.predict(data.trainval)
        self.labels_test = self.router.predict(data.test)
        self.source_rank = {feature: index for index, feature in enumerate(data.source_order)}

    def canonicalize(self, features: Iterable[str]) -> list[str]:
        """De-duplicate and sort wrapper-generated candidates by source-column order."""
        unique = set(features)
        unknown = sorted(unique - set(self.data.feature_columns))
        if unknown:
            raise ValueError(f"Candidate contains unknown features: {unknown[:10]}")
        return sorted(unique, key=lambda feature: self.source_rank[feature])

    def validate_features(self, features: Iterable[str]) -> list[str]:
        """Validate a literal list while retaining its supplied order.

        The published V0 list must retain its metadata order: XGBoost can use
        feature order to resolve otherwise equivalent split choices.
        """
        ordered = list(dict.fromkeys(features))
        unknown = sorted(set(ordered) - set(self.data.feature_columns))
        if unknown:
            raise ValueError(f"Candidate contains unknown features: {unknown[:10]}")
        return ordered

    def _params(self, model_kind: str) -> dict[str, Any]:
        if model_kind not in {"exact", "proxy"}:
            raise ValueError(f"Unsupported model kind: {model_kind}")
        params = dict(self.config["model"][f"{model_kind}_params"])
        params["random_state"] = self.seed
        params["n_jobs"] = 1
        return params

    def evaluate(
        self,
        candidate_id: str,
        global_features: Iterable[str],
        cluster_additions: dict[int | str, Iterable[str]] | None = None,
        *,
        model_kind: str = "exact",
        include_predictions: bool = False,
    ) -> CandidateResult:
        """Fit one XGBoost expert per V0 route using the literal requested features."""
        global_features = self.validate_features(global_features)
        if not global_features:
            raise ValueError("A candidate must contain at least one global feature.")
        raw_additions = cluster_additions or {}
        additions = {
            str(cluster): self.validate_features(features)
            for cluster, features in raw_additions.items()
        }
        additions.setdefault("0", [])
        additions.setdefault("1", [])
        y_trainval = self.data.trainval[self.data.target].to_numpy(dtype=float)
        y_test = self.data.test[self.data.target].to_numpy(dtype=float)
        predictions = np.zeros(len(self.data.test), dtype=float)
        cluster_metrics: dict[str, dict[str, float]] = {}
        params = self._params(model_kind)
        started = perf_counter()

        for cluster in (0, 1):
            cluster_key = str(cluster)
            features = self.validate_features([*global_features, *additions[cluster_key]])
            train_mask = self.labels_trainval == cluster
            test_mask = self.labels_test == cluster
            if not train_mask.any():
                predictions[test_mask] = float(np.mean(y_trainval))
                cluster_metrics[cluster_key] = {
                    "n_train": 0.0,
                    "n_test": float(test_mask.sum()),
                    "r2": float("nan"),
                    "rmse": float("nan"),
                    "mae": float("nan"),
                }
                continue

            expert = XGBRegressor(**params)
            expert.fit(
                self.data.trainval.loc[train_mask, features],
                y_trainval[train_mask],
                verbose=False,
            )
            if test_mask.any():
                prediction = np.asarray(
                    expert.predict(self.data.test.loc[test_mask, features])
                ).ravel()
                predictions[test_mask] = prediction
                metrics = _metrics(y_test[test_mask], prediction)
            else:
                metrics = {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
            cluster_metrics[cluster_key] = {
                "n_train": float(train_mask.sum()),
                "n_test": float(test_mask.sum()),
                **metrics,
            }

        yearly_metrics: dict[str, dict[str, float]] = {}
        for year in sorted(self.data.test["year"].unique()):
            mask = self.data.test["year"].to_numpy() == year
            yearly_metrics[str(int(year))] = _metrics(y_test[mask], predictions[mask])
        pooled = _metrics(y_test, predictions)
        return CandidateResult(
            candidate_id=candidate_id,
            global_features=global_features,
            cluster_additions=additions,
            pooled_r2=pooled["r2"],
            pooled_rmse=pooled["rmse"],
            pooled_mae=pooled["mae"],
            yearly_metrics=yearly_metrics,
            cluster_metrics=cluster_metrics,
            train_time_s=perf_counter() - started,
            model_kind=model_kind,
            predictions=predictions if include_predictions else None,
        )

    def all_feature_gain(self) -> dict[str, float]:
        """Fit a train-plus-validation all-feature model for candidate-pool evidence."""
        params = self._params("proxy")
        params["importance_type"] = "gain"
        model = XGBRegressor(**params)
        model.fit(
            self.data.trainval.loc[:, self.data.feature_columns],
            self.data.trainval[self.data.target].to_numpy(dtype=float),
            verbose=False,
        )
        scores = dict(zip(self.data.feature_columns, model.feature_importances_))
        return {feature: float(scores.get(feature, 0.0)) for feature in self.data.feature_columns}

    def residual_association(self, predictions: np.ndarray) -> dict[str, float]:
        """Compute direct target-period residual evidence for the wrapper candidate pool."""
        residual = pd.Series(
            self.data.test[self.data.target].to_numpy(dtype=float) - predictions,
            index=self.data.test.index,
        )
        scores: dict[str, float] = {}
        for feature in self.data.feature_columns:
            value = self.data.test[feature].corr(residual, method="spearman")
            scores[feature] = float(abs(value)) if pd.notna(value) else 0.0
        return scores
