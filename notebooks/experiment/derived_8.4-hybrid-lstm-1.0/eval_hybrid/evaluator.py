"""
Evaluator module for hybrid LSTM + XGBoost models.
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

from .data import HybridExperimentData


@dataclass
class HybridCandidateResult:
    model_name: str
    strategy_name: str
    candidate_id: str
    global_features: list[str]
    cluster_additions: dict[str, list[str]]
    pooled_r2: float
    pooled_rmse: float
    pooled_ubrmse: float
    pooled_bias: float
    pooled_mae: float
    pooled_pearson: float
    yearly_metrics: dict[str, dict[str, float]]
    cluster_metrics: dict[str, dict[str, float]]
    train_time_s: float
    predictions: np.ndarray | None = None

    def as_record(self) -> dict[str, Any]:
        c0_add = ";".join(self.cluster_additions.get("0", []))
        c1_add = ";".join(self.cluster_additions.get("1", []))
        return {
            "model_name": self.model_name,
            "strategy_name": self.strategy_name,
            "candidate_id": self.candidate_id,
            "pooled_r2": self.pooled_r2,
            "pooled_rmse": self.pooled_rmse,
            "pooled_ubrmse": self.pooled_ubrmse,
            "pooled_bias": self.pooled_bias,
            "pooled_mae": self.pooled_mae,
            "pooled_pearson": self.pooled_pearson,
            "global_feature_count": len(self.global_features),
            "cluster_0_additions": c0_add,
            "cluster_1_additions": c1_add,
            "cluster_0_feature_count": len(set(self.global_features) | set(self.cluster_additions.get("0", []))),
            "cluster_1_feature_count": len(set(self.global_features) | set(self.cluster_additions.get("1", []))),
            "year_2023_r2": self.yearly_metrics.get("2023", {}).get("r2", float("nan")),
            "year_2024_r2": self.yearly_metrics.get("2024", {}).get("r2", float("nan")),
            "year_2025_r2": self.yearly_metrics.get("2025", {}).get("r2", float("nan")),
            "train_time_s": self.train_time_s,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


class HybridStrategyEvaluator:
    def __init__(self, data: HybridExperimentData, config: dict[str, Any], strategy_name: str, models_dir: Path | None = None) -> None:
        self.data = data
        self.config = config
        self.strategy_name = strategy_name
        self.seed = int(config["model"]["seed"])
        self.models_dir = models_dir

        # Router fitting using V0FullRouter logic for Clustering_V0_Full_k2 or GlobalSingleRouter for Global_Single
        if strategy_name == "Clustering_V0_Full_k2":
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            features = data.v0_features
            vals_tr = data.trainval.loc[:, features].copy()
            self.v0_means = vals_tr.mean()
            vals_tr = vals_tr.fillna(self.v0_means)
            self.scaler = StandardScaler()
            scaled_tr = self.scaler.fit_transform(vals_tr)
            self.kmeans = KMeans(n_clusters=2, random_state=self.seed, n_init=10)
            self.kmeans.fit(scaled_tr)

            self.labels_trainval = self.kmeans.predict(scaled_tr)
            vals_te = data.test.loc[:, features].copy().fillna(self.v0_means)
            self.labels_test = self.kmeans.predict(self.scaler.transform(vals_te))
        else:
            self.labels_trainval = np.zeros(len(data.trainval), dtype=int)
            self.labels_test = np.zeros(len(data.test), dtype=int)

    def _params(self) -> dict[str, Any]:
        params = dict(self.config["model"]["exact_params"])
        params["random_state"] = self.seed
        params["n_jobs"] = 1
        import torch
        if not torch.cuda.is_available():
            params["device"] = "cpu"
        return params

    def fit_and_evaluate(
        self,
        model_name: str,
        candidate_id: str,
        global_features: list[str],
        cluster_additions: dict[str, list[str]] | None = None,
    ) -> HybridCandidateResult:
        cluster_additions = cluster_additions or {"0": [], "1": []}
        t0 = perf_counter()

        y_trainval = self.data.trainval[self.data.target].to_numpy(dtype=float)
        y_test = self.data.test[self.data.target].to_numpy(dtype=float)
        test_preds = np.zeros(len(self.data.test), dtype=float)

        models = {}
        unique_clusters = sorted(np.unique(self.labels_trainval))

        for cluster in unique_clusters:
            mask_tr = self.labels_trainval == cluster
            c_key = str(cluster)
            c_add = cluster_additions.get(c_key, [])
            features_c = list(dict.fromkeys(global_features + c_add))

            X_tr = self.data.trainval.loc[mask_tr, features_c]
            y_tr = y_trainval[mask_tr]

            model = XGBRegressor(**self._params())
            model.fit(X_tr, y_tr, verbose=False)
            models[c_key] = model

            mask_te = self.labels_test == cluster
            if np.any(mask_te):
                X_te = self.data.test.loc[mask_te, features_c]
                test_preds[mask_te] = model.predict(X_te)

            if self.models_dir is not None:
                booster_file = self.models_dir / f"{candidate_id}_cluster_{cluster}.json"
                model.save_model(str(booster_file))

        train_time = perf_counter() - t0
        pooled = compute_metrics(y_test, test_preds)

        # Yearly breakdown
        years = self.data.test["year"].to_numpy()
        yearly_metrics = {}
        for y in sorted(np.unique(years)):
            mask_y = years == y
            yearly_metrics[str(y)] = compute_metrics(y_test[mask_y], test_preds[mask_y])

        # Cluster breakdown
        cluster_metrics = {}
        for c in unique_clusters:
            mask_c = self.labels_test == c
            n_tr = int(np.sum(self.labels_trainval == c))
            n_te = int(np.sum(mask_c))
            m = compute_metrics(y_test[mask_c], test_preds[mask_c])
            m["n_train"] = n_tr
            m["n_test"] = n_te
            cluster_metrics[str(c)] = m

        return HybridCandidateResult(
            model_name=model_name,
            strategy_name=self.strategy_name,
            candidate_id=candidate_id,
            global_features=global_features,
            cluster_additions=cluster_additions,
            pooled_r2=pooled["r2"],
            pooled_rmse=pooled["rmse"],
            pooled_ubrmse=pooled["ubrmse"],
            pooled_bias=pooled["bias"],
            pooled_mae=pooled["mae"],
            pooled_pearson=pooled["pearson"],
            yearly_metrics=yearly_metrics,
            cluster_metrics=cluster_metrics,
            train_time_s=train_time,
            predictions=test_preds,
        )
