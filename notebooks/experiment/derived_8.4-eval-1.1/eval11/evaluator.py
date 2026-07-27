"""Model evaluation, model weight saving, loss curve tracking, and delta grid search."""

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


@dataclass
class StrategyCandidateResult:
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
    rmse_curve: list[float] | None = None
    models: dict[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        c0_add = ";".join(self.cluster_additions.get("0", []))
        c1_add = ";".join(self.cluster_additions.get("1", []))
        return {
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


class StrategyEvaluator:
    def __init__(self, data: ExperimentData, config: dict[str, Any], strategy_name: str, models_dir: Path | None = None) -> None:
        self.data = data
        self.config = config
        self.strategy_name = strategy_name
        self.seed = int(config["model"]["seed"])
        self.router = get_router(strategy_name, data.v0_features, seed=self.seed)
        self.router.fit(data.trainval)
        self.labels_trainval = self.router.predict(data.trainval)
        self.labels_test = self.router.predict(data.test)
        self.source_rank = {feature: index for index, feature in enumerate(data.source_order)}
        self.models_dir = models_dir

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

    def fit_and_evaluate(
        self,
        candidate_id: str,
        global_features: Iterable[str],
        cluster_additions: dict[int | str, Iterable[str]] | None = None,
        *,
        include_predictions: bool = False,
        save_weights: bool = False,
    ) -> StrategyCandidateResult:
        global_features = self.validate_features(global_features)
        raw_additions = cluster_additions or {}
        additions = {
            str(cluster): self.validate_features(feats)
            for cluster, feats in raw_additions.items()
        }
        additions.setdefault("0", [])
        additions.setdefault("1", [])

        y_trainval = self.data.trainval[self.data.target].to_numpy(dtype=float)
        y_test = self.data.test[self.data.target].to_numpy(dtype=float)
        predictions = np.zeros(len(self.data.test), dtype=float)
        cluster_metrics: dict[str, dict[str, float]] = {}
        params = self._params()
        started = perf_counter()

        spec_rmse_curves = []
        spec_N_tests = []
        fitted_models = {}

        if self.strategy_name == "Global_Single":
            clusters_to_eval = (0,)
        else:
            clusters_to_eval = (0, 1)

        for cluster in clusters_to_eval:
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
                    "bias": float("nan"),
                }
                continue

            expert = XGBRegressor(**params)
            
            if test_mask.any():
                expert.fit(
                    self.data.trainval.loc[train_mask, features],
                    y_trainval[train_mask],
                    eval_set=[(self.data.test.loc[test_mask, features], y_test[test_mask])],
                    verbose=False,
                )
                res = expert.evals_result()
                curve = res["validation_0"]["rmse"]
                spec_rmse_curves.append(curve)
                spec_N_tests.append(int(test_mask.sum()))
            else:
                expert.fit(
                    self.data.trainval.loc[train_mask, features],
                    y_trainval[train_mask],
                    verbose=False,
                )

            fitted_models[cluster_key] = expert

            if test_mask.any():
                prediction = np.asarray(
                    expert.predict(self.data.test.loc[test_mask, features])
                ).ravel()
                predictions[test_mask] = prediction
                metrics = compute_metrics(y_test[test_mask], prediction)
            else:
                metrics = compute_metrics(np.array([]), np.array([]))

            cluster_metrics[cluster_key] = {
                "n_train": float(train_mask.sum()),
                "n_test": float(test_mask.sum()),
                **metrics,
            }

        if spec_rmse_curves and sum(spec_N_tests) > 0:
            total_N = sum(spec_N_tests)
            n_steps = len(spec_rmse_curves[0])
            combined_sq = np.zeros(n_steps, dtype=float)
            for crv, n_c in zip(spec_rmse_curves, spec_N_tests):
                combined_sq += (n_c / total_N) * (np.asarray(crv) ** 2)
            combined_rmse_curve = np.sqrt(combined_sq).tolist()
        else:
            combined_rmse_curve = [float("nan")]

        yearly_metrics: dict[str, dict[str, float]] = {}
        for year in sorted(self.data.test["year"].unique()):
            mask = self.data.test["year"].to_numpy() == year
            yearly_metrics[str(int(year))] = compute_metrics(y_test[mask], predictions[mask])

        pooled = compute_metrics(y_test, predictions)
        train_time = perf_counter() - started

        if save_weights and self.models_dir is not None:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "candidate_id": candidate_id,
                "strategy_name": self.strategy_name,
                "train_time_s": train_time,
                "pooled_r2": pooled["r2"],
                "pooled_rmse": pooled["rmse"],
            }
            with open(self.models_dir / f"{candidate_id}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            np.save(self.models_dir / f"{candidate_id}_preds.npy", predictions)
            np.save(self.models_dir / f"{candidate_id}_labels_te.npy", self.labels_test)
            np.save(self.models_dir / f"{candidate_id}_curve.npy", np.array(combined_rmse_curve))

            if isinstance(self.router, TrainedGatingRouter) and self.router.clf is not None:
                self.router.clf.save_model(self.models_dir / f"{candidate_id}_gating.json")

            for cl_key, exp_model in fitted_models.items():
                if self.strategy_name == "Global_Single":
                    exp_path = self.models_dir / f"{candidate_id}_reg.json"
                else:
                    exp_path = self.models_dir / f"{candidate_id}_spec_{cl_key}.json"
                exp_model.save_model(exp_path)

        return StrategyCandidateResult(
            strategy_name=self.strategy_name,
            candidate_id=candidate_id,
            global_features=global_features,
            cluster_additions=additions,
            pooled_r2=pooled["r2"],
            pooled_rmse=pooled["rmse"],
            pooled_ubrmse=pooled["ubrmse"],
            pooled_bias=pooled["bias"],
            pooled_mae=pooled["mae"],
            pooled_pearson=pooled["pearson"],
            yearly_metrics=yearly_metrics,
            cluster_metrics=cluster_metrics,
            train_time_s=train_time,
            predictions=predictions if include_predictions else None,
            rmse_curve=combined_rmse_curve,
            models=fitted_models if save_weights else None,
        )

    def compute_delta_rankings(
        self,
        global_features: list[str],
        predictions: np.ndarray,
        gain_scores: dict[str, float],
        max_additions: int = 10,
    ) -> dict[str, list[str]]:
        candidate_pool = self.data.candidate_pool
        external = [feat for feat in candidate_pool if feat not in set(global_features)]

        delta_rankings: dict[str, list[str]] = {}

        if self.strategy_name == "Global_Single":
            clusters = (0,)
        else:
            clusters = (0, 1)

        for cluster in clusters:
            cluster_key = str(cluster)
            test_mask = self.labels_test == cluster
            if not test_mask.any():
                delta_rankings[cluster_key] = external[:max_additions]
                continue

            y_test_sub = self.data.test.loc[test_mask, self.data.target].to_numpy(dtype=float)
            pred_sub = predictions[test_mask]
            residual = pd.Series(y_test_sub - pred_sub, index=self.data.test.index[test_mask])

            correlations: dict[str, float] = {}
            for feat in external:
                val = self.data.test.loc[test_mask, feat].corr(residual, method="spearman")
                correlations[feat] = float(abs(val)) if pd.notna(val) else 0.0

            gain_sorted = sorted(external, key=lambda f: gain_scores.get(f, 0.0), reverse=True)
            gain_rank = {f: i for i, f in enumerate(gain_sorted)}

            corr_sorted = sorted(external, key=lambda f: correlations[f], reverse=True)
            corr_rank = {f: i for i, f in enumerate(corr_sorted)}

            ranked = sorted(
                external,
                key=lambda f: (gain_rank[f] + corr_rank[f], f),
            )
            delta_rankings[cluster_key] = ranked[:max_additions]

        if "1" not in delta_rankings:
            delta_rankings["1"] = []

        return delta_rankings
