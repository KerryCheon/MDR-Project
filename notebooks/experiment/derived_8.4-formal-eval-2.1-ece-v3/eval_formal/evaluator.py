"""Seed-aware model evaluation for derived_8.4-formal-eval-2.1-ece-v3.

Evaluates:
1. Full (temporal) evaluation on derived_8.4 test set (2023-2025, 7 WA stations).
2. Spatial evaluation on derived_8.4_ece_v3 (2026, 5 in-situ sensor stations in WA:
   ECE_BBG_Main_St, ECE_BBG_Lost_Meadow, ECE_Renton_Home, ECE_Renton_Garden_North, ECE_Renton_Garden_Shed;
   150 rows across 2026-07-20 to 2026-08-19).

Models and routers are trained strictly on the 7 Washington state stations (trainval).
The in-situ ECE dataset is completely unseen during training.
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
    """Pooled regression metrics including RMSE, Pearson r, R2, MAE, and bias."""
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


@dataclass
class SpatialEvalResult:
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
    """Evaluates one fixed configuration under temporal (WA test) or spatial (ECE) evaluation."""

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

        # Routing fix options
        fix_cfg = config.get("routing_fix", {})
        self.apply_routing_fix = bool(fix_cfg.get("enabled", True))
        self.tau = float(fix_cfg.get("tau_miss_rate", 0.10))
        self.temperature = float(fix_cfg.get("temperature", 0.25))

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

    def _stem(self, target_tag: str) -> str:
        return f"{self.config_id}__s{self.seed}__{target_tag}"

    def _router(self, apply_fix: bool = True):
        return get_router(
            self.strategy_name,
            self.data.v0_features,
            backbone_54=self.data.shared_backbone_54,
            seed=self.router_seed,
            device=self.config["model"]["exact_params"].get("device", "cuda"),
            apply_fix=apply_fix,
            tau=self.tau,
            temperature=self.temperature,
        )

    def evaluate_full(self) -> FullEvalResult:
        """Temporal evaluation: train on WA trainval, evaluate on WA test set."""
        trainval = self.data.trainval
        test = self.data.test
        y_trainval = trainval[self.data.target].to_numpy(dtype=float)
        y_test = test[self.data.target].to_numpy(dtype=float)

        router = self._router(apply_fix=False)
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
                      router=router, fitted_models=fitted_models,
                      keys=test[["station_id", "date"]])

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

    def evaluate_spatial_ece(self) -> SpatialEvalResult:
        """Spatial evaluation on derived_8.4_ece_v3 with the missingness-aware routing fix."""
        trainval = self.data.trainval
        ece_data = self.data.ece_all
        if ece_data.empty:
            raise ValueError("No in-situ ECE data found in ExperimentData.ece_all")

        y_trainval = trainval[self.data.target].to_numpy(dtype=float)
        y_ece = ece_data[self.data.target].to_numpy(dtype=float)

        router = self._router(apply_fix=self.apply_routing_fix)
        router.fit(trainval)
        labels_trainval = np.asarray(router.predict(trainval)).ravel().astype(int)
        labels_ece = np.asarray(router.predict(ece_data)).ravel().astype(int)

        # Checkpoint-reuse validity: spatial reuses *_full_* checkpoints trained
        # with apply_fix=False. Reuse is valid only if the gate is inactive on
        # clean WA trainval (labels identical either way). Log both rates.
        try:
            from .routers import availability_gate as _gate

            _router_feats: list[str] = getattr(router, "features", [])
            if _router_feats:
                _gated_tr, _, _ = _gate(trainval, _router_feats, self.tau)
                _gated_ece, _, _ = _gate(ece_data, _router_feats, self.tau)
                print(f"[spatial:{self.config_id}:s{self.seed}] gate rate "
                      f"trainval={float(_gated_tr.mean()):.4f} "
                      f"ece={float(_gated_ece.mean()):.4f}", flush=True)
                if float(_gated_tr.mean()) > 0:
                    print(f"[spatial:{self.config_id}:s{self.seed}] WARNING: gate active "
                          f"on WA trainval — *_full_* checkpoint partitions diverge "
                          "from spatial partitions; retrain instead of reusing.",
                          flush=True)
        except Exception as _e:  # diagnostics only; never break evaluation
            print(f"[spatial:{self.config_id}:s{self.seed}] gate diagnostic skipped: {_e}",
                  flush=True)

        # Soft blend weights if supported
        if hasattr(router, "predict_weights"):
            weights = np.asarray(router.predict_weights(ece_data), dtype=float)
        else:
            weights = np.column_stack([labels_ece == 0, labels_ece == 1]).astype(float)

        params = self._params()
        started = perf_counter()
        fitted_models: dict[str, Any] = {}
        expert_preds: dict[str, np.ndarray] = {}

        clusters_to_eval = (0,) if self.strategy_name == "Global_Single" else (0, 1)

        # Check if model checkpoints already exist from full run
        full_stem = self._stem("full")
        for cluster in clusters_to_eval:
            cluster_key = str(cluster)
            features = self.validate_features(
                [*self.global_features, *self.cluster_additions[cluster_key]]
            )
            train_mask = labels_trainval == cluster

            expert = None
            if self.models_dir is not None:
                if self.strategy_name == "Global_Single":
                    ckpt_path = self.models_dir / f"{full_stem}_reg.json"
                else:
                    ckpt_path = self.models_dir / f"{full_stem}_spec_{cluster_key}.json"
                if ckpt_path.exists():
                    try:
                        expert = XGBRegressor()
                        expert.load_model(ckpt_path)
                    except Exception:
                        expert = None

            if expert is None:
                if not train_mask.any():
                    fallback_val = float(np.mean(y_trainval))
                    expert_preds[cluster_key] = np.full(len(ece_data), fallback_val)
                    continue

                expert = XGBRegressor(**params)
                expert.fit(
                    trainval.loc[train_mask, features],
                    y_trainval[train_mask],
                    verbose=False,
                )

            fitted_models[cluster_key] = expert
            booster_feats = expert.get_booster().feature_names or features
            missing_ece = sorted(set(booster_feats) - set(ece_data.columns))
            if missing_ece:
                raise ValueError(
                    f"ECE schema missing {len(missing_ece)} booster feature(s) "
                    f"for {self.config_id}/{cluster_key}: {missing_ece[:10]}. "
                    "Checkpoint feature set diverged from ECE columns."
                )
            expert_preds[cluster_key] = np.asarray(
                expert.predict(ece_data.loc[:, booster_feats])
            ).ravel()

        # Combine predictions using weights
        if self.strategy_name == "Global_Single":
            predictions = expert_preds["0"]
        else:
            p0 = expert_preds.get("0", np.full(len(ece_data), float(np.mean(y_trainval))))
            p1 = expert_preds.get("1", np.full(len(ece_data), float(np.mean(y_trainval))))
            predictions = weights[:, 0] * p0 + weights[:, 1] * p1

        cluster_metrics: dict[str, dict[str, float]] = {}
        for cluster in clusters_to_eval:
            cluster_key = str(cluster)
            train_mask = labels_trainval == cluster
            ece_mask = labels_ece == cluster
            if ece_mask.any():
                cluster_metrics[cluster_key] = {
                    "n_train": float(train_mask.sum()),
                    "n_test": float(ece_mask.sum()),
                    **compute_metrics(y_ece[ece_mask], predictions[ece_mask]),
                }
            else:
                cluster_metrics[cluster_key] = {
                    "n_train": float(train_mask.sum()),
                    "n_test": 0.0,
                    **compute_metrics(np.array([]), np.array([])),
                }

        pooled = compute_metrics(y_ece, predictions)
        pooled["n"] = int(len(y_ece))

        station_ids = ece_data["station_id"].to_numpy()
        station_metrics: dict[str, dict[str, float]] = {}
        for s in sorted(ece_data["station_id"].unique()):
            mask = station_ids == s
            m = compute_metrics(y_ece[mask], predictions[mask])
            m["n"] = int(mask.sum())
            station_metrics[str(s)] = m

        yearly_metrics: dict[str, dict[str, float]] = {}
        for year in sorted(ece_data["year"].unique()):
            mask = ece_data["year"].to_numpy() == year
            yearly_metrics[str(int(year))] = compute_metrics(y_ece[mask], predictions[mask])

        train_time = perf_counter() - started

        stem = self._stem("ece")
        self._persist(stem, predictions, labels_ece, pooled, train_time,
                      n_train_total=int(len(trainval)),
                      router=router, fitted_models=fitted_models,
                      keys=ece_data[["station_id", "date"]])

        return SpatialEvalResult(
            config_id=self.config_id,
            strategy_name=self.strategy_name,
            seed=self.seed,
            n_train_total=int(len(trainval)),
            n_test=int(len(ece_data)),
            pooled=pooled,
            station_metrics=station_metrics,
            yearly_metrics=yearly_metrics,
            cluster_metrics=cluster_metrics,
            train_time_s=train_time,
            predictions=predictions,
            cluster_labels_test=labels_ece,
        )

    def _persist(self, stem: str, predictions: np.ndarray, labels_test: np.ndarray,
                 pooled: dict[str, float], train_time: float, *, n_train_total: int,
                 router, fitted_models: dict[str, Any],
                 keys: pd.DataFrame | None = None) -> None:
        """Save predictions, labels, model weights and a per-fold meta.json under `stem`.

        ``keys`` (station_id, date) is persisted as ``<stem>_keys.parquet`` so
        downstream plots join on keys instead of positional order.
        """
        if self.save_predictions and self.predictions_dir is not None:
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.predictions_dir / f"{stem}_preds.npy", predictions)
            np.save(self.predictions_dir / f"{stem}_labels_te.npy", labels_test)
            if keys is not None and len(keys) == len(predictions):
                try:
                    keys[["station_id", "date"]].to_parquet(
                        self.predictions_dir / f"{stem}_keys.parquet", index=False)
                except Exception as _e:
                    print(f"[persist:{stem}] keys parquet skipped: {_e}", flush=True)

        if self.save_weights and self.models_dir is not None:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(router, TrainedGatingRouter) and router.clf is not None:
                router.clf.save_model(self.models_dir / f"{stem}_gating.json")
            for cl_key, exp_model in fitted_models.items():
                if self.strategy_name == "Global_Single":
                    exp_path = self.models_dir / f"{stem}_reg.json"
                else:
                    exp_path = self.models_dir / f"{stem}_spec_{cl_key}.json"
                if not exp_path.exists():
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
            "pooled_pearson": pooled.get("pearson", float("nan")),
        }
        meta_path: Path | None = None
        if self.models_dir is not None:
            meta_path = self.models_dir / f"{stem}_meta.json"
        elif self.predictions_dir is not None:
            meta_path = self.predictions_dir / f"{stem}_meta.json"
        if meta_path is None:
            raise ValueError(
                "FormalEvaluator has neither models_dir nor predictions_dir; "
                "cannot persist per-job meta.json."
            )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
