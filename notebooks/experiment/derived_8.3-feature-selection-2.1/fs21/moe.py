"""Causal hard-expert and regime-delta utilities for MoE diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .artifacts import stable_json_hash
from .data import numeric_frame, ordered_feature_hash, resolve_repo_path
from .folds import FoldTask, build_inner_folds
from .ledger import validate_ledger
from .modeling import fit_model, model_configuration_id
from .router import FrozenRouter, fit_router


def load_historical_specialists(moe_config: Mapping[str, object]) -> dict[int, list[str]]:
    spec = dict(moe_config["historical_specialists"])
    path = resolve_repo_path(str(spec["source"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    strategy = str(spec["strategy"])
    clusters = payload.get("clusters", {}).get(strategy)
    if not isinstance(clusters, dict):
        raise ValueError(f"missing historical specialist strategy {strategy} in {path}")
    output = {}
    for regime in (0, 1):
        features = clusters.get(str(regime), {}).get("features")
        if not isinstance(features, list) or not features:
            raise ValueError(f"invalid historical specialist list for regime {regime}")
        output[regime] = list(features)
    return output


def reference_router(
    frame: pd.DataFrame,
    router_config: Mapping[str, object],
    v0_features: list[str],
) -> FrozenRouter:
    reference_years = set(int(value) for value in router_config["reference_years"])
    reference_frame = frame.loc[frame["_year"].isin(reference_years)]
    if set(reference_frame["_year"].unique()) != reference_years:
        raise ValueError("reference router is missing a 2017-2019 year")
    return fit_router(reference_frame, router_config, v0_features, reference=None)


def hard_expert_prediction_rows(
    frame: pd.DataFrame,
    task: FoldTask,
    *,
    candidate: str,
    shared_features: list[str],
    expert_features: Mapping[int, list[str]] | None,
    router_config: Mapping[str, object],
    v0_features: list[str],
    reference: FrozenRouter,
    beta: float,
    config: Mapping[str, object],
    device: str,
    smoke: bool,
) -> tuple[pd.DataFrame, dict]:
    train = frame.iloc[list(task.train_index)]
    validation = frame.iloc[list(task.validation_index)]
    router = fit_router(
        train,
        router_config,
        v0_features,
        reference=reference,
    )
    train_regime, train_distance = router.predict(train)
    validation_regime, validation_distance = router.predict(validation)
    train_router_values = (
        train[v0_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    validation_router_values = (
        validation[v0_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    prediction = np.full(len(validation), np.nan, dtype=float)
    expert_manifest = {}
    for regime in (0, 1):
        train_mask = train_regime == regime
        validation_mask = validation_regime == regime
        if not train_mask.any():
            raise ValueError(
                f"router regime {regime} has zero training rows in {task.fold_id}"
            )
        features = (
            list(expert_features[regime])
            if expert_features is not None
            else list(shared_features)
        )
        if not features or len(features) != len(set(features)):
            raise ValueError(f"invalid expert feature list for regime {regime}")
        model = fit_model(
            numeric_frame(train.loc[train_mask], features),
            train.loc[train_mask, str(config["data"]["target"])],
            train_years=train.loc[train_mask, "_year"],
            beta=beta,
            config=config,
            seed=task.learner_seed,
            device=device,
            smoke=smoke,
        )
        if validation_mask.any():
            prediction[validation_mask] = model.predict(
                numeric_frame(validation.loc[validation_mask], features)
            )
        expert_manifest[str(regime)] = {
            "features": features,
            "ordered_feature_hash": ordered_feature_hash(features),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
        }
    if not np.isfinite(prediction).all():
        raise ValueError(f"hard experts left predictions missing in {task.fold_id}")
    data = dict(config["data"])
    truth = validation[str(data["target"])].to_numpy(dtype=float)
    residual = truth - prediction
    shared_hash = ordered_feature_hash(shared_features)
    router_payload = router.to_dict()
    configuration_id = model_configuration_id(
        candidate=candidate,
        feature_hash=stable_json_hash(expert_manifest),
        beta=beta,
        learner_seed=task.learner_seed,
        device=device,
        kind="shared_hard_experts",
        router_hash=router_payload["router_hash"],
    )
    ledger = pd.DataFrame(
        {
            "model": "1.3-lite-hard-experts",
            "candidate": candidate,
            "path_source": "moe_causal",
            "endpoint": len(shared_features),
            "actual_count": max(
                len(row["features"]) for row in expert_manifest.values()
            ),
            "ordered_feature_hash": shared_hash,
            "fold_family": task.family,
            "outer_origin": task.origin,
            "fold_id": task.fold_id,
            "station_partition_seed": task.partition_seed,
            "learner_seed": task.learner_seed,
            "station": validation[str(data["station_col"])].astype(str).to_numpy(),
            "date": validation[str(data["time_col"])].astype(str).to_numpy(),
            "year": validation["_year"].to_numpy(dtype=int),
            "month": validation["_month"].to_numpy(dtype=int),
            "truth": truth,
            "prediction": prediction,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "squared_error": np.square(residual),
            "beta": float(beta),
            "model_config_id": configuration_id,
            "router_regime": validation_regime,
            "route_distance": validation_distance,
        }
    )
    metadata = {
        "candidate": candidate,
        "fold_id": task.fold_id,
        "router": router_payload,
        "expert_manifest": expert_manifest,
        "train_regime_counts": {
            str(regime): int((train_regime == regime).sum()) for regime in (0, 1)
        },
        "validation_regime_counts": {
            str(regime): int((validation_regime == regime).sum())
            for regime in (0, 1)
        },
        "train_route_distance_mean": float(np.mean(train_distance)),
        "validation_route_distance_mean": float(np.mean(validation_distance)),
        "centroid_drift_from_reference": _centroid_drift(router, reference),
        "router_feature_missingness": [
            {
                "feature": feature,
                "train_missing_rate": float(train_router_values[feature].isna().mean()),
                "validation_missing_rate": float(
                    validation_router_values[feature].isna().mean()
                ),
            }
            for feature in v0_features
        ],
    }
    return validate_ledger(ledger), metadata


def _centroid_drift(router: FrozenRouter, reference: FrozenRouter) -> dict[str, float]:
    raw_centers = router.centers * router.scaler_scale + router.scaler_mean
    reference_space = (raw_centers - reference.scaler_mean) / reference.scaler_scale
    return {
        str(router.label_mapping[raw]): float(
            np.linalg.norm(reference_space[raw] - reference.centers[aligned])
        )
        for raw, aligned in router.label_mapping.items()
    }


def _macro_rmse(
    validation: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
    station_col: str,
) -> float:
    table = pd.DataFrame(
        {
            "station": validation[station_col].astype(str).to_numpy(),
            "year": validation["_year"].to_numpy(dtype=int),
            "squared_error": np.square(truth - prediction),
        }
    )
    return float(
        table.groupby(["station", "year"])["squared_error"].mean().pow(0.5).mean()
    )


def rank_regime_additions(
    outer_training: pd.DataFrame,
    *,
    regime: int,
    shared_features: list[str],
    universe: list[str],
    source_family: str,
    partition_seed: int,
    learner_seed: int,
    router_config: Mapping[str, object],
    v0_features: list[str],
    reference: FrozenRouter,
    config: Mapping[str, object],
    device: str,
    smoke: bool,
    permutation_repeats: int,
) -> tuple[list[str], pd.DataFrame, dict]:
    """Rank unused inputs conditionally on the complete shared backbone."""
    unused = [feature for feature in universe if feature not in set(shared_features)]
    if not unused:
        return (
            [],
            pd.DataFrame(
                columns=[
                    "feature",
                    "original_position",
                    "importance_mean",
                    "importance_standard_error",
                    "importance_lcb",
                    "importance_observations",
                    "rank",
                ]
            ),
            {"usable_folds": 0, "reason": "no_unused_features"},
        )
    folds = build_inner_folds(
        outer_training,
        config,
        family=source_family,
        partition_seed=partition_seed,
    )
    positions = {feature: index for index, feature in enumerate(universe)}
    deltas = defaultdict(list)
    usable = []
    fold_details = []
    batch_size = int(config["ranking"]["permutation_batch_size"])
    data = dict(config["data"])
    for fold_number, fold in enumerate(folds):
        train = outer_training.iloc[list(fold.train_index)]
        validation = outer_training.iloc[list(fold.validation_index)]
        router = fit_router(train, router_config, v0_features, reference=reference)
        train_labels, _ = router.predict(train)
        validation_labels, _ = router.predict(validation)
        train_mask = train_labels == int(regime)
        validation_mask = validation_labels == int(regime)
        if train_mask.sum() < int(config["folds"]["minimum_train_rows"]):
            continue
        if validation_mask.sum() < int(config["folds"]["minimum_validation_rows"]):
            continue
        regime_train = train.loc[train_mask]
        regime_validation = validation.loc[validation_mask]
        shared_model = fit_model(
            numeric_frame(regime_train, shared_features),
            regime_train[str(data["target"])],
            train_years=regime_train["_year"],
            beta=0.0,
            config=config,
            seed=learner_seed,
            device=device,
            smoke=smoke,
        )
        train_shared_prediction = np.asarray(
            shared_model.predict(numeric_frame(regime_train, shared_features)),
            dtype=float,
        )
        training_residual = (
            regime_train[str(data["target"])].to_numpy(dtype=float)
            - train_shared_prediction
        )
        residual_model = fit_model(
            numeric_frame(regime_train, unused),
            training_residual,
            train_years=regime_train["_year"],
            beta=0.0,
            config=config,
            seed=learner_seed,
            device=device,
            smoke=smoke,
        )
        X_validation = numeric_frame(regime_validation, unused)
        truth = regime_validation[str(data["target"])].to_numpy(dtype=float)
        shared_prediction = np.asarray(
            shared_model.predict(
                numeric_frame(regime_validation, shared_features)
            ),
            dtype=float,
        )
        residual_prediction = np.asarray(
            residual_model.predict(X_validation), dtype=float
        )
        baseline_prediction = shared_prediction + residual_prediction
        baseline = _macro_rmse(
            regime_validation,
            truth,
            baseline_prediction,
            str(data["station_col"]),
        )
        shared_only = _macro_rmse(
            regime_validation,
            truth,
            shared_prediction,
            str(data["station_col"]),
        )
        usable.append(fold.fold_id)
        fold_details.append(
            {
                "fold_id": fold.fold_id,
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "training_residual_mean": float(np.mean(training_residual)),
                "training_residual_standard_deviation": float(
                    np.std(training_residual)
                ),
                "shared_only_validation_macro_rmse": shared_only,
                "full_residual_model_validation_macro_rmse": baseline,
            }
        )
        for repeat in range(int(permutation_repeats)):
            for start in range(0, len(unused), batch_size):
                batch = unused[start : start + batch_size]
                frames = []
                for feature in batch:
                    rng = np.random.default_rng(
                        learner_seed
                        + fold_number * 1_000_003
                        + repeat * 10_007
                        + positions[feature] * 101
                    )
                    permuted = X_validation.copy()
                    values = permuted[feature].to_numpy(copy=True)
                    permuted[feature] = values[rng.permutation(len(values))]
                    frames.append(permuted)
                predictions = np.asarray(
                    residual_model.predict(
                        pd.concat(frames, ignore_index=True)
                    ),
                    dtype=float,
                ).reshape(len(batch), len(X_validation))
                for index, feature in enumerate(batch):
                    score = _macro_rmse(
                        regime_validation,
                        truth,
                        shared_prediction + predictions[index],
                        str(data["station_col"]),
                    )
                    deltas[feature].append(score - baseline)
    rows = []
    for feature in unused:
        values = np.asarray(deltas.get(feature, []), dtype=float)
        if len(values):
            mean = float(np.mean(values))
            se = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            lcb = mean - 1.96 * se
        else:
            mean, se, lcb = float("-inf"), 0.0, float("-inf")
        rows.append(
            {
                "feature": feature,
                "original_position": positions[feature],
                "importance_mean": mean,
                "importance_standard_error": se,
                "importance_lcb": lcb,
                "importance_observations": len(values),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["importance_lcb", "importance_mean", "original_position"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    table["rank"] = np.arange(1, len(table) + 1)
    ordered = table["feature"].tolist() if usable else []
    return ordered, table.reset_index(drop=True), {
        "usable_folds": len(usable),
        "fold_ids": usable,
        "fold_details": fold_details,
        "conditional_on_complete_shared_backbone": True,
        "ranking_target": "training_frame_truth_minus_shared_prediction",
        "target_residual_sign": "truth_minus_prediction",
    }


def regime_coverage(ledger: pd.DataFrame) -> pd.DataFrame:
    return (
        ledger.groupby(["router_regime", "outer_origin"], sort=True)
        .agg(
            row_count=("truth", "size"),
            station_count=("station", "nunique"),
            target_standard_deviation=("truth", "std"),
        )
        .reset_index()
    )
