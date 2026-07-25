"""The frozen 1.3-lite learner and its only permitted training weights."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from xgboost import XGBRegressor

from .artifacts import stable_json_hash
from .constants import EXACT_LEARNER_PARAMS


def validate_exact_learner(config: Mapping[str, object]) -> None:
    observed = dict(config["learner"])
    if observed != EXACT_LEARNER_PARAMS:
        differences = {
            key: (observed.get(key), expected)
            for key, expected in EXACT_LEARNER_PARAMS.items()
            if observed.get(key) != expected
        }
        extras = sorted(set(observed).difference(EXACT_LEARNER_PARAMS))
        raise ValueError(
            f"learner is not exact 1.3-lite; differences={differences}, extras={extras}"
        )
    if int(observed["n_jobs"]) != 1:
        raise ValueError("every XGBoost fit must retain n_jobs=1")


def temporal_weights(years, beta: float) -> np.ndarray | None:
    """Return normalized recency weights, or no weights for beta zero."""
    beta = float(beta)
    if beta == 0.0:
        return None
    if beta != 0.2:
        raise ValueError(f"unsupported beta: {beta}")
    values = np.asarray(years, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("training years must be finite and nonempty")
    weights = np.exp(beta * (values - float(np.max(values))))
    return weights / float(np.mean(weights))


def learner_parameters(
    config: Mapping[str, object],
    *,
    seed: int,
    device: str,
    smoke: bool = False,
) -> dict:
    validate_exact_learner(config)
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported XGBoost device: {device}")
    params = dict(config["learner"])
    params.update({"random_state": int(seed), "device": device, "verbosity": 0})
    if smoke:
        smoke_config = dict(dict(config["runtime"])["smoke"])
        params["n_estimators"] = int(smoke_config["n_estimators"])
        params["max_depth"] = int(smoke_config["max_depth"])
    return params


def fit_model(
    X_train,
    y_train,
    *,
    train_years,
    beta: float,
    config: Mapping[str, object],
    seed: int,
    device: str,
    smoke: bool = False,
) -> XGBRegressor:
    """Fit without feature imputation; XGBoost receives NaN natively."""
    model = XGBRegressor(
        **learner_parameters(config, seed=seed, device=device, smoke=smoke)
    )
    weights = temporal_weights(train_years, beta)
    model.fit(X_train, np.asarray(y_train, dtype=float), sample_weight=weights)
    return model


def model_configuration_id(
    *,
    candidate: str,
    feature_hash: str,
    beta: float,
    learner_seed: int,
    device: str,
    kind: str = "single_global",
    router_hash: str | None = None,
) -> str:
    return stable_json_hash(
        {
            "candidate": candidate,
            "feature_hash": feature_hash,
            "beta": float(beta),
            "learner_seed": int(learner_seed),
            "device": device,
            "kind": kind,
            "router_hash": router_hash,
        }
    )

