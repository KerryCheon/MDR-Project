"""Causal permutation ranking and direct/progressive pruning paths."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .data import numeric_frame, ordered_feature_hash
from .folds import FoldTask
from .modeling import fit_model


def progressive_bridge_sizes(start_size: int, endpoints: list[int]) -> list[dict]:
    current = int(start_size)
    if any(int(value) <= 0 for value in endpoints):
        raise ValueError("endpoint sizes must be positive")
    rows = []
    for endpoint_value in endpoints:
        endpoint = int(endpoint_value)
        if endpoint >= current:
            if endpoint == current:
                rows.append({"size": endpoint, "endpoint": True})
            continue
        while current > endpoint:
            next_size = max(endpoint, current - endpoint)
            rows.append({"size": int(next_size), "endpoint": next_size == endpoint})
            current = next_size
    return rows


def direct_sizes(start_size: int, endpoints: list[int]) -> list[dict]:
    return [
        {"size": int(size), "endpoint": True}
        for size in endpoints
        if int(size) < int(start_size)
    ]


def _station_year_macro_rmse(
    context: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
    station_col: str,
) -> float:
    residual_sq = np.square(np.asarray(truth) - np.asarray(prediction))
    values = pd.DataFrame(
        {
            "station": context[station_col].astype(str).to_numpy(),
            "year": context["_year"].to_numpy(dtype=int),
            "squared_error": residual_sq,
        }
    )
    return float(
        values.groupby(["station", "year"], sort=True)["squared_error"]
        .mean()
        .pow(0.5)
        .mean()
    )


def rank_features(
    frame: pd.DataFrame,
    features: list[str],
    folds: list[FoldTask],
    *,
    config: Mapping[str, object],
    learner_seed: int,
    device: str,
    permutation_repeats: int,
    smoke: bool = False,
    original_positions: Mapping[str, int] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Rank by macro-RMSE importance LCB, mean, then original position."""
    if len(features) != len(set(features)):
        raise ValueError("ranking features contain duplicates")
    data = dict(config["data"])
    ranking = dict(config["ranking"])
    station_col = str(data["station_col"])
    target = str(data["target"])
    if original_positions is None:
        positions = {feature: index for index, feature in enumerate(features)}
    else:
        missing_positions = sorted(set(features).difference(original_positions))
        if missing_positions:
            raise ValueError(
                f"original positions are missing features: {missing_positions[:10]}"
            )
        positions = {
            feature: int(original_positions[feature]) for feature in features
        }
    importance = {feature: [] for feature in features}
    batch_size = max(1, int(ranking["permutation_batch_size"]))

    for fold_number, fold in enumerate(folds):
        train = frame.iloc[list(fold.train_index)]
        validation = frame.iloc[list(fold.validation_index)]
        X_train = numeric_frame(train, features)
        X_validation = numeric_frame(validation, features)
        truth = validation[target].to_numpy(dtype=float)
        model = fit_model(
            X_train,
            train[target].to_numpy(dtype=float),
            train_years=train["_year"].to_numpy(dtype=int),
            beta=0.0,
            config=config,
            seed=int(learner_seed),
            device=device,
            smoke=smoke,
        )
        baseline_prediction = np.asarray(model.predict(X_validation), dtype=float)
        baseline = _station_year_macro_rmse(
            validation, truth, baseline_prediction, station_col
        )
        for repeat in range(int(permutation_repeats)):
            for start in range(0, len(features), batch_size):
                batch = features[start : start + batch_size]
                permuted_frames = []
                for feature in batch:
                    rng = np.random.default_rng(
                        int(learner_seed)
                        + fold_number * 1_000_003
                        + repeat * 10_007
                        + positions[feature] * 101
                    )
                    permuted = X_validation.copy()
                    values = permuted[feature].to_numpy(copy=True)
                    permuted[feature] = values[rng.permutation(len(values))]
                    permuted_frames.append(permuted)
                predictions = np.asarray(
                    model.predict(pd.concat(permuted_frames, ignore_index=True)),
                    dtype=float,
                ).reshape(len(batch), len(validation))
                for index, feature in enumerate(batch):
                    score = _station_year_macro_rmse(
                        validation,
                        truth,
                        predictions[index],
                        station_col,
                    )
                    importance[feature].append(float(score - baseline))

    rows = []
    for feature in features:
        values = np.asarray(importance[feature], dtype=float)
        mean = float(np.mean(values))
        standard_error = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        lcb = mean - 1.96 * standard_error
        rows.append(
            {
                "feature": feature,
                "original_position": positions[feature],
                "importance_mean": mean,
                "importance_standard_error": standard_error,
                "importance_lcb": lcb,
                "importance_observations": len(values),
            }
        )
    detail = pd.DataFrame(rows).sort_values(
        ["importance_lcb", "importance_mean", "original_position"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    detail["rank"] = np.arange(1, len(detail) + 1)
    ordered = detail["feature"].tolist()
    return ordered, detail.reset_index(drop=True)


def pruning_path(
    frame: pd.DataFrame,
    universe: list[str],
    folds: list[FoldTask],
    *,
    method: str,
    endpoints: list[int],
    config: Mapping[str, object],
    learner_seed: int,
    device: str,
    permutation_repeats: int,
    smoke: bool = False,
) -> dict:
    if method == "direct":
        sizes = direct_sizes(len(universe), endpoints)
    elif method == "progressive":
        sizes = progressive_bridge_sizes(len(universe), endpoints)
    else:
        raise ValueError(f"unknown pruning method: {method}")
    current = list(universe)
    original_positions = {
        feature: position for position, feature in enumerate(universe)
    }
    steps = []
    endpoint_features = {}
    for step_number, step in enumerate(sizes):
        ordered, detail = rank_features(
            frame,
            current,
            folds,
            config=config,
            learner_seed=learner_seed,
            device=device,
            permutation_repeats=permutation_repeats,
            smoke=smoke,
            original_positions=original_positions,
        )
        current = ordered[: int(step["size"])]
        row = {
            "step": step_number,
            "size": len(current),
            "endpoint": bool(step["endpoint"]),
            "features": list(current),
            "ordered_feature_hash": ordered_feature_hash(current),
            "ranking": detail.to_dict(orient="records"),
        }
        steps.append(row)
        if step["endpoint"]:
            endpoint_features[str(len(current))] = list(current)
    return {
        "method": method,
        "start_size": len(universe),
        "endpoints": endpoint_features,
        "steps": steps,
    }


def complete_order_for_endpoint(path: Mapping[str, object], endpoint: int) -> list[str]:
    """Reconstruct a full-universe order from a nested refitted pruning path.

    The final reduction step only ranks the predictors that survived earlier
    steps.  Consensus percentile ranks nevertheless require every predictor.
    Survivors therefore come first in their latest refitted order, followed by
    eliminated cohorts in reverse elimination order.  Each cohort preserves
    the ranking from the step at which it was removed.
    """
    endpoint = int(endpoint)
    steps = []
    for raw_step in path["steps"]:
        step = dict(raw_step)
        steps.append(step)
        if int(step["size"]) == endpoint:
            break
    if not steps or int(steps[-1]["size"]) != endpoint:
        raise ValueError(f"pruning path has no endpoint {endpoint}")
    selected = list(path["endpoints"][str(endpoint)])
    ordered = list(selected)
    seen = set(ordered)
    for step in reversed(steps):
        retained = set(step["features"])
        ranked = [row["feature"] for row in step["ranking"]]
        for feature in ranked:
            if feature not in retained and feature not in seen:
                ordered.append(feature)
                seen.add(feature)
    expected_count = int(path["start_size"])
    if len(ordered) != expected_count or len(seen) != expected_count:
        raise ValueError(
            "could not reconstruct a complete pruning order: "
            f"expected {expected_count}, got {len(ordered)}"
        )
    return ordered


def consensus_features(
    rankings: list[dict],
    *,
    count: int,
    universe: list[str],
) -> tuple[list[str], pd.DataFrame]:
    """Use frequency, median rank, mean rank, and original position."""
    positions = {feature: index for index, feature in enumerate(universe)}
    universe_set = set(universe)
    rows = []
    for feature in universe:
        percentile_ranks = []
        selected = 0
        support_years = []
        for ranking in rankings:
            ordered = list(ranking["ordered"])
            if len(ordered) != len(universe) or set(ordered) != universe_set:
                raise ValueError(
                    "each consensus ranking must order the complete feature universe"
                )
            selected_features = set(ranking["selected"])
            rank_by_feature = {
                value: index + 1 for index, value in enumerate(ordered)
            }
            rank = rank_by_feature[feature]
            percentile_ranks.append(rank / len(ordered))
            if feature in selected_features:
                selected += 1
                support_years.append(int(ranking["year"]))
        rows.append(
            {
                "feature": feature,
                "selection_frequency": selected,
                "median_percentile_rank": float(np.median(percentile_ranks)),
                "mean_percentile_rank": float(np.mean(percentile_ranks)),
                "original_position": positions[feature],
                "support_years": "|".join(str(year) for year in support_years),
                "support_2020": 2020 in support_years,
                "support_2021": 2021 in support_years,
                "support_2022": 2022 in support_years,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        [
            "selection_frequency",
            "median_percentile_rank",
            "mean_percentile_rank",
            "original_position",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    ranking_years = {int(ranking["year"]) for ranking in rankings}
    table["selection_rate"] = table["selection_frequency"] / len(rankings)
    if {2020, 2021, 2022}.issubset(ranking_years):
        late_support = table[["support_2021", "support_2022"]].sum(axis=1)
        table["late_year_support_count"] = late_support
        table["late_support_change_vs_2020"] = (
            late_support / 2.0 - table["support_2020"].astype(float)
        )
        table["support_trend"] = np.select(
            [
                ~table["support_2020"] & (late_support > 0),
                table["support_2020"] & (late_support < 2),
                table["support_2020"] & (late_support == 2),
            ],
            ["gained_in_2021_2022", "lost_in_2021_2022", "sustained"],
            default="unsupported",
        )
    else:
        table["late_year_support_count"] = 0
        table["late_support_change_vs_2020"] = np.nan
        table["support_trend"] = "not_available_in_smoke"
    return table.head(int(count))["feature"].tolist(), table.reset_index(drop=True)
