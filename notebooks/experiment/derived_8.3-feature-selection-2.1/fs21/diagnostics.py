"""Ledger-derived station, month, temporal, and correlation diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .artifacts import atomic_write_csv
from .data import numeric_frame
from .folds import FoldTask
from .ledger import collapse_secondary_repeats, prediction_rows, validate_ledger
from .metrics import secondary_metric_tables
from .modeling import fit_model


def candidate_diagnostic_ids(candidate_features: Mapping[str, object]) -> dict[str, str]:
    challenger = str(candidate_features["benchmark_challenger"])
    if challenger == "V0":
        selected_id = "V0"
        union_id = "V0"
    else:
        form, source, endpoint = challenger.split("__")
        selected_id = f"selected_k__{source}__{endpoint}"
        union_id = f"v0_union_selected_k__{source}__{endpoint}"
    return {
        "V0": "V0",
        "selected_2_1": selected_id,
        "union_2_1_v0": union_id,
        "all_predictors": "all_predictors",
    }


def write_metric_tables(
    ledger: pd.DataFrame,
    candidate_ids: Mapping[str, str],
    output_dir: Path,
    *,
    variance_epsilon: float,
) -> dict[str, pd.DataFrame]:
    accumulated = {
        name: [] for name in ("overall", "year", "month", "station", "station_year")
    }
    for display_name, candidate in candidate_ids.items():
        tables = secondary_metric_tables(
            ledger,
            candidate,
            beta=0.0,
            variance_epsilon=variance_epsilon,
        )
        for table_name, table in tables.items():
            copy = table.copy()
            copy.insert(0, "model", display_name)
            copy.insert(1, "candidate", candidate)
            accumulated[table_name].append(copy)
    outputs = {}
    for name, frames in accumulated.items():
        outputs[name] = pd.concat(frames, ignore_index=True)
        atomic_write_csv(outputs[name], output_dir / f"metrics_by_{name}.csv")
    return outputs


def _station_rmse_interval(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict:
    keys = ["fold_family", "outer_origin", "station", "date", "truth"]
    merged = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("station diagnostic rows are not paired")
    rng = np.random.default_rng(int(seed))
    origins = sorted(merged["outer_origin"].unique().tolist())
    deltas = []
    for _ in range(int(replicates)):
        sampled_origins = rng.choice(origins, size=len(origins), replace=True)
        pieces = []
        for origin in sampled_origins:
            block = merged.loc[merged["outer_origin"] == int(origin)]
            indices = rng.integers(0, len(block), size=len(block))
            pieces.append(block.iloc[indices])
        sample = pd.concat(pieces, ignore_index=True)
        left_rmse = float(np.sqrt(sample["squared_error_left"].mean()))
        right_rmse = float(np.sqrt(sample["squared_error_right"].mean()))
        deltas.append(left_rmse - right_rmse)
    values = np.asarray(deltas, dtype=float)
    point = float(
        np.sqrt(merged["squared_error_left"].mean())
        - np.sqrt(merged["squared_error_right"].mean())
    )
    return {
        "delta_rmse": point,
        "ci_lower": float(np.quantile(values, 0.025)),
        "ci_upper": float(np.quantile(values, 0.975)),
        "bootstrap_standard_error": float(np.std(values, ddof=1)),
    }


def station_sufficiency_classification(
    ledger: pd.DataFrame,
    metric_tables: Mapping[str, pd.DataFrame],
    candidate_ids: Mapping[str, str],
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    collapsed = collapse_secondary_repeats(ledger)
    compact_id = candidate_ids["selected_2_1"]
    union_id = candidate_ids["union_2_1_v0"]
    all_id = candidate_ids["all_predictors"]
    v0_id = candidate_ids["V0"]
    station_metrics = metric_tables["station"]
    compact_metrics = station_metrics.loc[
        station_metrics["model"] == "selected_2_1"
    ].set_index("station")
    union_metrics = station_metrics.loc[
        station_metrics["model"] == "union_2_1_v0"
    ].set_index("station")
    all_metrics = station_metrics.loc[
        station_metrics["model"] == "all_predictors"
    ].set_index("station")
    v0_metrics = station_metrics.loc[
        station_metrics["model"] == "V0"
    ].set_index("station")
    target_std_threshold = float(compact_metrics["target_standard_deviation"].quantile(0.25))
    compact_median_rmse = float(compact_metrics["RMSE"].median())
    difficult_threshold = float(v0_metrics["RMSE"].quantile(0.9))
    interval_rows = []
    classifications = []
    for station in sorted(compact_metrics.index):
        compact_rows = collapsed.loc[
            (collapsed["candidate"] == compact_id)
            & (collapsed["station"] == station)
        ]
        all_rows = collapsed.loc[
            (collapsed["candidate"] == all_id)
            & (collapsed["station"] == station)
        ]
        comparison_rows = {
            "all_minus_selected": compact_rows,
            "all_minus_union": collapsed.loc[
                (collapsed["candidate"] == union_id)
                & (collapsed["station"] == station)
            ],
            "all_minus_v0": collapsed.loc[
                (collapsed["candidate"] == v0_id)
                & (collapsed["station"] == station)
            ],
        }
        intervals = {}
        for comparison_name, reference_rows in comparison_rows.items():
            interval = _station_rmse_interval(
                all_rows,
                reference_rows,
                replicates=replicates,
                seed=seed,
            )
            intervals[comparison_name] = interval
            interval_rows.append(
                {
                    "station": station,
                    "comparison": comparison_name,
                    **interval,
                }
            )
        interval = intervals["all_minus_selected"]
        compact = compact_metrics.loc[station]
        union = union_metrics.loc[station]
        all_row = all_metrics.loc[station]
        v0 = v0_metrics.loc[station]
        compact_is_best = float(compact["RMSE"]) <= float(v0["RMSE"])
        best_global = compact if compact_is_best else v0
        r2_value = best_global["R2"]
        best_global_rmse = float(best_global["RMSE"])
        best_global_standard_error = intervals[
            "all_minus_selected" if compact_is_best else "all_minus_v0"
        ]["bootstrap_standard_error"]
        low_variance_artifact = (
            (pd.isna(r2_value) or float(r2_value) < 0.0)
            and float(compact["target_standard_deviation"]) <= target_std_threshold
            and best_global_rmse <= compact_median_rmse
        )
        selection_failure = (
            interval["ci_upper"] < 0.0
            and intervals["all_minus_union"]["ci_upper"] < 0.0
            and intervals["all_minus_v0"]["ci_upper"] < 0.0
        )
        current_input_limitation = (
            min(
                float(compact["RMSE"]),
                float(union["RMSE"]),
                float(v0["RMSE"]),
                float(all_row["RMSE"]),
            )
            >= difficult_threshold
            and all(
                row["ci_lower"] <= 0.0 <= row["ci_upper"]
                for row in intervals.values()
            )
        )
        global_sufficient = (
            best_global_rmse
            <= float(all_row["RMSE"]) + best_global_standard_error
            and (pd.isna(r2_value) or float(r2_value) >= 0.0)
        )
        if low_variance_artifact:
            label = "low_target_variance_artifact"
        elif selection_failure:
            label = "selection_failure"
        elif current_input_limitation:
            label = "current_input_limitation"
        elif global_sufficient:
            label = "global_features_sufficient"
        else:
            label = "uncertain"
        classifications.append(
            {
                "station": station,
                "classification": label,
                "compact_R2": compact["R2"],
                "compact_R2_reason": compact["R2_reason"],
                "best_global_R2": r2_value,
                "best_global_R2_reason": best_global["R2_reason"],
                "compact_RMSE": compact["RMSE"],
                "compact_MAE": compact["MAE"],
                "all_predictor_RMSE": all_row["RMSE"],
                "union_RMSE": union["RMSE"],
                "v0_RMSE": v0["RMSE"],
                "target_count": compact["target_count"],
                "target_range": compact["target_range"],
                "target_variance": compact["target_variance"],
                "target_standard_deviation": compact[
                    "target_standard_deviation"
                ],
                "classification_thresholds_data_derived": True,
            }
        )
    return pd.DataFrame(classifications), pd.DataFrame(interval_rows)


def station_input_diagnostics(
    frame: pd.DataFrame,
    features: list[str],
    config: Mapping[str, object],
) -> pd.DataFrame:
    data = dict(config["data"])
    station_col = str(data["station_col"])
    lower_quantile, upper_quantile = (
        float(value) for value in config["diagnostics"]["input_distance_quantiles"]
    )
    rows = []
    for origin in config["folds"]["outer_origins"]:
        training = frame.loc[frame["_year"] < int(origin)]
        validation = frame.loc[frame["_year"] == int(origin)]
        train_values = numeric_frame(training, features)
        train_low = train_values.quantile(lower_quantile)
        train_high = train_values.quantile(upper_quantile)
        train_q25 = train_values.quantile(0.25)
        train_q75 = train_values.quantile(0.75)
        train_mean = train_values.mean()
        train_std = train_values.std().replace(0.0, np.nan)
        for station, station_frame in validation.groupby(station_col, sort=True):
            values = numeric_frame(station_frame, features)
            for feature in features:
                finite = values[feature].dropna()
                missing_rate = float(values[feature].isna().mean())
                if finite.empty or pd.isna(train_low[feature]):
                    out_of_range = float("nan")
                    distance = float("nan")
                    quantile_distance = float("nan")
                else:
                    out_of_range = float(
                        (
                            (finite < train_low[feature])
                            | (finite > train_high[feature])
                        ).mean()
                    )
                    scale = train_std[feature]
                    distance = (
                        float(abs(finite.mean() - train_mean[feature]) / scale)
                        if pd.notna(scale) and float(scale) > 0.0
                        else float("nan")
                    )
                    train_iqr = train_q75[feature] - train_q25[feature]
                    station_low = finite.quantile(lower_quantile)
                    station_high = finite.quantile(upper_quantile)
                    quantile_distance = (
                        float(
                            (
                                abs(station_low - train_low[feature])
                                + abs(station_high - train_high[feature])
                            )
                            / (2.0 * train_iqr)
                        )
                        if pd.notna(train_iqr) and float(train_iqr) > 0.0
                        else float("nan")
                    )
                rows.append(
                    {
                        "station": str(station),
                        "outer_origin": int(origin),
                        "feature": feature,
                        "missing_rate": missing_rate,
                        "out_of_training_range_rate": out_of_range,
                        "standardized_mean_distance": distance,
                        "normalized_quantile_distance": quantile_distance,
                        "training_range_lower_quantile": lower_quantile,
                        "training_range_upper_quantile": upper_quantile,
                    }
                )
    return pd.DataFrame(rows)


def residual_distribution(ledger: pd.DataFrame, candidate_ids: Mapping[str, str]) -> pd.DataFrame:
    collapsed = collapse_secondary_repeats(
        ledger.loc[ledger["candidate"].isin(set(candidate_ids.values()))]
    )
    reverse = {candidate: name for name, candidate in candidate_ids.items()}
    rows = []
    for (candidate, station, month), group in collapsed.groupby(
        ["candidate", "station", "month"], sort=True
    ):
        residual = group["residual"]
        rows.append(
            {
                "model": reverse[candidate],
                "candidate": candidate,
                "station": station,
                "month": int(month),
                "count": len(group),
                "residual_mean": float(residual.mean()),
                "residual_std": float(residual.std(ddof=1)),
                "residual_q05": float(residual.quantile(0.05)),
                "residual_q25": float(residual.quantile(0.25)),
                "residual_median": float(residual.median()),
                "residual_q75": float(residual.quantile(0.75)),
                "residual_q95": float(residual.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def seasonal_climatology(frame: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    data = dict(config["data"])
    return (
        frame.groupby([str(data["station_col"]), "_month"], sort=True)[
            str(data["target"])
        ]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .rename(columns={str(data["station_col"]): "station", "_month": "month"})
    )


def fit_window_ledger(
    frame: pd.DataFrame,
    candidate_features: Mapping[str, list[str]],
    config: Mapping[str, object],
    *,
    device: str,
    smoke: bool,
) -> pd.DataFrame:
    year_values = frame["_year"].to_numpy(dtype=int)
    rows = []
    for origin in config["folds"]["outer_origins"]:
        validation = np.flatnonzero(year_values == int(origin))
        for window in ("fixed_2017_2019", "expanding"):
            if window == "fixed_2017_2019":
                training = np.flatnonzero((year_values >= 2017) & (year_values <= 2019))
            else:
                training = np.flatnonzero(year_values < int(origin))
            task = FoldTask(
                family="forward_time",
                origin=int(origin),
                fold_id=f"{window}_{origin}",
                partition_seed=None,
                learner_seed=42,
                held_stations=(),
                train_index=tuple(int(value) for value in training),
                validation_index=tuple(int(value) for value in validation),
            )
            for name, features in candidate_features.items():
                ledger = prediction_rows(
                    frame,
                    task,
                    features,
                    candidate=f"{name}__{window}",
                    path_source="fit_window_diagnostic",
                    endpoint=len(features),
                    beta=0.0,
                    config=config,
                    device=device,
                    smoke=smoke,
                    model_name="1.3-lite-fit-window-diagnostic",
                )
                rows.append(ledger)
    return validate_ledger(pd.concat(rows, ignore_index=True))


def correlation_components(
    training: pd.DataFrame,
    features: list[str],
    *,
    threshold: float,
) -> tuple[list[list[str]], pd.DataFrame]:
    correlation = numeric_frame(training, features).corr(method="spearman")
    parent = {feature: feature for feature in features}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_rows = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            rho = correlation.loc[left, right]
            if pd.notna(rho) and abs(float(rho)) >= float(threshold):
                union(left, right)
                edge_rows.append({"feature_a": left, "feature_b": right, "spearman": rho})
    grouped = defaultdict(list)
    for feature in features:
        grouped[find(feature)].append(feature)
    components = [members for members in grouped.values() if len(members) > 1]
    return components, pd.DataFrame(edge_rows)


def joint_component_importance(
    frame: pd.DataFrame,
    features: list[str],
    components: list[list[str]],
    config: Mapping[str, object],
    *,
    origin: int,
    device: str,
    smoke: bool,
) -> pd.DataFrame:
    data = dict(config["data"])
    train = frame.loc[frame["_year"] < int(origin)]
    validation = frame.loc[frame["_year"] == int(origin)]
    model = fit_model(
        numeric_frame(train, features),
        train[str(data["target"])].to_numpy(dtype=float),
        train_years=train["_year"],
        beta=0.0,
        config=config,
        seed=42,
        device=device,
        smoke=smoke,
    )
    X_validation = numeric_frame(validation, features)
    truth = validation[str(data["target"])].to_numpy(dtype=float)
    baseline = float(np.sqrt(np.mean(np.square(truth - model.predict(X_validation)))))
    rows = []
    for index, component in enumerate(components):
        active = [feature for feature in component if feature in features]
        if not active:
            continue
        rng = np.random.default_rng(42 + index)
        permutation = rng.permutation(len(X_validation))
        permuted = X_validation.copy()
        permuted.loc[:, active] = permuted[active].to_numpy()[permutation]
        prediction = model.predict(permuted)
        rmse = float(np.sqrt(np.mean(np.square(truth - prediction))))
        rows.append(
            {
                "component_id": index,
                "features": "|".join(component),
                "active_features": "|".join(active),
                "component_size": len(component),
                "baseline_rmse": baseline,
                "joint_permutation_rmse": rmse,
                "importance_delta_rmse": rmse - baseline,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def feature_family_summary(features: list[str]) -> pd.DataFrame:
    rows = []
    for position, feature in enumerate(features):
        family = feature.split("_", maxsplit=1)[0]
        rows.append({"feature": feature, "family": family, "position": position})
    detail = pd.DataFrame(rows)
    return (
        detail.groupby("family", sort=True)
        .agg(feature_count=("feature", "size"), first_position=("position", "min"))
        .reset_index()
        .sort_values(["feature_count", "first_position"], ascending=[False, True])
    )
