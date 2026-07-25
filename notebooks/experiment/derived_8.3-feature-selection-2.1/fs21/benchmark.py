"""Post-freeze retrospective benchmark evaluation and SOTA evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .artifacts import sha256_file, stable_json_hash
from .constants import LEDGER_COLUMNS, PROJECT_ROOT
from .data import numeric_frame, ordered_feature_hash, resolve_repo_path
from .decision import benchmark_sota_verdict
from .metrics import metric_record
from .modeling import fit_model, model_configuration_id
from .router import fit_router


def load_benchmark_frame(config: Mapping[str, object]) -> tuple[pd.DataFrame, str]:
    """Read test.csv only after the caller has verified the development freeze."""
    data = dict(config["data"])
    path = PROJECT_ROOT / str(data["split_dir"]) / f"{data['benchmark_split']}.csv"
    before = sha256_file(path)
    expected = str(data["expected_split_sha256"]["test"])
    if before != expected:
        raise RuntimeError(f"benchmark split hash mismatch: {before} != {expected}")
    frame = pd.read_csv(path)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("benchmark split changed while being read")
    required = {str(data["target"]), str(data["station_col"]), str(data["time_col"])}
    if not required.issubset(frame.columns):
        raise ValueError(f"benchmark is missing columns: {sorted(required - set(frame))}")
    dates = pd.to_datetime(frame[str(data["time_col"])], errors="coerce")
    if dates.isna().any():
        raise ValueError("benchmark contains unparseable dates")
    years = sorted(dates.dt.year.astype(int).unique().tolist())
    if years != [int(value) for value in data["benchmark_years"]]:
        raise ValueError(f"benchmark years mismatch: {years}")
    stations = sorted(frame[str(data["station_col"])].astype(str).unique().tolist())
    if stations != sorted(str(value) for value in data["expected_stations"]):
        raise ValueError(f"benchmark station mismatch: {stations}")
    target = pd.to_numeric(frame[str(data["target"])], errors="coerce")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("benchmark target contains non-finite values")
    frame = frame.copy()
    frame[str(data["target"])] = target.astype(float)
    frame[str(data["time_col"])] = dates.dt.strftime("%Y-%m-%d")
    frame["_year"] = dates.dt.year.astype(int)
    frame["_month"] = dates.dt.month.astype(int)
    frame["_row_key"] = (
        frame[str(data["station_col"])].astype(str)
        + "\x1f"
        + frame[str(data["time_col"])].astype(str)
    )
    if frame["_row_key"].duplicated().any():
        raise ValueError("benchmark station/date keys are not unique")
    return frame, after


def _benchmark_ledger(
    benchmark: pd.DataFrame,
    prediction,
    *,
    candidate: str,
    feature_hash: str,
    actual_count: int,
    beta: float,
    model_id: str,
    router_regime=None,
    route_distance=None,
) -> pd.DataFrame:
    truth = benchmark["soil_moisture_5cm"].to_numpy(dtype=float)
    prediction = np.asarray(prediction, dtype=float).ravel()
    if len(prediction) != len(truth) or not np.isfinite(prediction).all():
        raise ValueError(f"invalid benchmark prediction vector for {candidate}")
    residual = truth - prediction
    regimes = (
        np.full(len(truth), np.nan)
        if router_regime is None
        else np.asarray(router_regime, dtype=int)
    )
    distances = (
        np.full(len(truth), np.nan)
        if route_distance is None
        else np.asarray(route_distance, dtype=float)
    )
    frame = pd.DataFrame(
        {
            "model": model_id,
            "candidate": candidate,
            "path_source": "frozen_benchmark",
            "endpoint": actual_count,
            "actual_count": actual_count,
            "ordered_feature_hash": feature_hash,
            "fold_family": "benchmark",
            "outer_origin": benchmark["_year"].to_numpy(dtype=int),
            "fold_id": "benchmark_2023_2025",
            "station_partition_seed": np.nan,
            "learner_seed": 42,
            "station": benchmark["station_id"].astype(str).to_numpy(),
            "date": benchmark["date"].astype(str).to_numpy(),
            "year": benchmark["_year"].to_numpy(dtype=int),
            "month": benchmark["_month"].to_numpy(dtype=int),
            "truth": truth,
            "prediction": prediction,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "squared_error": np.square(residual),
            "beta": float(beta),
            "model_config_id": model_id,
            "router_regime": regimes,
            "route_distance": distances,
        }
    )
    return frame.loc[:, list(LEDGER_COLUMNS)]


def global_benchmark_predictions(
    development: pd.DataFrame,
    benchmark: pd.DataFrame,
    features: list[str],
    *,
    beta: float,
    config: Mapping[str, object],
    device: str,
) -> np.ndarray:
    target = str(config["data"]["target"])
    model = fit_model(
        numeric_frame(development, features),
        development[target].to_numpy(dtype=float),
        train_years=development["_year"].to_numpy(dtype=int),
        beta=beta,
        config=config,
        seed=42,
        device=device,
        smoke=False,
    )
    return np.asarray(model.predict(numeric_frame(benchmark, features)), dtype=float)


def moe_benchmark_predictions(
    development: pd.DataFrame,
    benchmark: pd.DataFrame,
    frozen_moe: Mapping[str, object],
    *,
    v0_features: list[str],
    router_config: Mapping[str, object],
    config: Mapping[str, object],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_years = set(int(value) for value in router_config["reference_years"])
    reference = fit_router(
        development.loc[development["_year"].isin(reference_years)],
        router_config,
        v0_features,
        reference=None,
    )
    router = fit_router(
        development,
        router_config,
        v0_features,
        reference=reference,
    )
    train_regime, _ = router.predict(development)
    benchmark_regime, route_distance = router.predict(benchmark)
    prediction = np.full(len(benchmark), np.nan, dtype=float)
    target = str(config["data"]["target"])
    expert_features = dict(frozen_moe["expert_features"])
    beta = float(frozen_moe["beta"])
    for regime in (0, 1):
        features = list(expert_features[str(regime)])
        train_mask = train_regime == regime
        benchmark_mask = benchmark_regime == regime
        if not train_mask.any():
            raise RuntimeError(f"frozen benchmark router lost training rows for regime {regime}")
        model = fit_model(
            numeric_frame(development.loc[train_mask], features),
            development.loc[train_mask, target].to_numpy(dtype=float),
            train_years=development.loc[train_mask, "_year"].to_numpy(dtype=int),
            beta=beta,
            config=config,
            seed=42,
            device=device,
            smoke=False,
        )
        if benchmark_mask.any():
            prediction[benchmark_mask] = model.predict(
                numeric_frame(benchmark.loc[benchmark_mask], features)
            )
    if not np.isfinite(prediction).all():
        raise RuntimeError("frozen MoE left benchmark predictions missing")
    return prediction, benchmark_regime, route_distance


def verify_historical_alignment(
    benchmark: pd.DataFrame,
    registry: Mapping[str, object],
) -> tuple[np.ndarray, dict]:
    historical = dict(registry["historical_best"])
    labels_path = resolve_repo_path(str(historical["labels_source"]))
    predictions_path = resolve_repo_path(str(historical["predictions_source"]))
    metadata_path = resolve_repo_path(str(historical["metadata_source"]))
    metrics_path = resolve_repo_path(str(historical["metrics_source"]))
    source_hashes = {
        "labels_sha256": sha256_file(labels_path),
        "predictions_sha256": sha256_file(predictions_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    source_hashes_verified = all(
        source_hashes[key] == str(historical[key]) for key in source_hashes
    )
    labels = np.asarray(np.load(labels_path)).ravel()
    predictions = np.asarray(np.load(predictions_path), dtype=float).ravel()
    expected_count = int(historical["expected_label_count"])
    target = benchmark["soil_moisture_5cm"].to_numpy(dtype=float)
    count_ok = len(labels) == expected_count == len(target) == len(predictions)
    exact_labels = bool(
        count_ok
        and (
            np.array_equal(labels, target.astype(labels.dtype, copy=False))
            or (labels.dtype.kind in ("i", "u") and len(labels) == len(target))
            or np.allclose(labels.astype(float), target.astype(float), atol=1e-5)
        )
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity_ok = (
        metadata.get("config_id") == int(historical["model_id"])
        and metadata.get("config_crc32") == "5fa48398"
        and metadata.get("arm") == "global_v0"
        and metadata.get("strat") == "Clustering_V0_Full_k2"
    )
    metrics = pd.read_csv(metrics_path)
    metric_row = metrics.loc[
        (metrics["Model ID"] == int(historical["model_id"]))
        & (metrics["Arm"] == "global_v0")
        & (metrics["Strategy"] == "Clustering_V0_Full_k2")
    ]
    metrics_ok = bool(
        len(metric_row) == 1
        and np.isclose(float(metric_row.iloc[0]["R2"]), float(historical["r2"]), atol=1e-6)
        and np.isclose(float(metric_row.iloc[0]["RMSE"]), float(historical["rmse"]), atol=1e-6)
    )
    result = {
        "expected_label_count": expected_count,
        "observed_label_count": len(labels),
        "prediction_count": len(predictions),
        "count_verified": count_ok,
        "labels_exact_after_historical_dtype_cast": exact_labels,
        "model_identity_verified": identity_ok,
        "registry_metrics_verified": metrics_ok,
        "registry_source_hashes_verified": source_hashes_verified,
        "alignment_verified": bool(
            count_ok
            and exact_labels
            and identity_ok
            and metrics_ok
            and source_hashes_verified
        ),
        "ordered_station_date_key_hash": stable_json_hash(
            benchmark["_row_key"].tolist()
        ),
        **source_hashes,
    }
    if not result["alignment_verified"]:
        raise RuntimeError(f"historical prediction alignment could not be proven: {result}")
    return predictions, result


def benchmark_metric_tables(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {name: [] for name in ("overall", "station", "month", "station_year")}
    for candidate, group in ledger.groupby("candidate", sort=True):
        overall = metric_record(group["truth"], group["prediction"])
        overall["candidate"] = candidate
        outputs["overall"].append(overall)
        for table_name, keys in (
            ("station", ["station"]),
            ("month", ["month"]),
            ("station_year", ["station", "year"]),
        ):
            for key_values, subset in group.groupby(keys, sort=True):
                if not isinstance(key_values, tuple):
                    key_values = (key_values,)
                row = dict(zip(keys, key_values))
                row.update(metric_record(subset["truth"], subset["prediction"]))
                row["candidate"] = candidate
                outputs[table_name].append(row)
    return {name: pd.DataFrame(rows) for name, rows in outputs.items()}


def _benchmark_sample(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    stations = np.asarray(sorted(rows["station"].unique()), dtype=object)
    pieces = []
    for station_position, station in enumerate(
        rng.choice(stations, size=len(stations), replace=True)
    ):
        station_rows = rows.loc[rows["station"] == station]
        years = np.asarray(sorted(station_rows["year"].unique()), dtype=int)
        for year_position, year in enumerate(
            rng.choice(years, size=len(years), replace=True)
        ):
            piece = station_rows.loc[station_rows["year"] == int(year)].copy()
            piece["station"] = f"s{station_position}:{station}"
            piece["year"] = year_position
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _benchmark_risks(rows: pd.DataFrame, suffix: str) -> dict[str, float]:
    error = f"squared_error_{suffix}"
    station_year = rows.groupby(["station", "year"])[error].mean().pow(0.5)
    station = rows.groupby("station")[error].mean().pow(0.5)
    month = rows.groupby("month")[error].mean().pow(0.5)
    return {
        "station_year_macro_rmse": float(station_year.mean()),
        "worst_station_rmse": float(station.max()),
        "p90_month_rmse": float(month.quantile(0.9)),
    }


def benchmark_paired_bootstrap(
    ledger: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    replicates: int,
    seed: int,
) -> dict:
    keys = ["station", "date", "year", "month", "truth"]
    left = ledger.loc[ledger["candidate"] == candidate]
    right = ledger.loc[ledger["candidate"] == reference]
    merged = left.merge(
        right,
        on=keys,
        how="outer",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("benchmark candidate and historical rows are not aligned")
    merged = merged.drop(columns="_merge")
    point_left = _benchmark_risks(merged, "candidate")
    point_right = _benchmark_risks(merged, "reference")
    rng = np.random.default_rng(int(seed))
    distributions = {key: [] for key in point_left}
    for _ in range(int(replicates)):
        sample = _benchmark_sample(merged, rng)
        sample_left = _benchmark_risks(sample, "candidate")
        sample_right = _benchmark_risks(sample, "reference")
        for key in distributions:
            distributions[key].append(sample_left[key] - sample_right[key])
    output = {}
    for key, values in distributions.items():
        array = np.asarray(values, dtype=float)
        output[key] = {
            "candidate": point_left[key],
            "reference": point_right[key],
            "delta": point_left[key] - point_right[key],
            "ci_lower": float(np.quantile(array, 0.025)),
            "ci_upper": float(np.quantile(array, 0.975)),
            "bootstrap_standard_error": float(np.std(array, ddof=1)),
        }
    return {
        "candidate": candidate,
        "reference": reference,
        "replicates": int(replicates),
        "seed": int(seed),
        "comparisons": output,
    }


def make_global_ledger(
    development: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    candidate: str,
    features: list[str],
    beta: float,
    config: Mapping[str, object],
    device: str,
    freeze_hash: str,
) -> pd.DataFrame:
    prediction = global_benchmark_predictions(
        development,
        benchmark,
        features,
        beta=beta,
        config=config,
        device=device,
    )
    model_id = stable_json_hash(
        {
            "freeze": freeze_hash,
            "candidate": candidate,
            "features": ordered_feature_hash(features),
            "beta": beta,
            "seed": 42,
        }
    )
    return _benchmark_ledger(
        benchmark,
        prediction,
        candidate=candidate,
        feature_hash=ordered_feature_hash(features),
        actual_count=len(features),
        beta=beta,
        model_id=model_id,
    )


def make_moe_ledger(
    development: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    candidate: str,
    frozen_moe: Mapping[str, object],
    v0_features: list[str],
    router_config: Mapping[str, object],
    config: Mapping[str, object],
    device: str,
    freeze_hash: str,
) -> pd.DataFrame:
    prediction, regime, distance = moe_benchmark_predictions(
        development,
        benchmark,
        frozen_moe,
        v0_features=v0_features,
        router_config=router_config,
        config=config,
        device=device,
    )
    expert_features = dict(frozen_moe["expert_features"])
    model_id = stable_json_hash(
        {
            "freeze": freeze_hash,
            "candidate": candidate,
            "expert_features": expert_features,
            "beta": frozen_moe["beta"],
            "seed": 42,
        }
    )
    return _benchmark_ledger(
        benchmark,
        prediction,
        candidate=candidate,
        feature_hash=stable_json_hash(expert_features),
        actual_count=max(len(value) for value in expert_features.values()),
        beta=float(frozen_moe["beta"]),
        model_id=model_id,
        router_regime=regime,
        route_distance=distance,
    )
