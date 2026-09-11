"""Retrain and evaluate the no-SMAP ECE model-salvage experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
FORMAL_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.4-formal-eval-2.0-ece"
if str(FORMAL_DIR) not in sys.path:
    sys.path.insert(0, str(FORMAL_DIR))

from eval_formal.routers import (  # noqa: E402
    Backbone54Router,
    DynamicClusterRouter,
    GlobalSingleRouter,
    SeasonalBinaryRouter,
    UnivariateGAPIRouter,
    V0FullRouter,
)

DEFAULT_SEEDS = [42, 7, 13]
TARGET_DEFAULT = "soil_moisture_5cm"
CHECKPOINT_DIR = EXP_DIR / "artifacts/checkpoints"
PREDICTION_DIR = EXP_DIR / "artifacts/predictions"


@dataclass
class DataBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    trainval: pd.DataFrame
    test: pd.DataFrame
    ece: pd.DataFrame
    target: str
    v0_parent: list[str]
    v0_features: list[str]
    backbone_parent: list[str]
    backbone_features: list[str]
    dynamic_parent: list[str]
    dynamic_features: list[str]
    selected_feature_sets: dict[int, list[str]]
    selected_features: list[str]


class NoSmapTrainedGate:
    """Supervised two-class gate trained only from WA target labels."""

    def __init__(self, features: list[str], target: str, threshold: float,
                 params: dict[str, Any], seed: int) -> None:
        self.features = list(features)
        self.target = target
        self.threshold = float(threshold)
        self.params = dict(params)
        self.seed = int(seed)
        self.classifier: XGBClassifier | None = None

    def fit(self, train: pd.DataFrame) -> "NoSmapTrainedGate":
        params = dict(self.params)
        params["random_state"] = self.seed
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        self.classifier = XGBClassifier(**params)
        y = np.where(train[self.target].to_numpy(dtype=float) < self.threshold, 0, 1)
        self.classifier.fit(train.loc[:, self.features], y, verbose=False)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("NoSmapTrainedGate must be fitted before prediction.")
        return np.asarray(self.classifier.predict(frame.loc[:, self.features])).ravel().astype(int)


def load_configuration() -> dict[str, Any]:
    with (EXP_DIR / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _prepare_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    output = frame.copy()
    if "date" not in output.columns or "station_id" not in output.columns:
        raise ValueError("Every split must contain date and station_id columns.")
    output["date"] = pd.to_datetime(output["date"], errors="raise")
    output["month"] = output["date"].dt.month.astype(int)
    output["year"] = output["date"].dt.year.astype(int)
    if target not in output.columns:
        raise ValueError(f"Target column not found: {target}")
    if output[target].isna().any():
        raise ValueError("Target contains missing values.")
    excluded = {target, "station_id", "date", "month", "year"}
    for column in output.columns:
        if column not in excluded and not pd.api.types.is_numeric_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.replace([np.inf, -np.inf], np.nan)


def _load_v0_features(path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("derived_84_metadata_no_smap", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load feature metadata: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.OVERALL_SELECTED_FEATURES_V0)


def _load_backbone(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return list(yaml.safe_load(handle)["shared_backbone_54"])


def _resolve_experiment_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else EXP_DIR / path


def _load_selected_feature_sets(
    config: dict[str, Any], frames: dict[str, pd.DataFrame]
) -> dict[int, list[str]]:
    """Load and validate every completed size-specific selector manifest."""
    manifests = config.get("selected_feature_manifests", {})
    if not manifests:
        raise ValueError("Configuration has no selected_feature_manifests mapping.")
    selected_sets: dict[int, list[str]] = {}
    for raw_size, raw_path in manifests.items():
        size = int(raw_size)
        manifest_path = _resolve_experiment_path(raw_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Feature selection is incomplete; expected manifest: {manifest_path}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(
                "Feature selection is not complete: "
                f"status={payload.get('status')!r} ({manifest_path})"
            )
        selected = payload.get("global_features")
        if not isinstance(selected, list):
            raise ValueError(f"Manifest has no global_features list: {manifest_path}")
        selected = list(dict.fromkeys(map(str, selected)))
        if len(selected) != size:
            raise ValueError(
                f"Manifest {manifest_path} must contain exactly {size} features; "
                f"got {len(selected)}."
            )
        smap = [feature for feature in selected if "smap" in feature.lower()]
        if smap:
            raise ValueError(f"Selected feature manifest contains SMAP features: {smap}")
        additions = payload.get("cluster_additions", {})
        if any(additions.get(str(cluster), []) for cluster in (0, 1)):
            raise ValueError(f"Delta additions are not allowed in {manifest_path}.")
        for name, frame in frames.items():
            missing = sorted(set(selected) - set(frame.columns))
            if missing:
                raise ValueError(f"Selected features missing from {name}: {missing}")
        selected_sets[size] = selected
    expected_sizes = sorted(int(size) for size in config.get("selected_feature_sizes", []))
    if sorted(selected_sets) != expected_sizes:
        raise ValueError(
            f"Feature manifests cover {sorted(selected_sets)}, expected {expected_sizes}."
        )
    for smaller, larger in zip(expected_sizes, expected_sizes[1:]):
        if not set(selected_sets[smaller]).issubset(selected_sets[larger]):
            raise ValueError("Selected feature manifests must be nested by feature size.")
    return selected_sets


def drop_smap(features: Iterable[str]) -> list[str]:
    """Remove every explicitly SMAP-derived feature while preserving order."""
    return [feature for feature in features if "smap" not in str(feature).lower()]


def load_data(config: dict[str, Any]) -> DataBundle:
    data_cfg = config["data"]
    target = str(data_cfg.get("target", TARGET_DEFAULT))
    training_dir = PROJECT_ROOT / Path(data_cfg["training_dir"])
    train = _prepare_frame(pd.read_csv(training_dir / "train.csv", low_memory=False), target)
    val = _prepare_frame(pd.read_csv(training_dir / "val.csv", low_memory=False), target)
    test = _prepare_frame(pd.read_csv(training_dir / "test.csv", low_memory=False), target)
    ece = _prepare_frame(pd.read_csv(PROJECT_ROOT / Path(data_cfg["ece_v3_test"]), low_memory=False), target)

    v0_parent = _load_v0_features(PROJECT_ROOT / Path(data_cfg["v0_metadata"]))
    backbone_parent = _load_backbone(PROJECT_ROOT / Path(data_cfg["backbone_config"]))
    dynamic_parent = list(config["features"]["dynamic_parent"])
    v0_features = drop_smap(v0_parent)
    backbone_features = drop_smap(backbone_parent)
    dynamic_features = drop_smap(dynamic_parent)
    trainval = pd.concat([train, val], ignore_index=True)
    wa_stations = trainval["station_id"].astype(str).nunique()
    ece_stations = ece["station_id"].astype(str).nunique()
    if wa_stations != 7:
        raise ValueError(f"Expected exactly seven WA training stations, got {wa_stations}.")
    if ece_stations != 5:
        raise ValueError(f"Expected exactly five ECE evaluation stations, got {ece_stations}.")

    all_frames = {"trainval": trainval, "test": test, "ece": ece}
    for name, features in {
        "v0": v0_features,
        "backbone": backbone_features,
        "dynamic": dynamic_features,
    }.items():
        missing = sorted(set(features) - set(trainval.columns))
        if missing:
            raise ValueError(f"{name} features missing from WA trainval: {missing}")
        for frame_name, frame in all_frames.items():
            missing = sorted(set(features) - set(frame.columns))
            if missing:
                raise ValueError(f"{name} features missing from {frame_name}: {missing}")

    selected_feature_sets = _load_selected_feature_sets(
        config, {"wa_trainval": trainval, "wa_test": test, "ece": ece}
    )
    selected_features = selected_feature_sets[60]
    backbone_features = selected_features
    selected = v0_features + dynamic_features + [
        feature for features in selected_feature_sets.values() for feature in features
    ]
    smap_selected = [feature for feature in selected if "smap" in feature.lower()]
    if smap_selected:
        raise AssertionError(f"SMAP features survived filtering: {smap_selected}")
    return DataBundle(
        train=train,
        val=val,
        trainval=trainval,
        test=test,
        ece=ece,
        target=target,
        v0_parent=v0_parent,
        v0_features=v0_features,
        backbone_parent=backbone_parent,
        backbone_features=backbone_features,
        dynamic_parent=dynamic_parent,
        dynamic_features=dynamic_features,
        selected_feature_sets=selected_feature_sets,
        selected_features=selected_features,
    )


def feature_manifest(data: DataBundle, config: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "experiment": config["experiment"]["name"],
        "policy": "drop every feature whose name contains SMAP, case-insensitive",
        "v0_router": {"parent": data.v0_parent, "dropped": [f for f in data.v0_parent if f not in data.v0_features], "effective": data.v0_features},
        "legacy_backbone_parent": {"parent": data.backbone_parent, "dropped": [f for f in data.backbone_parent if f not in data.backbone_features], "effective": data.backbone_features},
        "selected_model_feature_sets": {
            str(size): {
                "parent": "local MI300 -> ElasticNet -> stability -> wrapper selection",
                "effective": features,
                "dropped": [],
                "delta_additions": [],
                "selection_manifest": str(_resolve_experiment_path(config["selected_feature_manifests"][str(size)])),
            }
            for size, features in sorted(data.selected_feature_sets.items())
        },
        "dynamic_router": {"parent": data.dynamic_parent, "dropped": [f for f in data.dynamic_parent if f not in data.dynamic_features], "effective": data.dynamic_features},
        "special_routers": {"gapi": ["G_API"], "seasonal": ["month"], "global": []},
    }
    manifest["counts"] = {
        "v0_parent": len(data.v0_parent),
        "v0_effective": len(data.v0_features),
        "backbone_parent": len(data.backbone_parent),
        "backbone_effective": len(data.backbone_features),
        "dynamic_parent": len(data.dynamic_parent),
        "dynamic_effective": len(data.dynamic_features),
        "selected_feature_sizes": {str(size): len(features) for size, features in sorted(data.selected_feature_sets.items())},
        "selected_backbone_effective": len(data.selected_features),
    }
    return manifest


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _hash_features(features: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()[:16]


def _base_params(config: dict[str, Any], smoke: bool, device_override: str | None) -> dict[str, Any]:
    params = dict(config["model"]["exact_params"])
    if smoke:
        params["n_estimators"] = 100
        params["device"] = "cpu"
    if device_override:
        params["device"] = device_override
    params["n_jobs"] = 1
    return params


def _model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(spec) for spec in config["models"]]


def _model_features(spec: dict[str, Any], data: DataBundle) -> list[str]:
    size = int(spec["feature_size"])
    try:
        return data.selected_feature_sets[size]
    except KeyError as error:
        raise ValueError(f"No selected feature manifest for model size {size}.") from error


def fit_router(spec: dict[str, Any], data: DataBundle, config: dict[str, Any],
               params: dict[str, Any]) -> Any:
    seed = int(config.get("router_seed", 42))
    router_type = spec["router"]
    if router_type == "v0":
        router = V0FullRouter(data.v0_features, seed=seed)
    elif router_type == "backbone54":
        router = Backbone54Router(data.backbone_features, seed=seed)
    elif router_type == "trained_gate":
        gate_params = dict(params)
        router = NoSmapTrainedGate(
            data.v0_features,
            data.target,
            float(config["model"]["trained_gate_threshold"]),
            gate_params,
            seed,
        )
    elif router_type == "gapi":
        router = UnivariateGAPIRouter("G_API")
    elif router_type == "dynamic":
        router = DynamicClusterRouter(features=data.dynamic_features, seed=seed)
    elif router_type == "seasonal":
        router = SeasonalBinaryRouter()
    elif router_type == "global":
        router = GlobalSingleRouter()
    else:
        raise ValueError(f"Unknown router type: {router_type}")
    router.fit(data.trainval)
    return router


def _router_feature_names(spec: dict[str, Any], data: DataBundle) -> list[str]:
    router_type = spec["router"]
    if router_type == "v0" or router_type == "trained_gate":
        return data.v0_features
    if router_type == "backbone54":
        return _model_features(spec, data)
    if router_type == "dynamic":
        return data.dynamic_features
    if router_type == "gapi":
        return ["G_API"]
    if router_type == "seasonal":
        return ["month"]
    return []


def _labels(router: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(router.predict(frame)).ravel().astype(int)


def fit_regressors(spec: dict[str, Any], data: DataBundle, router: Any,
                   params: dict[str, Any], seed: int) -> dict[int, XGBRegressor]:
    labels = _labels(router, data.trainval)
    y = data.trainval[data.target].to_numpy(dtype=float)
    params = dict(params)
    params["random_state"] = int(seed)
    model_features = _model_features(spec, data)
    models: dict[int, XGBRegressor] = {}
    clusters = (0, 1) if spec["two_regime"] else (0,)
    for cluster in clusters:
        mask = labels == cluster if spec["two_regime"] else np.ones(len(labels), dtype=bool)
        if not mask.any():
            continue
        model = XGBRegressor(**params)
        model.fit(data.trainval.loc[mask, model_features], y[mask], verbose=False)
        models[int(cluster)] = model
    return models


def predict_regressors(spec: dict[str, Any], data: DataBundle, router: Any,
                       models: dict[int, XGBRegressor], frame: pd.DataFrame,
                       fallback: float) -> tuple[np.ndarray, np.ndarray]:
    labels = _labels(router, frame)
    model_features = _model_features(spec, data)
    prediction = np.full(len(frame), float(fallback), dtype=float)
    clusters = (0, 1) if spec["two_regime"] else (0,)
    for cluster in clusters:
        mask = labels == cluster
        model = models.get(int(cluster))
        if mask.any() and model is not None:
            prediction[mask] = np.asarray(model.predict(frame.loc[mask, model_features])).ravel()
    return prediction, labels


def _prediction_path(model_id: str, seed: int) -> Path:
    return PREDICTION_DIR / f"{model_id}__s{seed}.csv"


def _checkpoint_dir(model_id: str, seed: int) -> Path:
    return CHECKPOINT_DIR / f"{model_id}__s{seed}"


def _make_prediction_frame(model_id: str, seed: int, dataset: str, window: str,
                           frame: pd.DataFrame, prediction: np.ndarray,
                           labels: np.ndarray, target: str) -> pd.DataFrame:
    return pd.DataFrame({
        "model_id": model_id,
        "seed": int(seed),
        "dataset": dataset,
        "window": window,
        "station_id": frame["station_id"].to_numpy(),
        "date": frame["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "target": frame[target].to_numpy(dtype=float),
        "prediction": np.asarray(prediction, dtype=float),
        "regime": np.asarray(labels, dtype=int),
    })


def _window_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = frame["date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
    return frame.loc[mask].copy()


def _save_models(models: dict[int, XGBRegressor], checkpoint: Path) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    for cluster, model in models.items():
        model.save_model(checkpoint / f"regime_{cluster}.json")


def _load_models(spec: dict[str, Any], checkpoint: Path) -> dict[int, XGBRegressor]:
    clusters = (0, 1) if spec["two_regime"] else (0,)
    models: dict[int, XGBRegressor] = {}
    for cluster in clusters:
        path = checkpoint / f"regime_{cluster}.json"
        if path.exists():
            model = XGBRegressor()
            model.load_model(path)
            models[int(cluster)] = model
    return models


def _run_one(spec: dict[str, Any], seed: int, data: DataBundle, router: Any,
             config: dict[str, Any], params: dict[str, Any], smoke: bool) -> None:
    model_id = spec["id"]
    prediction_path = _prediction_path(model_id, seed)
    checkpoint = _checkpoint_dir(model_id, seed)
    if prediction_path.exists() and (checkpoint / "meta.json").exists():
        try:
            meta = json.loads((checkpoint / "meta.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        compatible = (
            meta.get("smoke") is bool(smoke)
            and int(meta.get("n_estimators", -1)) == int(params["n_estimators"])
            and str(meta.get("device")) == str(params["device"])
            and int(meta.get("selected_feature_count", -1)) == int(spec["feature_size"])
            and str(meta.get("feature_hash")) == _hash_features(_model_features(spec, data))
        )
        if compatible:
            print(f"[resume] {model_id} seed={seed}")
            return

    started = time.perf_counter()
    models = fit_regressors(spec, data, router, params, seed)
    fallback = float(data.trainval[data.target].mean())
    rows: list[pd.DataFrame] = []
    temporal_full, temporal_summer, ece = data.test, _window_frame(data.test, "2025-07-20", "2025-08-19"), data.ece
    for dataset, window, frame in (
        ("wa_temporal", "temporal_full", temporal_full),
        ("wa_temporal", "temporal_summer_2025", temporal_summer),
        ("ece_spatial", "spatial_ece_v3_full", ece),
    ):
        prediction, labels = predict_regressors(spec, data, router, models, frame, fallback)
        rows.append(_make_prediction_frame(model_id, seed, dataset, window, frame, prediction, labels, data.target))

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(prediction_path, index=False)
    _save_models(models, checkpoint)
    _write_json(checkpoint / "meta.json", {
        "model_id": model_id,
        "seed": int(seed),
        "router_seed": int(config.get("router_seed", 42)),
        "router": spec["router"],
        "two_regime": bool(spec["two_regime"]),
        "n_trainval": int(len(data.trainval)),
        "effective_model_features": _model_features(spec, data),
        "selected_feature_manifest": str(_resolve_experiment_path(
            config["selected_feature_manifests"][str(spec["feature_size"])]
        )),
        "selected_feature_count": int(spec["feature_size"]),
        "feature_hash": _hash_features(_model_features(spec, data)),
        "n_estimators": int(params["n_estimators"]),
        "device": str(params["device"]),
        "train_time_s": time.perf_counter() - started,
        "smoke": bool(smoke),
    })
    print(f"[fit] {model_id} seed={seed} rows={len(pd.concat(rows))} time={time.perf_counter() - started:.1f}s")


def _metric_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def _diff_pearson(frame: pd.DataFrame) -> float:
    differences: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in frame.sort_values(["station_id", "date"]).groupby("station_id", sort=False):
        if len(group) > 1:
            differences.append((np.diff(group["target"].to_numpy(dtype=float)), np.diff(group["prediction"].to_numpy(dtype=float))))
    if not differences:
        return float("nan")
    true_diff = np.concatenate([item[0] for item in differences])
    pred_diff = np.concatenate([item[1] for item in differences])
    return _metric_pearson(true_diff, pred_diff)


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["target"].to_numpy(dtype=float)
    y_pred = frame["prediction"].to_numpy(dtype=float)
    if len(frame) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "ubrmse": np.nan,
                "r2": np.nan, "pearson": np.nan, "target_std": np.nan,
                "prediction_std": np.nan, "diff_pearson": np.nan}
    bias = float(np.mean(y_pred - y_true))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return {
        "n": int(len(frame)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": bias,
        "ubrmse": float(np.sqrt(max(0.0, rmse**2 - bias**2))),
        "r2": float(r2_score(y_true, y_pred)) if len(frame) > 1 else np.nan,
        "pearson": _metric_pearson(y_true, y_pred),
        "target_std": float(np.std(y_true)),
        "prediction_std": float(np.std(y_pred)),
        "diff_pearson": _diff_pearson(frame),
    }


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["model_id", "seed", "dataset", "window"], sort=True):
        model_id, seed, dataset, window = keys
        pooled = compute_metrics(group)
        rows.append({"model_id": model_id, "seed": int(seed), "dataset": dataset,
                     "window": window, "scope": "__pooled__", **pooled})
        for station_id, station in group.groupby("station_id", sort=True):
            rows.append({"model_id": model_id, "seed": int(seed), "dataset": dataset,
                         "window": window, "scope": str(station_id), **compute_metrics(station)})
    seed_metrics = pd.DataFrame(rows)
    metric_names = ["rmse", "mae", "bias", "ubrmse", "r2", "pearson", "target_std", "prediction_std", "diff_pearson"]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in seed_metrics.groupby(["model_id", "dataset", "window", "scope"], sort=True):
        model_id, dataset, window, scope = keys
        row: dict[str, Any] = {"model_id": model_id, "dataset": dataset, "window": window, "scope": scope,
                               "n_seeds": int(group["seed"].nunique()), "n_mean": float(group["n"].mean())}
        for metric in metric_names:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else np.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
        summary_rows.append(row)
    return seed_metrics, pd.DataFrame(summary_rows)


def load_reference_metrics(config: dict[str, Any]) -> pd.DataFrame:
    result_dir = PROJECT_ROOT / Path(config["data"]["original_results_dir"])
    rows: list[pd.DataFrame] = []
    for filename, dataset, window in (
        ("temporal_seed_summary.csv", "wa_temporal", "temporal_full"),
        ("spatial_seed_summary.csv", "ece_spatial", "spatial_ece_v3_full"),
    ):
        path = result_dir / filename
        if not path.exists():
            continue
        source = pd.read_csv(path, low_memory=False)
        for spec in config["models"]:
            reference_id = spec["parent_reference"]
            subset = source[(source["config_id"] == reference_id) & source["seed"].isin(config["seeds"])]
            if subset.empty:
                continue
            subset = subset.copy()
            subset["model_id"] = spec["id"]
            subset["dataset"] = dataset
            subset["window"] = window
            subset["scope"] = "__pooled__"
            subset["reference_config_id"] = reference_id
            subset["reference_only"] = True
            rows.append(subset[["model_id", "seed", "dataset", "window", "scope", "reference_config_id",
                                "reference_only", "rmse", "mae", "bias", "ubrmse", "r2", "pearson"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_reference_comparison(seed_metrics: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    if references.empty:
        return pd.DataFrame()
    metric_names = ["rmse", "mae", "bias", "ubrmse", "r2", "pearson"]
    left = seed_metrics[(seed_metrics["scope"] == "__pooled__") & seed_metrics["window"].isin(["temporal_full", "spatial_ece_v3_full"])].copy()
    merged = left.merge(references, on=["model_id", "seed", "dataset", "window", "scope"], suffixes=("_no_smap", "_original"))
    for metric in metric_names:
        merged[f"{metric}_delta_no_smap_minus_original"] = merged[f"{metric}_no_smap"] - merged[f"{metric}_original"]
    return merged


_EFFECT_SPLITS = {
    ("ece_spatial", "spatial_ece_v3_full"): ("ECE spatial", "ECE benefit"),
    ("wa_temporal", "temporal_full"): ("WA temporal", "WA degradation"),
}
_EFFECT_METRICS = ["rmse", "mae", "bias", "ubrmse", "r2", "pearson", "diff_pearson"]


def _mean_std(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan"), float("nan")
    return float(numeric.mean()), float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def _reference_value(row: pd.Series, metric: str, side: str) -> float:
    """Read a metric from either the current or a future reference artifact."""
    if metric == "diff_pearson":
        candidates = (
            ["diff_pearson_no_smap", "diff_pearson"]
            if side == "no_smap"
            else ["diff_pearson_original"]
        )
    else:
        candidates = [f"{metric}_{side}"]
    for column in candidates:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            return float(value) if pd.notna(value) else float("nan")
    return float("nan")


def build_old_vs_new_effects(
    reference_comparison: pd.DataFrame,
    expected_seeds: Iterable[int] | None = None,
    expected_model_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paired seed and summary effects from the reference comparison table.

    RMSE changes are retained in the generic ``no_smap - original`` direction.
    ``effect_rmse`` then uses the report-facing direction: ECE benefit is
    ``original - no_smap`` and WA degradation is ``no_smap - original``.
    """
    seed_columns = [
        "model_id", "seed", "split", "effect_label", "dataset", "window",
        "reference_config_id", "rmse_original", "rmse_no_smap",
        "rmse_change_no_smap_minus_original", "rmse_change_abs",
        "rmse_change_pct_no_smap_vs_original", "effect_rmse", "effect_rmse_pct",
        "improved", "worsened",
    ]
    summary_columns = [
        "model_id", "split", "effect_label", "dataset", "window", "reference_config_id",
        "n_seeds", "improved_seeds", "worsened_seeds",
    ]
    for metric in _EFFECT_METRICS:
        summary_columns.extend([
            f"{metric}_original_mean", f"{metric}_original_std",
            f"{metric}_no_smap_mean", f"{metric}_no_smap_std",
            f"{metric}_change_mean", f"{metric}_change_std",
        ])
    summary_columns.extend([
        "rmse_change_abs_mean", "rmse_change_abs_std",
        "rmse_change_pct_mean", "rmse_change_pct_std",
        "effect_rmse_mean", "effect_rmse_std",
        "effect_rmse_pct_mean", "effect_rmse_pct_std",
    ])

    if reference_comparison.empty:
        return pd.DataFrame(columns=seed_columns), pd.DataFrame(columns=summary_columns)

    required = {"model_id", "seed", "dataset", "window", "scope", "reference_config_id"}
    missing = sorted(required - set(reference_comparison.columns))
    if missing:
        raise ValueError(f"reference_comparison is missing required columns: {missing}")

    source = reference_comparison[
        reference_comparison["scope"].eq("__pooled__")
        & reference_comparison.set_index(["dataset", "window"]).index.isin(_EFFECT_SPLITS)
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=seed_columns), pd.DataFrame(columns=summary_columns)

    keys = ["model_id", "seed", "dataset", "window"]
    if source.duplicated(keys).any():
        duplicate_rows = source.loc[source.duplicated(keys, keep=False), keys].to_dict(orient="records")
        raise ValueError(f"Duplicate paired reference rows: {duplicate_rows}")

    model_ids = sorted(source["model_id"].astype(str).unique())
    if expected_model_ids is not None:
        expected_models = {str(model_id) for model_id in expected_model_ids}
        missing_models = sorted(expected_models - set(model_ids))
        if missing_models:
            raise ValueError(f"Missing model/reference comparisons: {missing_models}")
        unexpected_models = sorted(set(model_ids) - expected_models)
        if unexpected_models:
            raise ValueError(f"Unexpected model/reference comparisons: {unexpected_models}")

    mapping_counts = source.groupby("model_id")["reference_config_id"].nunique()
    invalid_mappings = mapping_counts[mapping_counts != 1]
    if not invalid_mappings.empty:
        raise ValueError("Each model must map to exactly one original reference configuration.")

    expected_seed_set = None if expected_seeds is None else {int(seed) for seed in expected_seeds}
    expected_splits = set(_EFFECT_SPLITS)
    expected_pairs = {(model_id, split) for model_id in model_ids for split in expected_splits}
    found_pairs = {
        (str(row["model_id"]), (str(row["dataset"]), str(row["window"])))
        for _, row in source.iterrows()
    }
    missing_pairs = sorted(expected_pairs - found_pairs)
    if missing_pairs:
        raise ValueError(f"Missing comparison splits: {missing_pairs}")

    seed_rows: list[dict[str, Any]] = []
    for _, row in source.sort_values(keys).iterrows():
        split, effect_label = _EFFECT_SPLITS[(str(row["dataset"]), str(row["window"]))]
        rmse_original = _reference_value(row, "rmse", "original")
        rmse_no_smap = _reference_value(row, "rmse", "no_smap")
        rmse_change = rmse_no_smap - rmse_original
        effect_rmse = -rmse_change if split == "ECE spatial" else rmse_change
        rmse_change_pct = rmse_change / rmse_original * 100.0 if rmse_original else float("nan")
        effect_pct = -rmse_change_pct if split == "ECE spatial" else rmse_change_pct
        seed_row: dict[str, Any] = {
            "model_id": str(row["model_id"]),
            "seed": int(row["seed"]),
            "split": split,
            "effect_label": effect_label,
            "dataset": str(row["dataset"]),
            "window": str(row["window"]),
            "reference_config_id": str(row["reference_config_id"]),
            "rmse_original": rmse_original,
            "rmse_no_smap": rmse_no_smap,
            "rmse_change_no_smap_minus_original": rmse_change,
            "rmse_change_abs": abs(rmse_change),
            "rmse_change_pct_no_smap_vs_original": rmse_change_pct,
            "effect_rmse": effect_rmse,
            "effect_rmse_pct": effect_pct,
            "improved": bool(rmse_change < 0),
            "worsened": bool(rmse_change > 0),
        }
        for metric in _EFFECT_METRICS:
            new_value = _reference_value(row, metric, "no_smap")
            original_value = _reference_value(row, metric, "original")
            seed_row[f"{metric}_original"] = original_value
            seed_row[f"{metric}_no_smap"] = new_value
            seed_row[f"{metric}_change"] = new_value - original_value
        seed_rows.append(seed_row)

    seed_effects = pd.DataFrame(seed_rows)
    if expected_seed_set is not None:
        actual_by_pair = seed_effects.groupby(["model_id", "split"])["seed"].agg(lambda values: set(values))
        invalid_pairs = {
            pair: sorted(expected_seed_set - set(seeds))
            for pair, seeds in actual_by_pair.items()
            if set(seeds) != expected_seed_set
        }
        if invalid_pairs:
            raise ValueError(f"Each model/split must contain exactly expected seeds: {invalid_pairs}")

    summary_rows: list[dict[str, Any]] = []
    for (model_id, split), group in seed_effects.groupby(["model_id", "split"], sort=True):
        first = group.iloc[0]
        summary_row: dict[str, Any] = {
            "model_id": model_id,
            "split": split,
            "effect_label": first["effect_label"],
            "dataset": first["dataset"],
            "window": first["window"],
            "reference_config_id": first["reference_config_id"],
            "n_seeds": int(group["seed"].nunique()),
            "improved_seeds": int(group["improved"].sum()),
            "worsened_seeds": int(group["worsened"].sum()),
        }
        for metric in _EFFECT_METRICS:
            for suffix, values in (
                ("original", group[f"{metric}_original"]),
                ("no_smap", group[f"{metric}_no_smap"]),
                ("change", group[f"{metric}_change"]),
            ):
                mean, std = _mean_std(values)
                summary_row[f"{metric}_{suffix}_mean"] = mean
                summary_row[f"{metric}_{suffix}_std"] = std
        for column in ["rmse_change_abs", "rmse_change_pct_no_smap_vs_original", "effect_rmse", "effect_rmse_pct"]:
            mean, std = _mean_std(group[column])
            output_name = {
                "rmse_change_pct_no_smap_vs_original": "rmse_change_pct",
                "effect_rmse": "effect_rmse",
                "effect_rmse_pct": "effect_rmse_pct",
            }.get(column, f"{column}_mean")
            if column == "rmse_change_abs":
                summary_row["rmse_change_abs_mean"] = mean
                summary_row["rmse_change_abs_std"] = std
            elif column == "rmse_change_pct_no_smap_vs_original":
                summary_row["rmse_change_pct_mean"] = mean
                summary_row["rmse_change_pct_std"] = std
            else:
                summary_row[f"{output_name}_mean"] = mean
                summary_row[f"{output_name}_std"] = std
        summary_rows.append(summary_row)

    return seed_effects[seed_columns + [
        f"{metric}_{suffix}" for metric in _EFFECT_METRICS for suffix in ("original", "no_smap", "change")
    ]], pd.DataFrame(summary_rows, columns=summary_columns)


def build_router_audit(specs: list[dict[str, Any]], routers: dict[str, Any], data: DataBundle,
                       config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        router = routers[spec["id"]]
        for dataset, frame in (("wa_trainval", data.trainval), ("wa_test", data.test), ("ece_v3", data.ece)):
            labels = _labels(router, frame)
            rows.append({
                "model_id": spec["id"],
                "router": spec["router"],
                "dataset": dataset,
                "router_seed": int(config.get("router_seed", 42)),
                "n": int(len(labels)),
                "regime_0_share": float(np.mean(labels == 0)),
                "regime_1_share": float(np.mean(labels == 1)),
                "router_feature_count": len(_router_feature_names(spec, data)),
                "router_features": ";".join(_router_feature_names(spec, data)),
            })
    return pd.DataFrame(rows)


def write_smap_invariance(data: DataBundle, specs: list[dict[str, Any]], routers: dict[str, Any],
                          config: dict[str, Any], params: dict[str, Any]) -> None:
    path = EXP_DIR / "smap_invariance.csv"
    rows: list[dict[str, Any]] = []
    altered = data.ece.copy()
    smap_columns = [column for column in altered.columns if "smap" in column.lower()]
    altered.loc[:, smap_columns] = 0.0
    for spec in specs:
        seed = 42
        checkpoint = _checkpoint_dir(spec["id"], seed)
        if not (checkpoint / "meta.json").exists():
            continue
        router = routers[spec["id"]]
        models = _load_models(spec, checkpoint)
        if not models:
            continue
        fallback = float(data.trainval[data.target].mean())
        original_pred, original_labels = predict_regressors(spec, data, router, models, data.ece, fallback)
        altered_pred, altered_labels = predict_regressors(spec, data, router, models, altered, fallback)
        rows.append({
            "model_id": spec["id"],
            "seed": seed,
            "smap_columns_altered": len(smap_columns),
            "max_abs_prediction_difference": float(np.max(np.abs(original_pred - altered_pred))),
            "changed_regime_labels": int(np.sum(original_labels != altered_labels)),
        })
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def make_trend_figures(predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create ECE trend figures; each panel contains at most five lines."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    architecture = [
        "Clustering_V0_Full_k2_no_smap_fs60",
        "Clustering_Backbone_k2_no_smap_fs60",
        "Trained_Gating_k2_no_smap_fs60",
        "Global_Single_60_no_smap_fs60",
    ]
    regime = [
        "Univariate_G_API_k2_no_smap_fs60",
        "Clustering_Dynamic_k2_no_smap_fs60",
        "Seasonal_Binary_k2_no_smap_fs60",
        "Global_Single_60_no_smap_fs60",
    ]
    global_sizes = [
        "Global_Single_40_no_smap_fs40",
        "Global_Single_50_no_smap_fs50",
        "Global_Single_60_no_smap_fs60",
        "Global_Single_69_no_smap_fs69",
    ]
    labels = {
        "Clustering_V0_Full_k2_no_smap_fs60": "V0 KMeans (60)",
        "Clustering_Backbone_k2_no_smap_fs60": "Backbone KMeans (60)",
        "Trained_Gating_k2_no_smap_fs60": "Trained gate (60)",
        "Univariate_G_API_k2_no_smap_fs60": "G_API gate (60)",
        "Clustering_Dynamic_k2_no_smap_fs60": "Dynamic gate (60)",
        "Seasonal_Binary_k2_no_smap_fs60": "Seasonal gate (60)",
        "Global_Single_40_no_smap_fs40": "Global (40)",
        "Global_Single_50_no_smap_fs50": "Global (50)",
        "Global_Single_60_no_smap_fs60": "Global (60)",
        "Global_Single_69_no_smap_fs69": "Global (69)",
    }
    ece = predictions[(predictions["dataset"] == "ece_spatial") & (predictions["seed"] == 42)].copy()
    paths: list[Path] = []
    for station in sorted(ece["station_id"].unique()):
        station_data = ece[ece["station_id"] == station].sort_values("date")
        observed = station_data.drop_duplicates("date")
        for suite_name, model_ids in (
            ("architecture", architecture),
            ("regime", regime),
            ("global_feature_sizes", global_sizes),
        ):
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(pd.to_datetime(observed["date"]), observed["target"], color="black", linewidth=2.2, label="Observed")
            for model_id in model_ids:
                model_data = station_data[station_data["model_id"] == model_id]
                ax.plot(pd.to_datetime(model_data["date"]), model_data["prediction"], linewidth=1.4, label=labels[model_id])
            ax.set_title(f"{station} — no-SMAP {suite_name} (seed 42)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Soil moisture")
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            path = output_dir / f"ece_{station}_{suite_name}_trend.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def make_old_vs_new_effect_figure(effect_summary: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the paired ECE benefit and WA degradation with seed error bars."""
    import matplotlib.pyplot as plt

    required = {"model_id", "split", "effect_rmse_mean", "effect_rmse_std"}
    missing = sorted(required - set(effect_summary.columns))
    if missing:
        raise ValueError(f"Effect summary is missing figure columns: {missing}")
    model_order = [
        "Clustering_V0_Full_k2_no_smap_fs60",
        "Clustering_Backbone_k2_no_smap_fs60",
        "Trained_Gating_k2_no_smap_fs60",
        "Univariate_G_API_k2_no_smap_fs60",
        "Clustering_Dynamic_k2_no_smap_fs60",
        "Seasonal_Binary_k2_no_smap_fs60",
        "Global_Single_40_no_smap_fs40",
        "Global_Single_50_no_smap_fs50",
        "Global_Single_60_no_smap_fs60",
        "Global_Single_69_no_smap_fs69",
    ]
    labels = {
        "Clustering_V0_Full_k2_no_smap_fs60": "V0 KMeans (60)",
        "Clustering_Backbone_k2_no_smap_fs60": "Backbone KMeans (60)",
        "Trained_Gating_k2_no_smap_fs60": "Trained gate (60)",
        "Univariate_G_API_k2_no_smap_fs60": "G_API gate (60)",
        "Clustering_Dynamic_k2_no_smap_fs60": "Dynamic gate (60)",
        "Seasonal_Binary_k2_no_smap_fs60": "Seasonal gate (60)",
        "Global_Single_40_no_smap_fs40": "Global (40)",
        "Global_Single_50_no_smap_fs50": "Global (50)",
        "Global_Single_60_no_smap_fs60": "Global (60)",
        "Global_Single_69_no_smap_fs69": "Global (69)",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    for ax, split, title, color in (
        (axes[0], "ECE spatial", "ECE benefit\n(original RMSE − no-SMAP RMSE)", "#2a9d8f"),
        (axes[1], "WA temporal", "WA degradation\n(no-SMAP RMSE − original RMSE)", "#e76f51"),
    ):
        subset = effect_summary[effect_summary["split"].eq(split)].set_index("model_id")
        missing_models = [model_id for model_id in model_order if model_id not in subset.index]
        if missing_models:
            raise ValueError(f"Effect summary is missing {split} models: {missing_models}")
        subset = subset.loc[model_order]
        x = np.arange(len(model_order))
        ax.bar(
            x,
            subset["effect_rmse_mean"].to_numpy(dtype=float),
            yerr=subset["effect_rmse_std"].to_numpy(dtype=float),
            color=color,
            alpha=0.88,
            capsize=4,
        )
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_title(title)
        ax.set_ylabel("RMSE difference")
        ax.set_xticks(x, [labels[model_id] for model_id in model_order], rotation=38, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("No-SMAP versus original SMAP-trained models (mean ± seed SD)")
    fig.tight_layout()
    path = output_dir / "old_vs_new_rmse_effect.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _prediction_source_path(base_dir: Path, model_id: str, seed: int) -> Path:
    """Resolve a per-seed prediction artifact from a completed experiment."""
    candidates = (
        base_dir / "artifacts" / "predictions" / f"{model_id}__s{seed}.csv",
        base_dir / "predictions" / f"{model_id}__s{seed}.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing prediction artifact for {model_id} seed={seed}; checked {candidates}"
    )


def global_version_source_status(
    data: DataBundle, config: dict[str, Any], seeds: Iterable[int] = (42, 7, 13)
) -> dict[str, Any]:
    """Return a non-throwing readiness audit for the seven-line ECE chart."""
    seed_list = [int(seed) for seed in seeds]
    formal_dir = PROJECT_ROOT / Path(config["data"]["original_results_dir"])
    one_zero_dir = PROJECT_ROOT / Path(config["data"]["salvage_1_0_results_dir"])
    global_specs = [
        spec for spec in config["models"]
        if spec["router"] == "global"
    ]
    missing: list[str] = []
    old_paths: list[str] = []
    one_zero_paths: list[str] = []
    one_one_paths: dict[str, list[str]] = {str(spec["feature_size"]): [] for spec in global_specs}
    for seed in seed_list:
        old = formal_dir / "predictions_spatial" / f"Global_Single_54__s{seed}__ece_preds.npy"
        old_paths.append(str(old))
        if not old.exists():
            missing.append(str(old))
        try:
            one_zero_paths.append(str(_prediction_source_path(one_zero_dir, "Global_Single_54_no_smap", seed)))
        except FileNotFoundError:
            missing.append(f"1.0 Global_Single_54_no_smap seed={seed}")
        for spec in global_specs:
            size = str(spec["feature_size"])
            try:
                one_one_paths[size].append(
                    str(_prediction_source_path(EXP_DIR, spec["id"], seed))
                )
            except FileNotFoundError:
                missing.append(f"1.1 {spec['id']} seed={seed}")
    return {
        "ready": not missing,
        "seeds": seed_list,
        "ece_rows": int(len(data.ece)),
        "ece_stations": sorted(data.ece["station_id"].astype(str).unique()),
        "ece_dates_per_station": int(data.ece.groupby("station_id")["date"].nunique().min()),
        "missing": missing,
        "original_paths": old_paths,
        "salvage_1_0_paths": one_zero_paths,
        "salvage_1_1_paths": one_one_paths,
    }


def _keyed_ece_prediction(
    path: Path, canonical: pd.DataFrame, *, model_id: str, seed: int
) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    source = source[
        source["model_id"].eq(model_id)
        & source["seed"].eq(int(seed))
        & source["dataset"].eq("ece_spatial")
        & source["window"].eq("spatial_ece_v3_full")
    ].copy()
    source["date"] = pd.to_datetime(source["date"], errors="raise").dt.strftime("%Y-%m-%d")
    source["station_id"] = source["station_id"].astype(str)
    keys = ["station_id", "date"]
    if source.duplicated(keys).any():
        raise ValueError(f"Duplicate ECE prediction keys in {path}")
    expected = set(map(tuple, canonical[keys].to_numpy()))
    actual = set(map(tuple, source[keys].to_numpy()))
    if actual != expected:
        raise ValueError(
            f"ECE prediction keys do not match the canonical test set for {path}: "
            f"expected {len(expected)}, got {len(actual)}"
        )
    return source[keys + ["prediction"]]


def make_global_version_charts(
    data: DataBundle,
    config: dict[str, Any],
    output_dir: Path,
    seeds: Iterable[int] = (42, 7, 13),
) -> list[Path]:
    """Generate seven-line old/1.0/size-grid/observed charts for every station."""
    import matplotlib.pyplot as plt

    seed_list = [int(seed) for seed in seeds]
    canonical = data.ece[["station_id", "date", data.target]].copy()
    canonical["station_id"] = canonical["station_id"].astype(str)
    canonical["date"] = pd.to_datetime(canonical["date"], errors="raise").dt.strftime("%Y-%m-%d")
    keys = ["station_id", "date"]
    if canonical.duplicated(keys).any():
        raise ValueError("Canonical ECE test contains duplicate station/date keys.")
    expected_rows = 5 * 30
    if len(canonical) != expected_rows or canonical["station_id"].nunique() != 5:
        raise ValueError(
            f"Expected five ECE stations with 30 dates each; got {len(canonical)} rows "
            f"and {canonical['station_id'].nunique()} stations."
        )
    status = global_version_source_status(data, config, seed_list)
    if not status["ready"]:
        raise FileNotFoundError(
            "Global-version chart inputs are incomplete: " + "; ".join(status["missing"])
        )
    formal_dir = PROJECT_ROOT / Path(config["data"]["original_results_dir"])
    one_zero_dir = PROJECT_ROOT / Path(config["data"]["salvage_1_0_results_dir"])
    global_specs = sorted(
        (spec for spec in config["models"] if spec["router"] == "global"),
        key=lambda spec: int(spec["feature_size"]),
    )
    source_frames: dict[str, list[pd.DataFrame]] = {
        "original": [],
        "salvage_1_0": [],
        **{f"salvage_1_1_{spec['feature_size']}": [] for spec in global_specs},
    }
    for seed in seed_list:
        old_values = np.asarray(
            np.load(formal_dir / "predictions_spatial" / f"Global_Single_54__s{seed}__ece_preds.npy")
        ).ravel()
        if len(old_values) != len(canonical):
            raise ValueError(f"Original ECE prediction length mismatch for seed {seed}.")
        source_frames["original"].append(
            canonical[keys].assign(prediction=old_values, seed=seed)
        )
        source_frames["salvage_1_0"].append(
            _keyed_ece_prediction(
                _prediction_source_path(one_zero_dir, "Global_Single_54_no_smap", seed),
                canonical,
                model_id="Global_Single_54_no_smap",
                seed=seed,
            ).assign(seed=seed)
        )
        for spec in global_specs:
            source_frames[f"salvage_1_1_{spec['feature_size']}"].append(
                _keyed_ece_prediction(
                    _prediction_source_path(EXP_DIR, spec["id"], seed),
                    canonical,
                    model_id=spec["id"],
                    seed=seed,
                ).assign(seed=seed)
            )
    combined = canonical.rename(columns={data.target: "target"})
    for source_name, frames in source_frames.items():
        averaged = (
            pd.concat(frames, ignore_index=True)
            .groupby(keys, as_index=False)["prediction"]
            .mean()
            .rename(columns={"prediction": source_name})
        )
        combined = combined.merge(averaged, on=keys, how="left", validate="one_to_one")
    line_columns = [
        "target", "original", "salvage_1_0",
        *[f"salvage_1_1_{spec['feature_size']}" for spec in global_specs],
    ]
    if combined[line_columns].isna().any().any():
        raise ValueError("Global-version chart inputs contain missing aligned values.")
    combined.to_csv(output_dir.parent / "global_version_predictions.csv", index=False)
    _write_json(
        output_dir.parent / "global_version_provenance.json",
        {
            "seeds": seed_list,
            "aggregation": "mean over common seeds [42, 7, 13]",
            "key": "station_id/date",
            "line_count": len(line_columns),
            "line_labels": [
                "Original Global_Single_54",
                "1.0 no-SMAP Global",
                *[f"1.1 Global ({spec['feature_size']} features)" for spec in global_specs],
                "Ground truth",
            ],
            "ece_stations": sorted(combined["station_id"].unique()),
            "dates_per_station": combined.groupby("station_id")["date"].nunique().to_dict(),
            "sources": status,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for station_id, station in combined.groupby("station_id", sort=True):
        station = station.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        dates = pd.to_datetime(station["date"])
        ax.plot(dates, station["original"], linewidth=1.7, label="Original Global_Single_54")
        ax.plot(dates, station["salvage_1_0"], linewidth=1.7, label="1.0 no-SMAP Global")
        for spec in global_specs:
            size = int(spec["feature_size"])
            ax.plot(
                dates,
                station[f"salvage_1_1_{size}"],
                linewidth=1.7,
                label=f"1.1 Global ({size} features)",
            )
        ax.plot(dates, station["target"], color="black", linewidth=2.2, label="Ground truth")
        if len(ax.lines) != 7:
            raise AssertionError("Global-version chart must contain exactly seven lines.")
        ax.set_title(f"{station_id} — global model version comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Soil moisture")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"ece_{station_id}_global_model_versions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _paired_metrics_from_predictions(
    new_predictions: pd.DataFrame,
    old_dir: Path,
    old_model_id: str,
    new_model_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    old_path = _prediction_source_path(old_dir, old_model_id, seed)
    old = pd.read_csv(old_path, low_memory=False)
    rows: list[dict[str, Any]] = []
    for dataset, window in (
        ("wa_temporal", "temporal_full"),
        ("ece_spatial", "spatial_ece_v3_full"),
    ):
        old_slice = old[(old["dataset"] == dataset) & (old["window"] == window)].copy()
        new_slice = new_predictions[
            (new_predictions["model_id"] == new_model_id)
            & (new_predictions["seed"] == int(seed))
            & (new_predictions["dataset"] == dataset)
            & (new_predictions["window"] == window)
        ].copy()
        old_metrics = compute_metrics(old_slice)
        new_metrics = compute_metrics(new_slice)
        row: dict[str, Any] = {
            "model_id": new_model_id,
            "old_model_id": old_model_id,
            "seed": int(seed),
            "dataset": dataset,
            "window": window,
        }
        for metric in ("rmse", "mae", "bias", "ubrmse", "r2", "pearson", "diff_pearson"):
            row[f"old_{metric}"] = old_metrics[metric]
            row[f"new_{metric}"] = new_metrics[metric]
            row[f"change_{metric}"] = new_metrics[metric] - old_metrics[metric]
        rows.append(row)
    return rows


def build_salvage_1_1_vs_1_0_comparison(
    predictions: pd.DataFrame, config: dict[str, Any], seeds: Iterable[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare completed 1.1 predictions with paired 1.0 predictions."""
    old_dir = PROJECT_ROOT / Path(config["data"]["salvage_1_0_results_dir"])
    rows: list[dict[str, Any]] = []
    for spec in config["models"]:
        new_model_id = str(spec["id"])
        old_model_id = str(spec["salvage_1_0_model_id"])
        for seed in seeds:
            try:
                rows.extend(
                    _paired_metrics_from_predictions(
                        predictions, old_dir, old_model_id, new_model_id, int(seed)
                    )
                )
            except FileNotFoundError:
                continue
    seed_frame = pd.DataFrame(rows)
    if seed_frame.empty:
        return seed_frame, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for keys, group in seed_frame.groupby(["model_id", "old_model_id", "dataset", "window"], sort=True):
        model_id, old_model_id, dataset, window = keys
        row: dict[str, Any] = {
            "model_id": model_id,
            "old_model_id": old_model_id,
            "dataset": dataset,
            "window": window,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in ("rmse", "mae", "bias", "ubrmse", "r2", "pearson", "diff_pearson"):
            row[f"old_{metric}_mean"] = group[f"old_{metric}"].mean()
            row[f"new_{metric}_mean"] = group[f"new_{metric}"].mean()
            row[f"change_{metric}_mean"] = group[f"change_{metric}"].mean()
        summary_rows.append(row)
    return seed_frame, pd.DataFrame(summary_rows)


def build_feature_selection_round_comparison(
    paired_seed: pd.DataFrame,
    config: dict[str, Any],
    expected_seeds: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the global-only before/after feature-selection comparison.

    ``paired_seed`` is produced by ``build_salvage_1_1_vs_1_0_comparison``;
    the old 1.0 model is the before-selection baseline and each 1.1 global
    size is an after-selection variant.  The stored ``rmse_effect`` follows
    the report-facing sign convention: ECE benefit is before-minus-after,
    while WA degradation is after-minus-before.
    """
    global_specs = sorted(
        (spec for spec in config["models"] if spec["router"] == "global"),
        key=lambda spec: int(spec["feature_size"]),
    )
    global_ids = [str(spec["id"]) for spec in global_specs]
    feature_sizes = {
        str(spec["id"]): int(spec["feature_size"]) for spec in global_specs
    }
    seed_columns = [
        "model_id", "feature_size", "before_model_id", "after_model_id", "seed",
        "split", "dataset", "window", "rmse_before", "rmse_after",
        "rmse_change_after_minus_before", "rmse_effect", "rmse_effect_pct",
        "pearson_before", "pearson_after", "pearson_change",
        "diff_pearson_before", "diff_pearson_after", "diff_pearson_change",
    ]
    summary_columns = [
        "model_id", "feature_size", "before_model_id", "after_model_id", "split",
        "dataset", "window", "n_seeds", "rmse_before_mean", "rmse_after_mean",
        "rmse_effect_mean", "rmse_effect_std", "rmse_effect_pct_mean",
        "rmse_effect_pct_std", "pearson_before_mean", "pearson_after_mean",
        "pearson_change_mean", "diff_pearson_before_mean", "diff_pearson_after_mean",
        "diff_pearson_change_mean",
    ]
    if paired_seed.empty:
        return pd.DataFrame(columns=seed_columns), pd.DataFrame(columns=summary_columns)

    required = {
        "model_id", "old_model_id", "seed", "dataset", "window", "old_rmse",
        "new_rmse", "old_pearson", "new_pearson", "change_pearson",
        "old_diff_pearson", "new_diff_pearson", "change_diff_pearson",
    }
    missing = sorted(required - set(paired_seed.columns))
    if missing:
        raise ValueError(f"1.0/1.1 paired comparison is missing columns: {missing}")

    source = paired_seed[
        paired_seed["model_id"].astype(str).isin(global_ids)
        & paired_seed.set_index(["dataset", "window"]).index.isin(_EFFECT_SPLITS)
    ].copy()
    if source.empty:
        return pd.DataFrame(columns=seed_columns), pd.DataFrame(columns=summary_columns)
    keys = ["model_id", "seed", "dataset", "window"]
    if source.duplicated(keys).any():
        duplicates = source.loc[source.duplicated(keys, keep=False), keys].to_dict(orient="records")
        raise ValueError(f"Duplicate feature-selection comparison rows: {duplicates}")

    expected_seed_set = None if expected_seeds is None else {int(seed) for seed in expected_seeds}
    expected_pairs = {
        (model_id, int(seed), split)
        for model_id in global_ids
        for seed in (expected_seed_set or set(source["seed"].astype(int).unique()))
        for split in _EFFECT_SPLITS
    }
    found_pairs = {
        (str(row["model_id"]), int(row["seed"]), (str(row["dataset"]), str(row["window"])))
        for _, row in source.iterrows()
    }
    if expected_seed_set is not None and found_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - found_pairs)
        unexpected_pairs = sorted(found_pairs - expected_pairs)
        raise ValueError(
            "Feature-selection comparison does not have exactly the requested pairs: "
            f"missing={missing_pairs}, unexpected={unexpected_pairs}"
        )

    rows: list[dict[str, Any]] = []
    for _, row in source.sort_values(keys).iterrows():
        model_id = str(row["model_id"])
        dataset = str(row["dataset"])
        window = str(row["window"])
        split, _ = _EFFECT_SPLITS[(dataset, window)]
        before_rmse = float(row["old_rmse"])
        after_rmse = float(row["new_rmse"])
        generic_change = after_rmse - before_rmse
        effect = -generic_change if split == "ECE spatial" else generic_change
        effect_pct = effect / before_rmse * 100.0 if before_rmse else float("nan")
        rows.append({
            "model_id": model_id,
            "feature_size": feature_sizes[model_id],
            "before_model_id": str(row["old_model_id"]),
            "after_model_id": model_id,
            "seed": int(row["seed"]),
            "split": split,
            "dataset": dataset,
            "window": window,
            "rmse_before": before_rmse,
            "rmse_after": after_rmse,
            "rmse_change_after_minus_before": generic_change,
            "rmse_effect": effect,
            "rmse_effect_pct": effect_pct,
            "pearson_before": float(row["old_pearson"]),
            "pearson_after": float(row["new_pearson"]),
            "pearson_change": float(row["change_pearson"]),
            "diff_pearson_before": float(row["old_diff_pearson"]),
            "diff_pearson_after": float(row["new_diff_pearson"]),
            "diff_pearson_change": float(row["change_diff_pearson"]),
        })

    seed_frame = pd.DataFrame(rows, columns=seed_columns)
    summary_rows: list[dict[str, Any]] = []
    for (model_id, split), group in seed_frame.groupby(["model_id", "split"], sort=True):
        first = group.iloc[0]
        summary_rows.append({
            "model_id": model_id,
            "feature_size": int(first["feature_size"]),
            "before_model_id": first["before_model_id"],
            "after_model_id": first["after_model_id"],
            "split": split,
            "dataset": first["dataset"],
            "window": first["window"],
            "n_seeds": int(group["seed"].nunique()),
            "rmse_before_mean": float(group["rmse_before"].mean()),
            "rmse_after_mean": float(group["rmse_after"].mean()),
            "rmse_effect_mean": float(group["rmse_effect"].mean()),
            "rmse_effect_std": float(group["rmse_effect"].std(ddof=1)) if len(group) > 1 else 0.0,
            "rmse_effect_pct_mean": float(group["rmse_effect_pct"].mean()),
            "rmse_effect_pct_std": float(group["rmse_effect_pct"].std(ddof=1)) if len(group) > 1 else 0.0,
            "pearson_before_mean": float(group["pearson_before"].mean()),
            "pearson_after_mean": float(group["pearson_after"].mean()),
            "pearson_change_mean": float(group["pearson_change"].mean()),
            "diff_pearson_before_mean": float(group["diff_pearson_before"].mean()),
            "diff_pearson_after_mean": float(group["diff_pearson_after"].mean()),
            "diff_pearson_change_mean": float(group["diff_pearson_change"].mean()),
        })
    return seed_frame, pd.DataFrame(summary_rows, columns=summary_columns)


def interpret_feature_selection_round(
    effect_summary: pd.DataFrame,
    expected_seeds: Iterable[int] = DEFAULT_SEEDS,
) -> str:
    """Create concise, data-driven prose for the README feature-selection section."""
    if effect_summary.empty:
        return "Interpretation is deferred until the complete 1.0/1.1 paired results are available."
    required = {
        "feature_size", "split", "n_seeds", "rmse_before_mean", "rmse_after_mean",
        "rmse_effect_mean", "rmse_effect_pct_mean", "pearson_change_mean",
        "diff_pearson_change_mean",
    }
    missing = sorted(required - set(effect_summary.columns))
    if missing:
        raise ValueError(f"Feature-selection effect summary is missing columns: {missing}")
    expected_n = len({int(seed) for seed in expected_seeds})
    if set(effect_summary["n_seeds"].astype(int)) != {expected_n}:
        raise ValueError(
            f"Feature-selection interpretation requires {expected_n} seeds per row."
        )

    def fmt(value: Any, digits: int = 4) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "NA" if pd.isna(numeric) else f"{float(numeric):.{digits}f}"

    ece = effect_summary[effect_summary["split"].eq("ECE spatial")].sort_values("feature_size")
    wa = effect_summary[effect_summary["split"].eq("WA temporal")].sort_values("feature_size")
    if ece.empty or wa.empty:
        raise ValueError("Feature-selection interpretation requires both ECE and WA splits.")
    best_ece = ece.sort_values(["rmse_after_mean", "feature_size"]).iloc[0]
    ece_effects = ece["rmse_effect_mean"].astype(float)
    if (ece_effects > 0).all():
        ece_statement = "all four selected-size variants improve pooled ECE RMSE"
    elif (ece_effects < 0).all():
        ece_statement = "all four selected-size variants worsen pooled ECE RMSE"
    else:
        ece_statement = "the ECE result is mixed across selected sizes"
    wa_effects = wa["rmse_effect_mean"].astype(float)
    wa_low = wa.loc[wa["rmse_effect_mean"].idxmin()]
    wa_high = wa.loc[wa["rmse_effect_mean"].idxmax()]
    ece_pearson = ece["pearson_change_mean"].dropna()
    ece_diff = ece["diff_pearson_change_mean"].dropna()
    wa_pearson = wa["pearson_change_mean"].dropna()
    wa_diff = wa["diff_pearson_change_mean"].dropna()
    trend_parts = []
    if not ece_pearson.empty:
        trend_parts.append(
            f"ECE Pearson changes range from {fmt(ece_pearson.min())} to {fmt(ece_pearson.max())}"
        )
    if not ece_diff.empty:
        trend_parts.append(
            f"ECE first-difference Pearson changes range from {fmt(ece_diff.min())} to {fmt(ece_diff.max())}"
        )
    if not wa_pearson.empty:
        trend_parts.append(
            f"WA Pearson changes range from {fmt(wa_pearson.min())} to {fmt(wa_pearson.max())}"
        )
    if not wa_diff.empty:
        trend_parts.append(
            f"WA first-difference Pearson changes range from {fmt(wa_diff.min())} to {fmt(wa_diff.max())}"
        )
    lines = [
        f"- Best pooled ECE RMSE is the {int(best_ece['feature_size'])}-feature model: "
        f"{fmt(best_ece['rmse_after_mean'])} after selection versus "
        f"{fmt(best_ece['rmse_before_mean'])} before, an ECE benefit of "
        f"{fmt(best_ece['rmse_effect_mean'])} ({fmt(best_ece['rmse_effect_pct_mean'], 2)}%).",
        f"- The feature-selection round means {ece_statement}; ECE benefit spans "
        f"{fmt(ece['rmse_effect_mean'].min())} to {fmt(ece['rmse_effect_mean'].max())} RMSE units.",
        f"- Pooled WA degradation ranges from {fmt(wa_effects.min())} "
        f"({int(wa_low['feature_size'])} features) to {fmt(wa_effects.max())} "
        f"({int(wa_high['feature_size'])} features); negative values indicate "
        "improvement and positive values indicate degradation.",
        f"- " + ("; ".join(trend_parts) + "." if trend_parts else "Trend correlations were unavailable."),
        f"- These are pooled comparisons across five ECE stations and seven WA stations, averaged over {expected_n} common seeds; they describe association, not a causal effect of feature selection.",
    ]
    return "\n".join(lines)


def _load_saved_predictions(
    config: dict[str, Any], seeds: Iterable[int]
) -> pd.DataFrame:
    """Load exactly the configured model/seed prediction artifacts."""
    specs = _model_specs(config)
    expected = {(_spec["id"], int(seed)) for _spec in specs for seed in seeds}
    frames: list[pd.DataFrame] = []
    found: set[tuple[str, int]] = set()
    for model_id, seed in sorted(expected):
        path = _prediction_path(model_id, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing saved prediction artifact: {path}")
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            raise ValueError(f"Saved prediction artifact is empty: {path}")
        pair = (str(frame["model_id"].iloc[0]), int(frame["seed"].iloc[0]))
        if pair != (str(model_id), int(seed)):
            raise ValueError(f"Prediction identity mismatch in {path}: {pair}")
        found.add(pair)
        frames.append(frame)
    if found != expected:
        raise RuntimeError(f"Saved prediction coverage is incomplete: {sorted(expected - found)}")
    predictions = pd.concat(frames, ignore_index=True)
    key_columns = ["model_id", "seed", "dataset", "window", "station_id", "date"]
    if predictions.duplicated(key_columns).any():
        raise RuntimeError("Saved prediction artifacts contain duplicate prediction keys.")
    return predictions


def rebuild_reports_from_saved_predictions(
    config: dict[str, Any], seeds: list[int]
) -> None:
    """Rebuild derived reports without refitting models or routers."""
    data = load_data(config)
    predictions = _load_saved_predictions(config, seeds)
    _write_json(EXP_DIR / "feature_manifest.json", feature_manifest(data, config))
    _write_json(EXP_DIR / "selection_provenance.json", {
        "manifest_paths": {
            str(size): str(_resolve_experiment_path(path))
            for size, path in config["selected_feature_manifests"].items()
        },
        "manifest_sha256": {
            str(size): hashlib.sha256(
                _resolve_experiment_path(path).read_bytes()
            ).hexdigest()
            for size, path in config["selected_feature_manifests"].items()
        },
        "selected_feature_hashes": {
            str(size): _hash_features(features)
            for size, features in sorted(data.selected_feature_sets.items())
        },
        "selected_feature_counts": {
            str(size): len(features)
            for size, features in sorted(data.selected_feature_sets.items())
        },
        "selected_features_by_size": {
            str(size): features
            for size, features in sorted(data.selected_feature_sets.items())
        },
        "fit_scope": "WA trainval only",
        "ece_target_used_for_fit": False,
        "delta_additions": [],
    })
    _write_json(EXP_DIR / "input_audit.json", {
        "train_rows": len(data.train), "val_rows": len(data.val), "trainval_rows": len(data.trainval),
        "wa_test_rows": len(data.test), "ece_rows": len(data.ece),
        "wa_train_stations": sorted(data.trainval["station_id"].astype(str).unique()),
        "ece_stations": sorted(data.ece["station_id"].astype(str).unique()),
        "ece_date_min": str(data.ece["date"].min().date()), "ece_date_max": str(data.ece["date"].max().date()),
        "ece_target_used_for_fit": False,
    })
    seed_metrics, summary = summarize_predictions(predictions)
    predictions.to_csv(EXP_DIR / "predictions.csv", index=False)
    seed_metrics.to_csv(EXP_DIR / "seed_metrics.csv", index=False)
    summary.to_csv(EXP_DIR / "summary.csv", index=False)
    summary[summary["scope"] != "__pooled__"].to_csv(EXP_DIR / "station_summary.csv", index=False)
    references = load_reference_metrics(config)
    references.to_csv(EXP_DIR / "reference_metrics.csv", index=False)
    comparison = build_reference_comparison(seed_metrics, references)
    comparison.to_csv(EXP_DIR / "reference_comparison.csv", index=False)
    effect_seed, effect_summary = build_old_vs_new_effects(
        comparison,
        expected_seeds=seeds,
        expected_model_ids=[spec["id"] for spec in _model_specs(config)],
    )
    effect_seed.to_csv(EXP_DIR / "old_vs_new_effect_seed.csv", index=False)
    effect_summary.to_csv(EXP_DIR / "old_vs_new_effect_summary.csv", index=False)
    one_zero_seed, one_zero_summary = build_salvage_1_1_vs_1_0_comparison(
        predictions, config, seeds
    )
    one_zero_seed.to_csv(EXP_DIR / "salvage_1_1_vs_1_0_seed.csv", index=False)
    one_zero_summary.to_csv(EXP_DIR / "salvage_1_1_vs_1_0_summary.csv", index=False)
    feature_selection_seed, feature_selection_summary = build_feature_selection_round_comparison(
        one_zero_seed, config, expected_seeds=seeds
    )
    feature_selection_seed.to_csv(EXP_DIR / "feature_selection_round_effect_seed.csv", index=False)
    feature_selection_summary.to_csv(EXP_DIR / "feature_selection_round_effect_summary.csv", index=False)
    version_status = global_version_source_status(data, config, (42, 7, 13))
    _write_json(EXP_DIR / "global_version_provenance.json", version_status)
    if version_status["ready"]:
        make_global_version_charts(data, config, EXP_DIR / "figures", (42, 7, 13))
    print(
        f"[reports-only] models={len(_model_specs(config))} seeds={len(seeds)} "
        f"prediction_rows={len(predictions)}"
    )


def run_experiment(config: dict[str, Any], seeds: list[int], smoke: bool,
                   resume: bool, device_override: str | None = None) -> None:
    global PREDICTION_DIR, CHECKPOINT_DIR
    data = load_data(config)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(EXP_DIR / "feature_manifest.json", feature_manifest(data, config))
    selection_manifests = {
        str(size): _resolve_experiment_path(path)
        for size, path in config["selected_feature_manifests"].items()
    }
    _write_json(EXP_DIR / "selection_provenance.json", {
        "manifest_paths": {size: str(path) for size, path in selection_manifests.items()},
        "manifest_sha256": {
            size: hashlib.sha256(path.read_bytes()).hexdigest()
            for size, path in selection_manifests.items()
        },
        "selected_feature_hashes": {
            str(size): _hash_features(features)
            for size, features in sorted(data.selected_feature_sets.items())
        },
        "selected_feature_counts": {
            str(size): len(features)
            for size, features in sorted(data.selected_feature_sets.items())
        },
        "selected_features_by_size": {
            str(size): features
            for size, features in sorted(data.selected_feature_sets.items())
        },
        "fit_scope": "WA trainval only",
        "ece_target_used_for_fit": False,
        "delta_additions": [],
    })
    _write_json(EXP_DIR / "input_audit.json", {
        "train_rows": len(data.train), "val_rows": len(data.val), "trainval_rows": len(data.trainval),
        "wa_test_rows": len(data.test), "ece_rows": len(data.ece),
        "wa_train_stations": sorted(data.trainval["station_id"].astype(str).unique()),
        "ece_stations": sorted(data.ece["station_id"].astype(str).unique()),
        "ece_date_min": str(data.ece["date"].min().date()), "ece_date_max": str(data.ece["date"].max().date()),
        "ece_target_used_for_fit": False,
    })
    params = _base_params(config, smoke, device_override)
    specs = _model_specs(config)
    routers = {spec["id"]: fit_router(spec, data, config, params) for spec in specs}
    audit = build_router_audit(specs, routers, data, config)
    audit.to_csv(EXP_DIR / "routing_audit.csv", index=False)
    _write_json(EXP_DIR / "routing_audit.json", {"rows": audit.to_dict(orient="records")})

    for spec in specs:
        for seed in seeds:
            if not resume:
                prediction_path = _prediction_path(spec["id"], seed)
                checkpoint = _checkpoint_dir(spec["id"], seed)
                if prediction_path.exists():
                    prediction_path.unlink()
                if checkpoint.exists():
                    for child in checkpoint.iterdir():
                        if child.is_file():
                            child.unlink()
            _run_one(spec, seed, data, routers[spec["id"]], config, params, smoke)

    expected = {(_spec["id"], seed) for _spec in specs for seed in seeds}
    found = set()
    prediction_frames = []
    expected_paths = {
        _prediction_path(model_id, seed)
        for model_id, seed in expected
    }
    for path in sorted(expected_paths):
        if not path.exists():
            continue
        source = pd.read_csv(path, low_memory=False)
        if source.empty:
            continue
        found.add((str(source["model_id"].iloc[0]), int(source["seed"].iloc[0])))
        prediction_frames.append(source)
    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(f"Missing model-seed outputs: {missing}")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_keys = [
        "model_id", "seed", "dataset", "window", "station_id", "date"
    ]
    if predictions.duplicated(prediction_keys).any():
        raise RuntimeError("Duplicate model/seed/dataset/window/station/date predictions found.")
    expected_windows = {
        ("wa_temporal", "temporal_full"),
        ("wa_temporal", "temporal_summer_2025"),
        ("ece_spatial", "spatial_ece_v3_full"),
    }
    for model_id, seed in expected:
        found_windows = set(
            map(tuple, predictions[
                (predictions["model_id"] == model_id) & (predictions["seed"] == seed)
            ][["dataset", "window"]].drop_duplicates().to_numpy())
        )
        if found_windows != expected_windows:
            raise RuntimeError(
                f"Incomplete evaluation windows for {model_id} seed={seed}: {found_windows}"
            )
    predictions.to_csv(EXP_DIR / "predictions.csv", index=False)
    seed_metrics, summary = summarize_predictions(predictions)
    seed_metrics.to_csv(EXP_DIR / "seed_metrics.csv", index=False)
    summary.to_csv(EXP_DIR / "summary.csv", index=False)
    station_summary = summary[summary["scope"] != "__pooled__"].copy()
    station_summary.to_csv(EXP_DIR / "station_summary.csv", index=False)
    references = load_reference_metrics(config)
    references.to_csv(EXP_DIR / "reference_metrics.csv", index=False)
    comparison = build_reference_comparison(seed_metrics, references)
    comparison.to_csv(EXP_DIR / "reference_comparison.csv", index=False)
    effect_seed, effect_summary = build_old_vs_new_effects(
        comparison,
        expected_seeds=seeds,
        expected_model_ids=[spec["id"] for spec in specs],
    )
    effect_seed.to_csv(EXP_DIR / "old_vs_new_effect_seed.csv", index=False)
    effect_summary.to_csv(EXP_DIR / "old_vs_new_effect_summary.csv", index=False)
    one_zero_seed, one_zero_summary = build_salvage_1_1_vs_1_0_comparison(
        predictions, config, seeds
    )
    one_zero_seed.to_csv(EXP_DIR / "salvage_1_1_vs_1_0_seed.csv", index=False)
    one_zero_summary.to_csv(EXP_DIR / "salvage_1_1_vs_1_0_summary.csv", index=False)
    feature_selection_seed, feature_selection_summary = build_feature_selection_round_comparison(
        one_zero_seed, config, expected_seeds=seeds
    )
    feature_selection_seed.to_csv(EXP_DIR / "feature_selection_round_effect_seed.csv", index=False)
    feature_selection_summary.to_csv(EXP_DIR / "feature_selection_round_effect_summary.csv", index=False)
    version_status = global_version_source_status(data, config, (42, 7, 13))
    _write_json(EXP_DIR / "global_version_provenance.json", version_status)
    if version_status["ready"]:
        make_global_version_charts(data, config, EXP_DIR / "figures", (42, 7, 13))
    else:
        print(
            "[global-version] chart deferred until old and 1.0 common-seed artifacts "
            f"are complete; missing={len(version_status['missing'])}"
        )
    write_smap_invariance(data, specs, routers, config, params)
    print(f"[complete] models={len(specs)} seeds={len(seeds)} prediction_rows={len(predictions)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run seed 42 with 100 CPU trees.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Optional XGBoost device override.")
    parser.add_argument("--no-resume", action="store_true", help="Refit outputs even when checkpoints exist.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild reports from saved predictions without fitting.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_configuration()
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else list(config.get("seeds", DEFAULT_SEEDS))
    if args.smoke:
        seeds = [42]
    if args.report_only:
        rebuild_reports_from_saved_predictions(config, seeds)
    else:
        run_experiment(config, seeds, args.smoke, resume=not args.no_resume, device_override=args.device)


if __name__ == "__main__":
    main()
