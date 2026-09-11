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

DEFAULT_SEEDS = [42, 7, 13, 101, 123]
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

    selected = v0_features + backbone_features + dynamic_features
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
    )


def feature_manifest(data: DataBundle, config: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "experiment": config["experiment"]["name"],
        "policy": "drop every feature whose name contains SMAP, case-insensitive",
        "v0_router": {"parent": data.v0_parent, "dropped": [f for f in data.v0_parent if f not in data.v0_features], "effective": data.v0_features},
        "backbone_54_model_and_router": {"parent": data.backbone_parent, "dropped": [f for f in data.backbone_parent if f not in data.backbone_features], "effective": data.backbone_features},
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
        return data.backbone_features
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
    models: dict[int, XGBRegressor] = {}
    clusters = (0, 1) if spec["two_regime"] else (0,)
    for cluster in clusters:
        mask = labels == cluster if spec["two_regime"] else np.ones(len(labels), dtype=bool)
        if not mask.any():
            continue
        model = XGBRegressor(**params)
        model.fit(data.trainval.loc[mask, data.backbone_features], y[mask], verbose=False)
        models[int(cluster)] = model
    return models


def predict_regressors(spec: dict[str, Any], data: DataBundle, router: Any,
                       models: dict[int, XGBRegressor], frame: pd.DataFrame,
                       fallback: float) -> tuple[np.ndarray, np.ndarray]:
    labels = _labels(router, frame)
    prediction = np.full(len(frame), float(fallback), dtype=float)
    clusters = (0, 1) if spec["two_regime"] else (0,)
    for cluster in clusters:
        mask = labels == cluster
        model = models.get(int(cluster))
        if mask.any() and model is not None:
            prediction[mask] = np.asarray(model.predict(frame.loc[mask, data.backbone_features])).ravel()
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
        "effective_model_features": data.backbone_features,
        "feature_hash": _hash_features(data.backbone_features),
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
        "Clustering_V0_Full_k2_no_smap",
        "Clustering_Backbone54_k2_no_smap",
        "Trained_Gating_k2_no_smap",
        "Global_Single_54_no_smap",
    ]
    regime = [
        "Univariate_G_API_k2_no_smap",
        "Clustering_Dynamic_k2_no_smap",
        "Seasonal_Binary_k2_no_smap",
        "Global_Single_54_no_smap",
    ]
    labels = {
        "Clustering_V0_Full_k2_no_smap": "V0 KMeans",
        "Clustering_Backbone54_k2_no_smap": "Backbone54 KMeans",
        "Trained_Gating_k2_no_smap": "Trained gate",
        "Univariate_G_API_k2_no_smap": "G_API gate",
        "Clustering_Dynamic_k2_no_smap": "Dynamic gate",
        "Seasonal_Binary_k2_no_smap": "Seasonal gate",
        "Global_Single_54_no_smap": "Global 54-lineage",
    }
    ece = predictions[(predictions["dataset"] == "ece_spatial") & (predictions["seed"] == 42)].copy()
    paths: list[Path] = []
    for station in sorted(ece["station_id"].unique()):
        station_data = ece[ece["station_id"] == station].sort_values("date")
        observed = station_data.drop_duplicates("date")
        for suite_name, model_ids in (("architecture", architecture), ("regime", regime)):
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


def run_experiment(config: dict[str, Any], seeds: list[int], smoke: bool,
                   resume: bool, device_override: str | None = None) -> None:
    global PREDICTION_DIR, CHECKPOINT_DIR
    data = load_data(config)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(EXP_DIR / "feature_manifest.json", feature_manifest(data, config))
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

    prediction_files = sorted(PREDICTION_DIR.glob("*.csv"))
    expected = {(_spec["id"], seed) for _spec in specs for seed in seeds}
    found = set()
    prediction_frames = []
    for path in prediction_files:
        source = pd.read_csv(path, low_memory=False)
        if source.empty:
            continue
        found.add((str(source["model_id"].iloc[0]), int(source["seed"].iloc[0])))
        prediction_frames.append(source)
    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(f"Missing model-seed outputs: {missing}")
    predictions = pd.concat(prediction_frames, ignore_index=True)
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
    write_smap_invariance(data, specs, routers, config, params)
    print(f"[complete] models={len(specs)} seeds={len(seeds)} prediction_rows={len(predictions)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run seed 42 with 100 CPU trees.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Optional XGBoost device override.")
    parser.add_argument("--no-resume", action="store_true", help="Refit outputs even when checkpoints exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_configuration()
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else list(config.get("seeds", DEFAULT_SEEDS))
    if args.smoke:
        seeds = [42]
    run_experiment(config, seeds, args.smoke, resume=not args.no_resume, device_override=args.device)


if __name__ == "__main__":
    main()
