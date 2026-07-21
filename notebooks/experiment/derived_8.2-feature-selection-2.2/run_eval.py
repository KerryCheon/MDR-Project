"""Evaluate frozen 2.2 feature sets without casually consuming the test split."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from artifact_state import (
    artifact_is_complete,
    atomic_write_csv,
    atomic_write_json,
    capture_source_state,
    invalidate_completion,
    write_completion_marker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.yaml"
TARGET = "soil_moisture_5cm"
SEED = 42
EVALUATION_REQUIRED_FILES = (
    "metrics_summary.csv",
    "metrics_by_year.csv",
    "metrics_by_station.csv",
    "evaluation_manifest.json",
)
HAND_MDR_V25 = [
    "SMAP_sm_pm_interp_ema02", "V_rollmin_LST_modis_kobs30", "D_sin_DOY",
    "G_rain_sum_3d", "V_ema_G_API_kobs7", "V_rollmin_G_API_kobs30",
    "G_rain_sum_7d", "C_lag_LST_modis_kobs30", "C_lag_G_API_kobs1",
    "V_ema_G_API_kobs14", "V_rollmean_G_API_kobs14", "G_API", "G_DSLR",
    "SMAP_ampm_diff_interp", "V_rollmax_G_API_kobs30",
    "V_ema_G_API_kobs30", "V_rollmean_s2_b11_kobs7",
    "V_ema_LST_modis_kobs7", "V_rollmean_G_API_kobs7",
    "C_lag_s2_b11_kobs30", "A_d_E_SAR_diff_kobs14",
    "C_lag_LST_modis_kobs6", "A_d_LST_modis_kobs14",
    "A_d_SMAP_sm_interp_kobs14", "V_rollstd_SMAP_sm_interp_kobs30",
    "SMAP_sm_interp_grad7", "year_frac", "sin_year", "cos_year",
    "API_x_year", "SMAP_x_year", "slope", "elev", "K_slope_sin",
    "K_slope_cos", "K_aspect_cos", "J_clay_wfrac_b0",
    "J_sand_wfrac_b0",
]


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _probe_device() -> str:
    requested = os.environ.get("XGB_DEVICE", "").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested not in {"", "cuda"}:
        raise ValueError("XGB_DEVICE must be 'cpu' or 'cuda'")
    try:
        model = xgb.XGBRegressor(n_estimators=1, device="cuda", verbosity=0)
        model.fit(np.asarray([[0.0], [1.0]]), np.asarray([0.0, 1.0]))
        return "cuda"
    except Exception:
        return "cpu"


def _metrics(y_true, prediction) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    residual = y_true - prediction
    return {
        "R2": float(r2_score(y_true, prediction)),
        "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "ubRMSE": float(np.std(residual)),
        "Bias": float(np.mean(residual)),
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "Med|Err|": float(np.median(np.abs(residual))),
        "Pearson": float(np.corrcoef(y_true, prediction)[0, 1]),
    }


def _weights(dates: pd.Series, beta: float) -> np.ndarray | None:
    if float(beta) == 0.0:
        return None
    years = pd.to_datetime(dates).dt.year.to_numpy(dtype=float)
    values = np.exp(float(beta) * (years - float(np.max(years))))
    return values / float(np.mean(values))


def _finite_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Match selection preprocessing while retaining XGBoost native missing values."""
    return (
        frame[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def _fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    beta: float,
    params: dict,
):
    missing = [feature for feature in features if feature not in train.columns]
    if missing:
        raise ValueError(f"Missing selected features: {missing[:10]}")
    train_target = pd.to_numeric(train[TARGET], errors="coerce")
    validation_target = pd.to_numeric(validation[TARGET], errors="coerce")
    train_ok = train_target.notna()
    validation_ok = validation_target.notna()
    model = xgb.XGBRegressor(**params)
    model.fit(
        _finite_features(train.loc[train_ok], features),
        train_target.loc[train_ok],
        sample_weight=_weights(train.loc[train_ok, "date"], beta),
    )
    prediction = np.asarray(
        model.predict(_finite_features(validation.loc[validation_ok], features))
    ).ravel()
    return (
        validation_target.loc[validation_ok].to_numpy(dtype=float),
        prediction,
        validation.loc[validation_ok, ["station_id", "date"]].copy(),
    )


def _load_features(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as stream:
        return list(json.load(stream)["features"])


def _v3_features() -> list[str]:
    path = PROJECT_ROOT / "data/splits/derived_8.2/dataset_metadata.py"
    spec = importlib.util.spec_from_file_location("derived_8_2_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.OVERALL_SELECTED_FEATURES_V3)


def _dynamic_router(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    router_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    columns = list(router_config["columns"])
    train_values = train[columns].apply(pd.to_numeric, errors="coerce")
    validation_values = validation[columns].apply(pd.to_numeric, errors="coerce")
    means = train_values.mean()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values.fillna(means))
    validation_scaled = scaler.transform(validation_values.fillna(means))
    model = KMeans(
        n_clusters=int(router_config["n_clusters"]),
        n_init=int(router_config["n_init"]),
        random_state=SEED,
    )
    return model.fit_predict(train_scaled), model.predict(validation_scaled)


def _frozen_router(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    router_artifact: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the train-only router used to define regime feature semantics."""
    columns = list(router_artifact["columns"])
    means = pd.Series(router_artifact["means"], dtype=float).reindex(columns)
    center = np.asarray(router_artifact["scaler_mean"], dtype=float)
    scale = np.asarray(router_artifact["scaler_scale"], dtype=float)
    scale = np.where(scale == 0.0, 1.0, scale)
    centroids = np.asarray(router_artifact["centers"], dtype=float)

    def _labels(frame: pd.DataFrame) -> np.ndarray:
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(means)
        scaled = (values.to_numpy(dtype=float) - center) / scale
        distances = np.square(scaled[:, None, :] - centroids[None, :, :]).sum(axis=2)
        return np.argmin(distances, axis=1)

    return _labels(train), _labels(validation)


def _evaluate_moe(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    regime_features: dict[int, list[str]],
    router_config: dict,
    beta: float,
    params: dict,
    router_artifact: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if router_artifact is None:
        train_labels, validation_labels = _dynamic_router(
            train,
            validation,
            router_config,
        )
    else:
        train_labels, validation_labels = _frozen_router(
            train,
            validation,
            router_artifact,
        )
    target = pd.to_numeric(validation[TARGET], errors="coerce")
    valid_target = target.notna().to_numpy()
    prediction = np.full(len(validation), np.nan, dtype=float)

    for regime, features in sorted(regime_features.items()):
        train_mask = train_labels == regime
        validation_mask = (validation_labels == regime) & valid_target
        model = xgb.XGBRegressor(**params)
        train_target = pd.to_numeric(
            train.loc[train_mask, TARGET],
            errors="coerce",
        )
        train_nonmissing = train_target.notna()
        train_rows = train.loc[train_mask].loc[train_nonmissing]
        model.fit(
            _finite_features(train_rows, features),
            train_target.loc[train_nonmissing],
            sample_weight=_weights(train_rows["date"], beta),
        )
        prediction[validation_mask] = model.predict(
            _finite_features(validation.loc[validation_mask], features)
        )

    if np.isnan(prediction[valid_target]).any():
        raise RuntimeError("MoE left validation rows without predictions")
    metadata = validation.loc[valid_target, ["station_id", "date"]].copy()
    return (
        target.loc[valid_target].to_numpy(dtype=float),
        prediction[valid_target],
        metadata,
    )


def _metric_rows(
    dataset: str,
    model_name: str,
    beta: float,
    y_true: np.ndarray,
    prediction: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[dict, list[dict], list[dict]]:
    overall = {
        "dataset": dataset,
        "model": model_name,
        "beta": beta,
        **_metrics(y_true, prediction),
    }
    years = pd.to_datetime(metadata["date"]).dt.year.to_numpy()
    stations = metadata["station_id"].astype(str).to_numpy()
    year_rows = []
    station_rows = []
    for year in sorted(np.unique(years)):
        mask = years == year
        year_rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "beta": beta,
                "year": int(year),
                **_metrics(y_true[mask], prediction[mask]),
            }
        )
    for station in sorted(np.unique(stations)):
        mask = stations == station
        station_rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "beta": beta,
                "station_id": station,
                **_metrics(y_true[mask], prediction[mask]),
            }
        )
    return overall, year_rows, station_rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Consume the held-out 2023-2025 test split after feature sets are frozen.",
    )
    parser.add_argument(
        "--retrospective-test",
        action="store_true",
        help=(
            "Evaluate nested-revision artifacts on the already-consumed test split; "
            "diagnostic only, never an unbiased SOTA claim."
        ),
    )
    parser.add_argument(
        "--artifact-set",
        choices=(
            "final",
            "smoke",
            "nested",
            "nested_smoke",
            "nested_locked_outer",
            "crossed_candidates_locked_outer",
        ),
        default="final",
        help="Selection artifact tree to evaluate.",
    )
    args = parser.parse_args(argv)

    if args.confirm_final and args.retrospective_test:
        parser.error("--confirm-final and --retrospective-test are mutually exclusive")
    if args.confirm_final and args.artifact_set != "final":
        parser.error("--confirm-final is reserved for the frozen original final artifacts")
    retrospective_sets = {
        "nested",
        "nested_locked_outer",
        "crossed_candidates_locked_outer",
    }
    if args.retrospective_test and args.artifact_set not in retrospective_sets:
        parser.error("--retrospective-test requires a completed revision artifact set")

    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    config = _load_config()
    device = _probe_device()
    source_state = capture_source_state(PROJECT_ROOT, EXP_DIR)
    params = dict(config["evaluation"]["xgb_params"])
    params["device"] = device
    artifact_root = EXP_DIR / "artifacts" / args.artifact_set
    uses_test = args.confirm_final or args.retrospective_test
    if args.confirm_final:
        output_name = "final_test_eval"
    elif args.retrospective_test:
        output_name = "retrospective_test_eval"
    else:
        output_name = "validation_eval"
    output_root = artifact_root / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    invalidate_completion(output_root)

    overall_rows = []
    year_rows = []
    station_rows = []
    parallel_workers = max(1, int(os.environ.get("XGB_PARALLEL_WORKERS", "16")))
    for dataset in ("derived_8.0", "derived_8.2"):
        split_dir = PROJECT_ROOT / "data/splits" / dataset
        train = pd.read_csv(split_dir / "train.csv")
        val = pd.read_csv(split_dir / "val.csv")
        test = pd.read_csv(split_dir / "test.csv")
        if uses_test:
            fit_frame = pd.concat([train, val], ignore_index=True)
            score_frame = test
        else:
            fit_frame = train
            score_frame = val

        selected_path = artifact_root / dataset / "global" / "selected_features.json"
        if not artifact_is_complete(selected_path.parent, [selected_path.name]):
            raise FileNotFoundError(
                f"Run selection first; incomplete artifact at {selected_path.parent}"
            )
        feature_sets = {"2.2_global": _load_features(selected_path)}
        if dataset == "derived_8.0":
            feature_sets["hand_mdr_v25"] = list(HAND_MDR_V25)
        else:
            feature_sets["V3"] = _v3_features()
            c1_path = (
                PROJECT_ROOT
                / "notebooks/experiment/derived_8.2-feature-selection-2.1"
                / "artifacts/derived_8.2/c1_baseline_bypass_off/selected_features.json"
            )
            feature_sets["2.1_c1"] = _load_features(c1_path)

        regime_features = None
        router_artifact = None
        regime_paths = [
            artifact_root / dataset / f"regime_{regime}/selected_features.json"
            for regime in (0, 1)
        ]
        if dataset == "derived_8.2" and all(
            artifact_is_complete(path.parent, [path.name])
            for path in regime_paths
        ):
            regime_features = {
                regime: _load_features(regime_paths[regime])
                for regime in (0, 1)
            }
            router_path = artifact_root / dataset / "router.json"
            if router_path.exists():
                with open(router_path, encoding="utf-8") as stream:
                    router_artifact = json.load(stream)

        tasks = []
        for beta in config["evaluation"]["temporal_betas"]:
            for model_name, features in feature_sets.items():
                tasks.append(("global", float(beta), model_name, features, None))
            if regime_features is not None:
                tasks.append(
                    (
                        "moe",
                        float(beta),
                        "2.2_clustering_dynamic_k2_shared_plus_delta",
                        regime_features,
                        router_artifact,
                    )
                )
                if router_artifact is not None:
                    shared_features = {
                        regime: feature_sets["2.2_global"]
                        for regime in regime_features
                    }
                    tasks.extend(
                        [
                            (
                                "moe",
                                float(beta),
                                "2.2_clustering_frozen_k2_shared_only",
                                shared_features,
                                router_artifact,
                            ),
                            (
                                "moe",
                                float(beta),
                                "2.2_clustering_refit_k2_shared_plus_delta",
                                regime_features,
                                None,
                            ),
                        ]
                    )

        def _run_eval_task(task):
            kind, beta, model_name, features, task_router = task
            if kind == "global":
                y_true, prediction, metadata = _fit_predict(
                    fit_frame,
                    score_frame,
                    features,
                    beta,
                    params,
                )
            else:
                y_true, prediction, metadata = _evaluate_moe(
                    fit_frame,
                    score_frame,
                    features,
                    config["regime_delta"]["router"],
                    beta,
                    params,
                    router_artifact=task_router,
                )
            return _metric_rows(
                dataset,
                model_name,
                beta,
                y_true,
                prediction,
                metadata,
            )

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            for overall, years, stations in executor.map(_run_eval_task, tasks):
                overall_rows.append(overall)
                year_rows.extend(years)
                station_rows.extend(stations)

    summary = pd.DataFrame(overall_rows)
    atomic_write_csv(summary, output_root / "metrics_summary.csv")
    atomic_write_csv(pd.DataFrame(year_rows), output_root / "metrics_by_year.csv")
    atomic_write_csv(
        pd.DataFrame(station_rows),
        output_root / "metrics_by_station.csv",
    )

    gates = {}
    if args.confirm_final and args.artifact_set == "final":
        lookup = summary.set_index(["dataset", "model", "beta"])["R2"]
        gates = {
            "derived_8.0_drift": {
                "value": float(lookup[("derived_8.0", "2.2_global", 0.2)]),
                "threshold": 0.8253479076167946,
            },
            "derived_8.0_no_drift": {
                "value": float(lookup[("derived_8.0", "2.2_global", 0.0)]),
                "threshold": 0.8222,
            },
            "derived_8.2_global": {
                "value": float(lookup[("derived_8.2", "2.2_global", 0.0)]),
                "threshold": 0.6648,
            },
            "derived_8.2_moe": {
                "value": float(
                    lookup[
                        (
                            "derived_8.2",
                            "2.2_clustering_dynamic_k2_shared_plus_delta",
                            0.0,
                        )
                    ]
                ),
                "threshold": 0.6672,
            },
        }
        for gate in gates.values():
            gate["pass"] = bool(gate["value"] > gate["threshold"])

    atomic_write_json(
        output_root / "evaluation_manifest.json",
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "parallel_workers": parallel_workers,
            "artifact_set": args.artifact_set,
            "final_test": args.confirm_final,
            "retrospective_test": args.retrospective_test,
            "unbiased_sota_eligible": bool(
                args.confirm_final and args.artifact_set == "final"
            ),
            "source_state": source_state,
            "gates": gates,
        },
    )
    write_completion_marker(
        output_root,
        EVALUATION_REQUIRED_FILES,
        source_state=source_state,
    )
    print(summary.sort_values(["dataset", "beta", "R2"]).to_string(index=False))
    if gates:
        print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
