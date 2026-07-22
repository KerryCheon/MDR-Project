"""Evaluate frozen 2.0 feature sets with retrospective-test safeguards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
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
    invalidate_completion,
    write_completion_marker,
)
from data_loading import load_v0_features, resolve_router_config, router_provenance
from runtime import add_runtime_arguments, validate_workers
from split_provenance import read_hashed_csv


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
        config = yaml.safe_load(stream)
    config["regime_delta"]["router"] = resolve_router_config(
        PROJECT_ROOT,
        config["regime_delta"]["router"],
    )
    return config


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


def _fit_router(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    router_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    columns = list(router_config["columns"])
    for label, frame in (("fit", train), ("score", validation)):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing {label} router features: {missing[:10]}")
    train_values = (
        train[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    validation_values = (
        validation[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    means = train_values.mean()
    if means.isna().any():
        missing_means = means.index[means.isna()].tolist()
        raise ValueError(f"Router features contain no finite values: {missing_means}")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values.fillna(means))
    validation_scaled = scaler.transform(validation_values.fillna(means))
    model = KMeans(
        n_clusters=int(router_config["n_clusters"]),
        n_init=int(router_config["n_init"]),
        random_state=int(router_config["random_state"]),
    )
    return model.fit_predict(train_scaled), model.predict(validation_scaled)


def _frozen_router(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    router_artifact: dict,
    router_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the train-only router used to define regime feature semantics."""
    columns = list(router_artifact["columns"])
    expected = router_provenance(
        router_config,
        fit_scope="inner_training_split",
        frozen_for_evaluation=True,
    )
    for key in (
        "kind",
        "feature_source",
        "feature_source_sha256",
        "columns",
        "feature_count",
        "imputation",
        "scaler",
        "n_clusters",
        "n_init",
        "random_state",
        "fit_scope",
        "frozen_for_evaluation",
    ):
        if router_artifact.get(key) != expected[key]:
            raise ValueError(f"frozen router provenance mismatch for {key}")
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
        train_labels, validation_labels = _fit_router(
            train,
            validation,
            router_config,
        )
    else:
        train_labels, validation_labels = _frozen_router(
            train,
            validation,
            router_artifact,
            router_config,
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
        help=(
            "Disabled for this rerun because derived 8.3 test data were already "
            "consumed by derived_8.3-eval-1.0."
        ),
    )
    parser.add_argument(
        "--retrospective-test",
        action="store_true",
        help=(
            "Evaluate an artifact set on the already-consumed test split; "
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
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)

    if args.confirm_final:
        parser.error(
            "--confirm-final is unavailable: all derived 8.3 test evaluations "
            "are retrospective"
        )
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    config = _load_config()
    device = args.device
    parallel_workers = validate_workers(args.workers)
    serial_revision_retro = bool(
        args.retrospective_test
        and args.artifact_set
        in {"nested", "nested_locked_outer", "crossed_candidates_locked_outer"}
    )
    task_workers = 1 if serial_revision_retro else parallel_workers
    params = dict(config["evaluation"]["xgb_params"])
    params["device"] = device
    artifact_root = EXP_DIR / "artifacts" / args.artifact_set
    uses_test = args.retrospective_test
    if args.retrospective_test:
        output_name = "retrospective_test_eval"
    else:
        output_name = "validation_eval"
    output_root = artifact_root / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    invalidate_completion(output_root)

    overall_rows = []
    year_rows = []
    station_rows = []
    split_hashes = {}
    for dataset in ("derived_8.0", "derived_8.3"):
        split_dir = PROJECT_ROOT / "data/splits" / dataset
        train, train_hash = read_hashed_csv(split_dir / "train.csv")
        val, val_hash = read_hashed_csv(split_dir / "val.csv")
        dataset_hashes = {"train": train_hash, "val": val_hash}
        if uses_test:
            test, test_hash = read_hashed_csv(split_dir / "test.csv")
            dataset_hashes["test"] = test_hash
            fit_frame = pd.concat([train, val], ignore_index=True)
            score_frame = test
        else:
            fit_frame = train
            score_frame = val
        split_hashes[dataset] = dataset_hashes

        selected_path = artifact_root / dataset / "global" / "selected_features.json"
        if not artifact_is_complete(selected_path.parent, [selected_path.name]):
            raise FileNotFoundError(
                f"Run selection first; incomplete artifact at {selected_path.parent}"
            )
        feature_sets = {"2.0_global": _load_features(selected_path)}
        if dataset == "derived_8.0":
            feature_sets["hand_mdr_v25"] = list(HAND_MDR_V25)
        else:
            feature_sets["V0"] = load_v0_features(PROJECT_ROOT)

        regime_features = None
        router_artifact = None
        regime_paths = [
            artifact_root / dataset / f"regime_{regime}/selected_features.json"
            for regime in (0, 1)
        ]
        if dataset == "derived_8.3" and all(
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
                        "2.0_clustering_v0_full_k2_shared_plus_delta",
                        regime_features,
                        router_artifact,
                    )
                )
                if router_artifact is not None:
                    shared_features = {
                        regime: feature_sets["2.0_global"]
                        for regime in regime_features
                    }
                    tasks.extend(
                        [
                            (
                                "moe",
                                float(beta),
                                "2.0_clustering_v0_full_k2_frozen_shared_only",
                                shared_features,
                                router_artifact,
                            ),
                            (
                                "moe",
                                float(beta),
                                "2.0_clustering_v0_full_k2_refit_shared_plus_delta",
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

        if task_workers == 1:
            task_results = (_run_eval_task(task) for task in tasks)
        else:
            executor = ThreadPoolExecutor(max_workers=task_workers)
            task_results = executor.map(_run_eval_task, tasks)
        try:
            for task_index, (overall, years, stations) in enumerate(task_results, start=1):
                overall_rows.append(overall)
                year_rows.extend(years)
                station_rows.extend(stations)
                print(
                    f"{dataset}: completed evaluation task {task_index}/{len(tasks)}",
                    flush=True,
                )
        finally:
            if task_workers != 1:
                executor.shutdown(wait=True)

    summary = pd.DataFrame(overall_rows)
    atomic_write_csv(summary, output_root / "metrics_summary.csv")
    atomic_write_csv(pd.DataFrame(year_rows), output_root / "metrics_by_year.csv")
    atomic_write_csv(
        pd.DataFrame(station_rows),
        output_root / "metrics_by_station.csv",
    )

    gates = {}

    atomic_write_json(
        output_root / "evaluation_manifest.json",
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "parallel_workers": parallel_workers,
            "task_workers": task_workers,
            "artifact_set": args.artifact_set,
            "final_test": False,
            "retrospective_test": args.retrospective_test,
            "split_hashes": split_hashes,
            "router_config": config["regime_delta"]["router"],
            "unbiased_sota_eligible": False,
            "ineligibility_reason": (
                "derived_8.3-eval-1.0 already consumed the 2023-2025 test split"
            ),
            "gates": gates,
        },
    )
    write_completion_marker(
        output_root,
        EVALUATION_REQUIRED_FILES,
    )
    print(summary.sort_values(["dataset", "beta", "R2"]).to_string(index=False))
    if gates:
        print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
