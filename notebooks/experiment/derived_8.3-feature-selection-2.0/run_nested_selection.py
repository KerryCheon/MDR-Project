"""Run nested inner-ranking/outer-time feature selection for the 2.0 rerun.

The original ``artifacts/final`` tree remains immutable.  This runner writes to
``artifacts/nested`` and never reads the already-consumed 2023-2025 test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from artifact_state import (
    atomic_write_csv,
    atomic_write_json,
    invalidate_completion,
    sha256_file,
    write_completion_marker,
)
from data_loading import resolve_router_config, router_provenance
from runtime import add_runtime_arguments, validate_workers


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "nested_config.yaml"
TARGET = "soil_moisture_5cm"
DATASETS = ("derived_8.0", "derived_8.3")
NESTED_REQUIRED_FILES = (
    "selected_features.json",
    "inner_selection.json",
    "outer_selection.json",
    "outer_fold_metrics.csv",
)


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["regime_delta"]["router"] = resolve_router_config(
        PROJECT_ROOT,
        config["regime_delta"]["router"],
    )
    return config


def _load_inner_outer(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    split_dir = PROJECT_ROOT / "data" / "splits" / dataset
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    train = pd.read_csv(train_path)
    outer = pd.read_csv(val_path)
    hashes = {
        "train": sha256_file(train_path),
        "val": sha256_file(val_path),
    }
    return train, outer, hashes


def _prepare_xy(
    frame: pd.DataFrame,
    station_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split

    X, y, _, _ = preprocess_split(
        frame,
        TARGET,
        drop_cols=[station_col, time_col],
    )
    context = frame.loc[X.index, [station_col, time_col]].copy()
    return X, y, context


def _router_labels(
    train: pd.DataFrame,
    outer: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    columns = list(config["columns"])
    for label, frame in (("inner training", train), ("outer", outer)):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing {label} router features: {missing[:10]}")
    train_values = (
        train[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    outer_values = (
        outer[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    means = train_values.mean()
    if means.isna().any():
        missing_means = means.index[means.isna()].tolist()
        raise ValueError(f"Router features contain no finite values: {missing_means}")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values.fillna(means))
    outer_scaled = scaler.transform(outer_values.fillna(means))
    model = KMeans(
        n_clusters=int(config["n_clusters"]),
        n_init=int(config["n_init"]),
        random_state=int(config["random_state"]),
    )
    train_labels = model.fit_predict(train_scaled)
    outer_labels = model.predict(outer_scaled)
    metadata = router_provenance(
        config,
        fit_scope="inner_training_split",
        frozen_for_evaluation=True,
    )
    metadata.update(
        {
            "fit_row_count": len(train),
            "means": {key: float(value) for key, value in means.items()},
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "centers": model.cluster_centers_.tolist(),
        }
    )
    return train_labels, outer_labels, metadata


def _nested_configs(
    config: dict, smoke: bool, device: str, workers: int
) -> tuple[dict, dict]:
    inner = json.loads(json.dumps(config["inner_selection"]))
    outer = json.loads(json.dumps(config["outer_selection"]))
    inner["model_params"]["device"] = device
    outer["model_params"]["device"] = device
    inner["parallel_workers"] = workers
    outer["parallel_workers"] = workers
    if smoke:
        inner.update(
            {
                "candidate_sizes": [12, 8],
                "n_station_folds": 2,
                "max_validation_years": 1,
                "min_train_rows": 40,
                "min_validation_rows": 8,
                "train_weight_betas": [0.0],
            }
        )
        outer.update(
            {
                "n_station_folds": 2,
                "max_validation_years": 1,
                "min_train_rows": 40,
                "min_validation_rows": 8,
                "train_weight_betas": [0.0],
            }
        )
        inner["model_params"].update({"n_estimators": 20, "max_depth": 3})
        outer["model_params"].update({"n_estimators": 20, "max_depth": 3})
    return inner, outer


def _write_nested_artifact(
    out_dir: Path,
    *,
    dataset: str,
    scope: str,
    inner_result: dict,
    outer_result: dict,
    split_hashes: dict,
    device: str,
    write_completion: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    invalidate_completion(out_dir)
    payload = {
        "version": "2.0-nested",
        "dataset": dataset,
        "scope": scope,
        "created": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "split_hashes": split_hashes,
        "n_features": len(outer_result["selected"]),
        "features": list(outer_result["selected"]),
        "inner_stopping_reason": inner_result["stopping_reason"],
        "outer_stopping_reason": outer_result["stopping_reason"],
        "inner_candidate_sizes": [
            candidate["n_features"] for candidate in inner_result["selection_path"]
        ],
        "outer_candidate_summaries": outer_result["candidate_summaries"],
        "outer_folds": outer_result["folds"],
        "inner_config": inner_result["config"],
        "outer_config": outer_result["config"],
    }
    atomic_write_json(out_dir / "selected_features.json", payload)
    atomic_write_json(out_dir / "inner_selection.json", inner_result)
    atomic_write_json(out_dir / "outer_selection.json", outer_result)

    rows = []
    for candidate in outer_result["candidate_summaries"]:
        for fold in candidate["fold_metrics"]:
            rows.append(
                {
                    "n_features": candidate["n_features"],
                    "candidate_mean_nrmse": candidate["mean_nrmse"],
                    "candidate_ucb": candidate["upper_confidence_bound"],
                    **fold,
                }
            )
    atomic_write_csv(pd.DataFrame(rows), out_dir / "outer_fold_metrics.csv")
    if write_completion:
        write_completion_marker(
            out_dir,
            NESTED_REQUIRED_FILES,
        )


def _run_nested(
    X_inner: pd.DataFrame,
    y_inner: pd.Series,
    context_inner: pd.DataFrame,
    X_outer: pd.DataFrame,
    y_outer: pd.Series,
    context_outer: pd.DataFrame,
    inner_config: dict,
    outer_config: dict,
    required_features: list[str] | None = None,
) -> tuple[dict, dict]:
    from Modeling.Src.soilmoist_fl.Selectors.grouped_oof import (
        evaluate_forward_station_time_candidates,
        select_grouped_oof,
    )

    inner_result = select_grouped_oof(
        X_inner,
        y_inner,
        context_inner,
        config=inner_config,
        required_features=required_features,
    )
    candidates = [
        candidate["features"] for candidate in inner_result["selection_path"]
    ]
    if required_features:
        candidates.append(required_features)
    outer_result = evaluate_forward_station_time_candidates(
        X_inner,
        y_inner,
        context_inner,
        X_outer,
        y_outer,
        context_outer,
        candidates,
        config=outer_config,
        required_features=required_features,
    )
    return inner_result, outer_result


def run_dataset(
    dataset: str,
    config: dict,
    smoke: bool,
    device: str,
    workers: int,
) -> dict:
    train, outer, hashes = _load_inner_outer(dataset)
    station_col = config["data"]["station_col"]
    time_col = config["data"]["time_col"]
    X_inner, y_inner, context_inner = _prepare_xy(train, station_col, time_col)
    X_outer, y_outer, context_outer = _prepare_xy(outer, station_col, time_col)
    inner_config, outer_config = _nested_configs(config, smoke, device, workers)
    inner_result, outer_result = _run_nested(
        X_inner,
        y_inner,
        context_inner,
        X_outer,
        y_outer,
        context_outer,
        inner_config,
        outer_config,
    )

    root = EXP_DIR / "artifacts" / ("nested_smoke" if smoke else "nested")
    _write_nested_artifact(
        root / dataset / "global",
        dataset=dataset,
        scope="2.0_global",
        inner_result=inner_result,
        outer_result=outer_result,
        split_hashes=hashes,
        device=device,
    )
    shared = list(outer_result["selected"])
    summary = {
        "dataset": dataset,
        "global_n_features": len(shared),
        "outer_stopping_reason": outer_result["stopping_reason"],
        "regimes": {},
    }
    if dataset != "derived_8.3":
        return summary

    inner_labels, outer_labels, router_metadata = _router_labels(
        train,
        outer,
        config["regime_delta"]["router"],
    )
    atomic_write_json(root / dataset / "router.json", router_metadata)
    summary["router"] = {
        "kind": router_metadata["kind"],
        "feature_source": router_metadata["feature_source"],
        "feature_count": router_metadata["feature_count"],
        "fit_scope": router_metadata["fit_scope"],
        "frozen_for_evaluation": router_metadata["frozen_for_evaluation"],
    }

    additions = list(config["regime_delta"]["additions"])
    if smoke:
        additions = [0, 3]
    delta_inner = json.loads(json.dumps(inner_config))
    delta_inner["candidate_sizes"] = [
        min(X_inner.shape[1], len(shared) + int(addition))
        for addition in additions
    ]

    for regime in sorted(np.unique(inner_labels)):
        inner_mask = inner_labels == regime
        outer_mask = outer_labels == regime
        regime_inner, regime_outer = _run_nested(
            X_inner.loc[inner_mask].reset_index(drop=True),
            y_inner.loc[inner_mask].reset_index(drop=True),
            context_inner.loc[inner_mask].reset_index(drop=True),
            X_outer.loc[outer_mask].reset_index(drop=True),
            y_outer.loc[outer_mask].reset_index(drop=True),
            context_outer.loc[outer_mask].reset_index(drop=True),
            delta_inner,
            outer_config,
            required_features=shared,
        )
        _write_nested_artifact(
            root / dataset / f"regime_{int(regime)}",
            dataset=dataset,
            scope=f"2.0_clustering_v0_full_k2_regime_{int(regime)}",
            inner_result=regime_inner,
            outer_result=regime_outer,
            split_hashes=hashes,
            device=device,
        )
        summary["regimes"][str(int(regime))] = {
            "n_features": len(regime_outer["selected"]),
            "n_delta": len(regime_outer["selected"]) - len(shared),
            "outer_stopping_reason": regime_outer["stopping_reason"],
        }
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="Dataset to run; repeat for both. Default: both.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced configuration under artifacts/nested_smoke.",
    )
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)

    config = _load_config()
    workers = validate_workers(args.workers)
    datasets = args.dataset or list(DATASETS)
    summaries = [
        run_dataset(dataset, config, args.smoke, args.device, workers)
        for dataset in datasets
    ]
    root = EXP_DIR / "artifacts" / ("nested_smoke" if args.smoke else "nested")
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        root / "selection_summary.json",
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "workers": workers,
            "smoke": args.smoke,
            "datasets": summaries,
        },
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
