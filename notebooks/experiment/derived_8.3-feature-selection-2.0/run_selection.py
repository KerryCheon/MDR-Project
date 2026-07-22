"""Run grouped OOF global and shared-plus-delta feature selection for 2.0."""

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
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = EXP_DIR / "config.yaml"
TARGET = "soil_moisture_5cm"
DATASETS = ("derived_8.0", "derived_8.3")
SELECTION_REQUIRED_FILES = (
    "selected_features.json",
    "importance_detail.json",
    "fold_metrics.csv",
)


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["regime_delta"]["router"] = resolve_router_config(
        PROJECT_ROOT,
        config["regime_delta"]["router"],
    )
    return config


def _load_development_data(dataset: str) -> tuple[pd.DataFrame, dict]:
    split_dir = PROJECT_ROOT / "data" / "splits" / dataset
    paths = {name: split_dir / f"{name}.csv" for name in ("train", "val")}
    frames = [
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["val"]),
    ]
    development = pd.concat(frames, ignore_index=True)
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return development, hashes


def _selection_config(config: dict, smoke: bool, device: str, workers: int) -> dict:
    stage = json.loads(json.dumps(config["selection"]["stages"][0]))
    stage.pop("kind", None)
    stage["model_params"]["device"] = device
    stage["parallel_workers"] = workers
    if smoke:
        stage.update(
            {
                "candidate_sizes": [12, 8],
                "n_station_folds": 2,
                "max_validation_years": 1,
                "permutation_repeats": 1,
            }
        )
        stage["model_params"].update({"n_estimators": 20, "max_depth": 3})
        stage["train_weight_betas"] = [0.0]
    return stage


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


def _write_selection(
    out_dir: Path,
    *,
    dataset: str,
    scope: str,
    result: dict,
    split_hashes: dict,
    config: dict,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    invalidate_completion(out_dir)
    payload = {
        "version": "2.0",
        "dataset": dataset,
        "scope": scope,
        "created": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "split_hashes": split_hashes,
        "n_features": len(result["selected"]),
        "features": list(result["selected"]),
        "stopping_reason": result["stopping_reason"],
        "scores": result.get("scores", {}),
        "selection_path": result.get("selection_path", []),
        "baseline_required": result.get("baseline_required"),
        "folds": result.get("folds", []),
        "config": config,
    }
    atomic_write_json(out_dir / "selected_features.json", payload)
    atomic_write_json(
        out_dir / "importance_detail.json",
        result.get("importance_detail", {}),
    )
    fold_rows = []
    for candidate in result.get("selection_path", []):
        for row in candidate.get("fold_metrics", []):
            fold_rows.append(
                {
                    "n_features": candidate["n_features"],
                    "candidate_ucb": candidate["upper_confidence_bound"],
                    **row,
                }
            )
    atomic_write_csv(pd.DataFrame(fold_rows), out_dir / "fold_metrics.csv")
    write_completion_marker(
        out_dir,
        SELECTION_REQUIRED_FILES,
    )


def _fit_router_labels(
    frame: pd.DataFrame,
    router_config: dict,
) -> tuple[np.ndarray, dict]:
    columns = list(router_config["columns"])
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing router features: {missing[:10]}")
    values = (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    means = values.mean()
    if means.isna().any():
        missing_means = means.index[means.isna()].tolist()
        raise ValueError(f"Router features contain no finite values: {missing_means}")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.fillna(means))
    model = KMeans(
        n_clusters=int(router_config["n_clusters"]),
        n_init=int(router_config["n_init"]),
        random_state=int(router_config["random_state"]),
    )
    labels = model.fit_predict(scaled)
    provenance = router_provenance(
        router_config,
        fit_scope="combined_development_pool",
        frozen_for_evaluation=False,
    )
    provenance.update(
        {
            "fit_row_count": len(frame),
            "means": {key: float(value) for key, value in means.items()},
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "centers": model.cluster_centers_.tolist(),
        }
    )
    return labels, provenance


def run_dataset(
    dataset: str,
    config: dict,
    smoke: bool,
    device: str,
    workers: int,
) -> dict:
    from Modeling.Src.soilmoist_fl.Selectors.grouped_oof import select_grouped_oof

    development, hashes = _load_development_data(dataset)
    station_col = config["data"]["station_col"]
    time_col = config["data"]["time_col"]
    X, y, context = _prepare_xy(development, station_col, time_col)
    grouped_config = _selection_config(config, smoke, device, workers)
    result = select_grouped_oof(
        X,
        y,
        context,
        config=grouped_config,
    )

    root = EXP_DIR / "artifacts" / ("smoke" if smoke else "final")
    global_dir = root / dataset / "global"
    _write_selection(
        global_dir,
        dataset=dataset,
        scope="2.0_global",
        result=result,
        split_hashes=hashes,
        config=grouped_config,
        device=device,
    )

    summary = {
        "dataset": dataset,
        "global_n_features": len(result["selected"]),
        "global_stopping_reason": result["stopping_reason"],
        "regimes": {},
    }
    if dataset != "derived_8.3":
        return summary

    delta_config = json.loads(json.dumps(config["regime_delta"]["selection"]))
    delta_config["model_params"]["device"] = device
    delta_config["parallel_workers"] = workers
    additions = list(config["regime_delta"]["additions"])
    if smoke:
        delta_config.update(
            {
                "n_station_folds": 2,
                "max_validation_years": 1,
                "min_train_rows": 40,
                "min_validation_rows": 8,
            }
        )
        delta_config["model_params"].update({"n_estimators": 20, "max_depth": 3})
        delta_config["train_weight_betas"] = [0.0]
        additions = [0, 3]

    shared = list(result["selected"])
    delta_config["candidate_sizes"] = [
        min(X.shape[1], len(shared) + int(addition))
        for addition in additions
    ]
    labels, provenance = _fit_router_labels(
        development,
        config["regime_delta"]["router"],
    )
    atomic_write_json(root / dataset / "router_provenance.json", provenance)
    summary["router"] = {
        "kind": provenance["kind"],
        "feature_source": provenance["feature_source"],
        "feature_count": provenance["feature_count"],
        "fit_scope": provenance["fit_scope"],
        "frozen_for_evaluation": provenance["frozen_for_evaluation"],
    }
    for regime in sorted(np.unique(labels)):
        mask = labels == regime
        regime_result = select_grouped_oof(
            X.loc[mask].reset_index(drop=True),
            y.loc[mask].reset_index(drop=True),
            context.loc[mask].reset_index(drop=True),
            config=delta_config,
            required_features=shared,
        )
        regime_dir = root / dataset / f"regime_{int(regime)}"
        _write_selection(
            regime_dir,
            dataset=dataset,
            scope=f"2.0_clustering_v0_full_k2_regime_{int(regime)}",
            result=regime_result,
            split_hashes=hashes,
            config=delta_config,
            device=device,
        )
        summary["regimes"][str(int(regime))] = {
            "n_features": len(regime_result["selected"]),
            "n_delta": len(regime_result["selected"]) - len(shared),
            "stopping_reason": regime_result["stopping_reason"],
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
        help="Use a small, isolated configuration and write under artifacts/smoke.",
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
    root = EXP_DIR / "artifacts" / ("smoke" if args.smoke else "final")
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
