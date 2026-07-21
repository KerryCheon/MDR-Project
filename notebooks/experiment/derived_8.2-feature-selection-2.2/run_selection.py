"""Run grouped OOF global and shared-plus-delta feature selection for 2.2."""

from __future__ import annotations

import argparse
import json
import os
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
    capture_source_state,
    invalidate_completion,
    sha256_file,
    write_completion_marker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.yaml"
TARGET = "soil_moisture_5cm"
DATASETS = ("derived_8.0", "derived_8.2")
SELECTION_REQUIRED_FILES = (
    "selected_features.json",
    "importance_detail.json",
    "fold_metrics.csv",
)


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


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


def _probe_device() -> str:
    import xgboost as xgb

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


def _selection_config(config: dict, smoke: bool, device: str) -> dict:
    stage = json.loads(json.dumps(config["selection"]["stages"][0]))
    stage.pop("kind", None)
    stage["model_params"]["device"] = device
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
    source_state: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    invalidate_completion(out_dir)
    payload = {
        "version": "2.2",
        "dataset": dataset,
        "scope": scope,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": source_state["git_commit"],
        "source_state": source_state,
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
        source_state=source_state,
    )


def _dynamic_labels(frame: pd.DataFrame, router_config: dict) -> np.ndarray:
    columns = list(router_config["columns"])
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    means = values.mean()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.fillna(means))
    model = KMeans(
        n_clusters=int(router_config["n_clusters"]),
        n_init=int(router_config["n_init"]),
        random_state=42,
    )
    return model.fit_predict(scaled)


def run_dataset(
    dataset: str,
    config: dict,
    smoke: bool,
    device: str,
    source_state: dict,
) -> dict:
    from Modeling.Src.soilmoist_fl.Selectors.grouped_oof import select_grouped_oof

    development, hashes = _load_development_data(dataset)
    station_col = config["data"]["station_col"]
    time_col = config["data"]["time_col"]
    X, y, context = _prepare_xy(development, station_col, time_col)
    grouped_config = _selection_config(config, smoke, device)
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
        scope="global",
        result=result,
        split_hashes=hashes,
        config=grouped_config,
        device=device,
        source_state=source_state,
    )

    summary = {
        "dataset": dataset,
        "global_n_features": len(result["selected"]),
        "global_stopping_reason": result["stopping_reason"],
        "regimes": {},
    }
    if dataset != "derived_8.2":
        return summary

    delta_config = json.loads(json.dumps(config["regime_delta"]["selection"]))
    delta_config["model_params"]["device"] = device
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
    labels = _dynamic_labels(development, config["regime_delta"]["router"])
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
            scope=f"clustering_dynamic_k2_regime_{int(regime)}",
            result=regime_result,
            split_hashes=hashes,
            config=delta_config,
            device=device,
            source_state=source_state,
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
    args = parser.parse_args(argv)

    config = _load_config()
    device = _probe_device()
    source_state = capture_source_state(PROJECT_ROOT, EXP_DIR)
    datasets = args.dataset or list(DATASETS)
    summaries = [
        run_dataset(dataset, config, args.smoke, device, source_state)
        for dataset in datasets
    ]
    root = EXP_DIR / "artifacts" / ("smoke" if args.smoke else "final")
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        root / "selection_summary.json",
        {
            "created": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "smoke": args.smoke,
            "source_state": source_state,
            "datasets": summaries,
        },
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
