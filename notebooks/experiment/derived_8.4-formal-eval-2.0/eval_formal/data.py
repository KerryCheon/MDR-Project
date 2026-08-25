"""Data loading for derived_8.4-formal-eval-2.0.

Loads:
1. derived_8.4 splits (7 Washington stations: train 2017-2020, val 2021-2022, test 2023-2025).
2. derived_8.4-oos splits (10 Out-of-State stations across OR, ID, CA, CO, WY, MT:
   25,176 rows across 2017-2025; completely unseen during training).
Exposes the shared 54-feature backbone, the V0-50 baseline features, and candidate pool.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ExperimentData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    trainval: pd.DataFrame
    oos_train: pd.DataFrame
    oos_val: pd.DataFrame
    oos_test: pd.DataFrame
    oos_all: pd.DataFrame
    oos_stations: list[str]
    feature_columns: list[str]
    source_order: list[str]
    target: str
    v0_features: list[str]
    shared_backbone_54: list[str]
    candidate_pool: list[str]


def _load_v0_features(metadata_path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("derived_84_metadata", metadata_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metadata module: {metadata_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.OVERALL_SELECTED_FEATURES_V0)


def _prepare_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    output = frame.copy()
    if "date" not in output.columns:
        raise ValueError("Expected a date column in every split.")
    output["date"] = pd.to_datetime(output["date"], errors="raise")
    output["month"] = output["date"].dt.month.astype(int)
    output["year"] = output["date"].dt.year.astype(int)
    if target not in output.columns:
        raise ValueError(f"Target column not found: {target}")
    if output[target].isna().any():
        raise ValueError("Target contains missing values.")

    excluded = {target, "station_id", "date", "month", "year"}
    for column in output.columns:
        if column in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.replace([float("inf"), float("-inf")], float("nan"))
    return output


def load_experiment_data(project_root: Path, config: dict[str, Any]) -> ExperimentData:
    data_cfg = config["data"]
    target = str(data_cfg["target"])

    # 1. Washington State split (derived_8.4)
    frames_wa = {
        name: _prepare_frame(pd.read_csv(project_root / path), target)
        for name, path in data_cfg["splits"].items()
    }
    train = frames_wa["train"]
    val = frames_wa["val"]
    test = frames_wa["test"]
    trainval = pd.concat([train, val], axis=0, ignore_index=True)

    # 2. Out-of-State split (derived_8.4-oos)
    oos_cfg = data_cfg.get("spatial_oos", {})
    if "splits" in oos_cfg:
        frames_oos = {
            name: _prepare_frame(pd.read_csv(project_root / path), target)
            for name, path in oos_cfg["splits"].items()
        }
        oos_train = frames_oos["train"]
        oos_val = frames_oos["val"]
        oos_test = frames_oos["test"]
        oos_all = pd.concat([oos_train, oos_val, oos_test], axis=0, ignore_index=True)
    else:
        # Fallback if single file or direct load
        oos_train = pd.DataFrame()
        oos_val = pd.DataFrame()
        oos_test = pd.DataFrame()
        oos_all = pd.DataFrame()

    oos_stations = sorted(oos_all["station_id"].unique()) if not oos_all.empty else []

    excluded = {target, "station_id", "date", "month", "year"}
    feature_columns = [column for column in train.columns if column not in excluded]
    source_order = list(feature_columns)

    v0_features = _load_v0_features(project_root / Path(data_cfg["metadata_path"]))
    shared_backbone = list(config["shared_backbone_54"])

    candidate_pool_path = project_root / Path(config["candidate_pool_file"])
    if candidate_pool_path.exists():
        df_pool = pd.read_csv(candidate_pool_path)
        candidate_pool = list(df_pool["feature"].dropna())
    else:
        candidate_pool = list(feature_columns)

    return ExperimentData(
        train=train,
        val=val,
        test=test,
        trainval=trainval,
        oos_train=oos_train,
        oos_val=oos_val,
        oos_test=oos_test,
        oos_all=oos_all,
        oos_stations=oos_stations,
        feature_columns=feature_columns,
        source_order=source_order,
        target=target,
        v0_features=v0_features,
        shared_backbone_54=shared_backbone,
        candidate_pool=candidate_pool,
    )
