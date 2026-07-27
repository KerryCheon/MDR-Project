"""Data loading, numeric coercion, and source-order handling."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExperimentData:
    """All split frames and feature metadata needed by the local experiment."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    trainval: pd.DataFrame
    feature_columns: list[str]
    source_order: list[str]
    target: str
    v0_features: list[str]


def _load_v0_features(metadata_path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("derived_84_metadata", metadata_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metadata module: {metadata_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.OVERALL_SELECTED_FEATURES_V0)


def _prepare_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Add reporting calendar fields and coerce model candidate columns to numeric."""
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
    output = output.replace([np.inf, -np.inf], np.nan)
    return output


def load_experiment_data(config: dict[str, Any]) -> ExperimentData:
    """Load derived_8.4 unchanged and verify split and V0 feature contracts."""
    data_cfg = config["data"]
    target = str(data_cfg["target"])
    frames = {
        name: _prepare_frame(pd.read_csv(path), target)
        for name, path in data_cfg["splits"].items()
    }
    train = frames["train"]
    val = frames["val"]
    test = frames["test"]
    trainval = pd.concat([train, val], axis=0, ignore_index=True)
    excluded = {target, "station_id", "date", "month", "year"}
    feature_columns = [column for column in train.columns if column not in excluded]
    source_order = list(feature_columns)

    for split_name, frame in frames.items():
        missing = sorted(set(feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{split_name} is missing feature columns: {missing[:10]}")

    v0_features = _load_v0_features(Path(data_cfg["metadata_path"]))
    missing_v0 = [feature for feature in v0_features if feature not in feature_columns]
    if missing_v0:
        raise ValueError(f"V0 features missing from data: {missing_v0}")
    if len(v0_features) != 50:
        raise ValueError(f"Expected 50 V0 features, got {len(v0_features)}")

    return ExperimentData(
        train=train,
        val=val,
        test=test,
        trainval=trainval,
        feature_columns=feature_columns,
        source_order=source_order,
        target=target,
        v0_features=v0_features,
    )
