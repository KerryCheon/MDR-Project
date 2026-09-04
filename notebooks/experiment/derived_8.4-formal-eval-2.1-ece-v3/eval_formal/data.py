"""Data loading for derived_8.4-formal-eval-2.1-ece-v3.

Loads:
1. derived_8.4 splits (7 Washington stations: train 2017-2020, val 2021-2022, test 2023-2025).
2. derived_8.4_ece_v3 splits (5 in-situ ECE stations in WA: ECE_BBG_Main_St, ECE_BBG_Lost_Meadow,
   ECE_Renton_Home, ECE_Renton_Garden_North, ECE_Renton_Garden_Shed; 150 rows across 2026-07-20 to 2026-08-19;
   completely unseen during training).
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
    ece_train: pd.DataFrame
    ece_val: pd.DataFrame
    ece_test: pd.DataFrame
    ece_all: pd.DataFrame
    ece_stations: list[str]
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
    if output.empty:
        return output
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
        name: _prepare_frame(pd.read_csv(project_root / path, low_memory=False), target)
        for name, path in data_cfg["splits"].items()
    }
    train = frames_wa["train"]
    val = frames_wa["val"]
    test = frames_wa["test"]
    trainval = pd.concat([train, val], axis=0, ignore_index=True)

    # 2. In-situ ECE split (derived_8.4_ece_v3)
    ece_cfg = data_cfg.get("spatial_ece", {})
    if "splits" in ece_cfg:
        frames_ece = {}
        for name, path in ece_cfg["splits"].items():
            fpath = project_root / path
            if fpath.exists():
                df_raw = pd.read_csv(fpath, low_memory=False)
                frames_ece[name] = _prepare_frame(df_raw, target)
            else:
                frames_ece[name] = pd.DataFrame()
        ece_train = frames_ece.get("train", pd.DataFrame())
        ece_val = frames_ece.get("val", pd.DataFrame())
        ece_test = frames_ece.get("test", pd.DataFrame())
        non_empty = [df for df in [ece_train, ece_val, ece_test] if not df.empty]
        ece_all = pd.concat(non_empty, axis=0, ignore_index=True) if non_empty else pd.DataFrame()
        if not ece_all.empty:
            excluded_ece = {target, "station_id", "date", "month", "year"}
            for col in ece_all.columns:
                if col not in excluded_ece:
                    ece_all[col] = pd.to_numeric(ece_all[col], errors="coerce")
    else:
        ece_train = pd.DataFrame()
        ece_val = pd.DataFrame()
        ece_test = pd.DataFrame()
        ece_all = pd.DataFrame()

    ece_stations = sorted(ece_all["station_id"].unique()) if not ece_all.empty else []

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
        ece_train=ece_train,
        ece_val=ece_val,
        ece_test=ece_test,
        ece_all=ece_all,
        ece_stations=ece_stations,
        feature_columns=feature_columns,
        source_order=source_order,
        target=target,
        v0_features=v0_features,
        shared_backbone_54=shared_backbone,
        candidate_pool=candidate_pool,
    )
