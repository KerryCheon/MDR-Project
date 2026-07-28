"""
Data loading and hybrid CTX / Head Hidden / Head Pre-ReLU feature concatenation module for derived_8.4-hybrid-lstm-1.3.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HybridExperimentData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    trainval: pd.DataFrame
    feature_columns: list[str]
    source_order: list[str]
    target: str
    v0_features: list[str]
    shared_backbone_54: list[str]
    ctx_feature_cols: list[str]
    hybrid_backbone_214: list[str]
    ctx_80_feature_cols: list[str]
    hybrid_backbone_134: list[str]
    head_pre_relu_cols: list[str]
    hybrid_backbone_134_pre: list[str]
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
        raise ValueError("Expected a date column in split.")
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


def load_hybrid_experiment_data(project_root: Path, exp_root: Path, config: dict[str, Any]) -> HybridExperimentData:
    data_cfg = config["data"]
    target = str(data_cfg["target"])

    frames = {
        name: _prepare_frame(pd.read_csv(project_root / path), target)
        for name, path in data_cfg["splits"].items()
    }
    train, val, test = frames["train"], frames["val"], frames["test"]

    artifacts_dir = exp_root / config["artifacts"]["directory"]

    # Load frozen CTX vectors (160 dimensions)
    ctx_train = np.load(artifacts_dir / "ctx_train.npy")
    ctx_val   = np.load(artifacts_dir / "ctx_val.npy")
    ctx_test  = np.load(artifacts_dir / "ctx_test.npy")

    num_ctx = ctx_train.shape[1]
    ctx_cols = [f"ctx_{i}" for i in range(num_ctx)]

    df_ctx_tr = pd.DataFrame(ctx_train, columns=ctx_cols, index=train.index)
    df_ctx_va = pd.DataFrame(ctx_val,   columns=ctx_cols, index=val.index)
    df_ctx_te = pd.DataFrame(ctx_test,  columns=ctx_cols, index=test.index)

    train = pd.concat([train, df_ctx_tr], axis=1)
    val   = pd.concat([val,   df_ctx_va], axis=1)
    test  = pd.concat([test,  df_ctx_te], axis=1)

    # Load frozen head hidden vectors (80 dimensions, after head Linear→ReLU)
    hh_train = np.load(artifacts_dir / "head_hidden_train.npy")
    hh_val   = np.load(artifacts_dir / "head_hidden_val.npy")
    hh_test  = np.load(artifacts_dir / "head_hidden_test.npy")

    num_hh = hh_train.shape[1]
    hh_cols = [f"hh_{i}" for i in range(num_hh)]

    df_hh_tr = pd.DataFrame(hh_train, columns=hh_cols, index=train.index)
    df_hh_va = pd.DataFrame(hh_val,   columns=hh_cols, index=val.index)
    df_hh_te = pd.DataFrame(hh_test,  columns=hh_cols, index=test.index)

    train = pd.concat([train, df_hh_tr], axis=1)
    val   = pd.concat([val,   df_hh_va], axis=1)
    test  = pd.concat([test,  df_hh_te], axis=1)

    # Load frozen head pre-ReLU vectors (80 dimensions, after head Linear BEFORE ReLU)
    hp_train = np.load(artifacts_dir / "head_pre_relu_train.npy")
    hp_val   = np.load(artifacts_dir / "head_pre_relu_val.npy")
    hp_test  = np.load(artifacts_dir / "head_pre_relu_test.npy")

    num_hp = hp_train.shape[1]
    hp_cols = [f"hp_{i}" for i in range(num_hp)]

    df_hp_tr = pd.DataFrame(hp_train, columns=hp_cols, index=train.index)
    df_hp_va = pd.DataFrame(hp_val,   columns=hp_cols, index=val.index)
    df_hp_te = pd.DataFrame(hp_test,  columns=hp_cols, index=test.index)

    train = pd.concat([train, df_hp_tr], axis=1)
    val   = pd.concat([val,   df_hp_va], axis=1)
    test  = pd.concat([test,  df_hp_te], axis=1)

    trainval = pd.concat([train, val], axis=0, ignore_index=True)

    excluded = {target, "station_id", "date", "month", "year"}
    feature_columns = [col for col in train.columns if col not in excluded]
    source_order = list(feature_columns)

    v0_features = _load_v0_features(project_root / Path(data_cfg["metadata_path"]))
    shared_backbone_54 = list(config["shared_backbone_54"])
    hybrid_backbone_214 = shared_backbone_54 + ctx_cols
    hybrid_backbone_134 = shared_backbone_54 + hh_cols
    hybrid_backbone_134_pre = shared_backbone_54 + hp_cols

    candidate_pool_path = project_root / Path(config["candidate_pool_file"])
    if candidate_pool_path.exists():
        df_pool = pd.read_csv(candidate_pool_path)
        candidate_pool = list(df_pool["feature"].dropna())
    else:
        candidate_pool = list(feature_columns)

    return HybridExperimentData(
        train=train,
        val=val,
        test=test,
        trainval=trainval,
        feature_columns=feature_columns,
        source_order=source_order,
        target=target,
        v0_features=v0_features,
        shared_backbone_54=shared_backbone_54,
        ctx_feature_cols=ctx_cols,
        hybrid_backbone_214=hybrid_backbone_214,
        ctx_80_feature_cols=hh_cols,
        hybrid_backbone_134=hybrid_backbone_134,
        head_pre_relu_cols=hp_cols,
        hybrid_backbone_134_pre=hybrid_backbone_134_pre,
        candidate_pool=candidate_pool,
    )
