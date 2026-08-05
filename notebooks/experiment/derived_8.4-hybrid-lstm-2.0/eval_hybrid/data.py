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
    static_features: list[str]
    static_feature_count: int
    repr_feature_cols: list[str]
    hybrid_features: list[str]

    @property
    def repr_feature_count(self) -> int:
        return len(self.repr_feature_cols)

    @property
    def total_feature_count(self) -> int:
        return len(self.hybrid_features)


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


REPR_CONFIGS = {
    "ctx": {"prefix": "ctx_", "npy_name": "ctx"},
    "hh": {"prefix": "hh_", "npy_name": "head_hidden"},
    "hp": {"prefix": "hp_", "npy_name": "head_pre_relu"},
}


def verify_static_temporal_split(
    train_df: pd.DataFrame,
    static_features: list[str],
    temporal_features: list[str],
) -> dict[str, Any]:
    """Audit the strict static/temporal feature split (derived_8.4-hybrid-lstm-2.0).

    Asserts (and returns evidence for the notebook):
      - every feature exists in the split columns,
      - static and temporal sets are disjoint,
      - every static feature is constant per station (time-invariant),
      - the temporal set is non-trivial.
    """
    missing_static = [c for c in static_features if c not in train_df.columns]
    missing_temporal = [c for c in temporal_features if c not in train_df.columns]
    if missing_static or missing_temporal:
        raise ValueError(
            f"Missing features in split: static={missing_static} temporal={missing_temporal}"
        )

    overlap = sorted(set(static_features) & set(temporal_features))
    if overlap:
        raise ValueError(f"Static/temporal split is not disjoint: {overlap}")

    # Constant per station => time-invariant (static).
    nunique = train_df.groupby("station_id")[static_features].nunique()
    constant = (nunique == 1).all()
    non_constant = [f for f in static_features if not bool(constant[f])]
    if non_constant:
        raise ValueError(f"Static features vary within a station: {non_constant}")

    evidence = {
        "n_static": len(static_features),
        "n_temporal": len(temporal_features),
        "n_overlap": len(overlap),
        "static_not_constant": non_constant,
    }
    print(
        f"[SplitCheck] static={len(static_features)} (all constant per station), "
        f"temporal={len(temporal_features)}, disjoint={not overlap}",
        flush=True,
    )
    return evidence


def load_hybrid_experiment_data(
    project_root: Path,
    exp_root: Path,
    config: dict[str, Any],
    repr_type: str = "ctx",
    hidden_size: int = 40,
) -> HybridExperimentData:
    """Load tabular splits + one raw LSTM representation (no PCA) for a given hidden size.

    Representations are read from ``<artifacts>/h{hidden_size}/`` (e.g. h40/ctx_train.npy).
    The resulting LSTM feature columns are named ``ctx_0..ctx_{2H-1}``,
    ``hh_0..hh_{H-1}``, ``hp_0..hp_{H-1}``.

    In derived_8.4-hybrid-lstm-2.0 the direct tabular feature set is the
    STATIC feature list from ``config["static_features"]`` (no temporal/rolling
    features are fed to XGBoost directly; they live in the LSTM context only).
    """
    data_cfg = config["data"]
    target = str(data_cfg["target"])

    frames = {
        name: _prepare_frame(pd.read_csv(project_root / path), target)
        for name, path in data_cfg["splits"].items()
    }
    train, val, test = frames["train"], frames["val"], frames["test"]

    artifacts_dir = exp_root / config["artifacts"]["directory"] / f"h{hidden_size}"
    rcfg = REPR_CONFIGS[repr_type]

    base_name = rcfg["npy_name"]
    train_arr = np.load(artifacts_dir / f"{base_name}_train.npy")
    val_arr   = np.load(artifacts_dir / f"{base_name}_val.npy")
    test_arr  = np.load(artifacts_dir / f"{base_name}_test.npy")

    prefix = rcfg["prefix"]
    n_dim = train_arr.shape[1]
    cols = [f"{prefix}{i}" for i in range(n_dim)]

    df_tr = pd.DataFrame(train_arr, columns=cols, index=train.index)
    df_va = pd.DataFrame(val_arr,   columns=cols, index=val.index)
    df_te = pd.DataFrame(test_arr,  columns=cols, index=test.index)

    train = pd.concat([train, df_tr], axis=1)
    val   = pd.concat([val,   df_va], axis=1)
    test  = pd.concat([test,  df_te], axis=1)

    trainval = pd.concat([train, val], axis=0, ignore_index=True)

    excluded = {target, "station_id", "date", "month", "year"}
    feature_columns = [col for col in train.columns if col not in excluded]

    v0_features = _load_v0_features(project_root / Path(data_cfg["metadata_path"]))
    static_features = list(config["static_features"])
    hybrid_features = static_features + cols

    return HybridExperimentData(
        train=train,
        val=val,
        test=test,
        trainval=trainval,
        feature_columns=feature_columns,
        source_order=list(feature_columns),
        target=target,
        v0_features=v0_features,
        static_features=static_features,
        static_feature_count=len(static_features),
        repr_feature_cols=cols,
        hybrid_features=hybrid_features,
    )
