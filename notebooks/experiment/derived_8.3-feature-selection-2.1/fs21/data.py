"""Leakage-safe data and feature-source resolution for development."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from .artifacts import sha256_file, stable_json_hash
from .constants import PROJECT_ROOT


def read_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def resolve_repo_path(relative: str, *, must_exist: bool = True) -> Path:
    root = PROJECT_ROOT.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes project root: {relative}")
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_literal_string_list(source: str) -> tuple[list[str], str]:
    """Load a literal list without importing mutable dataset metadata."""
    try:
        relative, constant = source.split("::", maxsplit=1)
    except ValueError as error:
        raise ValueError("feature source must be '<path>::<constant>'") from error
    path = resolve_repo_path(relative)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            targets = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if constant not in targets:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{source} is not a literal string list")
        if not value or len(value) != len(set(value)):
            raise ValueError(f"{source} is empty or contains duplicates")
        return list(value), sha256_file(path)
    raise ValueError(f"missing constant {constant} in {path}")


def ordered_feature_hash(features: list[str] | tuple[str, ...]) -> str:
    return stable_json_hash(list(features))


def _guard_development_path(path: Path) -> None:
    """Make accidental benchmark reads fail before any file is opened."""
    if path.name.casefold() == "test.csv":
        raise PermissionError("development code may not access test.csv")


def _read_stable_csv(path: Path, **kwargs) -> tuple[pd.DataFrame, str]:
    before = sha256_file(path)
    frame = pd.read_csv(path, **kwargs)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"split changed while being read: {path}")
    return frame, after


def load_development(config: Mapping[str, object]) -> tuple[pd.DataFrame, dict]:
    """Read only train and val and enforce the 2017-2022 boundary."""
    data = dict(config["data"])
    split_dir = resolve_repo_path(
        str(data["split_dir"]) + "/train.csv"
    ).parent
    allowed_names = tuple(str(name) for name in data["development_splits"])
    if allowed_names != ("train", "val"):
        raise ValueError("development_splits must be exactly [train, val]")
    frames = []
    hashes = {}
    for name in allowed_names:
        path = split_dir / f"{name}.csv"
        _guard_development_path(path)
        frame, digest = _read_stable_csv(path)
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        "_split": np.repeat(name, len(frame)),
                        "_split_row": np.arange(len(frame), dtype=np.int64),
                    }
                ),
            ],
            axis=1,
        )
        frames.append(frame)
        hashes[name] = digest
    development = pd.concat(frames, ignore_index=True)
    return validate_development(development, config, split_hashes=hashes), hashes


def validate_development(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    *,
    split_hashes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    data = dict(config["data"])
    target = str(data["target"])
    station_col = str(data["station_col"])
    time_col = str(data["time_col"])
    required = {target, station_col, time_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"development data is missing columns: {missing}")

    output = frame.copy()
    dates = pd.to_datetime(output[time_col], errors="coerce")
    if dates.isna().any():
        raise ValueError("development dates contain unparseable values")
    years = dates.dt.year.astype(int)
    expected_years = [int(year) for year in data["development_years"]]
    observed_years = sorted(int(year) for year in years.unique())
    if observed_years != expected_years or int(years.max()) > 2022:
        raise ValueError(
            f"development years must be exactly {expected_years}; got {observed_years}"
        )

    stations = sorted(output[station_col].astype(str).unique().tolist())
    expected_stations = sorted(str(value) for value in data["expected_stations"])
    if stations != expected_stations:
        raise ValueError(f"station mismatch: expected {expected_stations}, got {stations}")
    if len(output) != int(data["expected_development_rows"]):
        raise ValueError(
            f"development row mismatch: {len(output)} != "
            f"{data['expected_development_rows']}"
        )
    if split_hashes is not None:
        all_expected_hashes = dict(data["expected_split_sha256"])
        expected_hashes = {
            name: all_expected_hashes[name] for name in ("train", "val")
        }
        if dict(split_hashes) != expected_hashes:
            raise ValueError(
                f"development split hash mismatch: {split_hashes} != {expected_hashes}"
            )

    target_values = pd.to_numeric(output[target], errors="coerce")
    if not np.isfinite(target_values.to_numpy(dtype=float)).all():
        raise ValueError("development target contains non-finite values")
    output[target] = target_values.astype(float)
    output[time_col] = dates.dt.strftime("%Y-%m-%d")
    output["_year"] = years.to_numpy()
    output["_month"] = dates.dt.month.astype(int).to_numpy()
    output["_row_key"] = (
        output[station_col].astype(str) + "\x1f" + output[time_col].astype(str)
    )
    if output["_row_key"].duplicated().any():
        duplicates = output.loc[
            output["_row_key"].duplicated(keep=False), "_row_key"
        ].head().tolist()
        raise ValueError(f"duplicate station/date development keys: {duplicates}")
    return output


def predictor_columns(frame: pd.DataFrame, config: Mapping[str, object]) -> list[str]:
    data = dict(config["data"])
    excluded = {
        str(data["target"]),
        str(data["station_col"]),
        str(data["time_col"]),
        "_split",
        "_split_row",
        "_year",
        "_month",
        "_row_key",
    }
    columns = [column for column in frame.columns if column not in excluded]
    nonnumeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        raise ValueError(f"predictor universe contains nonnumeric columns: {nonnumeric}")
    expected = int(data["expected_predictor_count"])
    if len(columns) != expected:
        raise ValueError(f"expected {expected} predictors; found {len(columns)}")
    return columns


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce infinities to NaN while preserving native XGBoost missing values."""
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def load_control_features(config: Mapping[str, object]) -> tuple[dict, dict]:
    feature_config = dict(config["features"])
    v0, v0_source_hash = resolve_literal_string_list(str(feature_config["v0_source"]))
    if len(v0) != int(feature_config["v0_count"]):
        raise ValueError(f"V0 must have {feature_config['v0_count']} features")
    controls = {"V0": v0}
    provenance = {
        "V0": {
            "source": feature_config["v0_source"],
            "source_sha256": v0_source_hash,
            "ordered_feature_hash": ordered_feature_hash(v0),
            "promotable": False,
        }
    }
    for control_id, spec_value in dict(
        feature_config["diagnostic_controls"]
    ).items():
        spec = dict(spec_value)
        path = resolve_repo_path(str(spec["source"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features")
        if not isinstance(features, list) or not all(
            isinstance(item, str) for item in features
        ):
            raise ValueError(f"invalid feature list in {path}")
        if not features or len(features) != len(set(features)):
            raise ValueError(f"feature list is empty or duplicated in {path}")
        controls[str(control_id)] = list(features)
        provenance[str(control_id)] = {
            "source": spec["source"],
            "source_sha256": sha256_file(path),
            "ordered_feature_hash": ordered_feature_hash(features),
            "promotable": bool(spec.get("promotable", False)),
        }
    return controls, provenance


def development_coverage(frame: pd.DataFrame, config: Mapping[str, object]):
    data = dict(config["data"])
    return (
        frame.groupby([str(data["station_col"]), "_year"], sort=True)
        .size()
        .rename("row_count")
        .reset_index()
        .rename(columns={"_year": "year"})
    )
