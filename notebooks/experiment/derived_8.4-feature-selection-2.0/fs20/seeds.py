"""Historical seed-list loading without mutating historical experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from .data import ExperimentData
from .selection import SelectionResult


def _load_json_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"Expected a top-level features list: {path}")
    return [str(feature) for feature in features]


def _load_metadata_features(path: Path, attribute: str) -> list[str]:
    spec = importlib.util.spec_from_file_location("seed_metadata", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import metadata seed: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features = getattr(module, attribute)
    return [str(feature) for feature in features]


def _available(features: list[str], data: ExperimentData) -> list[str]:
    valid = set(data.feature_columns)
    return [feature for feature in features if feature in valid]


def load_seed_sets(
    data: ExperimentData,
    config: dict[str, Any],
    global_selection_results: dict[str, SelectionResult],
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Collect static and locally generated seed sets with visible availability notes."""
    seeds: dict[str, list[str]] = {"v0": list(data.v0_features)}
    notes: list[dict[str, str]] = []
    for name, path in config.get("seeds", {}).get("files", {}).items():
        if not Path(path).exists():
            notes.append({"seed": name, "status": "missing", "detail": str(path)})
            continue
        features = _available(_load_json_features(Path(path)), data)
        if not features:
            notes.append({"seed": name, "status": "empty_after_schema_check", "detail": str(path)})
            continue
        seeds[name] = features
        notes.append({"seed": name, "status": "loaded", "detail": str(len(features))})

    for name, metadata in config.get("seeds", {}).get("metadata", {}).items():
        path = Path(metadata["path"])
        if not path.exists():
            notes.append({"seed": name, "status": "missing", "detail": str(path)})
            continue
        features = _available(_load_metadata_features(path, str(metadata["attribute"])), data)
        if not features:
            notes.append({"seed": name, "status": "empty_after_schema_check", "detail": str(path)})
            continue
        seeds[name] = features
        notes.append({"seed": name, "status": "loaded", "detail": str(len(features))})

    for profile in ("mi300", "legacy_forced_bypass", "mi300_repaired"):
        if profile in global_selection_results:
            features = _available(global_selection_results[profile].selected, data)
            if features:
                seeds[f"local_{profile}"] = features
                notes.append({"seed": f"local_{profile}", "status": "generated", "detail": str(len(features))})
    return seeds, notes
