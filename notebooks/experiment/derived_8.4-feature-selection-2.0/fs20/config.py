"""Configuration loading and invariant checks for the isolated experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

FORBIDDEN_TEMPORAL_KEYS = {
    "temporal_weight",
    "temporal_weights",
    "recency_weight",
    "recency_weights",
    "drift",
    "drift_weight",
    "year_weight",
    "year_weights",
}


def _walk_keys(value: Any, path: str = "") -> list[str]:
    """Return dotted paths for banned configuration keys."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            if str(key).lower() in FORBIDDEN_TEMPORAL_KEYS:
                found.append(key_path)
            found.extend(_walk_keys(child, key_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_keys(child, f"{path}[{index}]"))
    return found


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load config and make filesystem references explicit project-root paths."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    forbidden = _walk_keys(config)
    if forbidden:
        raise ValueError(
            "derived_8.4-feature-selection-2.0 forbids temporal weighting or drift "
            f"configuration: {', '.join(forbidden)}"
        )

    config = deepcopy(config)
    config["_config_path"] = config_path
    config["_project_root"] = PROJECT_ROOT
    config["_experiment_dir"] = EXPERIMENT_DIR

    for split_name, split_path in config["data"]["splits"].items():
        config["data"]["splits"][split_name] = _resolve_path(split_path)
    config["data"]["metadata_path"] = _resolve_path(config["data"]["metadata_path"])

    for seed_name, seed_path in config.get("seeds", {}).get("files", {}).items():
        config["seeds"]["files"][seed_name] = _resolve_path(seed_path)

    artifact_dir = config.get("artifacts", {}).get("directory", "artifacts")
    config.setdefault("artifacts", {})["directory"] = _resolve_path(
        EXPERIMENT_DIR / artifact_dir
    )

    canonical_mi_k = int(config["selection"]["canonical_mi_k"])
    if canonical_mi_k != 300:
        raise ValueError(
            "The canonical selector must use MI k=300; lower-k and no-MI profiles are "
            "diagnostic-only."
        )
    if config["selection"].get("canonical_profile") == "no_mi":
        raise ValueError("The no-MI profile cannot be canonical.")

    return config

