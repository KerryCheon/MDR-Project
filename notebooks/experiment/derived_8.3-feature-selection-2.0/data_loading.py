"""Shared, validated data-source resolution for the 2.0 rerun."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Mapping


V0_FEATURE_SOURCE = (
    "data/splits/derived_8.3/dataset_metadata.py::"
    "OVERALL_SELECTED_FEATURES_V0"
)
V0_FEATURE_COUNT = 50
ROUTER_KIND = "clustering_v0_full_k2"


def resolve_source_path(project_root: Path, source: str) -> tuple[Path, str]:
    """Resolve a repository-relative ``path::constant`` source safely."""
    try:
        relative_path, constant_name = source.split("::", maxsplit=1)
    except ValueError as error:
        raise ValueError(
            "feature source must use '<relative path>::<constant>' syntax"
        ) from error
    if not relative_path or not constant_name:
        raise ValueError(f"invalid feature source: {source!r}")

    root = project_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"feature source escapes the repository: {source!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, constant_name


def source_sha256(project_root: Path, source: str) -> str:
    """Hash the source file that defines a resolved feature list."""
    path, _ = resolve_source_path(project_root, source)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_string_list_source(project_root: Path, source: str) -> list[str]:
    """Load one literal string-list constant from a repository-relative source."""
    path, constant_name = resolve_source_path(project_root, source)

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            targets = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if constant_name not in targets:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{constant_name} is not a string list in {path}")
        if not value or len(value) != len(set(value)):
            raise ValueError(f"{constant_name} is empty or contains duplicates")
        return list(value)
    raise ValueError(f"missing constant {constant_name} in {path}")


def load_v0_features(project_root: Path) -> list[str]:
    """Return the exact ordered derived 8.3 V0 baseline feature list."""
    features = load_string_list_source(project_root, V0_FEATURE_SOURCE)
    if len(features) != V0_FEATURE_COUNT:
        raise ValueError(
            f"derived 8.3 V0 must contain {V0_FEATURE_COUNT} features; "
            f"found {len(features)}"
        )
    return features


def resolve_router_config(
    project_root: Path,
    router_config: Mapping[str, object],
) -> dict:
    """Resolve and validate the immutable Clustering_V0_Full_k2 contract."""
    resolved = deepcopy(dict(router_config))
    expected = {
        "kind": ROUTER_KIND,
        "feature_source": V0_FEATURE_SOURCE,
        "imputation": "mean",
        "scaler": "StandardScaler",
        "n_clusters": 2,
        "n_init": 10,
        "random_state": 42,
    }
    mismatches = {
        key: (resolved.get(key), value)
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid Clustering_V0_Full_k2 config: {mismatches}")

    columns = load_string_list_source(project_root, resolved["feature_source"])
    v0_features = load_v0_features(project_root)
    if columns != v0_features:
        raise ValueError("router columns do not match the exact ordered V0 list")
    resolved["columns"] = columns
    resolved["feature_count"] = len(columns)
    resolved["feature_source_sha256"] = source_sha256(
        project_root,
        resolved["feature_source"],
    )
    return resolved


def router_provenance(
    router_config: Mapping[str, object],
    *,
    fit_scope: str,
    frozen_for_evaluation: bool,
) -> dict:
    """Return the resolved router identity and preprocessing provenance."""
    return {
        "kind": router_config["kind"],
        "feature_source": router_config["feature_source"],
        "feature_source_sha256": router_config["feature_source_sha256"],
        "columns": list(router_config["columns"]),
        "feature_count": int(router_config["feature_count"]),
        "imputation": router_config["imputation"],
        "scaler": router_config["scaler"],
        "n_clusters": int(router_config["n_clusters"]),
        "n_init": int(router_config["n_init"]),
        "random_state": int(router_config["random_state"]),
        "fit_scope": fit_scope,
        "frozen_for_evaluation": bool(frozen_for_evaluation),
    }
