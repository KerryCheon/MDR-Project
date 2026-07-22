"""Validate data, router, protocol parity, and live-code substitutions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from data_loading import (
    ROUTER_KIND,
    V0_FEATURE_COUNT,
    V0_FEATURE_SOURCE,
    load_v0_features,
    resolve_router_config,
    source_sha256,
)


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
SOURCE_EXP_DIR = (
    PROJECT_ROOT
    / "notebooks/experiment/derived_8.2-feature-selection-2.2"
)
RUNTIME_FILES = (
    "config.yaml",
    "data_loading.py",
    "generate_results.py",
    "nested_config.yaml",
    "run_all.py",
    "run_candidate_diagnostics.py",
    "run_crossed_candidate_selection.py",
    "run_eval.py",
    "run_locked_outer_selection.py",
    "run_nested_selection.py",
    "run_selection.py",
)
BANNED_RUNTIME_TEXT = (
    "OVERALL_SELECTED_FEATURES_V3",
    '"2.1_c1"',
    '"global_c1"',
    "derived_8.2-feature-selection-2.2",
    "data/splits/derived_8.2",
    '"derived_8.2"',
    "clustering_dynamic_k2",
)


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def _validate_protocol_parity() -> None:
    source = _read_yaml(SOURCE_EXP_DIR / "config.yaml")
    target = _read_yaml(EXP_DIR / "config.yaml")
    for key in ("data", "selection", "evaluation", "logging"):
        if source[key] != target[key]:
            raise ValueError(f"2.2 protocol drift in config section: {key}")
    if source["regime_delta"]["additions"] != target["regime_delta"]["additions"]:
        raise ValueError("2.2 regime-delta candidate sizes changed")
    if source["regime_delta"]["selection"] != target["regime_delta"]["selection"]:
        raise ValueError("2.2 regime selector settings changed")

    source_nested = _read_yaml(SOURCE_EXP_DIR / "nested_config.yaml")
    target_nested = _read_yaml(EXP_DIR / "nested_config.yaml")
    for key in ("data", "inner_selection", "outer_selection"):
        if source_nested[key] != target_nested[key]:
            raise ValueError(f"2.2 nested protocol drift in section: {key}")
    if (
        source_nested["regime_delta"]["additions"]
        != target_nested["regime_delta"]["additions"]
    ):
        raise ValueError("2.2 nested regime-delta candidate sizes changed")


def _validate_runtime_substitutions() -> None:
    failures = []
    for filename in RUNTIME_FILES:
        path = EXP_DIR / filename
        text = path.read_text(encoding="utf-8")
        for banned in BANNED_RUNTIME_TEXT:
            if banned in text:
                failures.append(f"{filename}: {banned}")
    if failures:
        raise ValueError(f"stale live-code references: {failures}")


def _validate_existing_router_artifacts(features: list[str]) -> list[str]:
    expected_hash = source_sha256(PROJECT_ROOT, V0_FEATURE_SOURCE)
    candidates = (
        (
            EXP_DIR / "artifacts/smoke/derived_8.3/router_provenance.json",
            "combined_development_pool",
            False,
        ),
        (
            EXP_DIR / "artifacts/final/derived_8.3/router_provenance.json",
            "combined_development_pool",
            False,
        ),
        (
            EXP_DIR / "artifacts/nested_smoke/derived_8.3/router.json",
            "inner_training_split",
            True,
        ),
        (
            EXP_DIR / "artifacts/nested/derived_8.3/router.json",
            "inner_training_split",
            True,
        ),
    )
    validated = []
    for path, fit_scope, frozen in candidates:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        expected = {
            "kind": ROUTER_KIND,
            "feature_source": V0_FEATURE_SOURCE,
            "feature_source_sha256": expected_hash,
            "feature_count": V0_FEATURE_COUNT,
            "columns": features,
            "imputation": "mean",
            "scaler": "StandardScaler",
            "n_clusters": 2,
            "n_init": 10,
            "random_state": 42,
            "fit_scope": fit_scope,
            "frozen_for_evaluation": frozen,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"{path} router mismatch for {key}")
        validated.append(str(path.relative_to(EXP_DIR)))
    return validated


def main() -> None:
    features = load_v0_features(PROJECT_ROOT)
    if len(features) != V0_FEATURE_COUNT or len(features) != len(set(features)):
        raise ValueError("V0 is not exactly 50 unique features")

    split_columns = {}
    for split in ("train", "val", "test"):
        path = PROJECT_ROOT / "data/splits/derived_8.3" / f"{split}.csv"
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [feature for feature in features if feature not in columns]
        if missing:
            raise ValueError(f"{split} is missing V0 features: {missing}")
        split_columns[split] = len(columns)

    router_configs = {}
    for filename in ("config.yaml", "nested_config.yaml"):
        config = _read_yaml(EXP_DIR / filename)
        resolved = resolve_router_config(
            PROJECT_ROOT,
            config["regime_delta"]["router"],
        )
        if resolved["columns"] != features:
            raise ValueError(f"{filename} router order differs from V0")
        router_configs[filename] = {
            "kind": resolved["kind"],
            "feature_source": resolved["feature_source"],
            "feature_count": resolved["feature_count"],
            "imputation": resolved["imputation"],
            "scaler": resolved["scaler"],
            "n_clusters": resolved["n_clusters"],
            "n_init": resolved["n_init"],
            "random_state": resolved["random_state"],
        }
    if router_configs["config.yaml"] != router_configs["nested_config.yaml"]:
        raise ValueError("original and nested router contracts differ")
    if router_configs["config.yaml"]["kind"] != ROUTER_KIND:
        raise ValueError("unexpected router kind")
    if router_configs["config.yaml"]["feature_source"] != V0_FEATURE_SOURCE:
        raise ValueError("unexpected router feature source")

    _validate_protocol_parity()
    _validate_runtime_substitutions()
    router_artifacts = _validate_existing_router_artifacts(features)

    print(
        json.dumps(
            {
                "status": "ok",
                "v0_feature_count": len(features),
                "v0_feature_source": V0_FEATURE_SOURCE,
                "v0_source_sha256": source_sha256(
                    PROJECT_ROOT,
                    V0_FEATURE_SOURCE,
                ),
                "split_column_counts": split_columns,
                "router": router_configs["config.yaml"],
                "validated_router_artifacts": router_artifacts,
                "protocol": "unchanged derived_8.2-feature-selection-2.2 settings",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
