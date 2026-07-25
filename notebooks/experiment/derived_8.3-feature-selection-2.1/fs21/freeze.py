"""Create and verify the immutable development-to-benchmark handoff."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .artifacts import atomic_write_json, sha256_file, stable_json_hash
from .constants import (
    BENCHMARK_REGISTRY_PATH,
    DEVELOPMENT_FREEZE_PATH,
    EXP_DIR,
    GLOBAL_CONFIG_PATH,
    MOE_CONFIG_PATH,
    PROJECT_ROOT,
)
from .state import git_revision, runtime_input_paths


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def prediction_affecting_paths(extra_paths: Iterable[Path]) -> list[Path]:
    paths = list(runtime_input_paths())
    paths.extend([GLOBAL_CONFIG_PATH, MOE_CONFIG_PATH, BENCHMARK_REGISTRY_PATH])
    paths.extend(extra_paths)
    return sorted(set(path.resolve() for path in paths if path.is_file()))


def create_development_freeze(
    *,
    split_paths: Iterable[Path],
    artifact_paths: Iterable[Path],
    selection: dict,
    learner: dict,
    router_decision: dict,
    benchmark_challenger: dict,
) -> dict:
    split_paths = [path.resolve() for path in split_paths]
    artifact_paths = [path.resolve() for path in artifact_paths]
    hashed_inputs = prediction_affecting_paths(artifact_paths)
    payload = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "split_hashes": {_relative(path): sha256_file(path) for path in split_paths},
        "input_hashes": {_relative(path): sha256_file(path) for path in hashed_inputs},
        "selection": selection,
        "learner": learner,
        "router_and_expert_decision": router_decision,
        "benchmark_challenger": benchmark_challenger,
        "benchmark_feedback_may_change_configuration": False,
        "overall_selected_features_v0_overwritten": False,
        "retrospective_test": True,
        "benchmark_reused": True,
        "unbiased_sota_eligible": False,
        "unbiased_generalization_claim_eligible": False,
        "ece_external_confirmation_pending": True,
    }
    payload["freeze_sha256"] = stable_json_hash(payload)
    atomic_write_json(DEVELOPMENT_FREEZE_PATH, payload)
    return payload


def verify_development_freeze(path: Path = DEVELOPMENT_FREEZE_PATH) -> dict:
    """Verify every frozen byte before benchmark data can be opened."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError("valid development_freeze.json is required") from error
    recorded_hash = payload.get("freeze_sha256")
    unhashed = dict(payload)
    unhashed.pop("freeze_sha256", None)
    if recorded_hash != stable_json_hash(unhashed):
        raise RuntimeError("development freeze payload hash is invalid")
    mismatches = []
    for group in ("split_hashes", "input_hashes"):
        for relative, expected in payload.get(group, {}).items():
            candidate = (PROJECT_ROOT / relative).resolve()
            if not candidate.is_relative_to(PROJECT_ROOT.resolve()):
                mismatches.append(f"{relative}: escapes project root")
            elif not candidate.is_file():
                mismatches.append(f"{relative}: missing")
            elif sha256_file(candidate) != expected:
                mismatches.append(f"{relative}: SHA-256 mismatch")
    if mismatches:
        raise RuntimeError(f"development freeze verification failed: {mismatches}")
    mandatory_false = (
        "unbiased_sota_eligible",
        "unbiased_generalization_claim_eligible",
    )
    if any(payload.get(key) is not False for key in mandatory_false):
        raise RuntimeError("freeze contains an impermissible unbiased claim")
    mandatory_true = (
        "retrospective_test",
        "benchmark_reused",
        "ece_external_confirmation_pending",
    )
    if any(payload.get(key) is not True for key in mandatory_true):
        raise RuntimeError("development freeze is missing mandatory disclosure flags")
    required_sections = (
        "selection",
        "learner",
        "router_and_expert_decision",
        "benchmark_challenger",
    )
    if any(not isinstance(payload.get(key), dict) for key in required_sections):
        raise RuntimeError("development freeze is missing a frozen decision section")
    return payload
