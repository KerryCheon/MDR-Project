"""Run the complete feature-selection 2.2 experiment with safe resume."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from artifact_state import artifact_is_complete, atomic_write_json, sha256_file
from runtime import DEFAULT_DEVICE, DEFAULT_WORKERS, add_runtime_arguments, validate_workers


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
ARTIFACTS_DIR = EXP_DIR / "artifacts"
STATE_PATH = ARTIFACTS_DIR / "run_state.json"


def _stage(name: str, script: str, required: list[str], *args: str) -> dict:
    return {"name": name, "script": script, "args": list(args), "required": required}


STAGES = (
    _stage("selection", "run_selection.py", ["final/derived_8.0/global/completion.json", "final/derived_8.2/global/completion.json", "final/derived_8.2/regime_0/completion.json", "final/derived_8.2/regime_1/completion.json"]),
    _stage("validation", "run_eval.py", ["final/validation_eval/completion.json"], "--artifact-set", "final"),
    _stage("final_retrospective", "run_eval.py", ["final/retrospective_test_eval/completion.json"], "--artifact-set", "final", "--retrospective-test"),
    _stage("nested_selection", "run_nested_selection.py", ["nested/derived_8.0/global/completion.json", "nested/derived_8.2/global/completion.json", "nested/derived_8.2/regime_0/completion.json", "nested/derived_8.2/regime_1/completion.json"]),
    _stage("nested_locked_outer", "run_locked_outer_selection.py", ["nested_locked_outer/derived_8.0/global/completion.json", "nested_locked_outer/derived_8.2/global/completion.json"]),
    _stage("crossed_selection", "run_crossed_candidate_selection.py", ["crossed_candidates_locked_outer/derived_8.0/global/completion.json", "crossed_candidates_locked_outer/derived_8.2/global/completion.json"]),
    _stage("progressive_selection", "run_crossed_candidate_selection.py", ["progressive_crossed_locked_outer/derived_8.0/global/completion.json"], "--progressive", "--dataset", "derived_8.0"),
    _stage("nested_diagnostics", "run_candidate_diagnostics.py", ["nested/candidate_diagnostics/completion.json"], "--artifact-set", "nested"),
    _stage("crossed_diagnostics", "run_candidate_diagnostics.py", ["crossed_candidates_locked_outer/candidate_diagnostics/completion.json"], "--artifact-set", "crossed_candidates_locked_outer"),
    _stage("progressive_diagnostics", "run_candidate_diagnostics.py", ["progressive_crossed_locked_outer/candidate_diagnostics/completion.json"], "--artifact-set", "progressive_crossed_locked_outer"),
    _stage("nested_retrospective", "run_eval.py", ["nested/retrospective_test_eval/completion.json"], "--artifact-set", "nested", "--retrospective-test"),
    _stage("crossed_retrospective", "run_eval.py", ["crossed_candidates_locked_outer/retrospective_test_eval/completion.json"], "--artifact-set", "crossed_candidates_locked_outer", "--retrospective-test"),
    _stage("report", "generate_results.py", ["report/completion.json"]),
)


def _safe_clean() -> None:
    if ARTIFACTS_DIR.resolve().parent != EXP_DIR.resolve() or ARTIFACTS_DIR.name != "artifacts":
        raise RuntimeError("refusing to delete an artifact path outside this experiment")
    if ARTIFACTS_DIR.exists():
        shutil.rmtree(ARTIFACTS_DIR)
    ARTIFACTS_DIR.mkdir()


def _fingerprint(stage: dict) -> dict:
    paths = [EXP_DIR / stage["script"], EXP_DIR / "config.yaml", EXP_DIR / "nested_config.yaml"]
    return {str(path.relative_to(EXP_DIR)): sha256_file(path) for path in paths if path.exists()}


def _complete(stage: dict) -> bool:
    for relative in stage["required"]:
        path = ARTIFACTS_DIR / relative
        if not path.is_file():
            return False
        if path.name == "completion.json" and not artifact_is_complete(path.parent, []):
            return False
    return True


def _new_state(device: str, workers: int) -> dict:
    return {"status": "running", "device": device, "workers": workers, "stages": []}


def _compatible_fingerprint_update(previous: dict, stage: dict, fingerprint: dict) -> bool:
    """Allow the serial revision-retrospective safety fix to resume final stages."""
    if stage["name"] not in {"validation", "final_retrospective"}:
        return False
    old = previous.get("fingerprint", {})
    return (
        old.get("config.yaml") == fingerprint.get("config.yaml")
        and old.get("nested_config.yaml") == fingerprint.get("nested_config.yaml")
        and "run_eval.py" in old
    )


def _load_state() -> dict | None:
    try:
        with open(STATE_PATH, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--restart", action="store_true", help="Discard an incomplete run and start clean.")
    args = parser.parse_args(argv)
    workers = validate_workers(args.workers)
    state = _load_state()
    if args.restart or state is None or state.get("status") == "complete":
        _safe_clean()
        state = _new_state(args.device, workers)
    elif state.get("device") != args.device or state.get("workers") != workers:
        parser.error("resume requires the original --device and --workers; use --restart to change them")
    else:
        state["status"] = "running"

    for stage in STAGES:
        previous = next((item for item in state["stages"] if item["name"] == stage["name"]), None)
        fingerprint = _fingerprint(stage)
        if previous and previous.get("status") == "complete" and _complete(stage):
            if previous.get("fingerprint") != fingerprint:
                if not _compatible_fingerprint_update(previous, stage, fingerprint):
                    parser.error(f"{stage['name']} inputs changed; use --restart to avoid mixing runs")
                previous["fingerprint"] = fingerprint
                previous["compatibility_note"] = "serial revision-retrospective safety fix does not affect this final evaluation stage"
                atomic_write_json(STATE_PATH, state)
            continue
        command = [sys.executable, str(EXP_DIR / stage["script"]), *stage["args"]]
        if stage["script"] != "generate_results.py":
            command.extend(["--device", args.device, "--workers", str(workers)])
        record = {"name": stage["name"], "command": command, "status": "running", "fingerprint": fingerprint, "started": datetime.now(timezone.utc).isoformat()}
        state["stages"] = [item for item in state["stages"] if item["name"] != stage["name"]] + [record]
        atomic_write_json(STATE_PATH, state)
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(PROJECT_ROOT), existing_pythonpath) if value
        )
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
        if result.returncode != 0 or not _complete(stage):
            record["status"] = "failed"
            state["status"] = "failed"
            atomic_write_json(STATE_PATH, state)
            raise SystemExit(result.returncode or 1)
        record["status"] = "complete"
        record["completed"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(STATE_PATH, state)
    state["status"] = "complete"
    atomic_write_json(STATE_PATH, state)


if __name__ == "__main__":
    main()
