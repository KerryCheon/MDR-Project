"""Run the complete development-only 2.1 pipeline with resumable stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fs21.artifacts import (
    atomic_write_json,
    completion_is_valid,
    invalidate_completion,
    sha256_file,
    write_completion,
)
from fs21.constants import (
    DEVELOPMENT_FREEZE_PATH,
    DEVELOPMENT_STAGE_NAMES,
    EXP_DIR,
    GLOBAL_CONFIG_PATH,
    MOE_CONFIG_PATH,
    PROJECT_ROOT,
)
from fs21.data import read_yaml, resolve_repo_path
from fs21.freeze import create_development_freeze
from fs21.global_pipeline import build_context, run_stage
from fs21.state import (
    RunJournal,
    build_fingerprint,
    environment_record,
    safe_restart,
    utc_now,
)
from generate_results import generate_reports
from preflight import run_preflight
from run_moe_diagnostics import run_causal_stage, run_delta_stage
from run_station_diagnostics import run as run_station_diagnostics


RUN_ALL_RELATIVE_PATH = str(
    Path(__file__).resolve().relative_to(PROJECT_ROOT.resolve())
)


def _migrate_worker_resume(
    *,
    state_path: Path,
    artifact_root: Path,
    new_fingerprint: dict,
    new_environment: dict,
    configured_workers: int,
    requested_workers: int,
) -> dict:
    """Authorize reuse when only orchestration concurrency changed.

    The worker pool schedules independent fits; every fit still has a frozen
    seed and ``n_jobs=1``. This migration is intentionally narrow: all split,
    code, configuration, device, and external-input hashes must match except
    for this runner's own hash, which changes to add the audited override.
    """
    if not state_path.is_file():
        raise RuntimeError("worker resume override requires an existing run journal")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    old_fingerprint = dict(payload.get("fingerprint", {}))
    if old_fingerprint == new_fingerprint:
        return {
            "applied": False,
            "reason": "journal_already_uses_requested_worker_fingerprint",
        }
    old_workers = int(old_fingerprint.get("workers", -1))
    if old_workers < 1:
        raise RuntimeError("worker override source count is invalid")
    if int(new_fingerprint.get("workers", -1)) != int(requested_workers):
        raise RuntimeError("new fingerprint does not contain the requested worker count")
    invariant_keys = ("git_revision", "split_hashes", "device", "smoke")
    mismatches = [
        key
        for key in invariant_keys
        if old_fingerprint.get(key) != new_fingerprint.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            f"worker-only resume has non-worker fingerprint changes: {mismatches}"
        )
    old_inputs = dict(old_fingerprint.get("runtime_inputs", {}))
    new_inputs = dict(new_fingerprint.get("runtime_inputs", {}))
    changed_inputs = sorted(
        key
        for key in set(old_inputs) | set(new_inputs)
        if old_inputs.get(key) != new_inputs.get(key)
    )
    completed_stages = []
    for row in payload.get("stages", []):
        if row.get("status") != "complete":
            continue
        name = str(row["name"])
        stage = artifact_root / "stages" / name
        if not completion_is_valid(stage):
            raise RuntimeError(f"cannot reuse corrupt completed stage: {name}")
        completed_stages.append(name)
    rank_root = (
        artifact_root
        / "stages"
        / "05_robust_candidate_generation"
        / "rank_units"
    )
    completed_rank_units = []
    if rank_root.is_dir():
        for unit in sorted(path for path in rank_root.iterdir() if path.is_dir()):
            if (unit / "completion.json").is_file():
                if not completion_is_valid(unit):
                    raise RuntimeError(f"cannot reuse corrupt rank unit: {unit.name}")
                completed_rank_units.append(unit.name)
    record = {
        "applied": True,
        "timestamp": utc_now(),
        "scope": "execution_concurrency_only",
        "from_workers": old_workers,
        "to_workers": int(requested_workers),
        "xgboost_n_jobs": 1,
        "model_seeds_folds_and_parameters_unchanged": True,
        "runtime_input_changes": changed_inputs,
        "old_fingerprint_sha256": old_fingerprint.get("fingerprint_sha256"),
        "new_fingerprint_sha256": new_fingerprint.get("fingerprint_sha256"),
        "reused_completed_stages": sorted(set(completed_stages)),
        "reused_completed_stage5_rank_units": len(completed_rank_units),
    }
    override_path = artifact_root / "worker_resume_override.json"
    atomic_write_json(override_path, record)
    payload.setdefault("worker_resume_overrides", []).append(record)
    payload["fingerprint"] = new_fingerprint
    payload["environment"] = new_environment
    payload["status"] = "running"
    payload["failure"] = None
    payload["completed"] = None
    atomic_write_json(state_path, payload)
    return record


def _freeze_stage(context) -> Path:
    stage = context.artifact_root / "stages" / "13_development_freeze"
    required = ["freeze_manifest.json"]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    if context.smoke:
        atomic_write_json(
            stage / "freeze_manifest.json",
            {
                "canonical_freeze_created": False,
                "reason": "smoke artifacts are noncanonical and benchmark-ineligible",
                "benchmark_may_run": False,
            },
        )
        write_completion(stage, required)
        return stage
    candidate_features_path = (
        context.artifact_root / "stages" / "09_consensus" / "candidate_features.json"
    )
    global_decision_path = (
        context.artifact_root
        / "stages"
        / "07_global_decision"
        / "global_promotion_decision.json"
    )
    beta_decision_path = (
        context.artifact_root / "stages" / "08_beta_decision" / "beta_decision.json"
    )
    moe_decision_path = (
        context.artifact_root
        / "stages"
        / "12_regime_delta_moe_decision"
        / "moe_promotion_decision.json"
    )
    frozen_moe_path = (
        context.artifact_root
        / "stages"
        / "12_regime_delta_moe_decision"
        / "frozen_moe_features.json"
    )
    candidate_features = json.loads(candidate_features_path.read_text(encoding="utf-8"))
    global_decision = json.loads(global_decision_path.read_text(encoding="utf-8"))
    beta_decision = json.loads(beta_decision_path.read_text(encoding="utf-8"))
    moe_decision = json.loads(moe_decision_path.read_text(encoding="utf-8"))
    frozen_moe = json.loads(frozen_moe_path.read_text(encoding="utf-8"))
    if moe_decision["moe_promoted"]:
        challenger = {
            "model_id": moe_decision["winner"],
            "kind": "moe",
            "development_eligible": True,
            "features": frozen_moe,
        }
    else:
        challenger = {
            "model_id": candidate_features["benchmark_challenger"],
            "kind": "single_global",
            "development_eligible": bool(global_decision["global_gate_passed"]),
            "features": candidate_features["benchmark_challenger_features"],
            "ordered_feature_hash": candidate_features[
                "benchmark_challenger_ordered_feature_hash"
            ],
            "beta": candidate_features["benchmark_challenger_beta"],
        }
    artifact_paths = [
        resolve_repo_path(
            str(context.config["features"]["v0_source"]).split("::", maxsplit=1)[0]
        ),
        context.artifact_root
        / "stages"
        / "01_preflight"
        / "preflight.json",
        context.artifact_root
        / "stages"
        / "01_preflight"
        / "predictors.json",
        context.artifact_root
        / "stages"
        / "01_preflight"
        / "development_coverage.csv",
        context.artifact_root
        / "stages"
        / "02_fold_manifests"
        / "fold_manifest.csv",
        context.artifact_root
        / "stages"
        / "02_fold_manifests"
        / "coverage_manifest.csv",
        context.artifact_root
        / "stages"
        / "02_fold_manifests"
        / "partitions.json",
        candidate_features_path,
        global_decision_path,
        beta_decision_path,
        moe_decision_path,
        frozen_moe_path,
        EXP_DIR / "generate_results.py",
    ]
    worker_override_path = context.artifact_root / "worker_resume_override.json"
    if worker_override_path.is_file():
        artifact_paths.append(worker_override_path)
    split_dir = PROJECT_ROOT / str(context.config["data"]["split_dir"])
    freeze = create_development_freeze(
        split_paths=[split_dir / "train.csv", split_dir / "val.csv"],
        artifact_paths=artifact_paths,
        selection={
            "global_decision": global_decision,
            "candidate_features": candidate_features,
            "beta_decision": beta_decision,
        },
        learner={
            "parameters": dict(context.config["learner"]),
            "seed": 42,
            "device": context.device,
            "workers": int(context.workers),
            "configured_default_workers": int(
                context.config["runtime"]["canonical_workers"]
            ),
            "native_missing_handling": True,
        },
        router_decision={
            "moe_decision": moe_decision,
            "frozen_moe": frozen_moe,
        },
        benchmark_challenger=challenger,
    )
    atomic_write_json(
        stage / "freeze_manifest.json",
        {
            "canonical_freeze_created": True,
            "path": str(DEVELOPMENT_FREEZE_PATH.relative_to(PROJECT_ROOT)),
            "freeze_sha256": sha256_file(DEVELOPMENT_FREEZE_PATH),
            "freeze_payload_sha256": freeze["freeze_sha256"],
            "benchmark_challenger": challenger,
        },
    )
    write_completion(stage, required)
    return stage


def _report_stage(context) -> Path:
    generate_reports(check=False, smoke=context.smoke)
    return context.artifact_root / "stages" / "14_development_report"


def _run_stage(context, name: str, moe_config: dict) -> Path:
    if name == "01_preflight":
        return run_preflight(
            device=context.device,
            workers=context.workers,
            smoke=context.smoke,
            allow_worker_resume=True,
        )
    if name in {
        "02_fold_manifests",
        "03_control_ledgers",
        "04_path_screen",
        "05_robust_candidate_generation",
        "06_candidate_oof",
        "07_global_decision",
        "08_beta_decision",
        "09_consensus",
    }:
        return run_stage(context, name)
    if name == "10_station_temporal_diagnostics":
        run_station_diagnostics(
            device=context.device,
            workers=context.workers,
            smoke=context.smoke,
        )
        return context.artifact_root / "stages" / name
    if name == "11_moe_causal_matrix":
        return run_causal_stage(context, moe_config)
    if name == "12_regime_delta_moe_decision":
        return run_delta_stage(context, moe_config)
    if name == "13_development_freeze":
        return _freeze_stage(context)
    if name == "14_development_report":
        return _report_stage(context)
    raise ValueError(f"unknown development stage: {name}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--allow-worker-resume",
        action="store_true",
        help=(
            "Audit and resume an existing canonical run with a different "
            "orchestration worker count; model-level n_jobs remains 1"
        ),
    )
    parser.add_argument(
        "--restart",
        nargs="?",
        const="01_preflight",
        choices=DEVELOPMENT_STAGE_NAMES,
        help="Clear the named 2.1 stage and downstream dependents; no value restarts all.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = read_yaml(GLOBAL_CONFIG_PATH)
    moe_config = read_yaml(MOE_CONFIG_PATH)
    if not args.smoke:
        if args.device != str(config["runtime"]["canonical_device"]):
            parser.error("canonical run requires --device cuda")
        if (
            args.workers != int(config["runtime"]["canonical_workers"])
            and not args.allow_worker_resume
        ):
            parser.error("canonical run requires --workers 4")
    if args.allow_worker_resume and args.smoke:
        parser.error("worker resume override is only valid for canonical development")
    artifact_root = EXP_DIR / "artifacts" / ("smoke" if args.smoke else "development")
    if args.restart is not None:
        safe_restart(artifact_root, args.restart)
    split_dir = PROJECT_ROOT / str(config["data"]["split_dir"])
    metadata_path = resolve_repo_path(
        str(config["features"]["v0_source"]).split("::", maxsplit=1)[0]
    )
    external_development_sources = [metadata_path]
    external_development_sources.extend(
        resolve_repo_path(str(spec["source"]))
        for spec in config["features"]["diagnostic_controls"].values()
    )
    external_development_sources.append(
        resolve_repo_path(str(moe_config["historical_specialists"]["source"]))
    )
    fingerprint = build_fingerprint(
        device=args.device,
        workers=args.workers,
        split_paths=[split_dir / "train.csv", split_dir / "val.csv"],
        extra_paths=external_development_sources,
        smoke=args.smoke,
    )
    state_path = artifact_root / "run_state.json"
    runtime_environment = environment_record(
        device=args.device,
        workers=args.workers,
    )
    if args.allow_worker_resume and state_path.is_file():
        migration = _migrate_worker_resume(
            state_path=state_path,
            artifact_root=artifact_root,
            new_fingerprint=fingerprint,
            new_environment=runtime_environment,
            configured_workers=int(config["runtime"]["canonical_workers"]),
            requested_workers=args.workers,
        )
        print(json.dumps({"worker_resume_override": migration}, indent=2))
    journal = RunJournal(
        state_path,
        command=[sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        fingerprint=fingerprint,
        environment=runtime_environment,
    )
    context = build_context(
        device=args.device,
        workers=args.workers,
        smoke=args.smoke,
    )
    for name in DEVELOPMENT_STAGE_NAMES:
        stage_dir = artifact_root / "stages" / name
        if journal.stage_status(name) == "complete" and completion_is_valid(stage_dir):
            continue
        command = [name, "--device", args.device, "--workers", str(args.workers)]
        if args.smoke:
            command.append("--smoke")
        journal.stage_started(name, command)
        try:
            output = _run_stage(context, name, moe_config)
            if not completion_is_valid(output):
                raise RuntimeError(f"stage returned without a valid completion marker: {output}")
        except BaseException as error:
            journal.stage_failed(name, error)
            raise
        journal.stage_completed(name)
    journal.complete()
    print(
        json.dumps(
            {
                "status": "complete",
                "development_only": True,
                "benchmark_invoked": False,
                "artifact_root": str(artifact_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
