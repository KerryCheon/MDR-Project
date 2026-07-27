#!/usr/bin/env python3
"""Run the isolated derived_8.4 feature-selection 2.0 experiment.

Run from the repository root with the notebook uv environment:

    uv run --project notebooks python notebooks/experiment/derived_8.4-feature-selection-2.0/run_search.py --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from fs20.audit import run_audit
from fs20.config import load_config
from fs20.data import load_experiment_data
from fs20.evaluate import ModelEvaluator
from fs20.search import DirectSearch, SearchIncompleteError


def _write_run_settings(config: dict, artifact_dir: Path, args: argparse.Namespace) -> None:
    """Persist plain run settings without hashes or a provenance registry."""
    payload = {
        "config_path": str(config["_config_path"]),
        "stage": args.stage,
        "workers": args.workers,
        "deadline_minutes": args.deadline_minutes,
        "audit_bootstrap": config["audit"]["stability_n_boot"],
        "search_rounds": config["search"]["max_rounds"],
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "run_settings.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _smoke_config(config: dict) -> dict:
    """Produce a bounded smoke configuration without changing the checked-in config."""
    smoke = deepcopy(config)
    smoke["audit"]["profiles"] = ["mi300"]
    smoke["audit"]["stability_n_boot"] = 2
    # The smoke path still exercises the mandatory 0/5/10 grid, so the pool
    # must hold the 50-feature backbone plus ten external specialists.
    smoke["search"]["candidate_pool_size"] = 60
    smoke["search"]["max_rounds"] = 1
    smoke["search"]["exact_attempts_per_round"] = 1
    smoke["search"]["global_feature_min"] = 50
    smoke["search"]["global_feature_max"] = 50
    smoke["search"]["final_reserve_minutes"] = 1
    smoke["model"]["exact_params"]["n_estimators"] = 50
    smoke["model"]["proxy_params"]["n_estimators"] = 50
    smoke["seeds"]["files"] = {
        name: path
        for name, path in smoke["seeds"]["files"].items()
        if name in {"legacy_8_2_forced_bypass", "current_8_4_c0", "current_8_4_c1_true_off"}
    }
    smoke["calibration"]["r2_tolerance"] = 1.0
    smoke["calibration"]["rmse_tolerance"] = 1.0
    smoke["artifacts"]["directory"] = Path(config["artifacts"]["directory"]) / "smoke"
    return smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "config.yaml",
        help="Experiment configuration path.",
    )
    parser.add_argument(
        "--stage",
        choices=("audit", "calibration", "search", "all", "report-data"),
        default="all",
        help="Run the audit, calibration, search (with a fresh audit), everything, or print existing data paths.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Candidate-model workers.")
    parser.add_argument("--deadline-minutes", type=int, default=None, help="Whole search budget.")
    parser.add_argument("--smoke", action="store_true", help="Run a small integration smoke path.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.smoke:
        config = _smoke_config(config)
    args.workers = args.workers or int(config["search"]["workers"])
    args.deadline_minutes = args.deadline_minutes or int(config["search"]["deadline_minutes"])
    artifact_dir = Path(config["artifacts"]["directory"])
    _write_run_settings(config, artifact_dir, args)

    if args.stage == "report-data":
        for filename in ("collapse_audit.csv", "candidate_pool.csv", "search_results.csv", "selected_features.json"):
            path = artifact_dir / ("audit/" + filename if filename == "collapse_audit.csv" else filename)
            print(path)
        return 0

    data = load_experiment_data(config)
    print(
        "Loaded derived_8.4: "
        f"train={len(data.train)}, val={len(data.val)}, test={len(data.test)}, "
        f"features={len(data.feature_columns)}, V0={len(data.v0_features)}"
    )
    if args.stage == "calibration":
        evaluator = ModelEvaluator(data, config)
        calibration = evaluator.evaluate(
            "baseline_v0_calibration",
            data.v0_features,
            model_kind="exact",
            include_predictions=False,
        )
        payload = calibration.as_record()
        (artifact_dir / "calibration.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
        return 0
    audit_results, audit_summary = run_audit(data, config, artifact_dir)
    print("=== Collapse audit ===")
    print(audit_summary.to_string(index=False))
    if args.stage == "audit":
        return 0

    search = DirectSearch(
        data,
        config,
        audit_results,
        workers=args.workers,
        deadline_minutes=args.deadline_minutes,
    )
    try:
        summary = search.run()
    except SearchIncompleteError as error:
        print(f"Search checkpointed as incomplete: {error}", file=sys.stderr)
        print(f"Wrote incomplete artifacts to {artifact_dir}", file=sys.stderr)
        return 2
    print("=== Direct-search winner ===")
    print(json.dumps(summary["winner"], indent=2))
    print(f"Wrote artifacts to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
