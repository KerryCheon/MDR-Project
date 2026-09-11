#!/usr/bin/env python3
"""Run the isolated no-SMAP feature-selection stage for salvage 1.1.

Run from the repository root with the notebook uv environment:

    cd notebooks/experiment/derived_8.4-ece-model-salvage-1.1
    uv run --no-sync python run_feature_selection.py --stage all
"""

from __future__ import annotations

import argparse
import hashlib
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


def _write_provenance(config: dict, artifact_dir: Path, data, summary: dict) -> None:
    selected_by_size = summary.get("selected_feature_sizes", {})
    candidate_path = artifact_dir / "candidate_pool.csv"
    config_path = Path(config["_config_path"])
    candidate_count = None
    if candidate_path.exists():
        import pandas as pd

        candidate_count = int(len(pd.read_csv(candidate_path)))
    payload = {
        "status": summary.get("status"),
        "run_mode": "smoke" if config.get("_smoke", False) else "full",
        "selection_pipeline": "local fs20 copy: MI -> ElasticNet -> stability -> wrapper",
        "selection_period": "WA 2023-01-01 through 2025-12-31",
        "fit_scope": "WA trainval only",
        "ece_rows_or_targets_used": False,
        "feature_policy": "case-insensitive removal of every SMAP feature",
        "selected_feature_sizes": {
            str(size): {
                "feature_count": int(record["feature_count"]),
                "feature_hash": str(record["feature_hash"]),
                "features": list(record["features"]),
            }
            for size, record in selected_by_size.items()
        },
        "selected_feature_counts": {
            str(size): len(record["features"])
            for size, record in selected_by_size.items()
        },
        "smap_selected_count": sum(
            "smap" in feature.lower()
            for record in selected_by_size.values()
            for feature in record["features"]
        ),
        "delta_addition_counts": config["search"].get("delta_addition_counts", [0]),
        "candidate_pool_count": candidate_count,
        "candidate_pool_artifact": str(candidate_path),
        "candidate_pool_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest() if candidate_path.exists() else None,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "wa_train_rows": int(len(data.train)),
        "wa_val_rows": int(len(data.val)),
        "wa_test_rows": int(len(data.test)),
    }
    (artifact_dir / "selection_provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _smoke_config(config: dict) -> dict:
    """Produce a bounded smoke configuration without changing the checked-in config."""
    smoke = deepcopy(config)
    smoke["_smoke"] = True
    smoke["audit"]["profiles"] = ["mi300"]
    smoke["audit"]["stability_n_boot"] = 2
    # Smoke retains the variable-size/no-delta contract with a bounded search.
    smoke["search"]["candidate_pool_size"] = 69
    smoke["search"]["max_rounds"] = 1
    smoke["search"]["exact_attempts_per_round"] = 1
    smoke["search"]["global_feature_min"] = 40
    smoke["search"]["global_feature_max"] = 69
    smoke["search"]["normalization_sizes"] = [40, 50, 60, 69]
    smoke["search"]["delta_addition_counts"] = [0]
    smoke["search"]["final_reserve_minutes"] = 1
    smoke["model"]["exact_params"]["n_estimators"] = 50
    smoke["model"]["proxy_params"]["n_estimators"] = 50
    smoke["model"]["exact_params"]["device"] = "cpu"
    smoke["model"]["proxy_params"]["device"] = "cpu"
    smoke["seeds"]["files"] = {
        name: path
        for name, path in smoke["seeds"]["files"].items()
        if name in {"legacy_8_2_forced_bypass", "current_8_4_c0", "current_8_4_c1_true_off"}
    }
    smoke["calibration"]["r2_tolerance"] = 1.0
    smoke["calibration"]["rmse_tolerance"] = 1.0
    # Keep the manifest at the canonical path so the one-seed model smoke test
    # exercises the same selector/model contract as the full run.
    smoke["artifacts"]["directory"] = Path(config["artifacts"]["directory"])
    return smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "feature_selection_config.yaml",
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
    parser.add_argument("--resume", action="store_true", help="Reuse a completed variable-size selector stage.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.smoke:
        config = _smoke_config(config)
    args.workers = args.workers or int(config["search"]["workers"])
    args.deadline_minutes = args.deadline_minutes or int(config["search"]["deadline_minutes"])
    artifact_dir = Path(config["artifacts"]["directory"])
    _write_run_settings(config, artifact_dir, args)

    selected_path = artifact_dir / "selected_features_by_size.json"
    if args.resume and selected_path.exists():
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        selected_sizes = payload.get("feature_sizes", {})
        provenance_path = artifact_dir / "selection_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
        candidate_pool_count = provenance.get("candidate_pool_count")
        expected_pool_count = int(config["search"].get("candidate_pool_size", 0))
        expected_sizes = [str(size) for size in config["search"].get("normalization_sizes", [])]
        if (
            payload.get("status") == "complete"
            and sorted(selected_sizes) == sorted(expected_sizes)
            and all(
                len(selected_sizes[size].get("features", [])) == int(size)
                and not any("smap" in str(feature).lower() for feature in selected_sizes[size].get("features", []))
                for size in expected_sizes
            )
            and not any(
                payload.get("delta_additions", {}).get(str(cluster), [])
                for cluster in (0, 1)
            )
            and provenance.get("status") == "complete"
            and candidate_pool_count == expected_pool_count
            and provenance.get("run_mode") == "full"
        ):
            print(f"[resume] completed feature selection: {selected_path}")
            return 0

    if args.stage == "report-data":
        for filename in ("collapse_audit.csv", "candidate_pool.csv", "search_results.csv", "selected_features_by_size.json"):
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
    _write_provenance(config, artifact_dir, data, summary)
    print(f"Wrote artifacts to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
