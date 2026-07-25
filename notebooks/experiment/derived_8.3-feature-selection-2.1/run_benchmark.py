"""Evaluate the one frozen challenger on the reused 2023-2025 benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from fs21.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    completion_is_valid,
    invalidate_completion,
    sha256_file,
    write_completion,
)
from fs21.constants import (
    BENCHMARK_REGISTRY_PATH,
    DEVELOPMENT_FREEZE_PATH,
    EXP_DIR,
    GLOBAL_CONFIG_PATH,
    MOE_CONFIG_PATH,
)
from fs21.data import load_control_features, load_development, read_yaml
from fs21.freeze import verify_development_freeze


REQUIRED = [
    "benchmark_predictions.csv.gz",
    "metrics_overall.csv",
    "metrics_by_station.csv",
    "metrics_by_month.csv",
    "metrics_by_station_year.csv",
    "historical_alignment.json",
    "paired_bootstrap_intervals.json",
    "benchmark_claim.json",
    "benchmark_manifest.json",
]


def _read_evaluated(path) -> dict:
    if not path.is_file():
        return {"version": 1, "freezes": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("freezes"), dict):
        raise RuntimeError("benchmark evaluated-model registry is corrupt")
    return payload


def _model_unit(stage, freeze_hash: str, name, build) -> pd.DataFrame:
    unit = stage / "model_units" / freeze_hash / name
    required = ["predictions.csv.gz", "model.json"]
    if completion_is_valid(unit, required):
        return pd.read_csv(unit / "predictions.csv.gz")
    unit.mkdir(parents=True, exist_ok=True)
    invalidate_completion(unit)
    ledger = build()
    atomic_write_csv(ledger, unit / "predictions.csv.gz")
    atomic_write_json(
        unit / "model.json",
        {
            "candidate": str(ledger["candidate"].iloc[0]),
            "model_config_id": str(ledger["model_config_id"].iloc[0]),
            "rows": len(ledger),
            "learner_seed": 42,
        },
    )
    write_completion(unit, required)
    return ledger


def run_benchmark(*, device: str, workers: int) -> None:
    # This is intentionally the first operation that can authorize benchmark
    # imports. No test-data module is imported before the freeze verifies.
    freeze = verify_development_freeze(DEVELOPMENT_FREEZE_PATH)

    from fs21.benchmark import (
        _benchmark_ledger,
        benchmark_metric_tables,
        benchmark_paired_bootstrap,
        load_benchmark_frame,
        make_global_ledger,
        make_moe_ledger,
        verify_historical_alignment,
    )
    from fs21.decision import benchmark_sota_verdict
    from generate_results import _benchmark_report

    config = read_yaml(GLOBAL_CONFIG_PATH)
    moe_config = read_yaml(MOE_CONFIG_PATH)
    registry = read_yaml(BENCHMARK_REGISTRY_PATH)
    if device != str(config["runtime"]["canonical_device"]):
        raise ValueError("benchmark must use the frozen canonical CUDA device")
    if workers != int(config["runtime"]["canonical_workers"]):
        raise ValueError("benchmark command must retain --workers 4")
    if freeze["learner"]["device"] != device:
        raise RuntimeError("benchmark device differs from the development freeze")
    freeze_hash = str(freeze["freeze_sha256"])
    stage = EXP_DIR / "artifacts" / "benchmark"
    evaluated_path = stage / "evaluated_models.json"
    evaluated = _read_evaluated(evaluated_path)
    existing = evaluated["freezes"].get(freeze_hash)
    if existing and existing.get("status") == "complete":
        raise RuntimeError(
            "this frozen model ID has already been evaluated on the benchmark"
        )
    stage.mkdir(parents=True, exist_ok=True)
    evaluated["freezes"][freeze_hash] = {
        "status": "running",
        "started": existing.get("started") if existing else datetime.now(timezone.utc).isoformat(),
        "challenger": freeze["benchmark_challenger"]["model_id"],
    }
    atomic_write_json(evaluated_path, evaluated)
    invalidate_completion(stage)

    development, development_hashes = load_development(config)
    controls, control_provenance = load_control_features(config)
    benchmark, benchmark_hash = load_benchmark_frame(config)
    historical_prediction, alignment = verify_historical_alignment(
        benchmark, registry
    )
    atomic_write_json(stage / "historical_alignment.json", alignment)

    v0_ledger = _model_unit(
        stage,
        freeze_hash,
        "v0_1_3_lite",
        lambda: make_global_ledger(
            development,
            benchmark,
            candidate="V0_1_3_lite",
            features=list(controls["V0"]),
            beta=0.0,
            config=config,
            device=device,
            freeze_hash=freeze_hash,
        ),
    )
    challenger_spec = dict(freeze["benchmark_challenger"])
    if challenger_spec["kind"] == "single_global":
        challenger_ledger = _model_unit(
            stage,
            freeze_hash,
            "challenger_2_1",
            lambda: make_global_ledger(
                development,
                benchmark,
                candidate="challenger_2_1",
                features=list(challenger_spec["features"]),
                beta=float(challenger_spec["beta"]),
                config=config,
                device=device,
                freeze_hash=freeze_hash,
            ),
        )
    elif challenger_spec["kind"] == "moe":
        challenger_ledger = _model_unit(
            stage,
            freeze_hash,
            "challenger_2_1",
            lambda: make_moe_ledger(
                development,
                benchmark,
                candidate="challenger_2_1",
                frozen_moe=dict(challenger_spec["features"]),
                v0_features=list(controls["V0"]),
                router_config=dict(moe_config["router"]),
                config=config,
                device=device,
                freeze_hash=freeze_hash,
            ),
        )
    else:
        raise RuntimeError(f"unknown frozen challenger kind: {challenger_spec['kind']}")
    historical = _benchmark_ledger(
        benchmark,
        historical_prediction,
        candidate="historical_model16",
        feature_hash=str(control_provenance["V0"]["ordered_feature_hash"]),
        actual_count=50,
        beta=0.0,
        model_id="derived_8.3-eval-1.0:model16:5fa48398",
    )
    ledger = pd.concat([v0_ledger, challenger_ledger, historical], ignore_index=True)
    atomic_write_csv(ledger, stage / "benchmark_predictions.csv.gz")
    tables = benchmark_metric_tables(ledger)
    atomic_write_csv(tables["overall"], stage / "metrics_overall.csv")
    atomic_write_csv(tables["station"], stage / "metrics_by_station.csv")
    atomic_write_csv(tables["month"], stage / "metrics_by_month.csv")
    atomic_write_csv(tables["station_year"], stage / "metrics_by_station_year.csv")
    comparison = benchmark_paired_bootstrap(
        ledger,
        "challenger_2_1",
        "historical_model16",
        replicates=int(config["bootstrap"]["replicates"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    atomic_write_json(stage / "paired_bootstrap_intervals.json", comparison)
    challenger_metrics = tables["overall"].loc[
        tables["overall"]["candidate"] == "challenger_2_1"
    ].iloc[0]
    historical_config = dict(registry["historical_best"])
    risks = comparison["comparisons"]
    claim = benchmark_sota_verdict(
        global_gate_passed=bool(
            freeze["selection"]["global_decision"]["global_gate_passed"]
        ),
        challenger_development_eligible=bool(
            challenger_spec["development_eligible"]
        ),
        r2=float(challenger_metrics["R2"]),
        rmse=float(challenger_metrics["RMSE"]),
        historical_r2=float(historical_config["r2"]),
        historical_rmse=float(historical_config["rmse"]),
        r2_margin=float(historical_config["sota_r2_margin"]),
        paired_primary_ci_upper=float(
            risks["station_year_macro_rmse"]["ci_upper"]
        ),
        worst_station_delta=float(risks["worst_station_rmse"]["delta"]),
        worst_station_standard_error=float(
            risks["worst_station_rmse"]["bootstrap_standard_error"]
        ),
        p90_month_delta=float(risks["p90_month_rmse"]["delta"]),
        p90_month_standard_error=float(
            risks["p90_month_rmse"]["bootstrap_standard_error"]
        ),
        alignment_verified=bool(alignment["alignment_verified"]),
    )
    claim.update(
        {
            "freeze_sha256": freeze_hash,
            "challenger_model_id": challenger_spec["model_id"],
            "challenger_kind": challenger_spec["kind"],
            "challenger_development_eligible": challenger_spec[
                "development_eligible"
            ],
            "challenger_R2": float(challenger_metrics["R2"]),
            "challenger_RMSE": float(challenger_metrics["RMSE"]),
            "historical_R2": float(historical_config["r2"]),
            "historical_RMSE": float(historical_config["rmse"]),
            "benchmark_feedback_changed_configuration": False,
        }
    )
    atomic_write_json(stage / "benchmark_claim.json", claim)
    atomic_write_json(
        stage / "benchmark_manifest.json",
        {
            "freeze_path": str(DEVELOPMENT_FREEZE_PATH),
            "freeze_file_sha256": sha256_file(DEVELOPMENT_FREEZE_PATH),
            "freeze_payload_sha256": freeze_hash,
            "development_split_hashes": development_hashes,
            "benchmark_split_sha256": benchmark_hash,
            "benchmark_rows": len(benchmark),
            "benchmark_years": [2023, 2024, 2025],
            "models": ["V0_1_3_lite", "challenger_2_1"],
            "historical_comparator": "historical_model16",
            "learner_seed": 42,
            "device": device,
            "workers": workers,
            "retrospective_test": True,
            "benchmark_reused": True,
            "selection_artifacts_rewritten": False,
        },
    )
    write_completion(stage, REQUIRED)
    evaluated = _read_evaluated(evaluated_path)
    evaluated["freezes"][freeze_hash].update(
        {
            "status": "complete",
            "completed": datetime.now(timezone.utc).isoformat(),
            "benchmark_claim_sha256": sha256_file(stage / "benchmark_claim.json"),
            "completion_sha256": sha256_file(stage / "completion.json"),
        }
    )
    atomic_write_json(evaluated_path, evaluated)
    atomic_write_text(
        EXP_DIR / "BENCHMARK_RESULTS.md",
        _benchmark_report(EXP_DIR / "artifacts" / "development"),
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-benchmark", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.confirm_benchmark:
        parser.error(
            "the reused retrospective benchmark requires explicit --confirm-benchmark"
        )
    run_benchmark(device=args.device, workers=args.workers)
    print(
        json.dumps(
            {
                "status": "complete",
                "retrospective_test": True,
                "benchmark_reused": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
