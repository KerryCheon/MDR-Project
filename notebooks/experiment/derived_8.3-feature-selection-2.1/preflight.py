"""Validate 2.1 development data, protocol parity, coverage, and runtime."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from fs21.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    completion_is_valid,
    invalidate_completion,
    write_completion,
)
from fs21.constants import EXACT_LEARNER_PARAMS, EXP_DIR, GLOBAL_CONFIG_PATH
from fs21.data import development_coverage, ordered_feature_hash
from fs21.global_pipeline import build_context
from fs21.modeling import validate_exact_learner


DEVELOPMENT_ENTRYPOINTS = (
    "preflight.py",
    "run_global_selection.py",
    "run_station_diagnostics.py",
    "run_moe_diagnostics.py",
    "run_all.py",
)


def _audit_no_test_csv_read() -> list[str]:
    """Reject a literal test.csv argument to a development read function."""
    failures = []
    paths = [EXP_DIR / name for name in DEVELOPMENT_ENTRYPOINTS]
    paths.extend(sorted((EXP_DIR / "fs21").glob("*.py")))
    excluded = {"benchmark.py", "freeze.py"}
    for path in paths:
        if path.name in excluded or not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            function_name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if function_name not in {"read_csv", "_read_stable_csv", "open"}:
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.casefold().endswith("test.csv"):
                        failures.append(f"{path.name}:{node.lineno}")
    if failures:
        raise ValueError(f"development test.csv read paths found: {failures}")
    return [str(path.relative_to(EXP_DIR)) for path in paths if path.is_file()]


def _validate_device(device: str) -> dict:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([0.0, 1.0, 1.5, 2.0], dtype=float)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1,
        max_depth=1,
        tree_method="hist",
        n_jobs=1,
        device=device,
        random_state=42,
    )
    model.fit(X, y)
    prediction = model.predict(X)
    return {
        "device": device,
        "tiny_fit_completed": True,
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }


def run_preflight(
    *, device: str, workers: int, smoke: bool = False, allow_worker_resume: bool = False
) -> Path:
    context = build_context(device=device, workers=workers, smoke=smoke)
    canonical_workers = int(context.config["runtime"]["canonical_workers"])
    if not smoke and not allow_worker_resume and int(workers) != canonical_workers:
        raise ValueError(
            f"canonical development requires {canonical_workers} workers; got {workers}"
        )
    validate_exact_learner(context.config)
    if dict(context.config["learner"]) != EXACT_LEARNER_PARAMS:
        raise AssertionError("exact learner validation unexpectedly drifted")
    audited = _audit_no_test_csv_read()
    device_result = _validate_device(device)
    coverage = development_coverage(context.frame, context.config)
    if (coverage["row_count"] <= 0).any():
        raise ValueError("preflight found a zero-observation station-year")
    stage = context.artifact_root / "stages" / "01_preflight"
    required = ["preflight.json", "development_coverage.csv", "predictors.json"]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    atomic_write_csv(coverage, stage / "development_coverage.csv")
    atomic_write_json(
        stage / "predictors.json",
        {
            "predictors": context.universe,
            "count": len(context.universe),
            "ordered_predictor_hash": ordered_feature_hash(context.universe),
            "canonical_count": 496,
            "smoke_subset": bool(smoke),
        },
    )
    atomic_write_json(
        stage / "preflight.json",
        {
            "experiment": context.config["experiment"]["name"],
            "device": device,
            "workers": int(workers),
            "smoke": bool(smoke),
            "split_hashes": context.split_hashes,
            "development_rows": len(context.frame),
            "stations": sorted(context.frame["station_id"].unique().tolist()),
            "years": sorted(context.frame["_year"].unique().tolist()),
            "predictor_count": len(context.universe),
            "v0_count": len(context.controls["V0"]),
            "exact_learner": dict(context.config["learner"]),
            "native_xgboost_missing_values": True,
            "global_feature_imputation": None,
            "development_splits_read": ["train", "val"],
            "benchmark_split_read": False,
            "audited_development_sources": audited,
            "device_validation": device_result,
        },
    )
    write_completion(stage, required)
    return stage


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    path = run_preflight(
        device=args.device,
        workers=args.workers,
        smoke=args.smoke,
    )
    print(json.dumps({"status": "ok", "stage": str(path)}, indent=2))


if __name__ == "__main__":
    main()

