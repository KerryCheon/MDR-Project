"""Diagnose nested candidate sets with the locked final-model architecture.

The 2021-2022 outer results are valid development diagnostics.  The 2023-2025
results are explicitly retrospective because that test split was consumed by
the original 2.2 run; they must not be used for a new unbiased SOTA claim.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.metrics import r2_score

from artifact_state import (
    artifact_is_complete,
    atomic_write_csv,
    atomic_write_json,
    capture_source_state,
    invalidate_completion,
    write_completion_marker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
TARGET = "soil_moisture_5cm"
SEED = 42
DIAGNOSTIC_REQUIRED_FILES = ("global_candidates.csv", "manifest.json")


def _probe_device() -> str:
    requested = os.environ.get("XGB_DEVICE", "").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested not in {"", "cuda"}:
        raise ValueError("XGB_DEVICE must be 'cpu' or 'cuda'")
    try:
        model = xgb.XGBRegressor(n_estimators=1, device="cuda", verbosity=0)
        model.fit(np.asarray([[0.0], [1.0]]), np.asarray([0.0, 1.0]))
        return "cuda"
    except Exception:
        return "cpu"


def _weights(dates: pd.Series, beta: float) -> np.ndarray | None:
    if float(beta) == 0.0:
        return None
    years = pd.to_datetime(dates).dt.year.to_numpy(dtype=float)
    values = np.exp(float(beta) * (years - float(np.max(years))))
    return values / float(np.mean(values))


def _evaluate(
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
    beta: float,
    params: dict,
) -> dict:
    train_target = pd.to_numeric(train[TARGET], errors="coerce")
    score_target = pd.to_numeric(score[TARGET], errors="coerce")
    train_ok = train_target.notna()
    score_ok = score_target.notna()
    train_features = (
        train.loc[train_ok, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    score_features = (
        score.loc[score_ok, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    model = xgb.XGBRegressor(**params)
    model.fit(
        train_features,
        train_target.loc[train_ok],
        sample_weight=_weights(train.loc[train_ok, "date"], beta),
    )
    prediction = np.asarray(model.predict(score_features)).ravel()
    truth = score_target.loc[score_ok].to_numpy(dtype=float)
    residual = truth - prediction
    return {
        "R2": float(r2_score(truth, prediction)),
        "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "MAE": float(np.mean(np.abs(residual))),
        "Bias": float(np.mean(residual)),
    }


def _candidate_sets(
    dataset: str,
    artifact_set: str,
    *,
    artifact_root: Path | None = None,
) -> list[list[str]]:
    root = artifact_root or EXP_DIR / "artifacts"
    crossed = artifact_set in {
        "crossed_candidates_locked_outer",
        "progressive_crossed_locked_outer",
    }
    filename = "outer_selection.json" if crossed else "inner_selection.json"
    path = root / artifact_set / dataset / "global" / filename
    if not artifact_is_complete(path.parent, [filename]):
        raise FileNotFoundError(f"candidate source artifact is incomplete: {path.parent}")
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    candidate_key = "candidate_summaries" if crossed else "selection_path"
    candidates = []
    seen = set()
    for candidate in payload[candidate_key]:
        features = list(candidate["features"])
        key = tuple(features)
        if key not in seen:
            candidates.append(features)
            seen.add(key)
    return candidates


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-set",
        choices=(
            "nested",
            "crossed_candidates_locked_outer",
            "progressive_crossed_locked_outer",
        ),
        default="nested",
    )
    args = parser.parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    with open(EXP_DIR / "config.yaml", encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    source_state = capture_source_state(PROJECT_ROOT, EXP_DIR)
    params = dict(base_config["evaluation"]["xgb_params"])
    params["device"] = _probe_device()

    tasks = []
    candidate_counts = {}
    for dataset in ("derived_8.0", "derived_8.2"):
        global_dir = EXP_DIR / "artifacts" / args.artifact_set / dataset / "global"
        if not global_dir.exists():
            continue
        split_dir = PROJECT_ROOT / "data/splits" / dataset
        train = pd.read_csv(split_dir / "train.csv")
        val = pd.read_csv(split_dir / "val.csv")
        test = pd.read_csv(split_dir / "test.csv")
        trainval = pd.concat([train, val], ignore_index=True)
        candidate_sets = _candidate_sets(dataset, args.artifact_set)
        candidate_counts[dataset] = len(candidate_sets)
        for candidate_index, features in enumerate(candidate_sets):
            for beta in (0.0, 0.2):
                tasks.append(
                    (
                        dataset,
                        train,
                        val,
                        trainval,
                        test,
                        candidate_index,
                        features,
                        beta,
                    )
                )

    def _run_task(task):
        (
            dataset,
            train,
            val,
            trainval,
            test,
            candidate_index,
            features,
            beta,
        ) = task
        outer = _evaluate(train, val, features, beta, params)
        retrospective = _evaluate(trainval, test, features, beta, params)
        return {
            "dataset": dataset,
            "candidate_index": candidate_index,
            "n_features": len(features),
            "beta": beta,
            "outer_R2": outer["R2"],
            "outer_RMSE": outer["RMSE"],
            "outer_MAE": outer["MAE"],
            "outer_Bias": outer["Bias"],
            "retrospective_test_R2": retrospective["R2"],
            "retrospective_test_RMSE": retrospective["RMSE"],
            "retrospective_test_MAE": retrospective["MAE"],
            "retrospective_test_Bias": retrospective["Bias"],
        }

    parallel_workers = max(1, int(os.environ.get("XGB_PARALLEL_WORKERS", "16")))
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        rows = list(executor.map(_run_task, tasks))

    output_dir = EXP_DIR / "artifacts" / args.artifact_set / "candidate_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    invalidate_completion(output_dir)
    results = pd.DataFrame(rows).sort_values(
        ["dataset", "beta", "n_features", "candidate_index"]
    )
    atomic_write_csv(results, output_dir / "global_candidates.csv")
    manifest = {
        "device": params["device"],
        "parallel_workers": parallel_workers,
        "artifact_set": args.artifact_set,
        "candidate_counts": candidate_counts,
        "source_state": source_state,
        "outer_period": "2021-2022",
        "retrospective_test_period": "2023-2025",
        "unbiased_sota_eligible": False,
        "reason": "The original 2.2 run already consumed the test split.",
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    write_completion_marker(
        output_dir,
        DIAGNOSTIC_REQUIRED_FILES,
        source_state=source_state,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
