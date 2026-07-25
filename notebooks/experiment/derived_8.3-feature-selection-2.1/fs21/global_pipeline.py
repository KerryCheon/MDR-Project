"""Resumable global-selection stages for the 2.1 development protocol."""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .artifacts import (
    atomic_write_csv,
    atomic_write_json,
    completion_is_valid,
    invalidate_completion,
    stable_json_hash,
    write_completion,
)
from .bootstrap import paired_hierarchical_bootstrap
from .constants import EXP_DIR, GLOBAL_CONFIG_PATH
from .data import (
    development_coverage,
    load_control_features,
    load_development,
    ordered_feature_hash,
    predictor_columns,
    read_yaml,
)
from .decision import choose_beta, choose_global_candidate
from .folds import (
    FoldTask,
    build_inner_folds,
    build_outer_tasks,
    task_manifest_rows,
)
from .ledger import (
    assert_candidate_coverage,
    prediction_rows,
    relabel_candidate,
    validate_ledger,
)
from .metrics import primary_risk, secondary_metric_tables
from .ranking import (
    complete_order_for_endpoint,
    consensus_features,
    pruning_path,
)


@dataclass
class ExperimentContext:
    config: dict
    frame: pd.DataFrame
    split_hashes: dict
    universe: list[str]
    controls: dict[str, list[str]]
    control_provenance: dict
    tasks: list[FoldTask]
    coverage: pd.DataFrame
    partition_mappings: dict[int, dict[str, int]]
    artifact_root: Path
    device: str
    workers: int
    smoke: bool


def _smoke_frame(frame: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    limit = int(dict(dict(config["runtime"])["smoke"])["row_limit_per_station_year"])
    data = dict(config["data"])
    return (
        frame.sort_values([str(data["station_col"]), "_year", str(data["time_col"])])
        .groupby([str(data["station_col"]), "_year"], sort=True, group_keys=False)
        .head(limit)
        .reset_index(drop=True)
    )


def _smoke_config(config: dict) -> dict:
    output = deepcopy(config)
    smoke = dict(dict(output["runtime"])["smoke"])
    output["folds"]["outer_origins"] = list(smoke["origins"])
    output["folds"]["station_partitions"] = int(smoke["station_partitions"])
    output["folds"]["partition_seeds"] = list(smoke["partition_seeds"])
    output["folds"]["station_time_learner_seeds"] = list(smoke["learner_seeds"])
    output["folds"]["forward_time_learner_seeds"] = list(smoke["learner_seeds"])
    output["folds"]["minimum_train_rows"] = int(smoke["minimum_train_rows"])
    output["folds"]["minimum_validation_rows"] = int(
        smoke["minimum_validation_rows"]
    )
    output["ranking"]["endpoint_counts"] = list(smoke["endpoint_counts"])
    output["ranking"]["permutation_repeats"] = int(smoke["permutation_repeats"])
    output["ranking"]["screen_permutation_repeats"] = 1
    output["bootstrap"]["replicates"] = int(smoke["bootstrap_replicates"])
    return output


def build_context(*, device: str, workers: int, smoke: bool) -> ExperimentContext:
    canonical_config = read_yaml(GLOBAL_CONFIG_PATH)
    frame, split_hashes = load_development(canonical_config)
    universe = predictor_columns(frame, canonical_config)
    controls, control_provenance = load_control_features(canonical_config)
    config = _smoke_config(canonical_config) if smoke else canonical_config
    if smoke:
        frame = _smoke_frame(frame, config)
        smoke_limit = int(dict(dict(config["runtime"])["smoke"])["predictor_limit"])
        v0 = controls["V0"]
        extras = [feature for feature in universe if feature not in set(v0)]
        universe = (v0 + extras)[:smoke_limit]
    missing_controls = {
        name: sorted(set(features).difference(frame.columns))
        for name, features in controls.items()
        if set(features).difference(frame.columns)
    }
    if missing_controls:
        raise ValueError(f"control feature sources are incompatible: {missing_controls}")
    tasks, coverage, mappings = build_outer_tasks(frame, config)
    root = EXP_DIR / "artifacts" / ("smoke" if smoke else "development")
    return ExperimentContext(
        config=config,
        frame=frame,
        split_hashes=split_hashes,
        universe=universe,
        controls=controls,
        control_provenance=control_provenance,
        tasks=tasks,
        coverage=coverage,
        partition_mappings=mappings,
        artifact_root=root,
        device=device,
        workers=int(workers),
        smoke=smoke,
    )


def _stage_dir(context: ExperimentContext, name: str) -> Path:
    return context.artifact_root / "stages" / name


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _map_units(
    function: Callable,
    units: list,
    *,
    workers: int,
) -> list:
    if workers <= 1:
        return [function(unit) for unit in units]
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        return list(executor.map(function, units))


def run_fold_manifest_stage(context: ExperimentContext) -> Path:
    stage = _stage_dir(context, "02_fold_manifests")
    required = ["fold_manifest.csv", "coverage_manifest.csv", "partitions.json"]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    manifest = task_manifest_rows(context.frame, context.tasks)
    atomic_write_csv(manifest, stage / "fold_manifest.csv")
    atomic_write_csv(context.coverage, stage / "coverage_manifest.csv")
    atomic_write_json(
        stage / "partitions.json",
        {
            str(seed): mapping
            for seed, mapping in sorted(context.partition_mappings.items())
        },
    )
    write_completion(stage, required)
    return stage


def _control_specs(context: ExperimentContext) -> dict[str, list[str]]:
    controls = dict(context.controls)
    controls["all_predictors"] = list(context.universe)
    return controls


def _evaluate_one(
    context: ExperimentContext,
    *,
    stage: Path,
    task: FoldTask,
    candidate: str,
    features: list[str],
    path_source: str,
    endpoint: int | None,
    beta: float,
) -> Path:
    unit = stage / "units" / _slug(
        f"{candidate}__{task.fold_id}__beta_{float(beta):g}"
    )
    required = ["predictions.csv.gz", "unit.json"]
    if completion_is_valid(unit, required):
        return unit / "predictions.csv.gz"
    unit.mkdir(parents=True, exist_ok=True)
    invalidate_completion(unit)
    ledger = prediction_rows(
        context.frame,
        task,
        features,
        candidate=candidate,
        path_source=path_source,
        endpoint=endpoint,
        beta=beta,
        config=context.config,
        device=context.device,
        smoke=context.smoke,
    )
    atomic_write_csv(ledger, unit / "predictions.csv.gz")
    atomic_write_json(
        unit / "unit.json",
        {
            "candidate": candidate,
            "fold_id": task.fold_id,
            "fold_family": task.family,
            "origin": task.origin,
            "partition_seed": task.partition_seed,
            "learner_seed": task.learner_seed,
            "beta": float(beta),
            "features": features,
            "ordered_feature_hash": ordered_feature_hash(features),
        },
    )
    write_completion(unit, required)
    return unit / "predictions.csv.gz"


def _combine_ledgers(paths: Iterable[Path]) -> pd.DataFrame:
    paths = list(paths)
    if not paths:
        raise ValueError("cannot combine an empty ledger set")
    return validate_ledger(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))


def run_control_stage(context: ExperimentContext) -> Path:
    stage = _stage_dir(context, "03_control_ledgers")
    required = ["oof_predictions.csv.gz", "control_features.json"]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    specs = _control_specs(context)
    units = [
        (task, candidate, features)
        for task in context.tasks
        for candidate, features in specs.items()
    ]

    def run(unit):
        task, candidate, features = unit
        return _evaluate_one(
            context,
            stage=stage,
            task=task,
            candidate=candidate,
            features=features,
            path_source="fixed_control",
            endpoint=len(features),
            beta=0.0,
        )

    paths = _map_units(run, units, workers=context.workers)
    ledger = _combine_ledgers(paths)
    atomic_write_csv(ledger, stage / "oof_predictions.csv.gz")
    atomic_write_json(
        stage / "control_features.json",
        {
            "controls": {
                name: {
                    "features": features,
                    "count": len(features),
                    "ordered_feature_hash": ordered_feature_hash(features),
                    "promotable": False,
                    **context.control_provenance.get(name, {}),
                }
                for name, features in specs.items()
            },
            "all_predictor_count_interpretation": (
                "smoke predictor subset" if context.smoke else "all 496 numeric predictors"
            ),
        },
    )
    write_completion(stage, required)
    return stage


def _base_screen_tasks(tasks: list[FoldTask]) -> list[FoldTask]:
    return [
        task
        for task in tasks
        if task.learner_seed == 42
        and (task.family == "forward_time" or task.partition_seed == 42)
    ]


def _rank_unit(
    context: ExperimentContext,
    *,
    stage: Path,
    task: FoldTask,
    source: str,
    method: str,
    permutation_repeats: int,
) -> Path:
    unit = stage / "rank_units" / _slug(
        f"{method}__{source}__{task.fold_id}"
    )
    required = ["path.json"]
    if completion_is_valid(unit, required):
        return unit / "path.json"
    unit.mkdir(parents=True, exist_ok=True)
    invalidate_completion(unit)
    outer_training = context.frame.iloc[list(task.train_index)].reset_index(drop=True)
    partition_seed = 42 if task.partition_seed is None else int(task.partition_seed)
    folds = build_inner_folds(
        outer_training,
        context.config,
        family=source,
        partition_seed=partition_seed,
    )
    path = pruning_path(
        outer_training,
        list(context.universe),
        folds,
        method=method,
        endpoints=[int(value) for value in context.config["ranking"]["endpoint_counts"]],
        config=context.config,
        learner_seed=task.learner_seed,
        device=context.device,
        permutation_repeats=int(permutation_repeats),
        smoke=context.smoke,
    )
    path.update(
        {
            "outer_fold_id": task.fold_id,
            "outer_family": task.family,
            "outer_origin": task.origin,
            "outer_held_stations": list(task.held_stations),
            "candidate_generation_row_count": len(outer_training),
            "candidate_generation_row_keys_sha256": stable_json_hash(
                outer_training["_row_key"].tolist()
            ),
            "path_source": source,
            "learner_seed": task.learner_seed,
            "station_partition_seed": task.partition_seed,
        }
    )
    atomic_write_json(unit / "path.json", path)
    write_completion(unit, required)
    return unit / "path.json"


def _path_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_id(list_form: str, source: str, endpoint: int) -> str:
    return f"{list_form}__{source}__k{int(endpoint)}"


def _screen_candidate_id(
    method: str, list_form: str, source: str, endpoint: int
) -> str:
    return f"{method}__{_candidate_id(list_form, source, endpoint)}"


def _candidate_features(selected: list[str], v0: list[str], list_form: str) -> list[str]:
    if list_form == "selected_k":
        return list(selected)
    if list_form == "v0_union_selected_k":
        seen = set(v0)
        return list(v0) + [feature for feature in selected if feature not in seen]
    raise ValueError(f"unknown list form: {list_form}")


def _evaluate_paths(
    context: ExperimentContext,
    *,
    stage: Path,
    path_records: list[tuple[FoldTask, str, str, Path]],
    screen: bool,
) -> tuple[pd.DataFrame, dict]:
    specs = []
    feature_manifest = {}
    list_forms = [str(value) for value in context.config["ranking"]["list_forms"]]
    for task, method, source, path_file in path_records:
        payload = _path_payload(path_file)
        for endpoint_text, selected in payload["endpoints"].items():
            endpoint = int(endpoint_text)
            for list_form in list_forms:
                features = _candidate_features(selected, context.controls["V0"], list_form)
                candidate = (
                    _screen_candidate_id(method, list_form, source, endpoint)
                    if screen
                    else _candidate_id(list_form, source, endpoint)
                )
                specs.append((task, candidate, features, source, endpoint))
                feature_manifest.setdefault(candidate, {})[task.fold_id] = {
                    "features": features,
                    "actual_count": len(features),
                    "ordered_feature_hash": ordered_feature_hash(features),
                }

    def run(spec):
        task, candidate, features, source, endpoint = spec
        return _evaluate_one(
            context,
            stage=stage,
            task=task,
            candidate=candidate,
            features=features,
            path_source=source,
            endpoint=endpoint,
            beta=0.0,
        )

    paths = _map_units(run, specs, workers=context.workers)
    return _combine_ledgers(paths), feature_manifest


def _mean_pairwise_jaccard(feature_rows: list[list[str]]) -> float:
    if len(feature_rows) < 2:
        return 1.0
    scores = []
    for left, right in combinations(feature_rows, 2):
        left_set, right_set = set(left), set(right)
        scores.append(len(left_set & right_set) / len(left_set | right_set))
    return float(np.mean(scores))


def run_path_screen_stage(context: ExperimentContext) -> Path:
    stage = _stage_dir(context, "04_path_screen")
    required = [
        "screen_oof_predictions.csv.gz",
        "path_screen_decision.json",
        "direct_progressive_overlap.csv",
        "candidate_features.json",
    ]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    tasks = _base_screen_tasks(context.tasks)
    rank_specs = [
        (task, method, str(source))
        for task in tasks
        for method in ("direct", "progressive")
        for source in context.config["ranking"]["path_sources"]
    ]

    def rank_screen_unit(spec):
        task, method, source = spec
        path = _rank_unit(
            context,
            stage=stage,
            task=task,
            source=source,
            method=method,
            permutation_repeats=int(
                context.config["ranking"]["screen_permutation_repeats"]
            ),
        )
        return task, method, source, path

    path_records = _map_units(
        rank_screen_unit,
        rank_specs,
        workers=context.workers,
    )
    screen_ledger, feature_manifest = _evaluate_paths(
        context,
        stage=stage,
        path_records=path_records,
        screen=True,
    )
    control = pd.read_csv(
        _stage_dir(context, "03_control_ledgers") / "oof_predictions.csv.gz"
    )
    screen_fold_ids = {task.fold_id for task in tasks}
    v0 = control.loc[
        (control["candidate"] == "V0") & control["fold_id"].isin(screen_fold_ids)
    ]
    combined = validate_ledger(pd.concat([v0, screen_ledger], ignore_index=True))
    atomic_write_csv(combined, stage / "screen_oof_predictions.csv.gz")
    atomic_write_json(stage / "candidate_features.json", feature_manifest)
    risks = []
    for candidate in sorted(screen_ledger["candidate"].unique()):
        row = primary_risk(combined, candidate, beta=0.0)
        risks.append(
            {
                "candidate": candidate,
                "method": candidate.split("__", maxsplit=1)[0],
                "combined_primary_rmse": row["combined_primary_rmse"],
                "forward_time_rmse": row["forward_time_rmse"],
                "station_time_rmse": row["station_time_rmse"],
            }
        )
    best = {
        method: min(
            [row for row in risks if row["method"] == method],
            key=lambda row: (row["combined_primary_rmse"], row["candidate"]),
        )
        for method in ("direct", "progressive")
    }
    comparison = paired_hierarchical_bootstrap(
        combined,
        best["direct"]["candidate"],
        best["progressive"]["candidate"],
        beta=0.0,
        replicates=int(context.config["bootstrap"]["replicates"]),
        seed=int(context.config["bootstrap"]["seed"]),
    )
    selected_method = (
        "direct"
        if comparison["comparisons"]["combined_primary_rmse"]["ci_upper"] < 0.0
        else "progressive"
    )
    decision = {
        "selected_method": selected_method,
        "direct_selected_only_if_paired_upper_ci_below_zero": True,
        "best_by_method": best,
        "paired_direct_minus_progressive": comparison,
        "candidate_risks": risks,
        "permutation_repeats": int(
            context.config["ranking"]["screen_permutation_repeats"]
        ),
    }
    atomic_write_json(stage / "path_screen_decision.json", decision)
    overlaps = []
    indexed = {
        (task.fold_id, method, source): _path_payload(path)
        for task, method, source, path in path_records
    }
    for task in tasks:
        for source in context.config["ranking"]["path_sources"]:
            direct = indexed[(task.fold_id, "direct", str(source))]
            progressive = indexed[(task.fold_id, "progressive", str(source))]
            for endpoint in sorted(set(direct["endpoints"]) & set(progressive["endpoints"])):
                left = set(direct["endpoints"][endpoint])
                right = set(progressive["endpoints"][endpoint])
                overlaps.append(
                    {
                        "fold_id": task.fold_id,
                        "outer_origin": task.origin,
                        "outer_family": task.family,
                        "path_source": source,
                        "endpoint": int(endpoint),
                        "jaccard": len(left & right) / len(left | right),
                    }
                )
    atomic_write_csv(pd.DataFrame(overlaps), stage / "direct_progressive_overlap.csv")
    write_completion(stage, required)
    return stage


def _selected_method(context: ExperimentContext) -> str:
    path = _stage_dir(context, "04_path_screen") / "path_screen_decision.json"
    return str(json.loads(path.read_text(encoding="utf-8"))["selected_method"])


def run_robust_generation_stage(context: ExperimentContext) -> Path:
    stage = _stage_dir(context, "05_robust_candidate_generation")
    required = ["path_manifest.json", "feature_rank_summary.csv"]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    method = _selected_method(context)
    units = [
        (task, str(source))
        for task in context.tasks
        for source in context.config["ranking"]["path_sources"]
    ]

    def run(unit):
        task, source = unit
        return (
            task,
            source,
            _rank_unit(
                context,
                stage=stage,
                task=task,
                source=source,
                method=method,
                permutation_repeats=int(context.config["ranking"]["permutation_repeats"]),
            ),
        )

    records = _map_units(run, units, workers=context.workers)
    manifest = []
    summary_rows = []
    for task, source, path in records:
        payload = _path_payload(path)
        manifest.append(
            {
                "fold_id": task.fold_id,
                "fold_family": task.family,
                "outer_origin": task.origin,
                "path_source": source,
                "method": method,
                "path": str(path.relative_to(context.artifact_root)),
            }
        )
        for step in payload["steps"]:
            for row in step["ranking"]:
                summary_rows.append(
                    {
                        "fold_id": task.fold_id,
                        "fold_family": task.family,
                        "outer_origin": task.origin,
                        "path_source": source,
                        "method": method,
                        "reduction_size": step["size"],
                        "is_endpoint": step["endpoint"],
                        **row,
                    }
                )
    atomic_write_json(
        stage / "path_manifest.json",
        {"selected_method": method, "paths": manifest},
    )
    atomic_write_csv(pd.DataFrame(summary_rows), stage / "feature_rank_summary.csv")
    write_completion(stage, required)
    return stage


def _robust_path_records(
    context: ExperimentContext,
) -> list[tuple[FoldTask, str, str, Path]]:
    stage = _stage_dir(context, "05_robust_candidate_generation")
    manifest = json.loads((stage / "path_manifest.json").read_text(encoding="utf-8"))
    tasks = {task.fold_id: task for task in context.tasks}
    return [
        (
            tasks[row["fold_id"]],
            row["method"],
            row["path_source"],
            context.artifact_root / row["path"],
        )
        for row in manifest["paths"]
    ]


def run_candidate_oof_stage(context: ExperimentContext) -> Path:
    stage = _stage_dir(context, "06_candidate_oof")
    required = ["oof_predictions.csv.gz", "candidate_features.json"]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    ledger, feature_manifest = _evaluate_paths(
        context,
        stage=stage,
        path_records=_robust_path_records(context),
        screen=False,
    )
    controls = pd.read_csv(
        _stage_dir(context, "03_control_ledgers") / "oof_predictions.csv.gz"
    )
    combined = validate_ledger(pd.concat([controls, ledger], ignore_index=True))
    for candidate in sorted(ledger["candidate"].unique()):
        assert_candidate_coverage(combined, candidate, reference="V0")
    atomic_write_csv(combined, stage / "oof_predictions.csv.gz")
    atomic_write_csv(combined, context.artifact_root / "oof_predictions.csv.gz")
    atomic_write_json(stage / "candidate_features.json", feature_manifest)
    write_completion(stage, required)
    return stage


def _candidate_stability(manifest: dict, candidate: str) -> float:
    features = [row["features"] for row in manifest[candidate].values()]
    return _mean_pairwise_jaccard(features)


def run_global_decision_stage(context: ExperimentContext) -> Path:
    stage = _stage_dir(context, "07_global_decision")
    required = [
        "global_promotion_decision.json",
        "candidate_metrics.csv",
        "paired_bootstrap_intervals.json",
        "overall_metrics.csv",
    ]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    ledger = pd.read_csv(
        _stage_dir(context, "06_candidate_oof") / "oof_predictions.csv.gz"
    )
    feature_manifest = json.loads(
        (
            _stage_dir(context, "06_candidate_oof") / "candidate_features.json"
        ).read_text(encoding="utf-8")
    )
    candidate_summaries = []
    bootstrap_payload = {}
    for candidate in sorted(feature_manifest):
        risk = primary_risk(ledger, candidate, beta=0.0)
        comparison = paired_hierarchical_bootstrap(
            ledger,
            candidate,
            "V0",
            beta=0.0,
            replicates=int(context.config["bootstrap"]["replicates"]),
            seed=int(context.config["bootstrap"]["seed"]),
        )
        bootstrap_payload[candidate] = comparison
        form, source, endpoint_text = candidate.split("__")
        counts = [
            int(row["actual_count"]) for row in feature_manifest[candidate].values()
        ]
        summary = {
            "candidate": candidate,
            "list_form": form,
            "path_source": source,
            "endpoint": int(endpoint_text.removeprefix("k")),
            "actual_count": max(counts),
            "actual_count_min": min(counts),
            "actual_count_max": max(counts),
            "actual_count_interpretation": "maximum across causal outer-task lists",
            "selection_stability": _candidate_stability(feature_manifest, candidate),
            "combined_primary_rmse": risk["combined_primary_rmse"],
            "forward_time_rmse": risk["forward_time_rmse"],
            "station_time_rmse": risk["station_time_rmse"],
            "coverage_matches_v0": True,
            "promotable": True,
            "bootstrap": comparison,
        }
        candidate_summaries.append(summary)
    decision = choose_global_candidate(candidate_summaries)
    metric_rows = [
        {
            key: value
            for key, value in summary.items()
            if key not in {"bootstrap"}
        }
        for summary in decision["candidate_summaries"]
    ]
    atomic_write_json(stage / "global_promotion_decision.json", decision)
    atomic_write_json(
        context.artifact_root / "global_promotion_decision.json", decision
    )
    atomic_write_json(stage / "paired_bootstrap_intervals.json", bootstrap_payload)
    atomic_write_csv(pd.DataFrame(metric_rows), stage / "candidate_metrics.csv")

    overall_rows = []
    for candidate in sorted(ledger["candidate"].unique()):
        tables = secondary_metric_tables(
            ledger,
            candidate,
            beta=0.0,
            variance_epsilon=float(context.config["decision"]["r2_zero_variance_epsilon"]),
        )
        row = tables["overall"].iloc[0].to_dict()
        row["candidate"] = candidate
        overall_rows.append(row)
    atomic_write_csv(pd.DataFrame(overall_rows), stage / "overall_metrics.csv")
    write_completion(stage, required)
    return stage


def _decision(context: ExperimentContext) -> dict:
    return json.loads(
        (
            _stage_dir(context, "07_global_decision")
            / "global_promotion_decision.json"
        ).read_text(encoding="utf-8")
    )


def _features_for_candidate_task(
    context: ExperimentContext,
    candidate: str,
    task: FoldTask,
) -> list[str]:
    if candidate == "V0":
        return list(context.controls["V0"])
    manifest = json.loads(
        (
            _stage_dir(context, "06_candidate_oof") / "candidate_features.json"
        ).read_text(encoding="utf-8")
    )
    return list(manifest[candidate][task.fold_id]["features"])


def run_beta_stage(context: ExperimentContext) -> Path:
    stage = _stage_dir(context, "08_beta_decision")
    required = ["beta_oof_predictions.csv.gz", "beta_decision.json"]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    global_decision = _decision(context)
    selected = str(global_decision["winner"])
    base_ledger = pd.read_csv(
        _stage_dir(context, "06_candidate_oof") / "oof_predictions.csv.gz"
    )
    base = base_ledger.loc[
        (base_ledger["candidate"] == selected) & (base_ledger["beta"] == 0.0)
    ].copy()
    base = relabel_candidate(base, "beta_0_0")
    units = [
        (task, _features_for_candidate_task(context, selected, task))
        for task in context.tasks
    ]

    def run(unit):
        task, features = unit
        return _evaluate_one(
            context,
            stage=stage,
            task=task,
            candidate="beta_0_2",
            features=features,
            path_source=(
                "fixed_control" if selected == "V0" else selected.split("__")[1]
            ),
            endpoint=None if selected == "V0" else int(selected.rsplit("k", 1)[1]),
            beta=0.2,
        )

    paths = _map_units(run, units, workers=context.workers)
    recent = _combine_ledgers(paths)
    raw = validate_ledger(pd.concat([base, recent], ignore_index=True))
    atomic_write_csv(raw, stage / "beta_oof_predictions.csv.gz")
    comparison_ledger = raw.copy()
    comparison_ledger["beta"] = 0.0
    comparison = paired_hierarchical_bootstrap(
        comparison_ledger,
        "beta_0_2",
        "beta_0_0",
        beta=0.0,
        replicates=int(context.config["bootstrap"]["replicates"]),
        seed=int(context.config["bootstrap"]["seed"]),
    )
    decision = choose_beta(comparison)
    decision["selected_candidate"] = selected
    decision["beta_arms_pooled_as_folds"] = False
    atomic_write_json(stage / "beta_decision.json", decision)
    write_completion(stage, required)
    return stage


def _year_specific_path(
    context: ExperimentContext,
    *,
    year: int,
    method: str,
    source: str,
) -> dict:
    frame = context.frame.loc[context.frame["_year"] <= int(year)].reset_index(drop=True)
    config = deepcopy(context.config)
    config["folds"]["inner_max_validation_years"] = 1
    folds = build_inner_folds(
        frame,
        config,
        family=source,
        partition_seed=42,
    )
    if {fold.origin for fold in folds} != {int(year)}:
        raise AssertionError(f"consensus ranking did not end in {year}")
    return pruning_path(
        frame,
        list(context.universe),
        folds,
        method=method,
        endpoints=[int(value) for value in context.config["ranking"]["endpoint_counts"]],
        config=context.config,
        learner_seed=42,
        device=context.device,
        permutation_repeats=int(context.config["ranking"]["permutation_repeats"]),
        smoke=context.smoke,
    )


def _consensus_path_unit(
    context: ExperimentContext,
    *,
    stage: Path,
    year: int,
    method: str,
    source: str,
) -> dict:
    unit = stage / "rank_units" / _slug(
        f"year_{int(year)}__{method}__{source}"
    )
    required = ["path.json"]
    if completion_is_valid(unit, required):
        return _path_payload(unit / "path.json")
    unit.mkdir(parents=True, exist_ok=True)
    invalidate_completion(unit)
    path = _year_specific_path(
        context,
        year=int(year),
        method=method,
        source=source,
    )
    atomic_write_json(unit / "path.json", path)
    write_completion(unit, required)
    return path


def _rank_correlation(left: list[str], right: list[str]) -> float:
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return float("nan")
    left_rank = np.asarray([left.index(feature) for feature in common], dtype=float)
    right_rank = np.asarray([right.index(feature) for feature in common], dtype=float)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def run_consensus_stage(context: ExperimentContext) -> Path:
    stage = _stage_dir(context, "09_consensus")
    required = [
        "candidate_features.json",
        "consensus_support.csv",
        "origin_lists.json",
        "stability_summary.csv",
    ]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    global_decision = _decision(context)
    challenger = (
        global_decision["winner"]
        if global_decision["global_gate_passed"]
        else global_decision["best_failed_candidate"]
    )
    if challenger is None or challenger == "V0":
        challenger_features = list(context.controls["V0"])
        origin_payload = []
        support = pd.DataFrame(
            {
                "feature": challenger_features,
                "selection_frequency": 3,
                "median_percentile_rank": np.arange(1, len(challenger_features) + 1)
                / len(challenger_features),
                "mean_percentile_rank": np.arange(1, len(challenger_features) + 1)
                / len(challenger_features),
                "original_position": [
                    context.universe.index(feature) for feature in challenger_features
                ],
                "support_years": "2020|2021|2022",
            }
        )
        list_form = "selected_k"
        source = "fixed_v0"
        endpoint = len(challenger_features)
    else:
        list_form, source, endpoint_text = challenger.split("__")
        endpoint = int(endpoint_text.removeprefix("k"))
        method = _selected_method(context)
        consensus_years = [2020, 2021, 2022]
        if context.smoke:
            consensus_years = [int(context.config["folds"]["outer_origins"][-1])]
        rankings = []
        origin_payload = []
        for year in consensus_years:
            path = _consensus_path_unit(
                context,
                stage=stage,
                year=year,
                method=method,
                source=source,
            )
            selected = list(path["endpoints"][str(endpoint)])
            ordered = complete_order_for_endpoint(path, endpoint)
            rankings.append({"year": year, "selected": selected, "ordered": ordered})
            origin_payload.append(
                {
                    "year": year,
                    "selected": selected,
                    "ordered": ordered,
                    "selected_ordered_feature_hash": ordered_feature_hash(selected),
                    "complete_ranking_hash": ordered_feature_hash(ordered),
                    "path": path,
                }
            )
        selected_consensus, support = consensus_features(
            rankings,
            count=endpoint,
            universe=context.universe,
        )
        challenger_features = _candidate_features(
            selected_consensus,
            context.controls["V0"],
            list_form,
        )
    consensus_selected_features = (
        list(challenger_features)
        if source == "fixed_v0"
        else list(selected_consensus)
    )
    consensus_union_features = _candidate_features(
        consensus_selected_features,
        context.controls["V0"],
        "v0_union_selected_k",
    )
    active_features = (
        challenger_features
        if global_decision["global_gate_passed"]
        else list(context.controls["V0"])
    )
    beta_decision = json.loads(
        (_stage_dir(context, "08_beta_decision") / "beta_decision.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "global_gate_passed": global_decision["global_gate_passed"],
        "active_global_candidate": global_decision["winner"],
        "active_global_features": active_features,
        "active_global_ordered_feature_hash": ordered_feature_hash(active_features),
        "benchmark_challenger": challenger,
        "benchmark_challenger_features": challenger_features,
        "benchmark_challenger_ordered_feature_hash": ordered_feature_hash(
            challenger_features
        ),
        "consensus_selected_features": consensus_selected_features,
        "consensus_selected_ordered_feature_hash": ordered_feature_hash(
            consensus_selected_features
        ),
        "consensus_union_features": consensus_union_features,
        "consensus_union_ordered_feature_hash": ordered_feature_hash(
            consensus_union_features
        ),
        "benchmark_challenger_development_eligible": bool(
            global_decision["global_gate_passed"]
        ),
        "list_form": list_form,
        "path_source": source,
        "endpoint": endpoint,
        "actual_count": len(challenger_features),
        "feature_count_interpretation": (
            "V0 canonical order plus consensus features not already in V0"
            if list_form == "v0_union_selected_k"
            else "exact consensus endpoint count"
        ),
        "active_global_beta": float(beta_decision["selected_beta"]),
        "benchmark_challenger_beta": (
            float(beta_decision["selected_beta"])
            if global_decision["global_gate_passed"]
            else 0.0
        ),
        "learner_seed": 42,
        "V0_automatically_overwritten": False,
    }
    atomic_write_json(stage / "candidate_features.json", payload)
    atomic_write_json(context.artifact_root / "candidate_features.json", payload)
    atomic_write_csv(support, stage / "consensus_support.csv")
    atomic_write_json(stage / "origin_lists.json", {"origins": origin_payload})
    stability_rows = []
    for left, right in combinations(origin_payload, 2):
        left_set, right_set = set(left["selected"]), set(right["selected"])
        stability_rows.append(
            {
                "left_year": left["year"],
                "right_year": right["year"],
                "jaccard": len(left_set & right_set) / len(left_set | right_set),
                "rank_correlation": _rank_correlation(
                    left["ordered"], right["ordered"]
                ),
                "left_v0_overlap": len(left_set & set(context.controls["V0"])),
                "right_v0_overlap": len(right_set & set(context.controls["V0"])),
            }
        )
    stability = pd.DataFrame(
        stability_rows,
        columns=[
            "left_year",
            "right_year",
            "jaccard",
            "rank_correlation",
            "left_v0_overlap",
            "right_v0_overlap",
        ],
    )
    atomic_write_csv(stability, stage / "stability_summary.csv")
    write_completion(stage, required)
    return stage


GLOBAL_STAGE_FUNCTIONS = {
    "02_fold_manifests": run_fold_manifest_stage,
    "03_control_ledgers": run_control_stage,
    "04_path_screen": run_path_screen_stage,
    "05_robust_candidate_generation": run_robust_generation_stage,
    "06_candidate_oof": run_candidate_oof_stage,
    "07_global_decision": run_global_decision_stage,
    "08_beta_decision": run_beta_stage,
    "09_consensus": run_consensus_stage,
}


def run_stage(context: ExperimentContext, stage_name: str) -> Path:
    try:
        function = GLOBAL_STAGE_FUNCTIONS[stage_name]
    except KeyError as error:
        raise ValueError(f"unknown global stage: {stage_name}") from error
    return function(context)
