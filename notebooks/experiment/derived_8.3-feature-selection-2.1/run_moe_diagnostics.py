"""Run causal MoE ablations, regime deltas, and the gated MoE decision."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fs21.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    completion_is_valid,
    invalidate_completion,
    stable_json_hash,
    write_completion,
)
from fs21.bootstrap import paired_hierarchical_bootstrap
from fs21.data import ordered_feature_hash, read_yaml
from fs21.decision import choose_moe, global_candidate_eligible
from fs21.global_pipeline import build_context
from fs21.ledger import (
    assert_candidate_coverage,
    prediction_rows,
    relabel_candidate,
    validate_ledger,
)
from fs21.metrics import primary_risk, secondary_metric_tables
from fs21.moe import (
    hard_expert_prediction_rows,
    load_historical_specialists,
    rank_regime_additions,
    reference_router,
    regime_coverage,
)
from fs21.constants import MOE_CONFIG_PATH


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_ids(candidate_features: dict) -> dict[str, str]:
    challenger = str(candidate_features["benchmark_challenger"])
    if challenger == "V0":
        return {"selected": "V0", "union": "V0"}
    _, source, endpoint = challenger.split("__")
    return {
        "selected": f"selected_k__{source}__{endpoint}",
        "union": f"v0_union_selected_k__{source}__{endpoint}",
    }


def _task_features(context, candidate: str, task, candidate_features: dict) -> list[str]:
    if candidate == "V0":
        return list(context.controls["V0"])
    manifest = _read_json(
        context.artifact_root
        / "stages"
        / "06_candidate_oof"
        / "candidate_features.json"
    )
    return list(manifest[candidate][task.fold_id]["features"])


def _parallel(function, units, workers):
    if workers <= 1:
        return [function(unit) for unit in units]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, units))


def _evaluate_expert_unit(
    *,
    context,
    stage: Path,
    task,
    candidate: str,
    shared_features: list[str],
    expert_features,
    router_config,
    reference,
    beta: float,
) -> Path:
    unit = stage / "units" / _slug(
        f"{candidate}__{task.fold_id}__beta_{float(beta):g}"
    )
    required = ["predictions.csv.gz", "router_experts.json"]
    if completion_is_valid(unit, required):
        return unit / "predictions.csv.gz"
    unit.mkdir(parents=True, exist_ok=True)
    invalidate_completion(unit)
    ledger, metadata = hard_expert_prediction_rows(
        context.frame,
        task,
        candidate=candidate,
        shared_features=shared_features,
        expert_features=expert_features,
        router_config=router_config,
        v0_features=list(context.controls["V0"]),
        reference=reference,
        beta=float(beta),
        config=context.config,
        device=context.device,
        smoke=context.smoke,
    )
    atomic_write_csv(ledger, unit / "predictions.csv.gz")
    atomic_write_json(unit / "router_experts.json", metadata)
    write_completion(unit, required)
    return unit / "predictions.csv.gz"


def _evaluate_global_unit(
    *,
    context,
    stage: Path,
    task,
    candidate: str,
    features: list[str],
    beta: float,
) -> Path:
    unit = stage / "global_units" / _slug(
        f"{candidate}__{task.fold_id}__beta_{float(beta):g}"
    )
    required = ["predictions.csv.gz", "model.json"]
    if completion_is_valid(unit, required):
        return unit / "predictions.csv.gz"
    unit.mkdir(parents=True, exist_ok=True)
    invalidate_completion(unit)
    ledger = prediction_rows(
        context.frame,
        task,
        features,
        candidate=candidate,
        path_source="moe_causal_single_global",
        endpoint=len(features),
        beta=float(beta),
        config=context.config,
        device=context.device,
        smoke=context.smoke,
    )
    atomic_write_csv(ledger, unit / "predictions.csv.gz")
    atomic_write_json(
        unit / "model.json",
        {
            "candidate": candidate,
            "fold_id": task.fold_id,
            "beta": float(beta),
            "features": features,
            "ordered_feature_hash": ordered_feature_hash(features),
        },
    )
    write_completion(unit, required)
    return unit / "predictions.csv.gz"


def _combine(paths) -> pd.DataFrame:
    return validate_ledger(
        pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    )


def _candidate_beta(ledger: pd.DataFrame, candidate: str) -> float:
    values = sorted(
        float(value)
        for value in ledger.loc[ledger["candidate"] == candidate, "beta"].unique()
    )
    if len(values) != 1:
        raise ValueError(f"candidate {candidate} has ambiguous beta arms: {values}")
    return values[0]


def run_causal_stage(context, moe_config: dict) -> Path:
    stage = context.artifact_root / "stages" / "11_moe_causal_matrix"
    required = [
        "moe_oof_predictions.csv.gz",
        "causal_ablation_metrics.csv",
        "router_regime_populations.csv",
        "router_drift.csv",
        "router_feature_missingness.csv",
        "router_route_distance.csv",
        "causal_matrix_manifest.json",
    ]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    frozen = _read_json(
        context.artifact_root / "stages" / "09_consensus" / "candidate_features.json"
    )
    ids = _candidate_ids(frozen)
    feature_beta = float(frozen["benchmark_challenger_beta"])
    single_sources = {
        "v0_single_global": {"source": "V0", "beta": 0.0},
        "selected_2_1_single_global": {
            "source": ids["selected"],
            "beta": feature_beta,
        },
        "union_2_1_v0_single_global": {
            "source": ids["union"],
            "beta": feature_beta,
        },
    }
    global_units = []
    for arm, spec in single_sources.items():
        for task in context.tasks:
            features = _task_features(context, spec["source"], task, frozen)
            global_units.append((task, arm, features, float(spec["beta"])))

    def run_global(unit):
        task, candidate, features, beta = unit
        return _evaluate_global_unit(
            context=context,
            stage=stage,
            task=task,
            candidate=candidate,
            features=features,
            beta=beta,
        )

    single_global = _combine(
        _parallel(run_global, global_units, context.workers)
    )
    router_config = dict(moe_config["router"])
    if str(router_config["feature_source"]) != str(
        context.config["features"]["v0_source"]
    ):
        raise ValueError("MoE router source must be the exact global V0 source")
    reference = reference_router(
        context.frame,
        router_config,
        list(context.controls["V0"]),
    )
    historical = load_historical_specialists(moe_config)
    shared_specs = {
        "v0_shared_hard_experts": {"source": "V0", "beta": 0.0},
        "selected_2_1_shared_hard_experts": {
            "source": ids["selected"],
            "beta": feature_beta,
        },
        "union_2_1_v0_shared_hard_experts": {
            "source": ids["union"],
            "beta": feature_beta,
        },
    }
    units = []
    for candidate, spec in shared_specs.items():
        for task in context.tasks:
            shared = _task_features(context, spec["source"], task, frozen)
            units.append((task, candidate, shared, None, float(spec["beta"])))
    for task in context.tasks:
        units.append(
            (
                task,
                "eval_1_0_saved_specialists",
                list(context.controls["V0"]),
                historical,
                0.0,
            )
        )

    def run(unit):
        task, candidate, shared, expert_features, beta = unit
        return _evaluate_expert_unit(
            context=context,
            stage=stage,
            task=task,
            candidate=candidate,
            shared_features=shared,
            expert_features=expert_features,
            router_config=router_config,
            reference=reference,
            beta=beta,
        )

    paths = _parallel(run, units, context.workers)
    experts = _combine(paths)
    combined = validate_ledger(pd.concat([single_global, experts], ignore_index=True))
    for candidate in sorted(combined["candidate"].unique()):
        if candidate != "v0_single_global":
            assert_candidate_coverage(
                combined,
                candidate,
                reference="v0_single_global",
            )
    atomic_write_csv(combined, stage / "moe_oof_predictions.csv.gz")
    metric_rows = []
    for candidate in sorted(combined["candidate"].unique()):
        beta = _candidate_beta(combined, candidate)
        risk = primary_risk(combined, candidate, beta=beta)
        secondary = secondary_metric_tables(
            combined, candidate, beta=beta
        )["overall"].iloc[0]
        metric_rows.append(
            {
                "candidate": candidate,
                "beta": beta,
                "combined_primary_rmse": risk["combined_primary_rmse"],
                "forward_time_rmse": risk["forward_time_rmse"],
                "station_time_rmse": risk["station_time_rmse"],
                "R2": secondary["R2"],
                "RMSE": secondary["RMSE"],
                "MAE": secondary["MAE"],
                "Bias": secondary["Bias"],
                "worst_station_RMSE": secondary["worst_station_RMSE"],
                "p90_month_RMSE": secondary["p90_month_RMSE"],
            }
        )
    metrics = pd.DataFrame(metric_rows)
    atomic_write_csv(metrics, stage / "causal_ablation_metrics.csv")

    router_rows = experts.loc[
        experts["candidate"] == "v0_shared_hard_experts"
    ].copy()
    populations = (
        router_rows.groupby(
            ["outer_origin", "year", "month", "station", "router_regime"],
            sort=True,
        )
        .agg(
            row_count=("truth", "size"),
            route_distance_mean=("route_distance", "mean"),
            route_distance_std=("route_distance", "std"),
            target_standard_deviation=("truth", "std"),
        )
        .reset_index()
    )
    atomic_write_csv(populations, stage / "router_regime_populations.csv")
    route_distance = (
        router_rows.groupby(
            ["fold_family", "outer_origin", "router_regime"],
            sort=True,
        )["route_distance"]
        .agg(
            row_count="size",
            mean="mean",
            std="std",
            minimum="min",
            median="median",
            maximum="max",
        )
        .reset_index()
    )
    atomic_write_csv(route_distance, stage / "router_route_distance.csv")
    drift_rows = []
    missingness_rows = []
    for metadata_path in sorted((stage / "units").glob("*/router_experts.json")):
        payload = _read_json(metadata_path)
        for regime, drift in payload["centroid_drift_from_reference"].items():
            drift_rows.append(
                {
                    "fold_id": payload["fold_id"],
                    "aligned_regime": int(regime),
                    "centroid_drift": float(drift),
                    "train_route_distance_mean": payload[
                        "train_route_distance_mean"
                    ],
                    "validation_route_distance_mean": payload[
                        "validation_route_distance_mean"
                    ],
                }
            )
        if payload.get("candidate") == "v0_shared_hard_experts":
            for row in payload["router_feature_missingness"]:
                missingness_rows.append(
                    {
                        "fold_id": payload["fold_id"],
                        **row,
                    }
                )
    atomic_write_csv(pd.DataFrame(drift_rows), stage / "router_drift.csv")
    atomic_write_csv(
        pd.DataFrame(missingness_rows),
        stage / "router_feature_missingness.csv",
    )
    shared_candidates = list(shared_specs)
    single_candidates = list(single_sources)
    strongest_shared = metrics.loc[
        metrics["candidate"].isin(shared_candidates)
    ].sort_values(["combined_primary_rmse", "candidate"]).iloc[0]["candidate"]
    strongest_single = metrics.loc[
        metrics["candidate"].isin(single_candidates)
    ].sort_values(["combined_primary_rmse", "candidate"]).iloc[0]["candidate"]
    manifest = {
        "candidate_ids": ids,
        "single_sources": single_sources,
        "shared_sources": shared_specs,
        "strongest_shared_control": strongest_shared,
        "strongest_single_global": strongest_single,
        "historical_specialist_counts": {
            str(regime): len(features) for regime, features in historical.items()
        },
        "global_gate_passed": frozen["global_gate_passed"],
        "moe_diagnostic_runs_even_after_global_failure": True,
        "router_reference": reference.to_dict(),
    }
    atomic_write_json(stage / "causal_matrix_manifest.json", manifest)
    write_completion(stage, required)
    return stage


def _shared_source_id(strongest_shared: str, ids: dict) -> str:
    mapping = {
        "v0_shared_hard_experts": "V0",
        "selected_2_1_shared_hard_experts": ids["selected"],
        "union_2_1_v0_shared_hard_experts": ids["union"],
    }
    return mapping[str(strongest_shared)]


def _single_for_shared(shared_candidate: str) -> str:
    mapping = {
        "v0_shared_hard_experts": "v0_single_global",
        "selected_2_1_shared_hard_experts": "selected_2_1_single_global",
        "union_2_1_v0_shared_hard_experts": "union_2_1_v0_single_global",
    }
    return mapping[str(shared_candidate)]


def _consensus_shared_features(
    context,
    frozen: dict,
    source_id: str,
    ids: dict,
) -> list[str]:
    if source_id == "V0":
        return list(context.controls["V0"])
    if source_id == ids["selected"]:
        return list(frozen["consensus_selected_features"])
    if source_id == ids["union"]:
        return list(frozen["consensus_union_features"])
    raise ValueError(f"unknown shared source for consensus freeze: {source_id}")


def _mean_top_rank_jaccard(
    ranking_table: pd.DataFrame,
    *,
    regime: int,
    count: int,
) -> float:
    groups = []
    subset = ranking_table.loc[
        (ranking_table["regime"] == int(regime))
        & (ranking_table["usable_folds"] > 0)
    ]
    for _, group in subset.groupby("fold_id", sort=True):
        groups.append(
            set(group.nsmallest(int(count), "rank")["feature"].astype(str))
        )
    if len(groups) < 2:
        return float("nan")
    values = []
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values))


def _rank_delta_unit(
    *,
    context,
    stage,
    task,
    regime,
    shared_features,
    source_family,
    router_config,
    reference,
) -> Path:
    unit = stage / "ranking_units" / _slug(
        f"regime_{regime}__{source_family}__{task.fold_id}"
    )
    required = ["ranking.csv", "ranking.json"]
    if completion_is_valid(unit, required):
        return unit / "ranking.json"
    unit.mkdir(parents=True, exist_ok=True)
    invalidate_completion(unit)
    outer_training = context.frame.iloc[list(task.train_index)].reset_index(drop=True)
    ordered, table, metadata = rank_regime_additions(
        outer_training,
        regime=int(regime),
        shared_features=shared_features,
        universe=context.universe,
        source_family=source_family,
        partition_seed=42 if task.partition_seed is None else int(task.partition_seed),
        learner_seed=task.learner_seed,
        router_config=router_config,
        v0_features=list(context.controls["V0"]),
        reference=reference,
        config=context.config,
        device=context.device,
        smoke=context.smoke,
        permutation_repeats=int(context.config["ranking"]["permutation_repeats"]),
    )
    atomic_write_csv(table, unit / "ranking.csv")
    atomic_write_json(
        unit / "ranking.json",
        {
            "fold_id": task.fold_id,
            "outer_origin": task.origin,
            "regime": int(regime),
            "path_source": source_family,
            "ordered": ordered,
            "metadata": metadata,
        },
    )
    write_completion(unit, required)
    return unit / "ranking.json"


def _delta_candidate(regime: int, count: int) -> str:
    return f"regime_{regime}_delta_{count}"


def run_delta_stage(context, moe_config: dict) -> Path:
    stage = context.artifact_root / "stages" / "12_regime_delta_moe_decision"
    required = [
        "regime_delta_rankings.csv",
        "regime_delta_oof_predictions.csv.gz",
        "regime_delta_decision.json",
        "moe_promotion_decision.json",
        "frozen_moe_features.json",
        "regime_metrics.csv",
    ]
    if completion_is_valid(stage, required):
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    causal_stage = context.artifact_root / "stages" / "11_moe_causal_matrix"
    causal_manifest = _read_json(causal_stage / "causal_matrix_manifest.json")
    causal_ledger = pd.read_csv(causal_stage / "moe_oof_predictions.csv.gz")
    frozen = _read_json(
        context.artifact_root / "stages" / "09_consensus" / "candidate_features.json"
    )
    ids = causal_manifest["candidate_ids"]
    shared_candidate = str(causal_manifest["strongest_shared_control"])
    shared_source = _shared_source_id(shared_candidate, ids)
    shared_beta = _candidate_beta(causal_ledger, shared_candidate)
    source_family = str(frozen.get("path_source", "station_time"))
    if source_family not in {"station_time", "forward_time"}:
        source_family = "station_time"
    router_config = dict(moe_config["router"])
    if str(router_config["feature_source"]) != str(
        context.config["features"]["v0_source"]
    ):
        raise ValueError("MoE router source must be the exact global V0 source")
    reference = reference_router(
        context.frame,
        router_config,
        list(context.controls["V0"]),
    )
    base = causal_ledger.loc[causal_ledger["candidate"] == shared_candidate].copy()
    coverage = regime_coverage(base)
    minimum_origins = (
        1 if context.smoke else int(moe_config["regime_delta"]["minimum_origins"])
    )
    minimum_stations = (
        2 if context.smoke else int(moe_config["regime_delta"]["minimum_stations"])
    )
    regime_allowed = {}
    for regime in (0, 1):
        rows = coverage.loc[coverage["router_regime"] == regime]
        regime_allowed[regime] = bool(
            rows["outer_origin"].nunique() >= minimum_origins
            and (rows["station_count"] >= minimum_stations).all()
        )

    ranking_paths = {}
    ranking_rows = []
    for task in context.tasks:
        shared = _task_features(context, shared_source, task, frozen)
        for regime in (0, 1):
            if not regime_allowed[regime]:
                continue
            path = _rank_delta_unit(
                context=context,
                stage=stage,
                task=task,
                regime=regime,
                shared_features=shared,
                source_family=source_family,
                router_config=router_config,
                reference=reference,
            )
            payload = _read_json(path)
            ranking_paths[(task.fold_id, regime)] = payload["ordered"]
            table = pd.read_csv(path.parent / "ranking.csv")
            table.insert(0, "fold_id", task.fold_id)
            table.insert(1, "outer_origin", task.origin)
            table.insert(2, "regime", regime)
            table.insert(
                3,
                "usable_folds",
                int(payload["metadata"]["usable_folds"]),
            )
            ranking_rows.append(table)
    ranking_table = (
        pd.concat(ranking_rows, ignore_index=True)
        if ranking_rows
        else pd.DataFrame(
            columns=[
                "fold_id",
                "outer_origin",
                "regime",
                "usable_folds",
                "feature",
                "rank",
            ]
        )
    )
    atomic_write_csv(ranking_table, stage / "regime_delta_rankings.csv")

    additions = [int(value) for value in moe_config["regime_delta"]["additions"]]
    if context.smoke:
        additions = [0, min(2, max(additions))]
    evaluation_specs = []
    for regime in (0, 1):
        if not regime_allowed[regime]:
            continue
        for count in additions:
            if count == 0:
                continue
            for task in context.tasks:
                shared = _task_features(context, shared_source, task, frozen)
                deltas = ranking_paths.get((task.fold_id, regime), [])[:count]
                experts = {0: list(shared), 1: list(shared)}
                experts[regime] = list(shared) + [
                    feature for feature in deltas if feature not in set(shared)
                ]
                evaluation_specs.append(
                    (task, _delta_candidate(regime, count), shared, experts)
                )

    def run(spec):
        task, candidate, shared, experts = spec
        return _evaluate_expert_unit(
            context=context,
            stage=stage,
            task=task,
            candidate=candidate,
            shared_features=shared,
            expert_features=experts,
            router_config=router_config,
            reference=reference,
            beta=shared_beta,
        )

    evaluated_paths = _parallel(run, evaluation_specs, context.workers)
    evaluated = _combine(evaluated_paths) if evaluated_paths else base.iloc[0:0].copy()
    base_copy = relabel_candidate(base, "shared_only")
    delta_ledger = validate_ledger(pd.concat([base_copy, evaluated], ignore_index=True))
    atomic_write_csv(delta_ledger, stage / "regime_delta_oof_predictions.csv.gz")

    regime_decisions = {}
    regime_metric_rows = []
    selected_counts = {0: 0, 1: 0}
    for regime in (0, 1):
        if not regime_allowed[regime]:
            regime_decisions[str(regime)] = {
                "selected_delta_count": 0,
                "reason": "insufficient_origin_or_station_coverage",
            }
            continue
        stability = _mean_top_rank_jaccard(
            ranking_table,
            regime=regime,
            count=max(additions),
        )
        usable_task_count = int(
            ranking_table.loc[
                (ranking_table["regime"] == regime)
                & (ranking_table["usable_folds"] > 0),
                "fold_id",
            ].nunique()
        )
        stability_value = float(stability) if np.isfinite(stability) else None
        base_regime = delta_ledger.loc[
            (delta_ledger["candidate"] == "shared_only")
            & (delta_ledger["router_regime"] == regime)
        ]
        comparisons = []
        for count in additions:
            if count == 0:
                continue
            candidate = _delta_candidate(regime, count)
            candidate_regime = delta_ledger.loc[
                (delta_ledger["candidate"] == candidate)
                & (delta_ledger["router_regime"] == regime)
            ]
            comparison_ledger = pd.concat([base_regime, candidate_regime], ignore_index=True)
            comparison = paired_hierarchical_bootstrap(
                comparison_ledger,
                candidate,
                "shared_only",
                beta=shared_beta,
                replicates=int(context.config["bootstrap"]["replicates"]),
                seed=int(context.config["bootstrap"]["seed"]),
            )
            risk = primary_risk(
                comparison_ledger,
                candidate,
                beta=shared_beta,
            )
            eligible = (
                comparison["comparisons"]["combined_primary_rmse"]["ci_upper"] < 0.0
                and comparison["comparisons"]["worst_station_rmse"]["delta"] <= 0.0
            )
            row = {
                "regime": regime,
                "delta_count": count,
                "candidate": candidate,
                "combined_primary_rmse": risk["combined_primary_rmse"],
                "eligible": eligible,
                "comparison": comparison,
            }
            comparisons.append(row)
            regime_metric_rows.append(
                {
                    key: value for key, value in row.items() if key != "comparison"
                }
            )
        eligible_rows = [row for row in comparisons if row["eligible"]]
        if not eligible_rows:
            minimum_stability = float(
                moe_config["regime_delta"]["minimum_rank_jaccard_for_stability"]
            )
            regime_decisions[str(regime)] = {
                "selected_delta_count": 0,
                "reason": (
                    "insufficient_usable_ranking_folds"
                    if usable_task_count == 0
                    else "unstable_rankings"
                    if np.isfinite(stability) and stability < minimum_stability
                    else "no_measured_robust_improvement"
                ),
                "usable_outer_task_rankings": usable_task_count,
                "top_rank_jaccard": stability_value,
                "minimum_rank_jaccard_for_stability": minimum_stability,
                "comparisons": comparisons,
            }
            continue
        best = min(eligible_rows, key=lambda row: row["combined_primary_rmse"])
        threshold = (
            best["combined_primary_rmse"]
            + best["comparison"]["candidate_primary_bootstrap_standard_error"]
        )
        selected = min(
            [row for row in eligible_rows if row["combined_primary_rmse"] <= threshold],
            key=lambda row: (row["delta_count"], row["candidate"]),
        )
        selected_counts[regime] = int(selected["delta_count"])
        regime_decisions[str(regime)] = {
            "selected_delta_count": selected_counts[regime],
            "reason": "smallest_qualifying_within_one_standard_error",
            "one_standard_error_threshold": threshold,
            "usable_outer_task_rankings": usable_task_count,
            "top_rank_jaccard": stability_value,
            "comparisons": comparisons,
        }
    atomic_write_csv(
        pd.DataFrame(
            regime_metric_rows,
            columns=[
                "regime",
                "delta_count",
                "candidate",
                "combined_primary_rmse",
                "eligible",
            ],
        ),
        stage / "regime_metrics.csv",
    )

    final_candidate = "moe_shared_plus_selected_deltas"
    if any(selected_counts.values()):
        final_specs = []
        for task in context.tasks:
            shared = _task_features(context, shared_source, task, frozen)
            experts = {0: list(shared), 1: list(shared)}
            for regime in (0, 1):
                count = selected_counts[regime]
                additions_for_task = (
                    ranking_paths.get((task.fold_id, regime), [])[:count]
                    if count > 0
                    else []
                )
                experts[regime] = list(shared) + [
                    feature
                    for feature in additions_for_task
                    if feature not in set(shared)
                ]
            final_specs.append((task, final_candidate, shared, experts))
        final_paths = _parallel(run, final_specs, context.workers)
        final_ledger = _combine(final_paths)
    else:
        final_ledger = relabel_candidate(base, final_candidate)
    all_moe_ledger = validate_ledger(
        pd.concat([causal_ledger, delta_ledger, final_ledger], ignore_index=True)
    )
    # Consider causal shared controls and the selected-delta arm, never the
    # historical small specialists, as promotion candidates.
    promotion_candidates = [
        "v0_shared_hard_experts",
        "selected_2_1_shared_hard_experts",
        "union_2_1_v0_shared_hard_experts",
        final_candidate,
    ]
    corresponding_single = {
        "v0_shared_hard_experts": "v0_single_global",
        "selected_2_1_shared_hard_experts": "selected_2_1_single_global",
        "union_2_1_v0_shared_hard_experts": "union_2_1_v0_single_global",
        final_candidate: _single_for_shared(shared_candidate),
    }
    moe_summaries = []
    for candidate in promotion_candidates:
        comparator = corresponding_single[candidate]
        candidate_beta = _candidate_beta(all_moe_ledger, candidate)
        comparator_beta = _candidate_beta(all_moe_ledger, comparator)
        assert_candidate_coverage(
            all_moe_ledger,
            candidate,
            reference=comparator,
        )
        comparison = paired_hierarchical_bootstrap(
            all_moe_ledger,
            candidate,
            comparator,
            candidate_beta=candidate_beta,
            reference_beta=comparator_beta,
            replicates=int(context.config["bootstrap"]["replicates"]),
            seed=int(context.config["bootstrap"]["seed"]),
        )
        risk = primary_risk(
            all_moe_ledger,
            candidate,
            beta=candidate_beta,
        )
        summary = {
            "candidate": candidate,
            "corresponding_single_global": comparator,
            "candidate_beta": candidate_beta,
            "reference_beta": comparator_beta,
            "actual_count": int(
                all_moe_ledger.loc[
                    all_moe_ledger["candidate"] == candidate,
                    "actual_count",
                ].max()
            ),
            "list_form": "shared_hard_experts",
            "path_source": source_family,
            "combined_primary_rmse": risk["combined_primary_rmse"],
            "forward_time_rmse": risk["forward_time_rmse"],
            "station_time_rmse": risk["station_time_rmse"],
            "selection_stability": 0.0,
            "coverage_matches_v0": True,
            "promotable": bool(frozen["global_gate_passed"]),
            "bootstrap": comparison,
        }
        eligible, failures = global_candidate_eligible(summary)
        summary["eligible"] = eligible
        summary["failure_reasons"] = failures
        moe_summaries.append(summary)
    diagnostic_best = min(
        moe_summaries,
        key=lambda row: (row["combined_primary_rmse"], row["candidate"]),
    )
    eligible_moe = [row for row in moe_summaries if row["eligible"]]
    moe_summary = (
        min(
            eligible_moe,
            key=lambda row: (row["combined_primary_rmse"], row["candidate"]),
        )
        if eligible_moe
        else diagnostic_best
    )
    best_moe = str(moe_summary["candidate"])
    promotion = choose_moe(
        global_gate_passed=bool(frozen["global_gate_passed"]),
        single_global_id=str(frozen["active_global_candidate"]),
        moe_summary=moe_summary,
    )
    promotion["strongest_single_global_comparator"] = moe_summary[
        "corresponding_single_global"
    ]
    promotion["best_moe_development_candidate"] = best_moe
    promotion["candidate_comparisons"] = moe_summaries
    atomic_write_json(stage / "moe_promotion_decision.json", promotion)
    atomic_write_json(context.artifact_root / "moe_promotion_decision.json", promotion)
    atomic_write_json(
        stage / "regime_delta_decision.json",
        {
            "coverage": coverage.to_dict(orient="records"),
            "coverage_requirements": {
                "origins": minimum_origins,
                "stations": minimum_stations,
            },
            "regime_allowed": {str(key): value for key, value in regime_allowed.items()},
            "regimes": regime_decisions,
            "selected_counts": {
                str(key): value for key, value in selected_counts.items()
            },
            "standalone_expert_lists_created": False,
        },
    )

    best_shared_source = (
        shared_source
        if best_moe == final_candidate
        else _shared_source_id(best_moe, ids)
    )
    final_shared = _consensus_shared_features(
        context,
        frozen,
        best_shared_source,
        ids,
    )
    frozen_deltas = {}
    for regime in (0, 1):
        count = selected_counts[regime] if best_moe == final_candidate else 0
        if count == 0:
            frozen_deltas[str(regime)] = []
            continue
        rows = ranking_table.loc[
            (ranking_table["regime"] == regime)
            & (ranking_table["usable_folds"] > 0)
        ].copy()
        rows["selected"] = rows["rank"] <= count
        consensus = (
            rows.groupby("feature", sort=False)
            .agg(
                selection_frequency=("selected", "sum"),
                median_rank=("rank", "median"),
                mean_rank=("rank", "mean"),
                original_position=("original_position", "min"),
            )
            .reset_index()
            .sort_values(
                [
                    "selection_frequency",
                    "median_rank",
                    "mean_rank",
                    "original_position",
                ],
                ascending=[False, True, True, True],
            )
        )
        frozen_deltas[str(regime)] = consensus.head(count)["feature"].tolist()
    frozen_moe = {
        "promoted": promotion["moe_promoted"],
        "candidate": best_moe,
        "shared_source": best_shared_source,
        "shared_features": final_shared,
        "shared_ordered_feature_hash": ordered_feature_hash(final_shared),
        "regime_deltas": frozen_deltas,
        "expert_features": {
            str(regime): final_shared
            + [
                feature
                for feature in frozen_deltas[str(regime)]
                if feature not in set(final_shared)
            ]
            for regime in (0, 1)
        },
        "router": reference.to_dict(),
        "learner_seed": 42,
        "beta": _candidate_beta(all_moe_ledger, best_moe),
        "corresponding_single_global": moe_summary[
            "corresponding_single_global"
        ],
    }
    atomic_write_json(stage / "frozen_moe_features.json", frozen_moe)
    write_completion(stage, required)
    return stage


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "causal", "delta"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    context = build_context(device=args.device, workers=args.workers, smoke=args.smoke)
    moe_config = read_yaml(MOE_CONFIG_PATH)
    if args.stage in {"all", "causal"}:
        run_causal_stage(context, moe_config)
    if args.stage in {"all", "delta"}:
        run_delta_stage(context, moe_config)
    print(json.dumps({"status": "complete", "stage": args.stage}, indent=2))


if __name__ == "__main__":
    main()
