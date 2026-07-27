"""Bounded direct feature-wrapper search for the 2023–2025 target period."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .data import ExperimentData
from .evaluate import CandidateResult, ModelEvaluator
from .seeds import load_seed_sets
from .selection import SelectionResult


DELTA_ADDITION_COUNTS = (0, 5, 10)


@dataclass(frozen=True)
class Candidate:
    """A literal candidate supplied to the evaluator."""

    candidate_id: str
    global_features: tuple[str, ...]
    cluster_additions: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def additions_dict(self) -> dict[str, list[str]]:
        return {str(cluster): list(features) for cluster, features in self.cluster_additions}


@dataclass
class EvaluationBatch:
    """Logical candidates resolved by one bounded evaluator batch."""

    requested: int
    fitted: list[CandidateResult]
    resolved: dict[str, CandidateResult]
    reused: dict[str, str]
    missing: list[Candidate]
    late: list[Candidate]


class SearchIncompleteError(RuntimeError):
    """Raised after a checkpointed search cannot complete its required final grid."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    return value


class SearchRecorder:
    """Incremental plain CSV/JSON checkpoints with literal candidate lists."""

    RESULT_COLUMNS = [
        "candidate_id",
        "canonical_candidate_id",
        "model_kind",
        "phase",
        "completion_status",
        "pooled_r2",
        "pooled_rmse",
        "pooled_mae",
        "global_feature_count",
        "cluster_0_additions",
        "cluster_1_additions",
        "cluster_0_feature_count",
        "cluster_1_feature_count",
        "year_2023_r2",
        "year_2024_r2",
        "year_2025_r2",
        "train_time_s",
        "global_features",
    ]
    ALIAS_COLUMNS = [
        "candidate_id",
        "canonical_candidate_id",
        "model_kind",
        "phase",
        "reason",
        "global_features",
        "cluster_0_additions",
        "cluster_1_additions",
    ]
    DELTA_GRID_COLUMNS = [
        "candidate_id",
        "canonical_candidate_id",
        "evaluation_status",
        "cluster_0_addition_count",
        "cluster_1_addition_count",
        "cluster_0_additions",
        "cluster_1_additions",
        "pooled_r2",
        "pooled_rmse",
        "pooled_mae",
        "year_2023_r2",
        "year_2024_r2",
        "year_2025_r2",
    ]

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[CandidateResult] = []
        self.rows: list[dict[str, Any]] = []
        self.alias_rows: list[dict[str, Any]] = []
        self.flush()
        self.write_delta_grid([])

    def _write_csv(self, name: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
        path = self.artifact_dir / name
        temporary = path.with_suffix(".writing.csv")
        frame = pd.DataFrame(rows)
        frame = frame.reindex(columns=columns)
        frame.to_csv(temporary, index=False)
        temporary.replace(path)

    def add(
        self,
        result: CandidateResult,
        *,
        phase: str,
        completion_status: str = "on_time",
        eligible: bool = True,
    ) -> None:
        if eligible:
            self.results.append(result)
        row = result.as_record()
        row.update(
            {
                "canonical_candidate_id": result.candidate_id,
                "phase": phase,
                "completion_status": completion_status,
            }
        )
        self.rows.append(row)
        self.flush()

    def add_alias(
        self,
        candidate: Candidate,
        *,
        model_kind: str,
        canonical: CandidateResult,
        phase: str,
        reason: str,
    ) -> None:
        additions = candidate.additions_dict()
        self.alias_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "canonical_candidate_id": canonical.candidate_id,
                "model_kind": model_kind,
                "phase": phase,
                "reason": reason,
                "global_features": ";".join(candidate.global_features),
                "cluster_0_additions": ";".join(additions.get("0", [])),
                "cluster_1_additions": ";".join(additions.get("1", [])),
            }
        )
        self.flush()

    def flush(self) -> None:
        self._write_csv("search_results.csv", self.rows, self.RESULT_COLUMNS)
        self._write_csv("candidate_aliases.csv", self.alias_rows, self.ALIAS_COLUMNS)

    def write_delta_grid(self, rows: list[dict[str, Any]]) -> None:
        self._write_csv("delta_grid.csv", rows, self.DELTA_GRID_COLUMNS)

    def invalidate_final_outputs(self) -> None:
        """Remove a prior winner before this search can produce a replacement.

        A deadline failure must not leave a previous run's selected feature set in
        place. The caller immediately writes a ``running`` summary so report
        readers cannot mistake stale artifacts for the current run.
        """
        selected_path = self.artifact_dir / "selected_features.json"
        if selected_path.exists():
            selected_path.unlink()

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self.artifact_dir / name
        temporary = path.with_suffix(".writing.json")
        temporary.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


class DirectSearch:
    """A bounded direct wrapper search with exact final decisions."""

    def __init__(
        self,
        data: ExperimentData,
        config: dict[str, Any],
        audit_results: dict[str, dict[str, dict[str, SelectionResult]]],
        *,
        workers: int,
        deadline_minutes: int,
        evaluator: ModelEvaluator | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.data = data
        self.config = config
        self.audit_results = audit_results
        self.workers = max(1, int(workers))
        self.evaluator = evaluator or ModelEvaluator(data, config)
        self.clock = clock
        self.started = self.clock()
        self.deadline = self.started + max(1, int(deadline_minutes)) * 60
        reserve_seconds = int(config["search"]["final_reserve_minutes"]) * 60
        self.wrapper_launch_deadline = max(self.started, self.deadline - reserve_seconds)

        minimum_pool_size = int(config["search"]["global_feature_max"]) + max(
            DELTA_ADDITION_COUNTS
        )
        if int(config["search"]["candidate_pool_size"]) < minimum_pool_size:
            raise ValueError(
                "candidate_pool_size must leave room for the largest 0/5/10 specialist "
                f"delta: expected at least {minimum_pool_size}."
            )
        self.recorder = SearchRecorder(Path(config["artifacts"]["directory"]))
        self.recorder.invalidate_final_outputs()
        self.recorder.write_json(
            "search_summary.json",
            {
                "status": "running",
                "selection_goal": "unweighted_pooled_test_r2_2023_2025",
                "deadline": {
                    "budget_seconds": self.deadline - self.started,
                    "wrapper_launch_cutoff_seconds": self.wrapper_launch_deadline
                    - self.started,
                },
            },
        )
        self.evidence: pd.DataFrame | None = None
        self.result_cache: dict[tuple[str, tuple[Any, ...]], CandidateResult] = {}
        self.phase_stats: dict[str, dict[str, int]] = {}

    def _elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started)

    def _can_launch(self, launch_deadline: float) -> bool:
        return self.clock() < min(launch_deadline, self.deadline)

    def _canonical_candidate(self, candidate: Candidate) -> Candidate:
        additions = candidate.additions_dict()
        return Candidate(
            candidate.candidate_id,
            tuple(self.evaluator.validate_features(candidate.global_features)),
            tuple(
                (
                    cluster,
                    tuple(self.evaluator.validate_features(additions.get(cluster, []))),
                )
                for cluster in ("0", "1")
            ),
        )

    def _candidate_key(self, candidate: Candidate) -> tuple[Any, ...]:
        normalized = self._canonical_candidate(candidate)
        return normalized.global_features, normalized.cluster_additions

    def _result_key(self, result: CandidateResult) -> tuple[Any, ...]:
        return self._candidate_key(
            Candidate(
                result.candidate_id,
                tuple(result.global_features),
                tuple(
                    (cluster, tuple(result.cluster_additions.get(cluster, [])))
                    for cluster in ("0", "1")
                ),
            )
        )

    def _rank_key(self, result: CandidateResult) -> tuple[Any, ...]:
        additions = sum(len(features) for features in result.cluster_additions.values())
        return (
            -result.pooled_r2,
            result.pooled_rmse,
            len(result.global_features) + additions,
            tuple(result.global_features),
            tuple(tuple(result.cluster_additions.get(cluster, [])) for cluster in ("0", "1")),
        )

    def better(self, contender: CandidateResult, incumbent: CandidateResult) -> bool:
        return self._rank_key(contender) < self._rank_key(incumbent)

    def best(self, results: Iterable[CandidateResult]) -> CandidateResult:
        available = list(results)
        if not available:
            raise ValueError("No candidates were evaluated.")
        return min(available, key=self._rank_key)

    @staticmethod
    def _distinct_results(results: Iterable[CandidateResult]) -> list[CandidateResult]:
        unique: dict[str, CandidateResult] = {}
        for result in results:
            unique.setdefault(result.candidate_id, result)
        return list(unique.values())

    def _record_batch(self, phase: str, batch: EvaluationBatch) -> None:
        stats = self.phase_stats.setdefault(
            phase,
            {"requested": 0, "fitted": 0, "reused": 0, "missing": 0, "late": 0},
        )
        stats["requested"] += batch.requested
        stats["fitted"] += len(batch.fitted)
        stats["reused"] += len(batch.reused)
        stats["missing"] += len(batch.missing)
        stats["late"] += len(batch.late)

    def _evaluate_many(
        self,
        candidates: Iterable[Candidate],
        model_kind: str,
        *,
        phase: str,
        launch_deadline: float,
        include_predictions: bool = False,
    ) -> EvaluationBatch:
        """Evaluate unique candidates without allowing executor-queued late starts."""
        requested = [self._canonical_candidate(candidate) for candidate in candidates]
        batch = EvaluationBatch(
            requested=len(requested),
            fitted=[],
            resolved={},
            reused={},
            missing=[],
            late=[],
        )
        if not requested:
            self._record_batch(phase, batch)
            return batch

        grouped: dict[tuple[Any, ...], tuple[Candidate, list[Candidate]]] = {}
        unique: list[tuple[tuple[Any, ...], Candidate, list[Candidate]]] = []
        for candidate in requested:
            key = self._candidate_key(candidate)
            cache_key = (model_kind, key)
            cached = self.result_cache.get(cache_key)
            if cached is not None and (not include_predictions or cached.predictions is not None):
                self.recorder.add_alias(
                    candidate,
                    model_kind=model_kind,
                    canonical=cached,
                    phase=phase,
                    reason="already_evaluated",
                )
                batch.resolved[candidate.candidate_id] = cached
                batch.reused[candidate.candidate_id] = cached.candidate_id
                continue

            if key not in grouped:
                grouped[key] = (candidate, [])
                unique.append((key, candidate, grouped[key][1]))
            else:
                grouped[key][1].append(candidate)

        if unique:
            print(
                f"[search] evaluating {len(unique)} unique {model_kind} candidates "
                f"with {self.workers} workers",
                flush=True,
            )

        next_index = 0
        futures: dict[Any, tuple[tuple[Any, ...], Candidate, list[Candidate]]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            def evaluate_if_before_cutoff(candidate: Candidate) -> CandidateResult | None:
                """Make the launch decision in the worker immediately before fitting.

                ``submit`` alone is not a launch: an executor can delay a task after
                it has been accepted.  Checking here prevents an accepted-but-queued
                candidate from starting a model fit after the phase cutoff.
                """
                if not self._can_launch(launch_deadline):
                    return None
                return self.evaluator.evaluate(
                    candidate.candidate_id,
                    candidate.global_features,
                    candidate.additions_dict(),
                    model_kind=model_kind,
                    include_predictions=include_predictions,
                )

            def fill_worker_slots() -> None:
                nonlocal next_index
                while len(futures) < self.workers and next_index < len(unique):
                    if not self._can_launch(launch_deadline):
                        return
                    key, candidate, aliases = unique[next_index]
                    next_index += 1
                    future = pool.submit(evaluate_if_before_cutoff, candidate)
                    futures[future] = (key, candidate, aliases)

            fill_worker_slots()
            while futures:
                future = next(as_completed(futures))
                key, candidate, aliases = futures.pop(future)
                result = future.result()
                if result is None:
                    batch.missing.extend([candidate, *aliases])
                    fill_worker_slots()
                    continue
                if self.clock() > self.deadline:
                    self.recorder.add(
                        result,
                        phase=phase,
                        completion_status="late",
                        eligible=False,
                    )
                    batch.late.extend([candidate, *aliases])
                    batch.missing.extend([candidate, *aliases])
                else:
                    self.recorder.add(result, phase=phase)
                    self.result_cache[(model_kind, key)] = result
                    batch.fitted.append(result)
                    batch.resolved[candidate.candidate_id] = result
                    print(
                        f"[search] completed {result.candidate_id}: R2={result.pooled_r2:.5f}, "
                        f"RMSE={result.pooled_rmse:.5f}",
                        flush=True,
                    )
                    for alias in aliases:
                        self.recorder.add_alias(
                            alias,
                            model_kind=model_kind,
                            canonical=result,
                            phase=phase,
                            reason="duplicate_in_batch",
                        )
                        batch.resolved[alias.candidate_id] = result
                        batch.reused[alias.candidate_id] = result.candidate_id
                fill_worker_slots()

        for _, candidate, aliases in unique[next_index:]:
            batch.missing.extend([candidate, *aliases])
        if batch.missing:
            print(
                f"[search] {phase}: {len(batch.missing)} candidates were not launched or "
                "completed before the deadline",
                flush=True,
            )
        self._record_batch(phase, batch)
        return batch

    @staticmethod
    def _rank_scores(scores: dict[str, float], source_order: list[str]) -> dict[str, int]:
        ordered = sorted(source_order, key=lambda feature: (-scores.get(feature, 0.0), feature))
        return {feature: index + 1 for index, feature in enumerate(ordered)}

    def _candidate_pool(
        self,
        seeds: dict[str, list[str]],
        mi_ranked: list[str],
        gain_scores: dict[str, float],
        residual_scores: dict[str, float],
    ) -> list[str]:
        source_order = self.data.source_order
        pool_size = int(self.config["search"]["candidate_pool_size"])
        seed_frequency = {
            feature: sum(feature in features for features in seeds.values())
            for feature in source_order
        }
        mi_set = set(mi_ranked[: int(self.config["selection"]["canonical_mi_k"])])
        gain_rank = self._rank_scores(gain_scores, source_order)
        residual_rank = self._rank_scores(residual_scores, source_order)
        gain_top = {feature for feature, rank in gain_rank.items() if rank <= pool_size}
        residual_top = {feature for feature, rank in residual_rank.items() if rank <= pool_size}
        seed_rank = self._rank_scores(seed_frequency, source_order)
        mi_rank = {feature: index + 1 for index, feature in enumerate(mi_ranked)}
        records: list[dict[str, Any]] = []
        for feature in source_order:
            support = (
                int(feature in mi_set)
                + int(seed_frequency[feature] > 0)
                + int(feature in gain_top)
                + int(feature in residual_top)
            )
            ranks = [
                mi_rank.get(feature, len(source_order) + 1),
                seed_rank[feature],
                gain_rank[feature],
                residual_rank[feature],
            ]
            records.append(
                {
                    "feature": feature,
                    "support": support,
                    "consensus_rank": float(np.mean(ranks)),
                    "mi_rank": mi_rank.get(feature, np.nan),
                    "seed_frequency": seed_frequency[feature],
                    "gain_rank": gain_rank[feature],
                    "residual_rank": residual_rank[feature],
                    "gain": gain_scores.get(feature, 0.0),
                    "residual_association": residual_scores.get(feature, 0.0),
                }
            )
        evidence = pd.DataFrame(records).sort_values(
            ["support", "consensus_rank", "feature"], ascending=[False, True, True]
        )
        multi_source = evidence.loc[evidence["support"] >= 2]
        selected = multi_source.head(pool_size)
        if len(selected) < pool_size:
            selected = pd.concat(
                [
                    selected,
                    evidence.loc[~evidence["feature"].isin(selected["feature"])].head(
                        pool_size - len(selected)
                    ),
                ]
            )
        self.evidence = selected.reset_index(drop=True)
        self.evidence.to_csv(self.recorder.artifact_dir / "candidate_pool.csv", index=False)
        return self.evidence["feature"].tolist()

    def _normalize(self, features: Iterable[str], target_size: int, candidate_pool: list[str]) -> list[str]:
        selected = self.evaluator.canonicalize(features)
        pool = [feature for feature in candidate_pool if feature not in selected]
        if len(selected) < target_size:
            selected = self.evaluator.canonicalize(
                [*selected, *pool[: target_size - len(selected)]]
            )
        elif len(selected) > target_size:
            if self.evidence is None:
                raise RuntimeError("Candidate evidence must be built before normalization.")
            rank = dict(zip(self.evidence["feature"], self.evidence.index))
            selected = sorted(
                selected,
                key=lambda feature: (
                    rank.get(feature, len(rank) + self.data.source_order.index(feature)),
                    feature,
                ),
            )[:target_size]
            selected = self.evaluator.canonicalize(selected)
        return selected

    def _weakest_included(self, features: list[str]) -> list[str]:
        if self.evidence is None:
            raise RuntimeError("Candidate evidence must be built before local search.")
        rank = dict(zip(self.evidence["feature"], self.evidence.index))
        return sorted(
            features,
            key=lambda feature: (
                rank.get(feature, len(rank) + self.data.source_order.index(feature)),
                feature,
            ),
            reverse=True,
        )[:8]

    def _local_variants(
        self, current: CandidateResult, candidate_pool: list[str], round_index: int
    ) -> list[Candidate]:
        settings = self.config["search"]
        minimum = int(settings["global_feature_min"])
        maximum = int(settings["global_feature_max"])
        current_features = list(current.global_features)
        outside = [feature for feature in candidate_pool if feature not in current_features]
        generated: dict[tuple[str, ...], Candidate] = {}

        if len(current_features) < maximum:
            for feature in outside:
                proposal = self.evaluator.canonicalize([*current_features, feature])
                generated[tuple(proposal)] = Candidate(
                    f"round_{round_index:02d}_add_{feature}", tuple(proposal)
                )
        if len(current_features) > minimum:
            for feature in current_features:
                proposal = self.evaluator.canonicalize(
                    candidate for candidate in current_features if candidate != feature
                )
                generated[tuple(proposal)] = Candidate(
                    f"round_{round_index:02d}_drop_{feature}", tuple(proposal)
                )
        weakest = self._weakest_included(current_features)
        strongest = outside[:8]
        for remove in weakest:
            for add in strongest:
                proposal = self.evaluator.canonicalize(
                    [feature for feature in current_features if feature != remove] + [add]
                )
                generated[tuple(proposal)] = Candidate(
                    f"round_{round_index:02d}_swap_{remove}_for_{add}", tuple(proposal)
                )
        return list(generated.values())

    def _cluster_external_ranking(
        self,
        global_features: list[str],
        predictions: np.ndarray,
        gain_scores: dict[str, float],
        cluster: int,
        candidate_pool: list[str],
    ) -> list[str]:
        external = [feature for feature in candidate_pool if feature not in set(global_features)]
        if len(external) < max(DELTA_ADDITION_COUNTS):
            raise ValueError(
                f"Cluster {cluster} has only {len(external)} pool-bounded external features; "
                f"{max(DELTA_ADDITION_COUNTS)} are required for the delta grid."
            )
        test_mask = self.evaluator.labels_test == cluster
        residual = pd.Series(
            self.data.test.loc[test_mask, self.data.target].to_numpy(dtype=float)
            - predictions[test_mask],
            index=self.data.test.index[test_mask],
        )
        correlation = {
            feature: (
                float(
                    abs(
                        self.data.test.loc[test_mask, feature].corr(
                            residual, method="spearman"
                        )
                    )
                )
                if pd.notna(
                    self.data.test.loc[test_mask, feature].corr(residual, method="spearman")
                )
                else 0.0
            )
            for feature in external
        }
        gain_rank = self._rank_scores(gain_scores, external)
        correlation_rank = self._rank_scores(correlation, external)
        return sorted(
            external,
            key=lambda feature: (gain_rank[feature] + correlation_rank[feature], feature),
        )[: max(DELTA_ADDITION_COUNTS)]

    def _build_delta_candidates(
        self,
        global_features: list[str],
        delta_rankings: dict[str, list[str]],
    ) -> tuple[list[Candidate], dict[str, tuple[int, int]]]:
        candidates: list[Candidate] = []
        coordinates: dict[str, tuple[int, int]] = {}
        for count_0 in DELTA_ADDITION_COUNTS:
            for count_1 in DELTA_ADDITION_COUNTS:
                candidate_id = f"delta_c0_{count_0}_c1_{count_1}"
                additions = (
                    ("0", tuple(delta_rankings["0"][:count_0])),
                    ("1", tuple(delta_rankings["1"][:count_1])),
                )
                candidates.append(Candidate(candidate_id, tuple(global_features), additions))
                coordinates[candidate_id] = (count_0, count_1)
        return candidates, coordinates

    def _delta_grid_rows(
        self,
        candidates: list[Candidate],
        coordinates: dict[str, tuple[int, int]],
        batch: EvaluationBatch,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            count_0, count_1 = coordinates[candidate.candidate_id]
            additions = candidate.additions_dict()
            result = batch.resolved.get(candidate.candidate_id)
            if result is None:
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "canonical_candidate_id": "",
                        "evaluation_status": "missing",
                        "cluster_0_addition_count": count_0,
                        "cluster_1_addition_count": count_1,
                        "cluster_0_additions": ";".join(additions.get("0", [])),
                        "cluster_1_additions": ";".join(additions.get("1", [])),
                    }
                )
                continue
            record = result.as_record()
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "canonical_candidate_id": batch.reused.get(
                        candidate.candidate_id, result.candidate_id
                    ),
                    "evaluation_status": (
                        "reused" if candidate.candidate_id in batch.reused else "fit"
                    ),
                    "cluster_0_addition_count": count_0,
                    "cluster_1_addition_count": count_1,
                    "cluster_0_additions": ";".join(additions.get("0", [])),
                    "cluster_1_additions": ";".join(additions.get("1", [])),
                    "pooled_r2": record["pooled_r2"],
                    "pooled_rmse": record["pooled_rmse"],
                    "pooled_mae": record["pooled_mae"],
                    "year_2023_r2": record["year_2023_r2"],
                    "year_2024_r2": record["year_2024_r2"],
                    "year_2025_r2": record["year_2025_r2"],
                }
            )
        return rows

    def _validate_calibration(self, baseline: CandidateResult) -> None:
        calibration = self.config["calibration"]
        r2_delta = abs(baseline.pooled_r2 - float(calibration["expected_r2"]))
        rmse_delta = abs(baseline.pooled_rmse - float(calibration["expected_rmse"]))
        if r2_delta > float(calibration["r2_tolerance"]) or rmse_delta > float(
            calibration["rmse_tolerance"]
        ):
            raise RuntimeError(
                "Local Model 16 parity check failed: "
                f"R2={baseline.pooled_r2:.6f}, RMSE={baseline.pooled_rmse:.6f}."
            )

    def _summary(
        self,
        *,
        status: str,
        baseline: CandidateResult | None,
        current: CandidateResult | None,
        candidate_pool: list[str],
        rounds: list[dict[str, Any]],
        delta_rankings: dict[str, list[str]],
        delta_rows: list[dict[str, Any]],
        winner: CandidateResult | None = None,
        incomplete_reason: str | None = None,
    ) -> dict[str, Any]:
        exact_results = [result for result in self.recorder.results if result.model_kind == "exact"]
        missing_delta_ids = [
            row["candidate_id"]
            for row in delta_rows
            if row.get("evaluation_status") == "missing"
        ]
        reused_delta_ids = [
            row["candidate_id"]
            for row in delta_rows
            if row.get("evaluation_status") == "reused"
        ]
        fitted_delta_ids = [
            row["candidate_id"]
            for row in delta_rows
            if row.get("evaluation_status") == "fit"
        ]
        summary: dict[str, Any] = {
            "status": status,
            "selection_goal": "unweighted_pooled_test_r2_2023_2025",
            "deadline": {
                "budget_seconds": self.deadline - self.started,
                "wrapper_launch_cutoff_seconds": self.wrapper_launch_deadline - self.started,
                "elapsed_seconds": self._elapsed_seconds(),
                "phase_stats": self.phase_stats,
            },
            "baseline": baseline.as_record() if baseline is not None else None,
            "best_completed": current.as_record() if current is not None else None,
            "candidate_pool_size": len(candidate_pool),
            "rounds": rounds,
            "delta_rankings": delta_rankings,
            "delta_grid": {
                "status": (
                    "complete"
                    if len(delta_rows) == len(DELTA_ADDITION_COUNTS) ** 2
                    and not missing_delta_ids
                    else "incomplete"
                ),
                "expected_coordinates": [
                    {
                        "cluster_0_additions": count_0,
                        "cluster_1_additions": count_1,
                    }
                    for count_0 in DELTA_ADDITION_COUNTS
                    for count_1 in DELTA_ADDITION_COUNTS
                ],
                "fitted_candidate_ids": fitted_delta_ids,
                "reused_candidate_ids": reused_delta_ids,
                "missing_candidate_ids": missing_delta_ids,
            },
            "completed_exact_candidates": len(exact_results),
            "completed_proxy_candidates": sum(
                result.model_kind == "proxy" for result in self.recorder.results
            ),
            "reused_candidate_count": len(self.recorder.alias_rows),
        }
        if incomplete_reason is not None:
            summary["incomplete_reason"] = incomplete_reason
        if winner is not None:
            summary.update(
                {
                    "winner": winner.as_record(),
                    "winner_yearly_metrics": winner.yearly_metrics,
                    "winner_cluster_metrics": winner.cluster_metrics,
                    "global_features": winner.global_features,
                    "cluster_additions": winner.cluster_additions,
                    "beats_local_v0": (
                        baseline is not None and self.better(winner, baseline)
                    ),
                    "beats_reported_07703": bool(winner.pooled_r2 > 0.7703),
                }
            )
        return summary

    def _fail_incomplete(
        self,
        reason: str,
        *,
        baseline: CandidateResult | None,
        current: CandidateResult | None,
        candidate_pool: list[str],
        rounds: list[dict[str, Any]],
        delta_rankings: dict[str, list[str]],
        delta_rows: list[dict[str, Any]],
    ) -> None:
        summary = self._summary(
            status="incomplete_deadline",
            baseline=baseline,
            current=current,
            candidate_pool=candidate_pool,
            rounds=rounds,
            delta_rankings=delta_rankings,
            delta_rows=delta_rows,
            incomplete_reason=reason,
        )
        self.recorder.write_json("search_summary.json", summary)
        raise SearchIncompleteError(reason)

    def _record_backbone_alias(self, current: CandidateResult) -> None:
        candidate = Candidate(
            "global_backbone_for_deltas",
            tuple(current.global_features),
            tuple(
                (cluster, tuple(current.cluster_additions.get(cluster, [])))
                for cluster in ("0", "1")
            ),
        )
        self.recorder.add_alias(
            candidate,
            model_kind="exact",
            canonical=current,
            phase="final_grid",
            reason="backbone_reference",
        )

    def run(self) -> dict[str, Any]:
        """Run calibration, wrapper rounds, then the required 0/5/10 delta grid."""
        baseline_candidate = Candidate(
            "baseline_v0_calibration",
            tuple(self.data.v0_features),
        )
        baseline_batch = self._evaluate_many(
            [baseline_candidate],
            "exact",
            phase="calibration",
            launch_deadline=self.deadline,
            include_predictions=True,
        )
        baseline = baseline_batch.resolved.get("baseline_v0_calibration")
        if baseline is None:
            self._fail_incomplete(
                "baseline did not complete before the search deadline",
                baseline=None,
                current=None,
                candidate_pool=[],
                rounds=[],
                delta_rankings={},
                delta_rows=[],
            )
        self._validate_calibration(baseline)
        print(
            f"[search] V0 parity: R2={baseline.pooled_r2:.5f}, RMSE={baseline.pooled_rmse:.5f}",
            flush=True,
        )

        global_audit = self.audit_results["global"]["global"]
        seeds, seed_notes = load_seed_sets(self.data, self.config, global_audit)
        if self._can_launch(self.wrapper_launch_deadline):
            gain_scores = self.evaluator.all_feature_gain()
        else:
            gain_scores = {feature: 0.0 for feature in self.data.feature_columns}
            print(
                "[search] wrapper cutoff reached before all-feature gain; using zero gain "
                "only for the final candidate-pool fallback",
                flush=True,
            )
        residual_scores = self.evaluator.residual_association(baseline.predictions)
        candidate_pool = self._candidate_pool(
            seeds,
            global_audit["mi300"].mi_ranked,
            gain_scores,
            residual_scores,
        )
        print(f"[search] candidate pool contains {len(candidate_pool)} features", flush=True)
        self.recorder.write_json(
            "seed_inventory.json",
            {
                "seed_notes": seed_notes,
                "seed_sizes": {name: len(features) for name, features in seeds.items()},
            },
        )

        seed_candidates = [
            Candidate(f"seed_{name}", tuple(self.evaluator.validate_features(features)))
            for name, features in seeds.items()
        ]
        seed_batch = self._evaluate_many(
            seed_candidates,
            "exact",
            phase="seed",
            launch_deadline=self.wrapper_launch_deadline,
            include_predictions=True,
        )
        seed_results = self._distinct_results(seed_batch.resolved.values())
        parent_results = self._distinct_results([baseline, *seed_results])
        viable = [result for result in parent_results if 40 <= len(result.global_features) <= 60]
        parents = sorted(viable or parent_results, key=self._rank_key)[:3]

        normalized_candidates: list[Candidate] = []
        for parent in parents:
            for target_size in (40, 50, 60):
                features = self._normalize(parent.global_features, target_size, candidate_pool)
                normalized_candidates.append(
                    Candidate(f"normalized_{parent.candidate_id}_{target_size}", tuple(features))
                )
        normalized_batch = self._evaluate_many(
            normalized_candidates,
            "exact",
            phase="normalization",
            launch_deadline=self.wrapper_launch_deadline,
            include_predictions=True,
        )
        current = self.best(
            self._distinct_results(
                [baseline, *seed_results, *normalized_batch.resolved.values()]
            )
        )

        rounds: list[dict[str, Any]] = []
        for round_index in range(1, int(self.config["search"]["max_rounds"]) + 1):
            if not self._can_launch(self.wrapper_launch_deadline):
                rounds.append({"round": round_index, "status": "wrapper_deadline"})
                break
            proxy_candidates = self._local_variants(current, candidate_pool, round_index)
            print(
                f"[search] round {round_index}: screening {len(proxy_candidates)} local variants",
                flush=True,
            )
            proxy_batch = self._evaluate_many(
                proxy_candidates,
                "proxy",
                phase=f"round_{round_index}_proxy",
                launch_deadline=self.wrapper_launch_deadline,
            )
            if not proxy_batch.fitted:
                rounds.append({"round": round_index, "status": "no_proxy_results"})
                break
            exact_limit = int(self.config["search"]["exact_attempts_per_round"])
            best_proxy = sorted(proxy_batch.fitted, key=self._rank_key)[:exact_limit]
            exact_candidates = [
                Candidate(
                    f"{result.candidate_id}_exact",
                    tuple(result.global_features),
                )
                for result in best_proxy
            ]
            exact_batch = self._evaluate_many(
                exact_candidates,
                "exact",
                phase=f"round_{round_index}_exact",
                launch_deadline=self.wrapper_launch_deadline,
                include_predictions=True,
            )
            improved = [result for result in exact_batch.fitted if self.better(result, current)]
            if improved:
                current = self.best(improved)
                print(
                    f"[search] round {round_index}: adopted {current.candidate_id}",
                    flush=True,
                )
                rounds.append(
                    {
                        "round": round_index,
                        "status": "improved",
                        "candidate_id": current.candidate_id,
                        "pooled_r2": current.pooled_r2,
                        "pooled_rmse": current.pooled_rmse,
                    }
                )
            else:
                rounds.append({"round": round_index, "status": "no_exact_improvement"})
                break

        if current.predictions is None:
            self._fail_incomplete(
                "the final global backbone has no retained exact predictions",
                baseline=baseline,
                current=current,
                candidate_pool=candidate_pool,
                rounds=rounds,
                delta_rankings={},
                delta_rows=[],
            )

        self._record_backbone_alias(current)
        try:
            delta_rankings = {
                str(cluster): self._cluster_external_ranking(
                    current.global_features,
                    current.predictions,
                    gain_scores,
                    cluster,
                    candidate_pool,
                )
                for cluster in (0, 1)
            }
        except ValueError as error:
            self._fail_incomplete(
                str(error),
                baseline=baseline,
                current=current,
                candidate_pool=candidate_pool,
                rounds=rounds,
                delta_rankings={},
                delta_rows=[],
            )

        print("[search] evaluating the required 0/5/10 add-only delta grid", flush=True)
        delta_candidates, delta_coordinates = self._build_delta_candidates(
            current.global_features,
            delta_rankings,
        )
        delta_batch = self._evaluate_many(
            delta_candidates,
            "exact",
            phase="final_grid",
            launch_deadline=self.deadline,
            include_predictions=True,
        )
        delta_rows = self._delta_grid_rows(
            delta_candidates,
            delta_coordinates,
            delta_batch,
        )
        self.recorder.write_delta_grid(delta_rows)
        if any(row["evaluation_status"] == "missing" for row in delta_rows):
            self._fail_incomplete(
                "the required 0/5/10 delta grid did not complete before the deadline",
                baseline=baseline,
                current=current,
                candidate_pool=candidate_pool,
                rounds=rounds,
                delta_rankings=delta_rankings,
                delta_rows=delta_rows,
            )

        exact_results = [result for result in self.recorder.results if result.model_kind == "exact"]
        winner = self.best(exact_results)
        summary = self._summary(
            status="complete",
            baseline=baseline,
            current=current,
            candidate_pool=candidate_pool,
            rounds=rounds,
            delta_rankings=delta_rankings,
            delta_rows=delta_rows,
            winner=winner,
        )
        self.recorder.write_json("selected_features.json", summary)
        self.recorder.write_json("search_summary.json", summary)
        return summary
