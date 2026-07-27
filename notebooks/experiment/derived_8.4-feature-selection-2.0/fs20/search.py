"""Bounded direct feature-wrapper search for the 2023–2025 target period."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import ExperimentData
from .evaluate import CandidateResult, ModelEvaluator
from .seeds import load_seed_sets
from .selection import SelectionResult


@dataclass(frozen=True)
class Candidate:
    """A literal candidate supplied to the evaluator."""

    candidate_id: str
    global_features: tuple[str, ...]
    cluster_additions: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def additions_dict(self) -> dict[str, list[str]]:
        return {cluster: list(features) for cluster, features in self.cluster_additions}


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

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[CandidateResult] = []
        self.rows: list[dict[str, Any]] = []

    def add(self, result: CandidateResult) -> None:
        self.results.append(result)
        self.rows.append(result.as_record())
        self.flush()

    def flush(self) -> None:
        path = self.artifact_dir / "search_results.csv"
        temporary = path.with_suffix(".writing.csv")
        pd.DataFrame(self.rows).to_csv(temporary, index=False)
        temporary.replace(path)

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self.artifact_dir / name
        temporary = path.with_suffix(".writing.json")
        temporary.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


class DirectSearch:
    """A six-round direct wrapper search with exact final decisions."""

    def __init__(
        self,
        data: ExperimentData,
        config: dict[str, Any],
        audit_results: dict[str, dict[str, dict[str, SelectionResult]]],
        *,
        workers: int,
        deadline_minutes: int,
    ) -> None:
        self.data = data
        self.config = config
        self.audit_results = audit_results
        self.workers = max(1, int(workers))
        self.evaluator = ModelEvaluator(data, config)
        self.recorder = SearchRecorder(Path(config["artifacts"]["directory"]))
        self.started = time.monotonic()
        self.deadline = self.started + max(1, int(deadline_minutes)) * 60
        reserve_seconds = int(config["search"]["final_reserve_minutes"]) * 60
        self.last_launch = self.deadline - reserve_seconds
        self.evidence: pd.DataFrame | None = None

    @staticmethod
    def _result_key(result: CandidateResult) -> tuple[Any, ...]:
        additions = tuple(
            (cluster, tuple(features))
            for cluster, features in sorted(result.cluster_additions.items())
        )
        return (tuple(result.global_features), additions)

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

    def _can_launch(self) -> bool:
        return time.monotonic() < self.last_launch

    def _evaluate_many(
        self, candidates: Iterable[Candidate], model_kind: str, *, include_predictions: bool = False
    ) -> list[CandidateResult]:
        pending = list(candidates)
        if not pending:
            return []
        print(f"[search] evaluating {len(pending)} {model_kind} candidates with {self.workers} workers", flush=True)
        completed: list[CandidateResult] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {}
            for candidate in pending:
                if not self._can_launch():
                    break
                future = pool.submit(
                    self.evaluator.evaluate,
                    candidate.candidate_id,
                    candidate.global_features,
                    candidate.additions_dict(),
                    model_kind=model_kind,
                    include_predictions=include_predictions,
                )
                futures[future] = candidate
            for future in as_completed(futures):
                result = future.result()
                self.recorder.add(result)
                completed.append(result)
                print(
                    f"[search] completed {result.candidate_id}: R2={result.pooled_r2:.5f}, "
                    f"RMSE={result.pooled_rmse:.5f}",
                    flush=True,
                )
        return completed

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
            support = int(feature in mi_set) + int(seed_frequency[feature] > 0) + int(feature in gain_top) + int(feature in residual_top)
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
            selected = pd.concat([selected, evidence.loc[~evidence["feature"].isin(selected["feature"])].head(pool_size - len(selected))])
        self.evidence = selected.reset_index(drop=True)
        self.evidence.to_csv(self.recorder.artifact_dir / "candidate_pool.csv", index=False)
        return self.evidence["feature"].tolist()

    def _normalize(self, features: Iterable[str], target_size: int, candidate_pool: list[str]) -> list[str]:
        selected = self.evaluator.canonicalize(features)
        pool = [feature for feature in candidate_pool if feature not in selected]
        if len(selected) < target_size:
            selected = self.evaluator.canonicalize([*selected, *pool[: target_size - len(selected)]])
        elif len(selected) > target_size:
            if self.evidence is None:
                raise RuntimeError("Candidate evidence must be built before normalization.")
            rank = dict(zip(self.evidence["feature"], self.evidence.index))
            selected = sorted(
                selected,
                key=lambda feature: (rank.get(feature, len(rank) + self.data.source_order.index(feature)), feature),
            )[:target_size]
            selected = self.evaluator.canonicalize(selected)
        return selected

    def _weakest_included(self, features: list[str]) -> list[str]:
        if self.evidence is None:
            raise RuntimeError("Candidate evidence must be built before local search.")
        rank = dict(zip(self.evidence["feature"], self.evidence.index))
        return sorted(features, key=lambda feature: (rank.get(feature, len(rank) + self.data.source_order.index(feature)), feature), reverse=True)[:8]

    def _local_variants(self, current: CandidateResult, candidate_pool: list[str], round_index: int) -> list[Candidate]:
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
    ) -> list[str]:
        test_mask = self.evaluator.labels_test == cluster
        residual = pd.Series(
            self.data.test.loc[test_mask, self.data.target].to_numpy(dtype=float) - predictions[test_mask],
            index=self.data.test.index[test_mask],
        )
        correlation = {
            feature: float(abs(self.data.test.loc[test_mask, feature].corr(residual, method="spearman")))
            if pd.notna(self.data.test.loc[test_mask, feature].corr(residual, method="spearman"))
            else 0.0
            for feature in self.data.feature_columns
        }
        external = [feature for feature in self.data.feature_columns if feature not in global_features]
        gain_rank = self._rank_scores(gain_scores, external)
        correlation_rank = self._rank_scores(correlation, external)
        return sorted(
            external,
            key=lambda feature: (gain_rank[feature] + correlation_rank[feature], feature),
        )[:10]

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

    def run(self) -> dict[str, Any]:
        """Run calibration, seed evaluation, six wrapper rounds, then 0/5/10 deltas."""
        baseline = self.evaluator.evaluate(
            "baseline_v0_calibration",
            self.data.v0_features,
            model_kind="exact",
            include_predictions=True,
        )
        self.recorder.add(baseline)
        self._validate_calibration(baseline)
        print(
            f"[search] V0 parity: R2={baseline.pooled_r2:.5f}, RMSE={baseline.pooled_rmse:.5f}",
            flush=True,
        )

        global_audit = self.audit_results["global"]["global"]
        seeds, seed_notes = load_seed_sets(self.data, self.config, global_audit)
        gain_scores = self.evaluator.all_feature_gain()
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
            {"seed_notes": seed_notes, "seed_sizes": {name: len(features) for name, features in seeds.items()}},
        )

        seed_candidates = [
            Candidate(f"seed_{name}", tuple(self.evaluator.validate_features(features)))
            for name, features in seeds.items()
        ]
        seed_results = self._evaluate_many(seed_candidates, "exact")
        if not seed_results:
            raise RuntimeError("No seed candidates completed before the deadline.")
        viable = [result for result in seed_results if 40 <= len(result.global_features) <= 60]
        parents = sorted(viable or seed_results, key=self._rank_key)[:3]
        normalized_candidates: list[Candidate] = []
        for parent in parents:
            for target_size in (40, 50, 60):
                features = self._normalize(parent.global_features, target_size, candidate_pool)
                normalized_candidates.append(
                    Candidate(f"normalized_{parent.candidate_id}_{target_size}", tuple(features))
                )
        normalized_results = self._evaluate_many(normalized_candidates, "exact")
        current = self.best([baseline, *seed_results, *normalized_results])

        rounds: list[dict[str, Any]] = []
        for round_index in range(1, int(self.config["search"]["max_rounds"]) + 1):
            if not self._can_launch():
                rounds.append({"round": round_index, "status": "deadline"})
                break
            proxy_candidates = self._local_variants(current, candidate_pool, round_index)
            print(
                f"[search] round {round_index}: screening {len(proxy_candidates)} local variants",
                flush=True,
            )
            proxy_results = self._evaluate_many(proxy_candidates, "proxy")
            if not proxy_results:
                rounds.append({"round": round_index, "status": "no_proxy_results"})
                break
            exact_limit = int(self.config["search"]["exact_attempts_per_round"])
            best_proxy = sorted(proxy_results, key=self._rank_key)[:exact_limit]
            exact_candidates = [
                Candidate(
                    f"{result.candidate_id}_exact",
                    tuple(result.global_features),
                )
                for result in best_proxy
            ]
            exact_results = self._evaluate_many(exact_candidates, "exact")
            improved = [result for result in exact_results if self.better(result, current)]
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

        global_for_deltas = self.evaluator.evaluate(
            "global_backbone_for_deltas",
            current.global_features,
            model_kind="exact",
            include_predictions=True,
        )
        self.recorder.add(global_for_deltas)
        delta_rankings = {
            str(cluster): self._cluster_external_ranking(
                global_for_deltas.global_features,
                global_for_deltas.predictions,
                gain_scores,
                cluster,
            )
            for cluster in (0, 1)
        }
        print("[search] evaluating the 0/5/10 add-only delta grid", flush=True)
        delta_candidates: list[Candidate] = []
        for count_0 in (0, 5, 10):
            for count_1 in (0, 5, 10):
                additions = (
                    ("0", tuple(delta_rankings["0"][:count_0])),
                    ("1", tuple(delta_rankings["1"][:count_1])),
                )
                delta_candidates.append(
                    Candidate(
                        f"delta_c0_{count_0}_c1_{count_1}",
                        tuple(global_for_deltas.global_features),
                        additions,
                    )
                )
        delta_results = self._evaluate_many(delta_candidates, "exact")
        exact_results = [result for result in self.recorder.results if result.model_kind == "exact"]
        winner = self.best(exact_results)
        summary = {
            "selection_goal": "unweighted_pooled_test_r2_2023_2025",
            "baseline": baseline.as_record(),
            "winner": winner.as_record(),
            "winner_yearly_metrics": winner.yearly_metrics,
            "winner_cluster_metrics": winner.cluster_metrics,
            "global_features": winner.global_features,
            "cluster_additions": winner.cluster_additions,
            "candidate_pool_size": len(candidate_pool),
            "rounds": rounds,
            "delta_rankings": delta_rankings,
            "completed_exact_candidates": len(exact_results),
            "completed_proxy_candidates": sum(
                result.model_kind == "proxy" for result in self.recorder.results
            ),
            "beats_local_v0": self.better(winner, baseline),
            "beats_reported_07703": bool(winner.pooled_r2 > 0.7703),
        }
        self.recorder.write_json("selected_features.json", summary)
        self.recorder.write_json("search_summary.json", summary)
        return summary
