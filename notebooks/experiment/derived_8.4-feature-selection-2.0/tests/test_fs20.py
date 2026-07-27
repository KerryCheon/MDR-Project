"""Focused tests for the isolated feature-selection 2.0 implementation."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from fs20.audit import classify_collapse
from fs20.config import FORBIDDEN_TEMPORAL_KEYS, _walk_keys, load_config
from fs20.evaluate import CandidateResult, V0Router
from fs20.reporting import build_exact_leaderboard
from fs20.search import Candidate, DirectSearch, EvaluationBatch, SearchIncompleteError
from fs20 import selection
from fs20.selection import get_bypass_features, select_features


def _selector_frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "A_signal": rng.normal(size=48),
            "V_signal": rng.normal(size=48),
            "J_soil": np.repeat([0.1, 0.4, 0.7, 0.9], 12),
            "K_aspect": np.repeat([0.2, 0.3, 0.5, 0.8], 12),
            "DOY": np.tile(np.arange(12), 4),
            "G_API": rng.normal(size=48),
        }
    )
    target = 0.7 * frame["A_signal"] + 0.2 * frame["J_soil"] + rng.normal(scale=0.02, size=48)
    return frame, target


def test_legacy_bypass_is_explicit_and_true_off_is_empty() -> None:
    frame, target = _selector_frame()
    bypass = get_bypass_features(frame.columns)
    assert {"J_soil", "K_aspect", "DOY", "G_API"}.issubset(bypass)

    legacy = select_features(
        frame,
        target,
        "legacy_forced_bypass",
        top_k=6,
        elasticnet_k=6,
        bootstrap_k=6,
        n_boot=2,
        sample_fraction=0.8,
        min_freq=0.6,
        min_keep=3,
        random_state=42,
    )
    true_off = select_features(
        frame,
        target,
        "mi300",
        top_k=6,
        elasticnet_k=6,
        bootstrap_k=6,
        n_boot=2,
        sample_fraction=0.8,
        min_freq=0.6,
        min_keep=3,
        random_state=42,
    )
    assert set(bypass).issubset(legacy.candidate_features)
    assert true_off.bypass_features == []


def test_repaired_fallback_uses_pre_stability_candidates() -> None:
    frame, target = _selector_frame()
    result = select_features(
        frame,
        target,
        "mi300_repaired",
        top_k=5,
        elasticnet_k=5,
        bootstrap_k=5,
        n_boot=2,
        sample_fraction=0.8,
        min_freq=1.1,
        min_keep=3,
        random_state=42,
    )
    assert result.fallback_applied
    assert len(result.stable_selected) == 0
    assert len(result.repaired_selected) >= 3
    assert set(result.repaired_selected).issubset(result.candidate_features)


def test_router_uses_train_only_mean() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 1.0, 2.0, 2.0]})
    test = pd.DataFrame({"x": [999.0], "y": [999.0]})
    router = V0Router(["x", "y"], seed=42).fit(train)
    assert router.means is not None
    assert router.means["x"] == 1.5
    assert router.means["y"] == 1.5
    assert len(router.predict(test)) == 1


def test_collapse_labels_are_stable() -> None:
    assert classify_collapse(2) == "hard_collapsed"
    assert classify_collapse(20) == "truncated"
    assert classify_collapse(50) == "healthy"


def test_checked_in_config_has_no_forbidden_temporal_keys() -> None:
    config = load_config(EXPERIMENT_DIR / "config.yaml")
    assert _walk_keys(config) == []
    copied = deepcopy(config)
    copied["selection"]["temporal_weight"] = 0.2
    assert _walk_keys(copied) == ["selection.temporal_weight"]
    assert "temporal_weight" in FORBIDDEN_TEMPORAL_KEYS


def test_candidate_interface_can_only_supply_additions() -> None:
    candidate = Candidate(
        "delta",
        ("A_signal", "V_signal"),
        (("0", ("J_soil",)), ("1", ("G_API",))),
    )
    assert candidate.additions_dict() == {"0": ["J_soil"], "1": ["G_API"]}


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _ScriptedClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.last = values[-1]

    def __call__(self) -> float:
        try:
            self.last = next(self.values)
        except StopIteration:
            pass
        return self.last


class _Evaluator:
    def __init__(self, clock: _Clock, *, duration: float = 0.0) -> None:
        self.clock = clock
        self.duration = duration
        self.calls: list[str] = []
        self.labels_test = np.array([0, 0, 0, 0])

    @staticmethod
    def validate_features(features: object) -> list[str]:
        return list(dict.fromkeys(features))  # type: ignore[arg-type]

    @staticmethod
    def canonicalize(features: object) -> list[str]:
        return sorted(dict.fromkeys(features))  # type: ignore[arg-type]

    def evaluate(
        self,
        candidate_id: str,
        global_features: object,
        cluster_additions: dict[str, list[str]],
        *,
        model_kind: str,
        include_predictions: bool,
    ) -> CandidateResult:
        self.calls.append(candidate_id)
        self.clock.advance(self.duration)
        return CandidateResult(
            candidate_id=candidate_id,
            global_features=list(global_features),  # type: ignore[arg-type]
            cluster_additions={
                "0": list(cluster_additions.get("0", [])),
                "1": list(cluster_additions.get("1", [])),
            },
            pooled_r2=0.8,
            pooled_rmse=0.04,
            pooled_mae=0.03,
            yearly_metrics={
                "2023": {"r2": 0.8},
                "2024": {"r2": 0.8},
                "2025": {"r2": 0.8},
            },
            cluster_metrics={},
            train_time_s=self.duration,
            model_kind=model_kind,
            predictions=np.zeros(4) if include_predictions else None,
        )


def _search_config(tmp_path: Path) -> dict:
    return {
        "artifacts": {"directory": tmp_path},
        "search": {
            "final_reserve_minutes": 0,
            "candidate_pool_size": 12,
            "global_feature_min": 1,
            "global_feature_max": 2,
            "max_rounds": 1,
            "exact_attempts_per_round": 1,
        },
    }


def _search_data() -> SimpleNamespace:
    feature_columns = ["global", *[f"pool_{index}" for index in range(10)], "outside"]
    test = pd.DataFrame(
        {
            "target": [0.0, 1.0, 0.0, 1.0],
            **{
                feature: [0.0, 1.0, 0.0, 1.0]
                for feature in feature_columns
            },
        }
    )
    return SimpleNamespace(
        feature_columns=feature_columns,
        source_order=feature_columns,
        target="target",
        test=test,
    )


def test_bounded_scheduler_does_not_launch_work_after_phase_cutoff(tmp_path: Path) -> None:
    clock = _Clock()
    evaluator = _Evaluator(clock, duration=1.0)
    search = DirectSearch(
        _search_data(),
        _search_config(tmp_path),
        {},
        workers=1,
        deadline_minutes=1,
        evaluator=evaluator,
        clock=clock,
    )
    batch = search._evaluate_many(
        [Candidate("first", ("global",)), Candidate("second", ("pool_0",)), Candidate("third", ("pool_1",))],
        "exact",
        phase="test",
        launch_deadline=2.0,
    )

    assert evaluator.calls == ["first", "second"]
    assert [candidate.candidate_id for candidate in batch.missing] == ["third"]
    assert search.phase_stats["test"] == {
        "requested": 3,
        "fitted": 2,
        "reused": 0,
        "missing": 1,
        "late": 0,
    }


def test_worker_rechecks_cutoff_before_accepted_task_starts(tmp_path: Path) -> None:
    # The submission check sees 0.0, while the worker's own check sees 1.0.
    # This simulates an executor delay between accepting and starting the task.
    clock = _ScriptedClock([0.0, 0.0, 1.0])
    evaluator = _Evaluator(_Clock())
    search = DirectSearch(
        _search_data(),
        _search_config(tmp_path),
        {},
        workers=1,
        deadline_minutes=1,
        evaluator=evaluator,
        clock=clock,
    )
    batch = search._evaluate_many(
        [Candidate("accepted_but_late", ("global",))],
        "exact",
        phase="test",
        launch_deadline=0.5,
    )

    assert evaluator.calls == []
    assert [candidate.candidate_id for candidate in batch.missing] == ["accepted_but_late"]


def test_equivalent_configurations_are_fitted_once_and_recorded_as_aliases(tmp_path: Path) -> None:
    clock = _Clock()
    evaluator = _Evaluator(clock)
    search = DirectSearch(
        _search_data(),
        _search_config(tmp_path),
        {},
        workers=1,
        deadline_minutes=1,
        evaluator=evaluator,
        clock=clock,
    )
    batch = search._evaluate_many(
        [Candidate("canonical", ("global",)), Candidate("same_features", ("global",))],
        "exact",
        phase="test",
        launch_deadline=search.deadline,
        include_predictions=True,
    )

    assert evaluator.calls == ["canonical"]
    assert batch.reused == {"same_features": "canonical"}
    aliases = pd.read_csv(tmp_path / "candidate_aliases.csv")
    assert aliases.loc[0, "candidate_id"] == "same_features"
    assert aliases.loc[0, "canonical_candidate_id"] == "canonical"


def test_cluster_specialists_cannot_escape_candidate_pool(tmp_path: Path) -> None:
    clock = _Clock()
    evaluator = _Evaluator(clock)
    data = _search_data()
    search = DirectSearch(
        data,
        _search_config(tmp_path),
        {},
        workers=1,
        deadline_minutes=1,
        evaluator=evaluator,
        clock=clock,
    )
    candidate_pool = ["global", *[f"pool_{index}" for index in range(10)]]
    ranking = search._cluster_external_ranking(
        ["global"],
        np.array([0.0, 0.0, 0.0, 0.0]),
        {"outside": 10_000.0, **{feature: 0.0 for feature in candidate_pool}},
        0,
        candidate_pool,
    )

    assert len(ranking) == 10
    assert set(ranking).issubset(set(candidate_pool) - {"global"})
    assert "outside" not in ranking


def test_stage_funnel_records_actual_elasticnet_nonzero_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0], "c": [0.5, 0.5]})
    target = pd.Series([0.0, 1.0])
    monkeypatch.setattr(
        selection,
        "_rank_mi",
        lambda X, y, k, random_state: (["a", "b", "c"], {"a": 1.0, "b": 0.5, "c": 0.25}),
    )
    monkeypatch.setattr(
        selection,
        "_fit_elasticnet",
        lambda X, y, k, random_state, alpha=None, l1_ratio=None: (
            ["a"],
            ["a", "b", "c"],
            {"a": 1.0, "b": 0.5, "c": 0.25},
            0.1,
            0.5,
        ),
    )
    monkeypatch.setattr(selection, "_stability", lambda *args, **kwargs: (["a"], {"a": 1.0}))

    result = select_features(
        frame,
        target,
        "mi300",
        top_k=2,
        elasticnet_k=1,
        bootstrap_k=1,
        n_boot=2,
        sample_fraction=0.8,
        min_freq=0.6,
        min_keep=1,
        random_state=42,
    )

    assert result.enet_nonzero == 3
    assert result.stage_counts["elasticnet_nonzero"] == 3


def test_unique_delta_winner_has_a_reportable_singleton_alias_group() -> None:
    results = pd.DataFrame(
        [
            {
                "candidate_id": "delta_c0_5_c1_0",
                "model_kind": "exact",
                "completion_status": "on_time",
                "pooled_r2": 0.81,
                "pooled_rmse": 0.04,
                "global_feature_count": 2,
                "cluster_0_feature_count": 7,
                "cluster_1_feature_count": 2,
                "year_2023_r2": 0.8,
                "year_2024_r2": 0.81,
                "year_2025_r2": 0.82,
                "global_features": "a;b",
                "cluster_0_additions": "c;d;e;f;g",
                "cluster_1_additions": "",
            }
        ]
    )
    aliases = pd.DataFrame(
        columns=[
            "candidate_id",
            "canonical_candidate_id",
            "model_kind",
            "phase",
            "reason",
            "global_features",
            "cluster_0_additions",
            "cluster_1_additions",
        ]
    )

    report = build_exact_leaderboard(results, aliases, "delta_c0_5_c1_0")

    assert report.winner_aliases == ("delta_c0_5_c1_0",)
    assert report.alias_groups.empty


def test_incomplete_search_persists_status_without_selected_features(tmp_path: Path) -> None:
    clock = _Clock()
    evaluator = _Evaluator(clock)
    (tmp_path / "selected_features.json").write_text('{"status": "complete"}\n', encoding="utf-8")
    search = DirectSearch(
        _search_data(),
        _search_config(tmp_path),
        {},
        workers=1,
        deadline_minutes=1,
        evaluator=evaluator,
        clock=clock,
    )
    assert not (tmp_path / "selected_features.json").exists()
    running = json.loads((tmp_path / "search_summary.json").read_text(encoding="utf-8"))
    assert running["status"] == "running"
    with pytest.raises(SearchIncompleteError, match="delta grid"):
        search._fail_incomplete(
            "the required 0/5/10 delta grid did not complete before the deadline",
            baseline=None,
            current=None,
            candidate_pool=[],
            rounds=[],
            delta_rankings={},
            delta_rows=[{"candidate_id": "delta_c0_10_c1_10", "evaluation_status": "missing"}],
        )

    summary = json.loads((tmp_path / "search_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete_deadline"
    assert not (tmp_path / "selected_features.json").exists()
