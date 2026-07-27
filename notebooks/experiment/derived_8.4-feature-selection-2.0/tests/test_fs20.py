"""Focused tests for the isolated feature-selection 2.0 implementation."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from fs20.audit import classify_collapse
from fs20.config import FORBIDDEN_TEMPORAL_KEYS, _walk_keys, load_config
from fs20.evaluate import V0Router
from fs20.search import Candidate
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
