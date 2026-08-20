#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent


def validate(smoke: bool = False) -> None:
    artifacts = EXPERIMENT_DIR / ("artifacts_smoke" if smoke else "artifacts")
    required = [
        "screen_leaderboard.csv",
        "confirmation_runs.csv",
        "confirmation_summary.csv",
        "winner_test_runs.csv",
        "winner_ensemble_metrics.json",
        "run_manifest.json",
        "RESULTS.md",
    ]
    missing = [name for name in required if not (artifacts / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")

    manifest = json.loads((artifacts / "run_manifest.json").read_text(encoding="utf-8"))
    screen = pd.read_csv(artifacts / "screen_leaderboard.csv")
    confirmation = pd.read_csv(artifacts / "confirmation_summary.csv")
    final = pd.read_csv(artifacts / "winner_test_runs.csv")
    winner = manifest["winner"]

    if len(screen) != int(manifest["candidate_count"]):
        raise AssertionError("Candidate manifest and screen row count differ")
    if winner != confirmation.iloc[0]["candidate"]:
        raise AssertionError("Winner is not the highest validation-confirmed candidate")
    if set(final["candidate"]) != {winner}:
        raise AssertionError("A losing candidate was evaluated on test")
    if manifest["test_target_used_for_candidate_selection"]:
        raise AssertionError("Manifest reports test-target selection")
    if not manifest["state_free"]:
        raise AssertionError("Experiment is not marked state-free")
    if not screen["hybrid_validation_r2"].is_monotonic_decreasing:
        raise AssertionError("Screen leaderboard is not ordered by validation R2")
    print(
        f"validated {len(screen)} candidates; winner={winner}; "
        f"test_runs={len(final)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    validate(parser.parse_args().smoke)
