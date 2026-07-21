"""Generate temporal and station/time paths, then select on locked outer folds.

The two candidate generators isolate temporal generalization from joint
station/time extrapolation. Their candidate lists are unioned without using the
outer labels, then the locked final learner selects on future held-out stations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from artifact_state import (
    artifact_is_complete,
    atomic_write_json,
    capture_source_state,
    write_completion_marker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Modeling.Src.soilmoist_fl.Selectors.grouped_oof import (  # noqa: E402
    evaluate_forward_station_time_candidates,
    select_grouped_oof,
)
from run_nested_selection import (  # noqa: E402
    NESTED_REQUIRED_FILES,
    _load_inner_outer,
    _prepare_xy,
    _probe_device,
    _write_nested_artifact,
)


CROSSED_REQUIRED_FILES = NESTED_REQUIRED_FILES + (
    "candidate_sources.json",
    "forward_time_inner_selection.json",
)
TEMPORAL_COMPLETION_MARKER = "forward_time_inner_selection.complete.json"


def _unique_candidates(*paths: list[dict]) -> list[list[str]]:
    candidates = []
    seen = set()
    for path in paths:
        for candidate in path:
            features = list(candidate["features"])
            key = tuple(features)
            if key not in seen:
                candidates.append(features)
                seen.add(key)
    return candidates


def _merge_outer_results(station_result: dict, temporal_result: dict) -> dict:
    summaries = []
    seen = set()
    for result in (station_result, temporal_result):
        for summary in result["candidate_summaries"]:
            key = tuple(summary["features"])
            if key not in seen:
                summaries.append(summary)
                seen.add(key)
    winner = min(
        summaries,
        key=lambda item: (item["upper_confidence_bound"], item["n_features"]),
    )
    merged = dict(temporal_result)
    merged.update(
        {
            "selected": list(winner["features"]),
            "candidate_summaries": summaries,
            "stopping_reason": "minimum_crossed_candidate_outer_ucb",
        }
    )
    return merged


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Rescore bridge sizes so the initial pruning step is not one-shot.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("derived_8.0", "derived_8.2"),
        help="Limit the run; may be repeated. Defaults to both datasets.",
    )
    args = parser.parse_args(argv)
    with open(EXP_DIR / "nested_config.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    device = _probe_device()
    inner_config = json.loads(json.dumps(config["inner_selection"]))
    inner_config["fold_strategy"] = "forward_time"
    inner_config["progressive_elimination"] = bool(args.progressive)
    inner_config["model_params"]["device"] = device
    outer_config = json.loads(json.dumps(config["outer_selection"]))
    outer_config["model_params"]["device"] = device
    source_state = capture_source_state(PROJECT_ROOT, EXP_DIR)
    artifact_name = (
        "progressive_crossed_locked_outer"
        if args.progressive
        else "crossed_candidates_locked_outer"
    )
    output_root = EXP_DIR / "artifacts" / artifact_name
    summaries = []

    datasets = tuple(args.dataset or ("derived_8.0", "derived_8.2"))
    for dataset in datasets:
        scope_dir = output_root / dataset / "global"
        selected_path = scope_dir / "selected_features.json"
        sources_path = scope_dir / "candidate_sources.json"
        if artifact_is_complete(
            scope_dir,
            CROSSED_REQUIRED_FILES,
            expected_source_tree_sha256=source_state["source_tree_sha256"],
        ):
            with open(selected_path, encoding="utf-8") as stream:
                selected_payload = json.load(stream)
            with open(sources_path, encoding="utf-8") as stream:
                source_payload = json.load(stream)
            summary = {
                "dataset": dataset,
                "n_features": int(selected_payload["n_features"]),
                "selected_sources": source_payload["selected_sources"],
                "n_unique_candidates": source_payload["n_unique_candidates"],
            }
            summaries.append(summary)
            print(f"Reusing completed crossed result for {dataset}", flush=True)
            continue
        train, outer, hashes = _load_inner_outer(dataset)
        station_col = config["data"]["station_col"]
        time_col = config["data"]["time_col"]
        X_inner, y_inner, context_inner = _prepare_xy(
            train,
            station_col,
            time_col,
        )
        X_outer, y_outer, context_outer = _prepare_xy(
            outer,
            station_col,
            time_col,
        )

        scope_dir.mkdir(parents=True, exist_ok=True)
        temporal_path = scope_dir / "forward_time_inner_selection.json"
        if artifact_is_complete(
            scope_dir,
            [temporal_path.name],
            marker_name=TEMPORAL_COMPLETION_MARKER,
            expected_source_tree_sha256=source_state["source_tree_sha256"],
        ):
            with open(temporal_path, encoding="utf-8") as stream:
                temporal_inner = json.load(stream)
            print(f"Reusing forward-time inner result for {dataset}", flush=True)
        else:
            temporal_inner = select_grouped_oof(
                X_inner,
                y_inner,
                context_inner,
                config=inner_config,
            )
            atomic_write_json(temporal_path, temporal_inner)
            write_completion_marker(
                scope_dir,
                [temporal_path.name],
                source_state=source_state,
                marker_name=TEMPORAL_COMPLETION_MARKER,
            )
            print(f"Saved forward-time inner result for {dataset}", flush=True)

        station_path = (
            EXP_DIR
            / "artifacts/nested"
            / dataset
            / "global/inner_selection.json"
        )
        if not artifact_is_complete(
            station_path.parent,
            NESTED_REQUIRED_FILES,
        ):
            raise FileNotFoundError(
                "Run nested selection first; source artifact is incomplete: "
                f"{station_path.parent}"
            )
        with open(station_path, encoding="utf-8") as stream:
            station_inner = json.load(stream)
        candidates = _unique_candidates(
            station_inner["selection_path"],
            temporal_inner["selection_path"],
        )
        locked_station_path = (
            EXP_DIR
            / "artifacts/nested_locked_outer"
            / dataset
            / "global/outer_selection.json"
        )
        if artifact_is_complete(
            locked_station_path.parent,
            NESTED_REQUIRED_FILES,
        ):
            with open(locked_station_path, encoding="utf-8") as stream:
                station_outer = json.load(stream)
            temporal_candidates = _unique_candidates(
                temporal_inner["selection_path"]
            )
            temporal_outer = evaluate_forward_station_time_candidates(
                X_inner,
                y_inner,
                context_inner,
                X_outer,
                y_outer,
                context_outer,
                temporal_candidates,
                config=outer_config,
            )
            outer_result = _merge_outer_results(station_outer, temporal_outer)
        else:
            outer_result = evaluate_forward_station_time_candidates(
                X_inner,
                y_inner,
                context_inner,
                X_outer,
                y_outer,
                context_outer,
                candidates,
                config=outer_config,
            )
        _write_nested_artifact(
            scope_dir,
            dataset=dataset,
            scope=artifact_name,
            inner_result=temporal_inner,
            outer_result=outer_result,
            split_hashes=hashes,
            device=device,
            source_state=source_state,
            write_completion=False,
        )
        sources = []
        selected = outer_result["selected"]
        if any(selected == item["features"] for item in station_inner["selection_path"]):
            sources.append("station_time")
        if any(selected == item["features"] for item in temporal_inner["selection_path"]):
            sources.append("forward_time")
        source_manifest = {
            "created": datetime.now(timezone.utc).isoformat(),
            "source_state": source_state,
            "station_time_inner": str(station_path.relative_to(PROJECT_ROOT)),
            "forward_time_inner": "forward_time_inner_selection.json",
            "n_unique_candidates": len(candidates),
            "selected_sources": sources,
        }
        atomic_write_json(scope_dir / "candidate_sources.json", source_manifest)
        write_completion_marker(
            scope_dir,
            CROSSED_REQUIRED_FILES,
            source_state=source_state,
        )
        summary = {
            "dataset": dataset,
            "n_features": len(selected),
            "selected_sources": sources,
            "n_unique_candidates": len(candidates),
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    atomic_write_json(
        output_root / "selection_summary.json",
        {
            "device": device,
            "inner_parallel_workers": inner_config["parallel_workers"],
            "outer_parallel_workers": outer_config["parallel_workers"],
            "source_state": source_state,
            "datasets": summaries,
        },
    )


if __name__ == "__main__":
    main()
