import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "notebooks/experiment/derived_8.2-feature-selection-2.2/"
    "run_candidate_diagnostics.py"
)
EVAL_SCRIPT_PATH = SCRIPT_PATH.parent / "run_eval.py"


def _load_diagnostics_module():
    spec = importlib.util.spec_from_file_location(
        "feature_selection_candidate_diagnostics",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_PATH.parent))
    return module


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "feature_selection_evaluation",
        EVAL_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EVAL_SCRIPT_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EVAL_SCRIPT_PATH.parent))
    return module


def _write_payload(path: Path, key: str, feature_sets: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: [{"features": features} for features in feature_sets]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_crossed_candidate_diagnostics_load_outer_union(tmp_path):
    module = _load_diagnostics_module()
    artifact_set = "crossed_candidates_locked_outer"
    global_dir = tmp_path / artifact_set / "derived_8.2" / "global"
    _write_payload(
        global_dir / "outer_selection.json",
        "candidate_summaries",
        [["station"], ["temporal"], ["station"]],
    )
    _write_payload(
        global_dir / "forward_time_inner_selection.json",
        "selection_path",
        [["temporal"]],
    )
    module.write_completion_marker(
        global_dir,
        ["outer_selection.json"],
    )

    candidates = module._candidate_sets(
        "derived_8.2",
        artifact_set,
        artifact_root=tmp_path,
    )

    assert candidates == [["station"], ["temporal"]]


def test_nested_candidate_diagnostics_keep_inner_path(tmp_path):
    module = _load_diagnostics_module()
    global_dir = tmp_path / "nested" / "derived_8.0" / "global"
    _write_payload(
        global_dir / "inner_selection.json",
        "selection_path",
        [["large", "small"], ["large"]],
    )
    module.write_completion_marker(
        global_dir,
        ["inner_selection.json"],
    )

    candidates = module._candidate_sets(
        "derived_8.0",
        "nested",
        artifact_root=tmp_path,
    )

    assert candidates == [["large", "small"], ["large"]]


def test_evaluation_defaults_are_cuda_and_four_workers():
    module = _load_eval_module()
    parser = module.argparse.ArgumentParser()
    module.add_runtime_arguments(parser)
    args = parser.parse_args([])
    assert args.device == "cuda"
    assert args.workers == 4
