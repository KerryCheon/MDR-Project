import importlib.util
import json
from pathlib import Path
import stat
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "notebooks/experiment/derived_8.2-feature-selection-2.2/artifact_state.py"


def _module():
    spec = importlib.util.spec_from_file_location("feature_selection_artifact_state", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completion_marker_rejects_missing_or_changed_files(tmp_path):
    module = _module()
    module.atomic_write_json(tmp_path / "selected.json", {"selected": ["x"]})
    module.atomic_write_json(tmp_path / "outer.json", {"score": 1.0})
    module.write_completion_marker(tmp_path, ["selected.json", "outer.json"])
    assert module.artifact_is_complete(tmp_path, ["selected.json", "outer.json"])

    (tmp_path / "outer.json").write_text(json.dumps({"score": 2.0}), encoding="utf-8")
    assert not module.artifact_is_complete(tmp_path, ["selected.json", "outer.json"])

    module.atomic_write_json(tmp_path / "outer.json", {"score": 1.0})
    module.write_completion_marker(tmp_path, ["selected.json", "outer.json"])
    (tmp_path / "selected.json").unlink()
    assert not module.artifact_is_complete(tmp_path, ["selected.json", "outer.json"])


def test_completion_marker_without_requested_files_checks_all_recorded_files(tmp_path):
    module = _module()
    module.atomic_write_json(tmp_path / "result.json", {"value": 1})
    module.write_completion_marker(tmp_path, ["result.json"])
    assert module.artifact_is_complete(tmp_path, [])
    (tmp_path / "result.json").unlink()
    assert not module.artifact_is_complete(tmp_path, [])


def test_completion_marker_can_guard_an_output_outside_artifact_directory(tmp_path):
    module = _module()
    report_dir = tmp_path / "artifacts/report"
    report_dir.mkdir(parents=True)
    markdown = tmp_path / "RESULTS.md"
    markdown.write_text("generated\n", encoding="utf-8")
    module.write_completion_marker(report_dir, ["../../RESULTS.md"])
    assert module.artifact_is_complete(report_dir, [])

    markdown.write_text("manually edited\n", encoding="utf-8")
    assert not module.artifact_is_complete(report_dir, [])


def test_invalidated_completion_cannot_be_reused(tmp_path):
    module = _module()
    module.atomic_write_json(tmp_path / "result.json", {"value": 1})
    module.write_completion_marker(tmp_path, ["result.json"])
    module.invalidate_completion(tmp_path)
    assert not module.artifact_is_complete(tmp_path, ["result.json"])


def test_atomic_writes_preserve_or_repair_shared_file_modes(tmp_path):
    module = _module()

    existing = tmp_path / "report.md"
    existing.write_text("old", encoding="utf-8")
    existing.chmod(0o664)
    module.atomic_write_text(existing, "new")
    assert stat.S_IMODE(existing.stat().st_mode) == 0o664

    owner_only = tmp_path / "manifest.json"
    owner_only.write_text("{}", encoding="utf-8")
    owner_only.chmod(0o600)
    module.atomic_write_json(owner_only, {"value": 1})
    assert stat.S_IMODE(owner_only.stat().st_mode) & stat.S_IRGRP

    created = tmp_path / "evidence.csv"
    module.atomic_write_csv(pd.DataFrame({"value": [1]}), created)
    assert stat.S_IMODE(created.stat().st_mode) == module.DEFAULT_SHARED_FILE_MODE


def test_master_runner_stage_order_and_defaults():
    experiment_dir = MODULE_PATH.parent
    sys.path.insert(0, str(experiment_dir))
    try:
        import run_all
    finally:
        sys.path.remove(str(experiment_dir))
    assert [stage["name"] for stage in run_all.STAGES] == [
        "selection",
        "validation",
        "final_retrospective",
        "nested_selection",
        "nested_locked_outer",
        "crossed_selection",
        "progressive_selection",
        "nested_diagnostics",
        "crossed_diagnostics",
        "progressive_diagnostics",
        "nested_retrospective",
        "crossed_retrospective",
        "report",
    ]
    assert run_all.DEFAULT_DEVICE == "cuda"
    assert run_all.DEFAULT_WORKERS == 4
