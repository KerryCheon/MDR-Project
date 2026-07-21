import importlib.util
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "notebooks/experiment/derived_8.2-feature-selection-2.2/artifact_state.py"
)


def _load_artifact_state_module():
    spec = importlib.util.spec_from_file_location(
        "feature_selection_artifact_state",
        MODULE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_source_state_captures_tracked_and_untracked_changes(tmp_path):
    module = _load_artifact_state_module()
    experiment_dir = tmp_path / "notebooks/experiment/test"
    source_path = tmp_path / "Modeling/Src/example.py"
    config_path = experiment_dir / "config.yaml"
    source_path.parent.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    config_path.write_text("workers: 16\n", encoding="utf-8")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "initial")

    clean = module.capture_source_state(tmp_path, experiment_dir)
    assert clean["git_dirty"] is False

    source_path.write_text("VALUE = 2\n", encoding="utf-8")
    new_path = experiment_dir / "new_runner.py"
    new_path.write_text("VALUE = 3\n", encoding="utf-8")
    dirty = module.capture_source_state(tmp_path, experiment_dir)

    assert dirty["git_dirty"] is True
    assert dirty["source_tree_sha256"] != clean["source_tree_sha256"]
    assert dirty["tracked_diff_sha256"] != clean["tracked_diff_sha256"]
    assert str(new_path.relative_to(tmp_path)) in dirty["source_files"]
    assert dirty["dirty_entries"]


def test_completion_marker_rejects_missing_or_changed_files(tmp_path):
    module = _load_artifact_state_module()
    source_state = {"source_tree_sha256": "source-hash"}
    module.atomic_write_json(tmp_path / "selected.json", {"selected": ["x"]})
    module.atomic_write_json(tmp_path / "outer.json", {"score": 1.0})
    required = ["selected.json", "outer.json"]
    module.write_completion_marker(
        tmp_path,
        required,
        source_state=source_state,
    )
    assert module.artifact_is_complete(tmp_path, required)
    assert not module.artifact_is_complete(
        tmp_path,
        required,
        expected_source_tree_sha256="different-source",
    )

    (tmp_path / "outer.json").write_text(json.dumps({"score": 2.0}), encoding="utf-8")
    assert not module.artifact_is_complete(tmp_path, required)

    module.atomic_write_json(tmp_path / "outer.json", {"score": 1.0})
    module.write_completion_marker(
        tmp_path,
        required,
        source_state=source_state,
    )
    (tmp_path / "selected.json").unlink()
    assert not module.artifact_is_complete(tmp_path, required)


def test_invalidated_completion_cannot_be_reused(tmp_path):
    module = _load_artifact_state_module()
    source_state = {"source_tree_sha256": "source-hash"}
    module.atomic_write_json(tmp_path / "result.json", {"value": 1})
    module.write_completion_marker(
        tmp_path,
        ["result.json"],
        source_state=source_state,
    )
    module.invalidate_completion(tmp_path)
    assert not module.artifact_is_complete(tmp_path, ["result.json"])
