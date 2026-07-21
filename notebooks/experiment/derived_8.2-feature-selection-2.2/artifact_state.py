"""Reproducible source-state capture and transactional artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


COMPLETION_MARKER = "completion.json"
SOURCE_SUFFIXES = {".lock", ".py", ".toml", ".yaml", ".yml"}
EXCLUDED_PARTS = {
    ".ipynb_checkpoints",
    ".venv",
    "__pycache__",
    "artifacts",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(project_root: Path, args: list[str]) -> bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=project_root,
        stderr=subprocess.DEVNULL,
    )


def _source_paths(project_root: Path, experiment_dir: Path) -> list[Path]:
    candidates = [
        project_root / "Modeling" / "Src",
        project_root / "Modeling" / "Utils",
        experiment_dir,
        project_root / "notebooks" / "pyproject.toml",
        project_root / "notebooks" / "uv.lock",
    ]
    paths = []
    for candidate in candidates:
        if candidate.is_file():
            paths.append(candidate)
            continue
        if not candidate.is_dir():
            continue
        for path in candidate.rglob("*"):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(candidate).parts
            if EXCLUDED_PARTS.intersection(relative_parts):
                continue
            if path.suffix in SOURCE_SUFFIXES:
                paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(project_root).as_posix())


def capture_source_state(project_root: Path, experiment_dir: Path) -> dict:
    """Capture commit identity plus dirty and untracked runtime source content."""
    project_root = project_root.resolve()
    experiment_dir = experiment_dir.resolve()
    files = {}
    tree_digest = hashlib.sha256()
    for path in _source_paths(project_root, experiment_dir):
        relative = path.relative_to(project_root).as_posix()
        file_hash = sha256_file(path)
        files[relative] = file_hash
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(file_hash.encode("ascii"))
        tree_digest.update(b"\0")

    try:
        commit = _git_output(project_root, ["rev-parse", "HEAD"]).decode().strip()
        status = _git_output(
            project_root,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        )
        tracked_diff = _git_output(project_root, ["diff", "--binary", "HEAD", "--"])
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        status = b""
        tracked_diff = b""

    dirty_entries = [
        entry.decode("utf-8", errors="replace")
        for entry in status.split(b"\0")
        if entry
    ]
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_entries),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_tree_sha256": tree_digest.hexdigest(),
        "source_file_count": len(files),
        "source_files": files,
        "dirty_entries": dirty_entries,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_csv(frame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=index)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def invalidate_completion(
    artifact_dir: Path,
    *,
    marker_name: str = COMPLETION_MARKER,
) -> None:
    (artifact_dir / marker_name).unlink(missing_ok=True)


def write_completion_marker(
    artifact_dir: Path,
    required_files: Iterable[str],
    *,
    source_state: dict,
    marker_name: str = COMPLETION_MARKER,
) -> None:
    names = list(dict.fromkeys(required_files))
    missing = [name for name in names if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot complete artifact; missing files: {missing}")
    payload = {
        "version": 1,
        "source_tree_sha256": source_state["source_tree_sha256"],
        "files": {
            name: sha256_file(artifact_dir / name)
            for name in names
        },
    }
    atomic_write_json(artifact_dir / marker_name, payload)


def artifact_is_complete(
    artifact_dir: Path,
    required_files: Iterable[str],
    *,
    marker_name: str = COMPLETION_MARKER,
    expected_source_tree_sha256: str | None = None,
) -> bool:
    marker_path = artifact_dir / marker_name
    try:
        with open(marker_path, encoding="utf-8") as stream:
            marker = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    if (
        expected_source_tree_sha256 is not None
        and marker.get("source_tree_sha256") != expected_source_tree_sha256
    ):
        return False
    recorded = marker.get("files", {})
    for name in dict.fromkeys(required_files):
        path = artifact_dir / name
        if not path.is_file() or name not in recorded:
            return False
        if sha256_file(path) != recorded[name]:
            return False
    return True
