"""Transactional artifact helpers for resumable experiment stages."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Iterable


COMPLETION_MARKER = "completion.json"
DEFAULT_SHARED_FILE_MODE = 0o664


def _replacement_mode(path: Path) -> int:
    """Preserve a readable target mode and repair owner-only legacy outputs."""
    if not path.exists():
        return DEFAULT_SHARED_FILE_MODE
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & stat.S_IRGRP:
        return mode
    return mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        os.chmod(temporary_path, _replacement_mode(path))
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
        os.chmod(temporary_path, _replacement_mode(path))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, _replacement_mode(path))
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
    marker_name: str = COMPLETION_MARKER,
) -> None:
    names = list(dict.fromkeys(required_files))
    missing = [name for name in names if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot complete artifact; missing files: {missing}")
    payload = {
        "version": 1,
        "files": {name: sha256_file(artifact_dir / name) for name in names},
    }
    atomic_write_json(artifact_dir / marker_name, payload)


def artifact_is_complete(
    artifact_dir: Path,
    required_files: Iterable[str],
    *,
    marker_name: str = COMPLETION_MARKER,
) -> bool:
    marker_path = artifact_dir / marker_name
    try:
        with open(marker_path, encoding="utf-8") as stream:
            marker = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    recorded = marker.get("files", {})
    names = list(dict.fromkeys(required_files)) or list(recorded)
    for name in names:
        path = artifact_dir / name
        if not path.is_file() or name not in recorded:
            return False
        if sha256_file(path) != recorded[name]:
            return False
    return True
