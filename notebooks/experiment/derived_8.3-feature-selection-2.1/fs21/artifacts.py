"""Atomic, hash-addressed artifact helpers used by every 2.1 stage."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


COMPLETION_MARKER = "completion.json"
DEFAULT_MODE = 0o664


class _NumpyEncoder(json.JSONEncoder):
    def default(self, value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return super().default(value)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_hash(payload: object) -> str:
    text = json.dumps(
        payload,
        cls=_NumpyEncoder,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_text(text)


def _replacement_mode(path: Path) -> int:
    if not path.exists():
        return DEFAULT_MODE
    mode = stat.S_IMODE(path.stat().st_mode)
    return mode if mode & stat.S_IRGRP else mode | 0o044


def _temporary_path(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    return descriptor, Path(name)


def atomic_write_json(path: Path, payload: object) -> None:
    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, cls=_NumpyEncoder, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, _replacement_mode(path))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, _replacement_mode(path))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(frame, path: Path, *, index: bool = False) -> None:
    """Write CSV or deterministic gzip CSV without exposing a partial file."""
    descriptor, temporary = _temporary_path(path)
    os.close(descriptor)
    try:
        compression = None
        if path.suffix == ".gz":
            compression = {"method": "gzip", "mtime": 0}
        frame.to_csv(temporary, index=index, compression=compression)
        os.chmod(temporary, _replacement_mode(path))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def invalidate_completion(directory: Path) -> None:
    (directory / COMPLETION_MARKER).unlink(missing_ok=True)


def write_completion(directory: Path, required_files: Iterable[str]) -> dict:
    names = list(dict.fromkeys(required_files))
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot complete {directory}; missing {missing}")
    payload = {
        "version": 1,
        "files": {name: sha256_file(directory / name) for name in names},
    }
    atomic_write_json(directory / COMPLETION_MARKER, payload)
    return payload


def completion_is_valid(directory: Path, required_files: Iterable[str] = ()) -> bool:
    marker = directory / COMPLETION_MARKER
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded = payload.get("files")
    if not isinstance(recorded, dict):
        return False
    names = list(required_files) or list(recorded)
    for name in names:
        path = directory / name
        if name not in recorded or not path.is_file():
            return False
        if sha256_file(path) != recorded[name]:
            return False
    return True

