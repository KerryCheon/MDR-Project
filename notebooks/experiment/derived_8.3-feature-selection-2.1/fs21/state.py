"""Resumable run journal with code/config/split/runtime fingerprinting."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable

from .artifacts import atomic_write_json, sha256_file, stable_json_hash
from .constants import DEVELOPMENT_STAGE_NAMES, EXP_DIR, PROJECT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def runtime_input_paths() -> list[Path]:
    paths = []
    for pattern in ("*.py", "*.yaml", "fs21/*.py"):
        paths.extend(EXP_DIR.glob(pattern))
    return sorted(path for path in set(paths) if path.is_file())


def input_hashes(extra_paths: Iterable[Path] = ()) -> dict[str, str]:
    paths = list(runtime_input_paths()) + list(extra_paths)
    output = {}
    for path in sorted(set(path.resolve() for path in paths)):
        try:
            key = str(path.relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            key = str(path)
        output[key] = sha256_file(path)
    return output


def environment_record(*, device: str, workers: int) -> dict:
    package_names = ["numpy", "pandas", "scikit-learn", "xgboost", "pyyaml"]
    packages = {}
    for name in package_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "missing"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "device": device,
        "workers": int(workers),
    }


def build_fingerprint(
    *,
    device: str,
    workers: int,
    split_paths: Iterable[Path],
    extra_paths: Iterable[Path] = (),
    smoke: bool,
) -> dict:
    split_hashes = {
        str(path.resolve().relative_to(PROJECT_ROOT.resolve())): sha256_file(path)
        for path in split_paths
    }
    payload = {
        "git_revision": git_revision(),
        "runtime_inputs": input_hashes(extra_paths),
        "split_hashes": split_hashes,
        "device": device,
        "workers": int(workers),
        "smoke": bool(smoke),
    }
    payload["fingerprint_sha256"] = stable_json_hash(payload)
    return payload


class RunJournal:
    def __init__(
        self,
        path: Path,
        *,
        command: list[str],
        fingerprint: dict,
        environment: dict,
    ) -> None:
        self.path = path
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "resume fingerprint mismatch; use --restart after reviewing changes"
                )
            self.payload = payload
            self.payload["command"] = command
        else:
            self.payload = {
                "version": 1,
                "command": command,
                "fingerprint": fingerprint,
                "environment": environment,
                "started": utc_now(),
                "completed": None,
                "status": "running",
                "failure": None,
                "stages": [],
            }
            self._write()

    def _write(self) -> None:
        atomic_write_json(self.path, self.payload)

    def stage_status(self, name: str) -> str | None:
        rows = [row for row in self.payload["stages"] if row["name"] == name]
        return rows[-1]["status"] if rows else None

    def stage_started(self, name: str, command: list[str]) -> None:
        self.payload["stages"].append(
            {
                "name": name,
                "command": command,
                "status": "running",
                "started": utc_now(),
                "completed": None,
                "failure": None,
            }
        )
        self._write()

    def stage_completed(self, name: str) -> None:
        row = self._active_stage(name)
        row["status"] = "complete"
        row["completed"] = utc_now()
        self._write()

    def stage_failed(self, name: str, error: BaseException) -> None:
        row = self._active_stage(name)
        row["status"] = "failed"
        row["completed"] = utc_now()
        row["failure"] = {"type": type(error).__name__, "message": str(error)}
        self.payload["status"] = "failed"
        self.payload["failure"] = row["failure"]
        self._write()

    def _active_stage(self, name: str) -> dict:
        for row in reversed(self.payload["stages"]):
            if row["name"] == name and row["status"] == "running":
                return row
        raise ValueError(f"no active stage named {name}")

    def complete(self) -> None:
        self.payload["status"] = "complete"
        self.payload["completed"] = utc_now()
        self.payload["failure"] = None
        self._write()


def safe_restart(root: Path, stage: str | None) -> list[str]:
    """Remove only 2.1 stage directories at or downstream of a requested stage."""
    resolved_root = root.resolve()
    artifacts_root = (EXP_DIR / "artifacts").resolve()
    if not resolved_root.is_relative_to(artifacts_root):
        raise ValueError(f"restart root is outside 2.1 artifacts: {root}")
    if stage is None:
        start = 0
    else:
        if stage not in DEVELOPMENT_STAGE_NAMES:
            raise ValueError(f"unknown restart stage: {stage}")
        start = DEVELOPMENT_STAGE_NAMES.index(stage)
    removed = []
    for name in DEVELOPMENT_STAGE_NAMES[start:]:
        path = resolved_root / "stages" / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(name)
    state_path = resolved_root / "run_state.json"
    if state_path.exists():
        state_path.unlink()
    if resolved_root.name == "development" and start <= 12:
        freeze = EXP_DIR / "development_freeze.json"
        freeze.unlink(missing_ok=True)
    return removed

