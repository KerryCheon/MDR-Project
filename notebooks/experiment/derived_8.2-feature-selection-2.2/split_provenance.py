"""Stable split-file loading and SHA-256 validation helpers."""

from __future__ import annotations

from pathlib import Path
import string

import pandas as pd

from artifact_state import sha256_file


def read_hashed_csv(path: Path, **read_csv_kwargs) -> tuple[pd.DataFrame, str]:
    """Read one CSV and prove that its bytes stayed stable during the read."""
    before = sha256_file(path)
    frame = pd.read_csv(path, **read_csv_kwargs)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"split file changed while it was being read: {path}")
    return frame, after


def validate_sha256(value: object, *, source: str) -> str:
    """Return a normalized SHA-256 digest or reject malformed provenance."""
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in string.hexdigits for character in digest):
        raise ValueError(f"invalid SHA-256 digest in {source}: {value!r}")
    return digest
