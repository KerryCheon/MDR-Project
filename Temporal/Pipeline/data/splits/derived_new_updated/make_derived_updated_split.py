from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path("/Users/jbalkovec/Desktop")
BASE_DIR = REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/base"
DERIVED_NEW_DIR = REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived_new_updated"
SPLIT_DIR = REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived_updated"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train_derived_updated.csv"
VAL_PATH = SPLIT_DIR / "val_derived_updated.csv"
TEST_PATH = SPLIT_DIR / "test_derived_updated.csv"
META_PATH = SPLIT_DIR / "split_meta_derived_updated.json"

KEY_COLS = ["station_id", "date"]


def _load_split(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} split: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing key columns: {missing}")
    df["station_id"] = df["station_id"].astype(str)
    df["date"] = df["date"].astype(str)
    dup = df.duplicated(subset=KEY_COLS, keep=False)
    if dup.any():
        raise ValueError(f"{label} has duplicate (station_id, date) rows: {int(dup.sum())}")
    return df


def _merge_splits(derived_path: Path, base_path: Path, label: str) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    derived = _load_split(derived_path, f"derived_new {label}")
    base = _load_split(base_path, f"base {label}")

    base_extra_cols = [c for c in base.columns if c not in derived.columns and c not in KEY_COLS]
    base_extra = base[KEY_COLS + base_extra_cols]

    merged = derived.merge(base_extra, on=KEY_COLS, how="left", validate="one_to_one")

    if len(merged) != len(derived):
        raise ValueError(
            f"Row count mismatch after merge for {label}: derived={len(derived)} merged={len(merged)}"
        )

    missing_base = 0
    if base_extra_cols:
        missing_base = int(merged[base_extra_cols].isna().all(axis=1).sum())

    stats = {
        "rows": int(len(merged)),
        "base_extra_cols": int(len(base_extra_cols)),
        "missing_base_rows": missing_base,
    }
    return merged, stats, base_extra_cols


def main() -> None:
    splits = {
        "train": (DERIVED_NEW_DIR / "train_derived_new.csv", BASE_DIR / "train_base.csv"),
        "val": (DERIVED_NEW_DIR / "val_derived_new.csv", BASE_DIR / "val_base.csv"),
        "test": (DERIVED_NEW_DIR / "test_derived_new.csv", BASE_DIR / "test_base.csv"),
    }

    outputs = {}
    meta_rows = {}
    base_extra_union = set()

    for label, (derived_path, base_path) in splits.items():
        merged, stats, base_extra_cols = _merge_splits(derived_path, base_path, label)
        outputs[label] = merged
        meta_rows[label] = stats
        base_extra_union.update(base_extra_cols)

    outputs["train"].to_csv(TRAIN_PATH, index=False)
    outputs["val"].to_csv(VAL_PATH, index=False)
    outputs["test"].to_csv(TEST_PATH, index=False)

    meta = {
        "variant": "derived_updated",
        "description": "derived_new splits merged with base splits (append base-only columns)",
        "key_cols": KEY_COLS,
        "source": {
            "base_dir": str(BASE_DIR),
            "derived_new_dir": str(DERIVED_NEW_DIR),
        },
        "rows": meta_rows,
        "base_extra_cols": sorted(base_extra_union),
        "columns": {
            "train": int(outputs["train"].shape[1]),
            "val": int(outputs["val"].shape[1]),
            "test": int(outputs["test"].shape[1]),
        },
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved derived_updated splits:")
    print(f"  {TRAIN_PATH}")
    print(f"  {VAL_PATH}")
    print(f"  {TEST_PATH}")
    print(f"Meta: {META_PATH}")


if __name__ == "__main__":
    main()
