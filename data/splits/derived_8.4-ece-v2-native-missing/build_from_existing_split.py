"""Build the native-missing ECE split from the committed derived_8.4-ece split.

The raw and processed ECE station files are not present in this checkout. This
recovery builder first verifies against the committed satellite caches that
SMAP AM and PM are unavailable for every station/week, then reverses the known
zero-fill error in every SMAP-derived feature while preserving the 499-column
schema used by the trained models.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / "data" / "splits" / "derived_8.4-ece"
OUTPUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "src" / "pipeline" / "data" / "cache"

STATION_CACHE_NAMES = {
    "ECE_BBG_Lost_Meadow": "ece_bbg_lost_meadow_satellite_cache.json",
    "ECE_BBG_Main_St": "ece_bbg_main_st_satellite_cache.json",
    "ECE_Renton_Garden_North": "ece_renton_garden_north_satellite_cache.json",
    "ECE_Renton_Garden_Shed": "ece_renton_garden_shed_satellite_cache.json",
    "ECE_Renton_Home": "ece_renton_home_satellite_cache.json",
}


def load_and_verify_missing_smap(test_df: pd.DataFrame) -> dict[str, int]:
    """Verify every evaluation row is covered only by null SMAP cache entries."""
    verified_rows: dict[str, int] = {}
    dates = pd.to_datetime(test_df["date"], errors="raise")

    for station_id, cache_name in STATION_CACHE_NAMES.items():
        station_rows = test_df.loc[test_df["station_id"].eq(station_id)]
        station_dates = dates.loc[station_rows.index]
        with (CACHE_DIR / cache_name).open(encoding="utf-8") as handle:
            cache = json.load(handle)

        covered = pd.Series(False, index=station_rows.index)
        for date_key, entry in cache.items():
            start_text, end_text = date_key.split("_")
            start = pd.Timestamp(start_text)
            end = pd.Timestamp(end_text)
            in_interval = station_dates.ge(start) & station_dates.lt(end)
            if not in_interval.any():
                continue
            if entry.get("SMAP_sm_am") is not None or entry.get("SMAP_sm_pm") is not None:
                raise ValueError(f"Finite SMAP found for {station_id} in {date_key}; migration is unsafe")
            covered.loc[in_interval.index[in_interval]] = True

        if not covered.all():
            missing_dates = station_dates.loc[~covered].dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(f"Cache does not cover {station_id} dates: {missing_dates}")
        verified_rows[station_id] = int(covered.sum())

    return verified_rows


def restore_native_missing(test_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Restore all SMAP source and descendant columns to native missing values."""
    corrected = test_df.copy()
    smap_columns = [column for column in corrected.columns if "SMAP" in column]
    mask_columns = [column for column in smap_columns if column.endswith("_mask")]
    value_columns = [column for column in smap_columns if column not in mask_columns]

    corrected[value_columns] = np.nan
    corrected[mask_columns] = 0
    return corrected, smap_columns


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(SOURCE_DIR / "test.csv", low_memory=False)
    verified_rows = load_and_verify_missing_smap(test_df)
    corrected, smap_columns = restore_native_missing(test_df)

    for filename in ("train.csv", "val.csv"):
        shutil.copy2(SOURCE_DIR / filename, OUTPUT_DIR / filename)
    corrected.to_csv(OUTPUT_DIR / "test.csv", index=False)
    corrected.to_csv(OUTPUT_DIR / "eval.csv", index=False)
    shutil.copy2(SOURCE_DIR / "config.yaml", OUTPUT_DIR / "config.yaml")
    shutil.copy2(SOURCE_DIR / "dataset_metadata.py", OUTPUT_DIR / "dataset_metadata.py")

    with (SOURCE_DIR / "split_meta.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata.update({
        "parent_split": "derived_8.4-ece",
        "builder": "data/splits/derived_8.4-ece-v2-native-missing/build_from_existing_split.py",
        "missing_data_policy": "Native NaN for unavailable SMAP source and derived features; observation masks set to 0",
        "smap_columns_corrected": len(smap_columns),
        "cache_verified_rows_by_station": verified_rows,
    })
    with (OUTPUT_DIR / "split_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved corrected split to {OUTPUT_DIR}")
    print(f"Rows: {len(corrected)}")
    print(f"SMAP columns corrected: {len(smap_columns)}")
    smap_value_columns = [column for column in smap_columns if not column.endswith("_mask")]
    print(f"SMAP finite feature values after correction: {int(corrected[smap_value_columns].notna().sum().sum())}")


if __name__ == "__main__":
    main()
