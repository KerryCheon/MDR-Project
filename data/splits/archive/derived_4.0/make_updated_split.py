import json
import os

import pandas as pd

BASE_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/base"
DERIVED_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_3.0"
OUT_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_4.0"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_FILES = ("train_base.csv", "val_base.csv", "test_base.csv")
DERIVED_FILES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}

OUT_FILES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}

SOURCE_META_PATH = os.path.join(DERIVED_DIR, "split_meta.json")
OUT_META_PATH = os.path.join(OUT_DIR, "split_meta.json")

KEY_COLS = ["station_id", "date"]
UPDATE_COLS = ["elev", "slope"]


def _assert_no_dupes(df: pd.DataFrame, keys, label: str):
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        raise ValueError(
            f"{label} has duplicate rows for key {keys}: {int(dup.sum())} duplicate rows found."
        )


def _load_base_table() -> pd.DataFrame:
    frames = []
    for fname in BASE_FILES:
        path = os.path.join(BASE_DIR, fname)
        frames.append(pd.read_csv(path, usecols=KEY_COLS + UPDATE_COLS))

    base_df = pd.concat(frames, ignore_index=True)
    _assert_no_dupes(base_df, KEY_COLS, "Base split")
    return base_df


def _build_base_lookups(base_df: pd.DataFrame):
    nunique = base_df.groupby("station_id")[UPDATE_COLS].nunique(dropna=True)
    bad = nunique[(nunique["elev"] > 1) | (nunique["slope"] > 1)]
    if not bad.empty:
        raise ValueError(
            "Base split has non-constant elev/slope within at least one station. "
            "Cannot safely use station fallback. Bad stations: "
            + ", ".join(bad.index.tolist())
        )

    date_lookup = base_df.rename(
        columns={
            "elev": "elev_base_date",
            "slope": "slope_base_date",
        }
    )

    station_lookup = (
        base_df.sort_values(KEY_COLS)
        .drop_duplicates(subset=["station_id"], keep="first")
        .loc[:, ["station_id"] + UPDATE_COLS]
        .rename(
            columns={
                "elev": "elev_base_station",
                "slope": "slope_base_station",
            }
        )
    )

    return date_lookup, station_lookup


def _update_one_split(df: pd.DataFrame, date_lookup: pd.DataFrame, station_lookup: pd.DataFrame):
    missing = sorted(set(KEY_COLS + UPDATE_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Derived split is missing required columns: {missing}")

    original_cols = df.columns.tolist()

    merged = df.merge(date_lookup, on=KEY_COLS, how="left")
    merged = merged.merge(station_lookup, on="station_id", how="left")

    elev_source = pd.Series("original", index=merged.index)
    elev_source[merged["elev_base_station"].notna()] = "base_station"
    elev_source[merged["elev_base_date"].notna()] = "base_date"

    slope_source = pd.Series("original", index=merged.index)
    slope_source[merged["slope_base_station"].notna()] = "base_station"
    slope_source[merged["slope_base_date"].notna()] = "base_date"

    merged["elev"] = (
        merged["elev_base_date"]
        .combine_first(merged["elev_base_station"])
        .combine_first(merged["elev"])
    )
    merged["slope"] = (
        merged["slope_base_date"]
        .combine_first(merged["slope_base_station"])
        .combine_first(merged["slope"])
    )

    merged = merged.drop(
        columns=["elev_base_date", "slope_base_date", "elev_base_station", "slope_base_station"]
    )
    merged = merged.loc[:, original_cols]

    stats = {
        "rows": int(len(merged)),
        "elev_source_counts": {
            "base_date": int((elev_source == "base_date").sum()),
            "base_station": int((elev_source == "base_station").sum()),
            "original": int((elev_source == "original").sum()),
        },
        "slope_source_counts": {
            "base_date": int((slope_source == "base_date").sum()),
            "base_station": int((slope_source == "base_station").sum()),
            "original": int((slope_source == "original").sum()),
        },
    }
    return merged, stats


def _read_source_meta():
    if not os.path.exists(SOURCE_META_PATH):
        return {}
    with open(SOURCE_META_PATH, "r") as f:
        return json.load(f)


def main():
    base_df = _load_base_table()
    date_lookup, station_lookup = _build_base_lookups(base_df)

    split_stats = {}

    for split_name, in_file in DERIVED_FILES.items():
        in_path = os.path.join(DERIVED_DIR, in_file)
        out_path = os.path.join(OUT_DIR, OUT_FILES[split_name])

        derived_df = pd.read_csv(in_path)
        updated_df, stats = _update_one_split(derived_df, date_lookup, station_lookup)

        if len(updated_df) != len(derived_df):
            raise RuntimeError(
                f"Row count changed for {split_name}: {len(derived_df)} -> {len(updated_df)}"
            )

        updated_df.to_csv(out_path, index=False)
        split_stats[split_name] = stats

        print(
            f"[{split_name}] rows={stats['rows']} "
            f"| elev(base_date/base_station/original)="
            f"{stats['elev_source_counts']['base_date']}/"
            f"{stats['elev_source_counts']['base_station']}/"
            f"{stats['elev_source_counts']['original']} "
            f"| slope(base_date/base_station/original)="
            f"{stats['slope_source_counts']['base_date']}/"
            f"{stats['slope_source_counts']['base_station']}/"
            f"{stats['slope_source_counts']['original']}"
        )

    source_meta = _read_source_meta()
    meta = {
        "source_split": "derived_new",
        "updated_split": "derived_new_updated",
        "update_rule": "overwrite elev/slope from base split using (station_id,date), then station_id fallback",
        "keys_used": KEY_COLS,
        "updated_columns": UPDATE_COLS,
        "base_files": [os.path.join(BASE_DIR, f) for f in BASE_FILES],
        "derived_files": {k: os.path.join(DERIVED_DIR, v) for k, v in DERIVED_FILES.items()},
        "output_files": {k: os.path.join(OUT_DIR, v) for k, v in OUT_FILES.items()},
        "rows": {k: int(v["rows"]) for k, v in split_stats.items()},
        "source_counts": {
            split: {
                "elev": stats["elev_source_counts"],
                "slope": stats["slope_source_counts"],
            }
            for split, stats in split_stats.items()
        },
    }

    for k in ("seed", "split_by", "policy", "variant", "target", "meta_cols_kept", "stations"):
        if k in source_meta:
            meta[f"source_{k}"] = source_meta[k]

    with open(OUT_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved updated split files:")
    for split_name, out_file in OUT_FILES.items():
        print(f"  {split_name}: {os.path.join(OUT_DIR, out_file)}")
    print(f"Meta:\n  {OUT_META_PATH}")


if __name__ == "__main__":
    main()
