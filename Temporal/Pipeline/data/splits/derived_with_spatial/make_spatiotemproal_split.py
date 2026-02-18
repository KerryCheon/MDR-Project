# make_spatiotemporal_split_static_append.py
# Jakob Balkovec

# This script merges the static spatial features with the existing split

import os
import json
import pandas as pd


SPLIT_DIR_IN = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_new_updated"
TRAIN_IN = os.path.join(SPLIT_DIR_IN, "train_derived_new_updated.csv")
VAL_IN   = os.path.join(SPLIT_DIR_IN, "val_derived_new_updated.csv")
TEST_IN  = os.path.join(SPLIT_DIR_IN, "test_derived_new_updated.csv")

STATION_STATIC_IN = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_with_spatial/spatial/station_static_features.csv"

SPLIT_DIR_OUT = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_with_spatial"
os.makedirs(SPLIT_DIR_OUT, exist_ok=True)

TRAIN_OUT = os.path.join(SPLIT_DIR_OUT, "train_plus_static.csv")
VAL_OUT   = os.path.join(SPLIT_DIR_OUT, "val_plus_static.csv")
TEST_OUT  = os.path.join(SPLIT_DIR_OUT, "test_plus_static.csv")
META_OUT  = os.path.join(SPLIT_DIR_OUT, "split_meta_plus_static.json")

DATE_COL = "date"
GROUP_COL = "station_id"

CANON_IDS = {
    "Darrington": "Darrington",
    "Quinault": "Quinault",
    "Spokane": "Spokane",
    "SourdoughGulch": "SourdoughGulch_WA_985",
    "Sourdough": "SourdoughGulch_WA_985",
    "Touchet": "Touchet_WA_824",
}

def _require_cols(df, cols, name):
    missing = sorted([c for c in cols if c not in df.columns])
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def _to_utc_naive_midnight(s):
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None).dt.normalize()

def _canon_station_id_from_anything(x):
    s = str(x)

    # ISMN filename case: parse station name chunk
    if s.endswith(".stm") or s.count("_") >= 4:
        base = s[:-4] if s.endswith(".stm") else s
        parts = base.split("_")
        station_name = parts[2] if len(parts) >= 3 else s
    else:
        station_name = s

    for key, canon in CANON_IDS.items():
        if key in station_name:
            return canon

    return station_name


def _collapse_static_per_station(df, name):
    _require_cols(df, [GROUP_COL], name)
    d = df.copy()
    d[GROUP_COL] = d[GROUP_COL].map(_canon_station_id_from_anything)
    d = d.sort_values([GROUP_COL]).copy()
    d = d.groupby(GROUP_COL, as_index=False).first()
    return d


def _merge_static(train_df, val_df, test_df, static_df):
    def merge_one(split_df):
        dup = [c for c in static_df.columns if c in split_df.columns and c != GROUP_COL]
        static_use = static_df.drop(columns=dup, errors="ignore")
        merged = split_df.merge(static_use, on=GROUP_COL, how="left", validate="many_to_one")
        return merged, dup

    train_m, dup_tr = merge_one(train_df)
    val_m,   dup_va = merge_one(val_df)
    test_m,  dup_te = merge_one(test_df)

    dropped = sorted(set(dup_tr + dup_va + dup_te))
    return train_m, val_m, test_m, dropped


def main():
    train = pd.read_csv(TRAIN_IN, low_memory=False)
    val   = pd.read_csv(VAL_IN, low_memory=False)
    test  = pd.read_csv(TEST_IN, low_memory=False)

    _require_cols(train, [GROUP_COL, DATE_COL], "train")
    _require_cols(val,   [GROUP_COL, DATE_COL], "val")
    _require_cols(test,  [GROUP_COL, DATE_COL], "test")

    for d in (train, val, test):
        d[DATE_COL] = _to_utc_naive_midnight(d[DATE_COL])
        d[GROUP_COL] = d[GROUP_COL].map(_canon_station_id_from_anything)

    print("\n=== Temporal stations (canonical) ===")
    print(sorted(pd.concat([train[GROUP_COL], val[GROUP_COL], test[GROUP_COL]]).unique().tolist()))

    static = pd.read_csv(STATION_STATIC_IN, low_memory=False)
    static = _collapse_static_per_station(static, name="station_static_features")

    print("\n=== Static stations (canonical) ===")
    print(sorted(static[GROUP_COL].unique().tolist()))

    train2, val2, test2, dropped_static_dupes = _merge_static(train, val, test, static)

    jk_cols = [c for c in train2.columns if c.startswith("J_") or c.startswith("K_")]
    print("\n=== Coverage check (train) ===")
    if jk_cols:
        print("Family J/K missing rates (top 10):")
        print(train2[jk_cols].isna().mean().sort_values(ascending=False).head(10).to_string())
    else:
        print("No J_/K_ columns found after merge. Check station_static_features.csv columns.")

    train2.to_csv(TRAIN_OUT, index=False)
    val2.to_csv(VAL_OUT, index=False)
    test2.to_csv(TEST_OUT, index=False)

    meta = {
        "policy": "append_station_static_features_by_station_id_only (no resplit)",
        "date_col": DATE_COL,
        "group_col": GROUP_COL,
        "inputs": {
            "train_in": TRAIN_IN,
            "val_in": VAL_IN,
            "test_in": TEST_IN,
            "station_static_in": STATION_STATIC_IN,
        },
        "outputs": {
            "train_out": TRAIN_OUT,
            "val_out": VAL_OUT,
            "test_out": TEST_OUT,
        },
        "rows": {"train": int(len(train2)), "val": int(len(val2)), "test": int(len(test2))},
        "stations_temporal": sorted(pd.concat([train2[GROUP_COL], val2[GROUP_COL], test2[GROUP_COL]]).unique().tolist()),
        "stations_static": sorted(static[GROUP_COL].unique().tolist()),
        "dropped_duplicate_cols_from_static": dropped_static_dupes,
        "canon_ids": CANON_IDS,
    }

    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(" ", TRAIN_OUT)
    print(" ", VAL_OUT)
    print(" ", TEST_OUT)
    print("Meta:")
    print(" ", META_OUT)


if __name__ == "__main__":
    main()
