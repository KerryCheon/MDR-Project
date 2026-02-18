import os
import json
import pandas as pd

SPLIT_DIR_IN = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_new/"
TRAIN_IN = os.path.join(SPLIT_DIR_IN, "train_derived_new.csv")
VAL_IN   = os.path.join(SPLIT_DIR_IN, "val_derived_new.csv")
TEST_IN  = os.path.join(SPLIT_DIR_IN, "test_derived_new.csv")

SPATIAL_IN = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_with_spatial/spatial_features_extracted.csv"

SPLIT_DIR_OUT = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_with_spatial/"
os.makedirs(SPLIT_DIR_OUT, exist_ok=True)

TRAIN_OUT = os.path.join(SPLIT_DIR_OUT, "train_plus_spatial.csv")
VAL_OUT   = os.path.join(SPLIT_DIR_OUT, "val_plus_spatial.csv")
TEST_OUT  = os.path.join(SPLIT_DIR_OUT, "test_plus_spatial.csv")
META_OUT  = os.path.join(SPLIT_DIR_OUT, "split_meta_plus_spatial.json")

DATE_COL = "date"
GROUP_COL = "station_id"
VAL_TIME_FRAC = 0.15
TEST_TIME_FRAC = 0.15
SEED = 42

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

def _require_no_dupes(df, name="df"):
    key = [GROUP_COL, DATE_COL]
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        n = int(dup.sum())
        raise ValueError(f"Duplicate (station_id, date) rows found in {name}: {n}")

def _to_utc_naive_midnight(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None).dt.normalize()

def _canon_station_id_from_anything(x: str) -> str:
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

def _make_splits_global_temporal(df: pd.DataFrame):
    d = df.dropna(subset=[DATE_COL]).copy()
    d = d.sort_values(DATE_COL).copy()

    unique_dates = pd.Series(d[DATE_COL].unique()).sort_values().to_numpy()
    if len(unique_dates) < 3:
        raise ValueError(f"Not enough unique dates for a global temporal split: {len(unique_dates)}")

    n_test = int(round(TEST_TIME_FRAC * len(unique_dates)))
    n_val  = int(round(VAL_TIME_FRAC  * len(unique_dates)))

    n_test = max(n_test, 1)
    n_val  = max(n_val, 1)

    max_tail = len(unique_dates) - 1
    if (n_val + n_test) > max_tail:
        n_test = min(n_test, max_tail - 1)
        n_val  = min(n_val,  max_tail - n_test)
        if n_val < 1 or n_test < 1:
            raise ValueError("Global temporal split collapsed. Reduce split fractions.")

    test_dates  = set(unique_dates[-n_test:])
    val_dates   = set(unique_dates[-(n_test + n_val):-n_test])
    train_dates = set(unique_dates[:-(n_test + n_val)])

    train_df = d[d[DATE_COL].isin(train_dates)].copy()
    val_df   = d[d[DATE_COL].isin(val_dates)].copy()
    test_df  = d[d[DATE_COL].isin(test_dates)].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Empty split produced: train={len(train_df)} val={len(val_df)} test={len(test_df)}. "
            "Adjust split fractions."
        )

    return train_df, val_df, test_df

def main():
    # -------------------------
    # Load derived splits, concat
    # -------------------------
    train = pd.read_csv(TRAIN_IN, low_memory=False)
    val   = pd.read_csv(VAL_IN, low_memory=False)
    test  = pd.read_csv(TEST_IN, low_memory=False)

    _require_cols(train, [GROUP_COL, DATE_COL], "train")
    _require_cols(val,   [GROUP_COL, DATE_COL], "val")
    _require_cols(test,  [GROUP_COL, DATE_COL], "test")

    for d in (train, val, test):
        d[DATE_COL] = _to_utc_naive_midnight(d[DATE_COL])

    base_all = pd.concat(
        [train.assign(_src_split="train"), val.assign(_src_split="val"), test.assign(_src_split="test")],
        ignore_index=True,
    )

    # base_all should already be unique on (station_id, date)
    _require_no_dupes(base_all, name="base_all")

    # -------------------------
    # Load spatial extracted, canonicalize station_id, normalize date
    # -------------------------
    spatial = pd.read_csv(SPATIAL_IN, low_memory=False)

    if "datetime" in spatial.columns and DATE_COL not in spatial.columns:
        spatial = spatial.rename(columns={"datetime": DATE_COL})

    _require_cols(spatial, [GROUP_COL, DATE_COL], "spatial")

    spatial[DATE_COL] = _to_utc_naive_midnight(spatial[DATE_COL])
    spatial[GROUP_COL] = spatial[GROUP_COL].map(_canon_station_id_from_anything)

    print("\n=== Spatial canonical station_id values ===")
    print(sorted(spatial[GROUP_COL].unique().tolist()))

    # Spatial might have multiple records per day due to depth/sensor. Collapse to one row per key.
    # Choose the first non-null per column (deterministic after sort).
    spatial = spatial.sort_values([GROUP_COL, DATE_COL]).copy()
    spatial = spatial.groupby([GROUP_COL, DATE_COL], as_index=False).first()

    _require_no_dupes(spatial, name="spatial (collapsed)")

    # -------------------------
    # Clip base_all to spatial window (optional)
    # -------------------------
    sp_min = spatial[DATE_COL].min()
    sp_max = spatial[DATE_COL].max()
    before = len(base_all)
    base_all = base_all[(base_all[DATE_COL] >= sp_min) & (base_all[DATE_COL] <= sp_max)].copy()
    after = len(base_all)
    print(f"\nClipped base_all to spatial window {sp_min.date()} -> {sp_max.date()}: {before} -> {after}")

    # -------------------------
    # MERGE (this is the real fix)
    # -------------------------
    # Avoid duplicating columns like lat/lon if spatial also has them
    drop_dupe_cols = [c for c in spatial.columns if c in base_all.columns and c not in [GROUP_COL, DATE_COL]]
    spatial_merge = spatial.drop(columns=drop_dupe_cols, errors="ignore")

    combined = base_all.merge(spatial_merge, on=[GROUP_COL, DATE_COL], how="left", validate="one_to_one")

    print("\n=== Combined station_id values (should match temporal IDs) ===")
    print(sorted(combined[GROUP_COL].unique().tolist()))

    _require_no_dupes(combined, name="combined")

    # -------------------------
    # Resplit globally by time (70/15/15)
    # -------------------------
    train_new, val_new, test_new = _make_splits_global_temporal(combined)

    for d in (train_new, val_new, test_new):
        if "_src_split" in d.columns:
            d.drop(columns=["_src_split"], inplace=True, errors="ignore")

    # -------------------------
    # Save
    # -------------------------
    train_new.to_csv(TRAIN_OUT, index=False)
    val_new.to_csv(VAL_OUT, index=False)
    test_new.to_csv(TEST_OUT, index=False)

    meta = {
        "seed": SEED,
        "policy": "concat(train,val,test)->clip_to_spatial_window->merge_spatial_on(station_id,date)->global_temporal_resplit",
        "val_time_frac": VAL_TIME_FRAC,
        "test_time_frac": TEST_TIME_FRAC,
        "date_col": DATE_COL,
        "group_col": GROUP_COL,
        "inputs": {
            "train_in": TRAIN_IN,
            "val_in": VAL_IN,
            "test_in": TEST_IN,
            "spatial_in": SPATIAL_IN,
        },
        "outputs": {
            "train_out": TRAIN_OUT,
            "val_out": VAL_OUT,
            "test_out": TEST_OUT,
        },
        "rows": {
            "combined": int(len(combined)),
            "train": int(len(train_new)),
            "val": int(len(val_new)),
            "test": int(len(test_new)),
        },
        "stations": sorted(combined[GROUP_COL].unique().tolist()),
        "spatial_window": {"min": str(sp_min), "max": str(sp_max)},
        "dropped_duplicate_spatial_cols": drop_dupe_cols,
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
