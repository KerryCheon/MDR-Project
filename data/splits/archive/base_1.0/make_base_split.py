# Jakob Balkovec
# make_base_split_temporal.py
#
# Temporal split policy (no spatial holdout):
#   For EACH station:
#     - TRAIN: earliest (1 - VAL_FRAC - TEST_FRAC) of unique dates
#     - VAL:   next VAL_FRAC of unique dates
#     - TEST:  last TEST_FRAC of unique dates
#
# Then concatenate across stations.

import os
import json
import numpy as np
import pandas as pd

MASTER_CLEANED = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv"
SPLIT_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/base_1.0"
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
VAL_PATH   = os.path.join(SPLIT_DIR, "val.csv")
TEST_PATH  = os.path.join(SPLIT_DIR, "test.csv")
META_PATH  = os.path.join(SPLIT_DIR, "split_meta.json")

SEED = 42
SPLIT_BY = "station_id + time"

VAL_TIME_FRAC  = 0.15   # middle chunk
TEST_TIME_FRAC = 0.15   # last chunk
MIN_VAL_DATES_PER_STATION  = 5
MIN_TEST_DATES_PER_STATION = 5

TARGET_COL = "soil_moisture_5cm"

FEATURE_COLS = [
    "rain_mm",
    "precip_mm",
    "air_temp_mean",
    "rh_mean",
    "solar_radiation",
    "NDVI",
    "NDMI",
    "MSI",
    "s1_vv",
    "s1_vh",
    "SAR_ratio",
    "elev",
    "slope",
    "DOY",
]

KEEP_META_COLS = [
    "station_id",
    "date",
    "longitude",
    "latitude",
]

df = pd.read_csv(MASTER_CLEANED)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# sanity checks
required = set(KEEP_META_COLS + FEATURE_COLS + [TARGET_COL])
missing = sorted(list(required - set(df.columns)))
if missing:
    raise ValueError(f"Missing required columns in master_cleaned.csv: {missing}")

df = df[KEEP_META_COLS + FEATURE_COLS + [TARGET_COL]].copy()

# drop rows with missing base features/target
cols_to_check = FEATURE_COLS + [TARGET_COL]
before = len(df)
df = df.dropna(subset=cols_to_check).copy()
after = len(df)
print(f"Rows before dropna: {before}, after dropna (base features + target): {after}")

stations = sorted(df["station_id"].dropna().unique().tolist())
if len(stations) < 1:
    raise ValueError("No stations found.")

train_parts, val_parts, test_parts = [], [], []

for sid, g in df.groupby("station_id"):
    g = g.sort_values("date").copy()
    unique_dates = pd.Series(g["date"].dropna().unique()).sort_values().to_numpy()

    if len(unique_dates) < (1 + MIN_VAL_DATES_PER_STATION + MIN_TEST_DATES_PER_STATION):
        print(f"[WARN] {sid}: not enough unique dates ({len(unique_dates)}). Keeping all in TRAIN.")
        train_parts.append(g)
        continue

    # compute counts, enforce minimums, and ensure at least 1 date remains for train
    n_test = int(round(TEST_TIME_FRAC * len(unique_dates)))
    n_val  = int(round(VAL_TIME_FRAC  * len(unique_dates)))

    n_test = max(n_test, MIN_TEST_DATES_PER_STATION)
    n_val  = max(n_val,  MIN_VAL_DATES_PER_STATION)

    # cap so we keep at least 1 date for train
    max_tail = len(unique_dates) - 1
    if (n_val + n_test) > max_tail:
        # shrink proportionally but keep mins if possible
        n_test = min(n_test, max_tail - 1)
        n_val  = min(n_val,  max_tail - n_test)

        if n_val < 1 or n_test < 1:
            print(f"[WARN] {sid}: tail split collapsed. Keeping all in TRAIN.")
            train_parts.append(g)
            continue

    test_dates = set(unique_dates[-n_test:])
    val_dates  = set(unique_dates[-(n_test + n_val):-n_test])
    train_dates = set(unique_dates[:-(n_test + n_val)])

    g_train = g[g["date"].isin(train_dates)].copy()
    g_val   = g[g["date"].isin(val_dates)].copy()
    g_test  = g[g["date"].isin(test_dates)].copy()

    if len(g_train) == 0 or len(g_val) == 0 or len(g_test) == 0:
        print(f"[WARN] {sid}: got empty split (train/val/test). Keeping all in TRAIN.")
        train_parts.append(g)
        continue

    train_parts.append(g_train)
    val_parts.append(g_val)
    test_parts.append(g_test)

train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
val_df   = pd.concat(val_parts,   ignore_index=True) if val_parts   else df.iloc[0:0].copy()
test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else df.iloc[0:0].copy()

print("\nSplit policy: temporal-only within each station (no station holdout)")
print(f"Stations: {len(stations)} | {stations}")
print("\nRow counts:")
print(f"  train rows: {len(train_df)}")
print(f"  val rows:   {len(val_df)}")
print(f"  test rows:  {len(test_df)}")

if len(val_df) == 0 or len(test_df) == 0:
    raise ValueError(
        "VAL or TEST ended up empty. Try lowering VAL_TIME_FRAC/TEST_TIME_FRAC "
        "or minimum date settings, or check that 'date' parses correctly."
    )

train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

meta = {
    "seed": SEED,
    "split_by": SPLIT_BY,
    "policy": "temporal_only: per-station time split into train/val/test, then concatenate",
    "val_time_frac": VAL_TIME_FRAC,
    "test_time_frac": TEST_TIME_FRAC,
    "min_val_dates_per_station": MIN_VAL_DATES_PER_STATION,
    "min_test_dates_per_station": MIN_TEST_DATES_PER_STATION,
    "target": TARGET_COL,
    "features": FEATURE_COLS,
    "meta_cols_kept": KEEP_META_COLS,
    "stations": stations,
    "rows": {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    },
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved splits to:\n  {TRAIN_PATH}\n  {VAL_PATH}\n  {TEST_PATH}\nMeta:\n  {META_PATH}")
