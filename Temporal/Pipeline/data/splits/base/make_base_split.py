# Jakob Balkovec
# make_base_split.py

# Script that makes the test/train/val split
#   60/20/20
#   by station_id

import os
import json
import numpy as np
import pandas as pd

MASTER_CLEANED = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv"
SPLIT_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/base/"
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(SPLIT_DIR, "train_base.csv")
VAL_PATH = os.path.join(SPLIT_DIR, "val_base.csv")
TEST_PATH = os.path.join(SPLIT_DIR, "test_base.csv")
META_PATH = os.path.join(SPLIT_DIR, "split_meta.json")

SEED = 42
SPLIT_FRAC = (0.60, 0.20, 0.20)  # train/val/test
SPLIT_BY = "station_id"

TARGET_COL = "soil_moisture_5cm"

FEATURE_COLS = [
    # atmosphere / forcing
    "rain_mm",
    "precip_mm",
    "air_temp_mean",
    "rh_mean",
    "solar_radiation",

    # vegetation / surface
    "NDVI",   # Sentinel-2 derived NDVI, not MODIS NDVI
    "NDMI",
    "MSI",

    # radar
    "s1_vv",
    "s1_vh",
    "SAR_ratio",

    # static terrain
    "elev",
    "slope",

    # seasonality proxy
    "DOY",
]

# extra columns to keep for analysis/debug (will not be used as features)
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

# drop rows with missing base features/target (baseline model should be clean)
cols_to_check = FEATURE_COLS + [TARGET_COL]
before = len(df)
df = df.dropna(subset=cols_to_check).copy()
after = len(df)
print(f"Rows before dropna: {before}, after dropna (base features + target): {after}")

rng = np.random.default_rng(SEED)
stations = df["station_id"].dropna().unique().tolist()
stations = sorted(stations)  # deterministic order before shuffle
rng.shuffle(stations)

n = len(stations)
n_train = int(round(SPLIT_FRAC[0] * n))
n_val = int(round(SPLIT_FRAC[1] * n))

# make sure everything is assigned
n_test = n - n_train - n_val

train_stations = set(stations[:n_train])
val_stations = set(stations[n_train:n_train + n_val])
test_stations = set(stations[n_train + n_val:])

train_df = df[df["station_id"].isin(train_stations)].copy()
val_df = df[df["station_id"].isin(val_stations)].copy()
test_df = df[df["station_id"].isin(test_stations)].copy()

print("\nStation split:")
print(f"  stations total: {n}")
print(f"  train stations: {len(train_stations)} | rows: {len(train_df)}")
print(f"  val stations:   {len(val_stations)}   | rows: {len(val_df)}")
print(f"  test stations:  {len(test_stations)}  | rows: {len(test_df)}")

train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

meta = {
    "seed": SEED,
    "split_frac": {"train": SPLIT_FRAC[0], "val": SPLIT_FRAC[1], "test": SPLIT_FRAC[2]},
    "split_by": SPLIT_BY,
    "target": TARGET_COL,
    "features": FEATURE_COLS,
    "meta_cols_kept": KEEP_META_COLS,
    "stations": {
        "train": sorted(list(train_stations)),
        "val": sorted(list(val_stations)),
        "test": sorted(list(test_stations)),
    },
    "rows": {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved splits to:\n  {TRAIN_PATH}\n  {VAL_PATH}\n  {TEST_PATH}\nMeta:\n  {META_PATH}")
