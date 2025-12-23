# Jakob Balkovec
# make_base_split.py
#
# 3-station split policy:
#   - TEST: hold out 1 entire station (station generalization)
#   - TRAIN: remaining stations
#   - VAL: time-based holdout (last VAL_TIME_FRAC of dates) within TRAIN stations

import os
import json
import numpy as np
import pandas as pd

MASTER_CLEANED = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv"
SPLIT_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/test/"
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(SPLIT_DIR, "train_base.csv")
VAL_PATH   = os.path.join(SPLIT_DIR, "val_base.csv")
TEST_PATH  = os.path.join(SPLIT_DIR, "test_base.csv")
META_PATH  = os.path.join(SPLIT_DIR, "split_meta.json")

SEED = 42
SPLIT_BY = "station_id"

# With 3 stations, this is the sane choice:
N_TEST_STATIONS = 1

# Validation is by time within the TRAIN stations:
VAL_TIME_FRAC = 0.20                 # last 20% of unique dates per station goes to val
MIN_VAL_DATES_PER_STATION = 5        # ensure val isn't tiny/empty per station (adjust if needed)

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

# station list
rng = np.random.default_rng(SEED)
stations = df["station_id"].dropna().unique().tolist()
stations = sorted(stations)  # deterministic order before shuffle
rng.shuffle(stations)

n = len(stations)
if n < 3:
    raise ValueError(f"Expected >= 3 stations for this split policy, found {n}: {stations}")

if N_TEST_STATIONS >= n:
    raise ValueError(f"N_TEST_STATIONS must be < number of stations. Got {N_TEST_STATIONS} with n={n}")

# hold out test station(s)
test_stations = set(stations[:N_TEST_STATIONS])
train_stations = set(stations[N_TEST_STATIONS:])

train_pool = df[df["station_id"].isin(train_stations)].copy()
test_df = df[df["station_id"].isin(test_stations)].copy()

# make time-based val inside train stations
train_parts = []
val_parts = []

for sid, g in train_pool.groupby("station_id"):
    g = g.sort_values("date").copy()
    unique_dates = pd.Series(g["date"].dropna().unique()).sort_values().to_numpy()

    if len(unique_dates) < 2:
        # not enough time points to split; keep all in train
        train_parts.append(g)
        continue

    n_val_dates = int(round(VAL_TIME_FRAC * len(unique_dates)))
    n_val_dates = max(n_val_dates, MIN_VAL_DATES_PER_STATION)
    n_val_dates = min(n_val_dates, len(unique_dates) - 1)  # leave at least 1 date for train

    val_dates = set(unique_dates[-n_val_dates:])

    g_val = g[g["date"].isin(val_dates)].copy()
    g_trn = g[~g["date"].isin(val_dates)].copy()

    # if something went weird, fall back safely
    if len(g_val) == 0 or len(g_trn) == 0:
        train_parts.append(g)
    else:
        train_parts.append(g_trn)
        val_parts.append(g_val)

train_df = pd.concat(train_parts, ignore_index=True) if train_parts else train_pool.iloc[0:0].copy()
val_df   = pd.concat(val_parts,   ignore_index=True) if val_parts   else train_pool.iloc[0:0].copy()

print("\nSplit policy: holdout station(s) for TEST + time-holdout for VAL within TRAIN")
print("Station split:")
print(f"  stations total: {n}")
print(f"  train stations: {len(train_stations)} | {sorted(list(train_stations))}")
print(f"  test stations:  {len(test_stations)}  | {sorted(list(test_stations))}")

print("\nRow counts:")
print(f"  train rows: {len(train_df)}")
print(f"  val rows:   {len(val_df)}")
print(f"  test rows:  {len(test_df)}")

if len(val_df) == 0:
    raise ValueError(
        "VAL ended up empty. Try lowering MIN_VAL_DATES_PER_STATION "
        "or VAL_TIME_FRAC, or check that 'date' parses correctly."
    )

train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

meta = {
    "seed": SEED,
    "split_by": SPLIT_BY,
    "policy": "3_stations: TEST=heldout_station(s), VAL=time_holdout_within_train",
    "n_test_stations": N_TEST_STATIONS,
    "val_time_frac": VAL_TIME_FRAC,
    "min_val_dates_per_station": MIN_VAL_DATES_PER_STATION,
    "target": TARGET_COL,
    "features": FEATURE_COLS,
    "meta_cols_kept": KEEP_META_COLS,
    "stations": {
        "train": sorted(list(train_stations)),
        "val": "time-holdout within train stations (see params above)",
        "test": sorted(list(test_stations)),
    },
    "rows": {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    },
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved splits to:\n  {TRAIN_PATH}\n  {VAL_PATH}\n  {TEST_PATH}\nMeta:\n  {META_PATH}")
