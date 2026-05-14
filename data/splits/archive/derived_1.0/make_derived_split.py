import os
import json
import numpy as np
import pandas as pd

from utils.derived_feature_math import (
    compute_ndmi,
    compute_sar_ratio,
    compute_api,
    compute_days_since_last_rain,
    rolling_std,
    temporal_gradient,
    train_only_monthly_anomaly,
)

MASTER_CLEANED = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv"
SPLIT_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_1.0"
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
VAL_PATH   = os.path.join(SPLIT_DIR, "val.csv")
TEST_PATH  = os.path.join(SPLIT_DIR, "test.csv")
META_PATH  = os.path.join(SPLIT_DIR, "split_meta.json")

SEED = 42
SPLIT_BY = "station_id + time"

VAL_TIME_FRAC  = 0.15
TEST_TIME_FRAC = 0.15
MIN_VAL_DATES_PER_STATION  = 5
MIN_TEST_DATES_PER_STATION = 5

TARGET_COL = "soil_moisture_5cm"

KEEP_META_COLS = ["station_id", "date", "longitude", "latitude"]

# Base exogenous columns we allow (no ground sensors)
BASE_COLS = [
    # precip
    "precip_mm",

    # sentinel-1
    "s1_vv",
    "s1_vh",

    # sentinel-2
    "s2_b8",
    "s2_b11",

    # modis LST
    "LST_modis",

    # static + time
    "elev",
    "slope",
    "DOY",
]

# Hyperparams (v1 defaults)
EPS = 1e-6
K_GRAD = 7
K_STD = 7
API_DECAY = 0.90
RAIN_THR_MM = 0.5

def _enforce_no_ground_sensors(selected_cols):
    """
    Forbid ground-sensor columns besides soil_moisture_5cm, while allowing precip_mm.
    """
    forbidden_prefixes = ("air_temp_", "rh_", "soil_temp_", "sur_temp_")
    forbidden_exact = {"solar_radiation", "precipitation"}

    bad = []
    for c in selected_cols:
        if c in forbidden_exact:
            bad.append(c)
        if c.startswith(forbidden_prefixes):
            bad.append(c)
        if c.startswith("soil_moisture_") and c != TARGET_COL:
            bad.append(c)

    if bad:
        raise ValueError(
            "Ground-sensor columns are not allowed in this split (except soil_moisture_5cm). "
            f"Forbidden columns selected: {sorted(set(bad))}"
        )

def _make_splits_temporal(df):
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

        n_test = int(round(TEST_TIME_FRAC * len(unique_dates)))
        n_val  = int(round(VAL_TIME_FRAC  * len(unique_dates)))

        n_test = max(n_test, MIN_TEST_DATES_PER_STATION)
        n_val  = max(n_val,  MIN_VAL_DATES_PER_STATION)

        max_tail = len(unique_dates) - 1
        if (n_val + n_test) > max_tail:
            n_test = min(n_test, max_tail - 1)
            n_val  = min(n_val,  max_tail - n_test)

            if n_val < 1 or n_test < 1:
                print(f"[WARN] {sid}: tail split collapsed. Keeping all in TRAIN.")
                train_parts.append(g)
                continue

        test_dates  = set(unique_dates[-n_test:])
        val_dates   = set(unique_dates[-(n_test + n_val):-n_test])
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

    return train_df, val_df, test_df, stations


def _add_derived_features(train_df, val_df, test_df):
    """
    Compute derived features with strict train-only seasonal baselines.
    All features computed per-station for temporal ops.
    """
    # Build a combined frame for features that need consistent computation (e.g., NDMI, SAR_ratio, API, DSLR)
    full = pd.concat(
        [train_df.assign(_split="train"), val_df.assign(_split="val"), test_df.assign(_split="test")],
        ignore_index=True
    )

    # Base derived: NDMI, SAR_ratio, API, DSLR
    full["NDMI"] = compute_ndmi(full, nir_col="s2_b8", swir_col="s2_b11", eps=EPS)
    full["SAR_ratio"] = compute_sar_ratio(full, vv_col="s1_vv", vh_col="s1_vh", eps=EPS)
    full["API"] = compute_api(full, precip_col="precip_mm", decay=API_DECAY)
    full["DSLR"] = compute_days_since_last_rain(full, precip_col="precip_mm", threshold_mm=RAIN_THR_MM)

    # Temporal gradient + rolling std on selected signals
    # (You can expand this list later in ablations.)
    for col in ["API", "NDMI", "SAR_ratio", "LST_modis"]:
        full[f"grad_{col}_{K_GRAD}"] = temporal_gradient(full, col=col, k=K_GRAD)
        full[f"rollstd_{col}_{K_STD}"] = rolling_std(full, col=col, window=K_STD)

    # Split back out first so we can compute train-only baselines cleanly
    train_full = full[full["_split"] == "train"].copy()
    val_full   = full[full["_split"] == "val"].copy()
    test_full  = full[full["_split"] == "test"].copy()

    # Seasonal anomaly (train-only monthly mean), applied to all splits
    # Apply to NDMI, SAR_ratio, LST_modis (your spec)
    for col in ["NDMI", "SAR_ratio", "LST_modis"]:
        # compute anomaly for each split using train-only baseline
        train_full[f"{col}_sa"] = train_only_monthly_anomaly(train_full, train_full, col=col)
        val_full[f"{col}_sa"]   = train_only_monthly_anomaly(train_full, val_full, col=col)
        test_full[f"{col}_sa"]  = train_only_monthly_anomaly(train_full, test_full, col=col)

    # Clean helper column
    train_full = train_full.drop(columns=["_split"])
    val_full   = val_full.drop(columns=["_split"])
    test_full  = test_full.drop(columns=["_split"])

    return train_full, val_full, test_full


def main():
    df = pd.read_csv(MASTER_CLEANED, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Required columns
    selected_cols = KEEP_META_COLS + BASE_COLS + [TARGET_COL]
    _enforce_no_ground_sensors(selected_cols)

    required = set(selected_cols)
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in master_cleaned.csv: {missing}")

    df = df[selected_cols].copy()

    # Drop rows missing the target (we assume exogenous is imputed, but target may be missing)
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    after = len(df)
    print(f"Rows before dropna(target): {before}, after: {after}")

    # Make temporal splits
    train_df, val_df, test_df, stations = _make_splits_temporal(df)

    print("\nSplit policy: temporal-only within each station")
    print(f"Stations: {len(stations)} | {stations}")
    print("\nRow counts (pre-derived):")
    print(f"  train rows: {len(train_df)}")
    print(f"  val rows:   {len(val_df)}")
    print(f"  test rows:  {len(test_df)}")

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("VAL or TEST ended up empty. Adjust split fractions or minimum date settings.")

    # Add derived features (train-only baselines)
    train_df, val_df, test_df = _add_derived_features(train_df, val_df, test_df)

    # Save
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    meta = {
        "seed": SEED,
        "split_by": SPLIT_BY,
        "policy": "temporal_only: per-station time split into train/val/test, then concatenate",
        "variant": "derived_no_ground",
        "target": TARGET_COL,
        "meta_cols_kept": KEEP_META_COLS,
        "base_cols": BASE_COLS,
        "derived_features": [
            "NDMI",
            "SAR_ratio",
            "API",
            "DSLR",
            f"grad_*_{K_GRAD}",
            f"rollstd_*_{K_STD}",
            "NDMI_sa",
            "SAR_ratio_sa",
            "LST_modis_sa",
        ],
        "params": {
            "k_grad": K_GRAD,
            "k_rollstd": K_STD,
            "api_decay": API_DECAY,
            "dslr_rain_threshold_mm": RAIN_THR_MM,
            "eps": EPS,
        },
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


if __name__ == "__main__":
    main()
