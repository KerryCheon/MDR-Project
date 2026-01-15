# make_derived_all_split.py

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.derived_features_all_math import (
    compute_ndvi,
    compute_ndmi,
    compute_msi,
    compute_sar_ratio,
    compute_sar_diff,
    compute_api,
    compute_days_since_last_rain,
    compute_rain_sums_days,
    compute_time_since_last_spike_past_only,

    series_lags,
    series_diffs,
    series_pct_change,
    series_gradient_kobs,

    rolling_mean,
    rolling_std,
    rolling_min,
    rolling_max,
    rolling_range,
    rolling_cv,
    rolling_corr,
    rolling_fft_dom_freq_and_entropy,
    rolling_mean_abs_change,

    ema,
    smm_index,

    train_only_monthly_anomaly_global,
    train_only_monthly_zscore_global)

MASTER_CLEANED = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv"
SPLIT_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_all/"
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(SPLIT_DIR, "train_derived_all.csv")
VAL_PATH   = os.path.join(SPLIT_DIR, "val_derived_all.csv")
TEST_PATH  = os.path.join(SPLIT_DIR, "test_derived_all.csv")
META_PATH  = os.path.join(SPLIT_DIR, "split_meta_derived_all.json")

SEED = 42
SPLIT_BY = "station_id + time"
VAL_TIME_FRAC  = 0.15
TEST_TIME_FRAC = 0.15
MIN_VAL_DATES_PER_STATION  = 5
MIN_TEST_DATES_PER_STATION = 5

DATE_COL = "date"
GROUP_COL = "station_id"
TARGET_COL = "soil_moisture_5cm"

KEEP_META_COLS = ["station_id", "date", "longitude", "latitude"]

BASE_COLS = [
    "precip_mm",
    "s1_vv",
    "s1_vh",
    "s2_b4",
    "s2_b8",
    "s2_b11",
    "s2_b12",
    "LST_modis",
    "elev",
    "slope",
    "aspect",
    "DOY",
]

EPS = 1e-6
RAIN_THR_MM = 0.5
API_DECAY = 0.90
SMM_ALPHA = 0.85

KOBS_SHORT = 1
KOBS_MED = 2
KOBS_LONG = 5

EXTRA_DIFFS = [5, 7, 14, 30]
EXTRA_LAGS  = [6, 12, 30]
FFT_WIN = 30
CORR_WINS = [7, 14]

WIN_OBS_7 = 7
WIN_OBS_14 = 14

RAIN_SUM_DAYS = (3, 7, 30)

SPIKE_COL = "s1_vv"
SPIKE_DIFF_COL = "E_dVV_1"
SPIKE_Z_THR = 2.0

PFX = {
    "META": "M",
    "BASE": "B",
    "MET": "G",
    "RAD": "E",
    "OPT": "F",
    "DYN": "A",
    "VOL": "V",
    "MEM": "C",
    "SEA": "D",
    "EVT": "I",
}

def _enforce_no_ground_sensors(selected_cols):
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
            "Ground-sensor columns are not allowed in this split (except target). "
            f"Forbidden columns selected: {sorted(set(bad))}"
        )

def _make_splits_temporal(df: pd.DataFrame):
    stations = sorted(df[GROUP_COL].dropna().unique().tolist())
    if len(stations) < 1:
        raise ValueError("No stations found.")

    train_parts, val_parts, test_parts = [], [], []

    for sid, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(DATE_COL).copy()
        unique_dates = pd.Series(g[DATE_COL].dropna().unique()).sort_values().to_numpy()

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

        g_train = g[g[DATE_COL].isin(train_dates)].copy()
        g_val   = g[g[DATE_COL].isin(val_dates)].copy()
        g_test  = g[g[DATE_COL].isin(test_dates)].copy()

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

def _check_temporal_split(train_df, val_df, test_df):
    for sid in train_df[GROUP_COL].unique():
        tr = train_df[train_df[GROUP_COL] == sid][DATE_COL]
        va = val_df[val_df[GROUP_COL] == sid][DATE_COL]
        te = test_df[test_df[GROUP_COL] == sid][DATE_COL]
        if len(va) and len(tr) and va.min() <= tr.max():
            print(f"[WARN] {sid}: val starts before/at train end")
        if len(te) and len(va) and te.min() <= va.max():
            print(f"[WARN] {sid}: test starts before/at val end")

def _require_no_dupes(df: pd.DataFrame):
    key = [GROUP_COL, DATE_COL]
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        n = int(dup.sum())
        raise ValueError(f"Duplicate (station_id, date) rows found: {n}. Deduplicate upstream.")

def _add_derived_features(train_df, val_df, test_df):
    full = pd.concat(
        [train_df.assign(_split="train"), val_df.assign(_split="val"), test_df.assign(_split="test")],
        ignore_index=True,
    )

    _require_no_dupes(full)

    # ----------------------------
    # 0) Core single-shot features
    # ----------------------------
    full[f"{PFX['OPT']}_NDVI"] = compute_ndvi(full, nir_col="s2_b8", red_col="s2_b4", eps=EPS)
    full[f"{PFX['OPT']}_NDMI"] = compute_ndmi(full, nir_col="s2_b8", swir_col="s2_b11", eps=EPS)
    full[f"{PFX['OPT']}_MSI"]  = compute_msi(full, swir_col="s2_b11", nir_col="s2_b8", eps=EPS)

    full[f"{PFX['RAD']}_SAR_ratio"] = compute_sar_ratio(full, vv_col="s1_vv", vh_col="s1_vh", eps=EPS)
    full[f"{PFX['RAD']}_SAR_diff"]  = compute_sar_diff(full, vv_col="s1_vv", vh_col="s1_vh")

    full[f"{PFX['MET']}_API"]  = compute_api(
        full, precip_col="precip_mm", decay=API_DECAY, group_col=GROUP_COL, date_col=DATE_COL
    )
    full[f"{PFX['MET']}_DSLR"] = compute_days_since_last_rain(
        full, precip_col="precip_mm", threshold_mm=RAIN_THR_MM, group_col=GROUP_COL, date_col=DATE_COL
    )

    for d in RAIN_SUM_DAYS:
        full[f"{PFX['MET']}_rain_sum_{d}d"] = compute_rain_sums_days(
            full, precip_col="precip_mm", window_days=d, group_col=GROUP_COL, date_col=DATE_COL
        )

    # ----------------------------
    # 1) Family parameters
    # ----------------------------
    DIFF_KOBS_LIST = [1, 2, 5, 7, 14, 30]         # A
    GRAD_KOBS_LIST = [WIN_OBS_7, WIN_OBS_14, 30]  # A
    LAG_KOBS_LIST  = [1, 2, 5, 6, 12, 30]         # C
    WIN_LIST       = [WIN_OBS_7, WIN_OBS_14, 30]  # B (+ used by H/E)
    FFT_WIN        = 30                           # D
    CORR_WINS      = [WIN_OBS_7, WIN_OBS_14]      # H

    INCLUDE_SWIR_DYNAMICS = True

    INCLUDE_NDVI_DYNAMICS = True
    INCLUDE_SARDIFF_DYNAMICS = True

    # ----------------------------
    # 2) Signals to derive A/B/C from
    # ----------------------------
    dyn_signals = {
        f"{PFX['MET']}_API": f"{PFX['MET']}_API",
        f"{PFX['OPT']}_NDMI": f"{PFX['OPT']}_NDMI",
        f"{PFX['RAD']}_SAR_ratio": f"{PFX['RAD']}_SAR_ratio",
        "LST_modis": "LST_modis",
    }
    if INCLUDE_NDVI_DYNAMICS:
        dyn_signals[f"{PFX['OPT']}_NDVI"] = f"{PFX['OPT']}_NDVI"
    if INCLUDE_SARDIFF_DYNAMICS:
        dyn_signals[f"{PFX['RAD']}_SAR_diff"] = f"{PFX['RAD']}_SAR_diff"
    if INCLUDE_SWIR_DYNAMICS:
        dyn_signals["s2_b11"] = "s2_b11"
        dyn_signals["s2_b12"] = "s2_b12"

    # ----------------------------
    # 3) Families A, B, C on selected signals
    # ----------------------------
    for i, col in enumerate(dyn_signals.values()):
        # A: n-step diffs
        if (i + 1) % 2 == 0:   # every 2 signals
            full = full.copy()

        for k in DIFF_KOBS_LIST:
            full[f"{PFX['DYN']}_d_{col}_kobs{k}"] = series_diffs(
                full, col=col, kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )

        # A: gradients
        for k in GRAD_KOBS_LIST:
            full[f"{PFX['DYN']}_grad_{col}_kobs{k}"] = series_gradient_kobs(
                full, col=col, kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )

        # A: percent change (1-step)
        full[f"{PFX['DYN']}_pct_{col}"] = series_pct_change(
            full, col=col, group_col=GROUP_COL, date_col=DATE_COL, eps=EPS
        )

        # B: rolling stats + EMA for multiple windows
        for w in WIN_LIST:
            full[f"{PFX['VOL']}_rollstd_{col}_kobs{w}"] = rolling_std(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, ddof=0, min_periods=w
            )
            full[f"{PFX['VOL']}_rollrng_{col}_kobs{w}"] = rolling_range(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            full[f"{PFX['VOL']}_rollcv_{col}_kobs{w}"] = rolling_cv(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, eps=EPS, ddof=0, min_periods=w
            )
            full[f"{PFX['VOL']}_rollmean_{col}_kobs{w}"] = rolling_mean(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            full[f"{PFX['VOL']}_rollmin_{col}_kobs{w}"] = rolling_min(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            full[f"{PFX['VOL']}_rollmax_{col}_kobs{w}"] = rolling_max(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            full[f"{PFX['VOL']}_ema_{col}_kobs{w}"] = ema(
                full, col=col, alpha=2.0 / (w + 1.0), group_col=GROUP_COL, date_col=DATE_COL
            )

        # C: lags (more horizons)
        for k in LAG_KOBS_LIST:
            full[f"{PFX['MEM']}_lag_{col}_kobs{k}"] = series_lags(
                full, col=col, lag_kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )

        # C: smoothed memory index (keep your original design)
        full[f"{PFX['MEM']}_smm_{col}_alpha{SMM_ALPHA}_n{KOBS_LONG}"] = smm_index(
            full, col=col, alpha=SMM_ALPHA, n_lags=KOBS_LONG, group_col=GROUP_COL, date_col=DATE_COL
        )

    # ----------------------------
    # 4) Radar event timing + roughness (E + I)
    # ----------------------------
    # E: VV 1-step diff used for event timing + roughness
    full[f"{PFX['RAD']}_dVV_1"] = series_diffs(full, col="s1_vv", kobs=1, group_col=GROUP_COL, date_col=DATE_COL)

    # E: roughness proxies (past-only rolling abs change)
    for w in CORR_WINS:
        full[f"{PFX['RAD']}_rough_s1_vv_kobs{w}"] = rolling_mean_abs_change(
            full, col="s1_vv", window=w, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, min_periods=w
        )
        full[f"{PFX['RAD']}_rough_s1_vh_kobs{w}"] = rolling_mean_abs_change(
            full, col="s1_vh", window=w, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, min_periods=w
        )

    # I: time since last spike (leakage-safe, past-only expanding zscore)
    full[f"{PFX['EVT']}_ts_spike_{SPIKE_COL}"] = compute_time_since_last_spike_past_only(
        full,
        diff_col=f"{PFX['RAD']}_dVV_1",
        zthr=SPIKE_Z_THR,
        group_col=GROUP_COL,
        date_col=DATE_COL,
        eps=EPS,
    )

    # ----------------------------
    # 5) Family H: cross-signal coupling (rolling correlations)
    # ----------------------------
    # Radar–Optical coupling (SAR ratio vs NDMI)
    for w in CORR_WINS:
        full[f"H_corr_{PFX['RAD']}_SAR_ratio__{PFX['OPT']}_NDMI_kobs{w}"] = rolling_corr(
            full,
            x_col=f"{PFX['RAD']}_SAR_ratio",
            y_col=f"{PFX['OPT']}_NDMI",
            window=w,
            group_col=GROUP_COL,
            date_col=DATE_COL,
            min_periods=w,
            past_only=True,
        )

    # Temperature–Moisture coupling (LST vs NDMI)
    for w in CORR_WINS:
        full[f"H_corr_LST_modis__{PFX['OPT']}_NDMI_kobs{w}"] = rolling_corr(
            full,
            x_col="LST_modis",
            y_col=f"{PFX['OPT']}_NDMI",
            window=w,
            group_col=GROUP_COL,
            date_col=DATE_COL,
            min_periods=w,
            past_only=True,
        )

    # ----------------------------
    # 6) Family D: train-only seasonal anomaly + z-score (your existing logic)
    # ----------------------------
    full = full.copy()
    train_full = full[full["_split"] == "train"].copy()
    val_full   = full[full["_split"] == "val"].copy()
    test_full  = full[full["_split"] == "test"].copy()

    for col in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        full[f"{PFX['SEA']}_sa_{col}"] = np.nan
        full[f"{PFX['SEA']}_z_{col}"] = np.nan

    for col in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        full.loc[train_full.index, f"{PFX['SEA']}_sa_{col}"] = train_only_monthly_anomaly_global(
            train_full, train_full, col=col, date_col=DATE_COL
        ).values
        full.loc[val_full.index,   f"{PFX['SEA']}_sa_{col}"] = train_only_monthly_anomaly_global(
            train_full, val_full, col=col, date_col=DATE_COL
        ).values
        full.loc[test_full.index,  f"{PFX['SEA']}_sa_{col}"] = train_only_monthly_anomaly_global(
            train_full, test_full, col=col, date_col=DATE_COL
        ).values

        full.loc[train_full.index, f"{PFX['SEA']}_z_{col}"] = train_only_monthly_zscore_global(
            train_full, train_full, col=col, date_col=DATE_COL, eps=EPS
        ).values
        full.loc[val_full.index,   f"{PFX['SEA']}_z_{col}"] = train_only_monthly_zscore_global(
            train_full, val_full, col=col, date_col=DATE_COL, eps=EPS
        ).values
        full.loc[test_full.index,  f"{PFX['SEA']}_z_{col}"] = train_only_monthly_zscore_global(
            train_full, test_full, col=col, date_col=DATE_COL, eps=EPS
        ).values

    # ----------------------------
    # 7) Family D: FFT dom frequency + spectral entropy (past-only rolling)
    # ----------------------------
    for sig in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        dom, ent = rolling_fft_dom_freq_and_entropy(
            full,
            col=sig,
            window=FFT_WIN,
            group_col=GROUP_COL,
            date_col=DATE_COL,
            past_only=True,
            eps=1e-12,
        )
        full[f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"] = dom
        full[f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"] = ent

    # ----------------------------
    # 8) Finalize: split back out + DSLR isnan flag
    # ----------------------------
    train_full = full[full["_split"] == "train"].drop(columns=["_split"]).copy()
    val_full   = full[full["_split"] == "val"].drop(columns=["_split"]).copy()
    test_full  = full[full["_split"] == "test"].drop(columns=["_split"]).copy()

    dslr_col = f"{PFX['MET']}_DSLR"
    full_isnan = full[dslr_col].isna().astype(int)
    train_full[f"{PFX['MET']}_DSLR_isnan"] = full_isnan[full["_split"] == "train"].values
    val_full[f"{PFX['MET']}_DSLR_isnan"]   = full_isnan[full["_split"] == "val"].values
    test_full[f"{PFX['MET']}_DSLR_isnan"]  = full_isnan[full["_split"] == "test"].values

    return train_full, val_full, test_full

def main():
    df = pd.read_csv(MASTER_CLEANED, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    selected_cols = KEEP_META_COLS + BASE_COLS + [TARGET_COL]
    _enforce_no_ground_sensors(selected_cols)

    missing = sorted(list(set(selected_cols) - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in master_cleaned.csv: {missing}")

    df = df[selected_cols].copy()

    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    after = len(df)
    print(f"Rows before dropna(target): {before}, after: {after}")

    train_df, val_df, test_df, stations = _make_splits_temporal(df)

    print("\nSplit policy: temporal-only within each station")
    print(f"Stations: {len(stations)}")
    print("Feature columns:", train_df.shape[1] - len(KEEP_META_COLS) - 1)
    print("\nRow counts (pre-derived):")
    print(f"  train rows: {len(train_df)}")
    print(f"  val rows:   {len(val_df)}")
    print(f"  test rows:  {len(test_df)}")

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("VAL or TEST ended up empty. Adjust split fractions or minimum date settings.")

    _check_temporal_split(train_df, val_df, test_df)

    train_df, val_df, test_df = _add_derived_features(train_df, val_df, test_df)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    meta = {
        "seed": SEED,
        "split_by": SPLIT_BY,
        "policy": "temporal_only: per-station tail split into train/val/test, then concatenate",
        "variant": "derived_all_no_ground",
        "target": TARGET_COL,
        "meta_cols_kept": KEEP_META_COLS,
        "base_cols": BASE_COLS,
        "prefixes": PFX,
        "params": {
            "eps": EPS,
            "api_decay": API_DECAY,
            "dslr_rain_threshold_mm": RAIN_THR_MM,
            "smm_alpha": SMM_ALPHA,
            "win_obs_7": WIN_OBS_7,
            "win_obs_14": WIN_OBS_14,
            "rain_sum_days": list(RAIN_SUM_DAYS),
            "spike_col": SPIKE_COL,
            "spike_diff_col": f"{PFX['RAD']}_dVV_1",
            "spike_method": "past_only_expanding_zscore",
            "spike_z_thr": SPIKE_Z_THR,
        },
        "stations": stations,
        "rows": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved splits to:\n  {TRAIN_PATH}\n  {VAL_PATH}\n  {TEST_PATH}\nMeta:\n  {META_PATH}")

if __name__ == "__main__":
    main()
