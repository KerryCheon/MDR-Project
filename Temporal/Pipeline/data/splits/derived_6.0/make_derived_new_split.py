# make_derived_all_split.py

import os
import json
import numpy as np
import pandas as pd

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
    train_only_monthly_zscore_global,
    add_smap_features,
)

MASTER_CLEANED = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv"
SPLIT_DIR = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_6.0"
STATIC_PATH = os.path.join(SPLIT_DIR, "station_static_features.csv")
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
VAL_PATH   = os.path.join(SPLIT_DIR, "val.csv")
TEST_PATH  = os.path.join(SPLIT_DIR, "test.csv")
META_PATH  = os.path.join(SPLIT_DIR, "split_meta.json")

SEED = 42
SPLIT_BY = "station_id + time"
VAL_TIME_FRAC  = 0.15
TEST_TIME_FRAC = 0.15

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
    "SMAP_sm_am_interp",
    "SMAP_sm_pm_interp",
]

EPS = 1e-6
RAIN_THR_MM = 0.5
API_DECAY = 0.90
SMM_ALPHA = 0.85

KOBS_LONG = 5
FFT_WIN = 30
CORR_WINS = [7, 14]
WIN_OBS_7 = 7
WIN_OBS_14 = 14
RAIN_SUM_DAYS = (3, 7, 30)

SPIKE_COL = "s1_vv"
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

def _load_and_merge_static(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(STATIC_PATH):
        raise FileNotFoundError(f"Missing station static file: {STATIC_PATH}")

    stat = pd.read_csv(STATIC_PATH, low_memory=False)

    stat["station_id"] = stat["station_id"].astype(str)

    if "landcover_label" in stat.columns:
        stat = stat.drop(columns=["landcover_label"])

    drop_dupe_meta = [c for c in ["latitude", "longitude"] if c in stat.columns]
    if drop_dupe_meta:
        stat = stat.drop(columns=drop_dupe_meta)

    if stat["station_id"].duplicated().any():
        dups = stat.loc[stat["station_id"].duplicated(), "station_id"].tolist()
        raise ValueError(f"Duplicate station_id rows in static file: {sorted(set(dups))}")

    merged = df.merge(stat, on="station_id", how="left")

    if "J_lc_code" in merged.columns:
        missing_static = merged["J_lc_code"].isna().groupby(merged["station_id"]).any()
        bad = missing_static[missing_static].index.tolist()
        if bad:
            raise ValueError(f"Static features missing after merge for station_id: {bad}")

    return merged

def _make_splits_global_temporal(df: pd.DataFrame):
    stations = sorted(df[GROUP_COL].dropna().unique().tolist())
    if len(stations) < 1:
        raise ValueError("No stations found.")

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

def _attach_cols(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    if not cols:
        return df
    block = pd.DataFrame(cols, index=df.index)
    return pd.concat([df, block], axis=1)

def _add_derived_features(train_df, val_df, test_df):
    full = pd.concat(
        [train_df.assign(_split="train"), val_df.assign(_split="val"), test_df.assign(_split="test")],
        ignore_index=True,
    )

    _require_no_dupes(full)

    full[DATE_COL] = pd.to_datetime(full[DATE_COL], errors="coerce")

    full[f"{PFX['SEA']}_sin_DOY"] = np.sin(2 * np.pi * full["DOY"] / 365.0)
    full[f"{PFX['SEA']}_cos_DOY"] = np.cos(2 * np.pi * full["DOY"] / 365.0)

    full[f"{PFX['OPT']}_NDVI"] = compute_ndvi(full, nir_col="s2_b8", red_col="s2_b4", eps=EPS)
    full[f"{PFX['OPT']}_NDMI"] = compute_ndmi(full, nir_col="s2_b8", swir_col="s2_b11", eps=EPS)
    full[f"{PFX['OPT']}_MSI"]  = compute_msi(full, swir_col="s2_b11", nir_col="s2_b8", eps=EPS)

    full[f"{PFX['RAD']}_SAR_ratio"] = compute_sar_ratio(full, vv_col="s1_vv", vh_col="s1_vh", eps=EPS)
    full[f"{PFX['RAD']}_SAR_diff"]  = compute_sar_diff(full, vv_col="s1_vv", vh_col="s1_vh")

    full[f"{PFX['MET']}_API"] = compute_api(
        full, precip_col="precip_mm", decay=API_DECAY, group_col=GROUP_COL, date_col=DATE_COL
    )
    full[f"{PFX['MET']}_DSLR"] = compute_days_since_last_rain(
        full, precip_col="precip_mm", threshold_mm=RAIN_THR_MM, group_col=GROUP_COL, date_col=DATE_COL
    )
    for d in RAIN_SUM_DAYS:
        full[f"{PFX['MET']}_rain_sum_{d}d"] = compute_rain_sums_days(
            full, precip_col="precip_mm", window_days=d, group_col=GROUP_COL, date_col=DATE_COL
        )

    full = add_smap_features(
        full,
        group_col=GROUP_COL,
        date_col=DATE_COL,
        imputed=True,
        make_combined=True,
        combined_col="SMAP_sm_interp",
        lags=(1, 7, 30),
        roll_windows=(7, 30),
        ema_alpha=0.2,
        add_ampm_diff=True,
    )

    DIFF_KOBS_LIST = [1, 2, 5, 7, 14, 30]
    GRAD_KOBS_LIST = [WIN_OBS_7, WIN_OBS_14, 30]
    LAG_KOBS_LIST  = [1, 2, 5, 6, 12, 30]
    WIN_LIST       = [WIN_OBS_7, WIN_OBS_14, 30]
    CORR_WINS      = [WIN_OBS_7, WIN_OBS_14]

    dyn_signals = {
        f"{PFX['MET']}_API": f"{PFX['MET']}_API",
        f"{PFX['OPT']}_NDMI": f"{PFX['OPT']}_NDMI",
        f"{PFX['RAD']}_SAR_ratio": f"{PFX['RAD']}_SAR_ratio",
        "LST_modis": "LST_modis",
        f"{PFX['OPT']}_NDVI": f"{PFX['OPT']}_NDVI",
        f"{PFX['RAD']}_SAR_diff": f"{PFX['RAD']}_SAR_diff",
        "s2_b11": "s2_b11",
        "s2_b12": "s2_b12",
    }
    if "SMAP_sm_interp" in full.columns:
        dyn_signals["SMAP_sm_interp"] = "SMAP_sm_interp"

    for col in dyn_signals.values():
        new_cols = {}

        for k in DIFF_KOBS_LIST:
            new_cols[f"{PFX['DYN']}_d_{col}_kobs{k}"] = series_diffs(
                full, col=col, kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )

        for k in GRAD_KOBS_LIST:
            new_cols[f"{PFX['DYN']}_grad_{col}_kobs{k}"] = series_gradient_kobs(
                full, col=col, kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )

        new_cols[f"{PFX['DYN']}_pct_{col}"] = series_pct_change(
            full, col=col, group_col=GROUP_COL, date_col=DATE_COL, eps=EPS
        )

        for w in WIN_LIST:
            new_cols[f"{PFX['VOL']}_rollstd_{col}_kobs{w}"] = rolling_std(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, ddof=0, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollrng_{col}_kobs{w}"] = rolling_range(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollcv_{col}_kobs{w}"] = rolling_cv(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, eps=EPS, ddof=0, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollmean_{col}_kobs{w}"] = rolling_mean(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollmin_{col}_kobs{w}"] = rolling_min(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollmax_{col}_kobs{w}"] = rolling_max(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_ema_{col}_kobs{w}"] = ema(
                full, col=col, alpha=2.0 / (w + 1.0), group_col=GROUP_COL, date_col=DATE_COL
            )

        for k in LAG_KOBS_LIST:
            new_cols[f"{PFX['MEM']}_lag_{col}_kobs{k}"] = series_lags(
                full, col=col, lag_kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )

        new_cols[f"{PFX['MEM']}_smm_{col}_alpha{SMM_ALPHA}_n{KOBS_LONG}"] = smm_index(
            full, col=col, alpha=SMM_ALPHA, n_lags=KOBS_LONG, group_col=GROUP_COL, date_col=DATE_COL
        )

        full = _attach_cols(full, new_cols)

    rad_cols = {}
    rad_cols[f"{PFX['RAD']}_dVV_1"] = series_diffs(
        full, col="s1_vv", kobs=1, group_col=GROUP_COL, date_col=DATE_COL
    )
    for w in CORR_WINS:
        rad_cols[f"{PFX['RAD']}_rough_s1_vv_kobs{w}"] = rolling_mean_abs_change(
            full, col="s1_vv", window=w, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, min_periods=w
        )
        rad_cols[f"{PFX['RAD']}_rough_s1_vh_kobs{w}"] = rolling_mean_abs_change(
            full, col="s1_vh", window=w, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, min_periods=w
        )
    full = _attach_cols(full, rad_cols)

    full[f"{PFX['EVT']}_ts_spike_{SPIKE_COL}"] = compute_time_since_last_spike_past_only(
        full, diff_col=f"{PFX['RAD']}_dVV_1", zthr=SPIKE_Z_THR, group_col=GROUP_COL, date_col=DATE_COL, eps=EPS
    )

    corr_cols = {}
    for w in CORR_WINS:
        corr_cols[f"H_corr_{PFX['RAD']}_SAR_ratio__{PFX['OPT']}_NDMI_kobs{w}"] = rolling_corr(
            full,
            x_col=f"{PFX['RAD']}_SAR_ratio",
            y_col=f"{PFX['OPT']}_NDMI",
            window=w,
            group_col=GROUP_COL,
            date_col=DATE_COL,
            min_periods=w,
            past_only=True,
        )
        corr_cols[f"H_corr_LST_modis__{PFX['OPT']}_NDMI_kobs{w}"] = rolling_corr(
            full,
            x_col="LST_modis",
            y_col=f"{PFX['OPT']}_NDMI",
            window=w,
            group_col=GROUP_COL,
            date_col=DATE_COL,
            min_periods=w,
            past_only=True,
        )
    full = _attach_cols(full, corr_cols)

    train_full = full[full["_split"] == "train"].copy()
    val_full   = full[full["_split"] == "val"].copy()
    test_full  = full[full["_split"] == "test"].copy()

    saz_cols = {}
    for col in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        saz_cols[f"{PFX['SEA']}_sa_{col}"] = pd.Series(np.nan, index=full.index, dtype=float)
        saz_cols[f"{PFX['SEA']}_z_{col}"] = pd.Series(np.nan, index=full.index, dtype=float)
    full = _attach_cols(full, saz_cols)

    for col in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        full.loc[train_full.index, f"{PFX['SEA']}_sa_{col}"] = train_only_monthly_anomaly_global(
            train_full, train_full, col=col, date_col=DATE_COL
        ).values
        full.loc[val_full.index, f"{PFX['SEA']}_sa_{col}"] = train_only_monthly_anomaly_global(
            train_full, val_full, col=col, date_col=DATE_COL
        ).values
        full.loc[test_full.index, f"{PFX['SEA']}_sa_{col}"] = train_only_monthly_anomaly_global(
            train_full, test_full, col=col, date_col=DATE_COL
        ).values

        full.loc[train_full.index, f"{PFX['SEA']}_z_{col}"] = train_only_monthly_zscore_global(
            train_full, train_full, col=col, date_col=DATE_COL, eps=EPS
        ).values
        full.loc[val_full.index, f"{PFX['SEA']}_z_{col}"] = train_only_monthly_zscore_global(
            train_full, val_full, col=col, date_col=DATE_COL, eps=EPS
        ).values
        full.loc[test_full.index, f"{PFX['SEA']}_z_{col}"] = train_only_monthly_zscore_global(
            train_full, test_full, col=col, date_col=DATE_COL, eps=EPS
        ).values

    fft_cols = {}
    for sig in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        dom, ent = rolling_fft_dom_freq_and_entropy(
            full, col=sig, window=FFT_WIN, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, eps=1e-12
        )
        fft_cols[f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"] = dom
        fft_cols[f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"] = ent
    full = _attach_cols(full, fft_cols)

    train_out = full[full["_split"] == "train"].drop(columns=["_split"]).copy()
    val_out   = full[full["_split"] == "val"].drop(columns=["_split"]).copy()
    test_out  = full[full["_split"] == "test"].drop(columns=["_split"]).copy()

    dslr_col = f"{PFX['MET']}_DSLR"
    isnan = full[dslr_col].isna().astype(int)
    train_out[f"{PFX['MET']}_DSLR_isnan"] = isnan[full["_split"] == "train"].values
    val_out[f"{PFX['MET']}_DSLR_isnan"]   = isnan[full["_split"] == "val"].values
    test_out[f"{PFX['MET']}_DSLR_isnan"]  = isnan[full["_split"] == "test"].values

    return train_out, val_out, test_out

def main():
    df = pd.read_csv(MASTER_CLEANED, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    selected_cols = KEEP_META_COLS + BASE_COLS + [TARGET_COL]
    _enforce_no_ground_sensors(selected_cols)

    missing = sorted(list(set(selected_cols) - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in master_cleaned.csv: {missing}")

    df = df[selected_cols].copy()
    df = df.dropna(subset=[TARGET_COL]).copy()

    df[GROUP_COL] = df[GROUP_COL].astype(str)
    df = _load_and_merge_static(df)

    train_df, val_df, test_df, stations = _make_splits_global_temporal(df)
    _check_temporal_split(train_df, val_df, test_df)

    train_df, val_df, test_df = _add_derived_features(train_df, val_df, test_df)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    meta = {
        "seed": SEED,
        "split_by": SPLIT_BY,
        "policy": "global_temporal_only: global date cutoffs into train/val/test, then derive features",
        "variant": "derived_all_new_no_ground_plus_static",
        "target": TARGET_COL,
        "meta_cols_kept": KEEP_META_COLS,
        "base_cols": BASE_COLS,
        "static_path": STATIC_PATH,
        "prefixes": PFX,
        "stations": stations,
        "rows": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved splits to:\n  {TRAIN_PATH}\n  {VAL_PATH}\n  {TEST_PATH}\nMeta:\n  {META_PATH}")

if __name__ == "__main__":
    main()
