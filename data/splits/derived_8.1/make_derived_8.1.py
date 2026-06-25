# Compiles Washington-only processed stations, merges LIA, derives 350+ features, and splits into train/val/test.

import os
import sys
import json
import numpy as np
import pandas as pd
import yaml

# Append local path so we can import utils safely
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

# Year Ranges
TRAIN_YEARS = set(range(2017, 2021)) # 2017-2020
VAL_YEARS   = set(range(2021, 2023)) # 2021-2022
TEST_YEARS  = set(range(2023, 2026)) # 2023-2025

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

# Standard constants from derived_6.0
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

STATION_NAME_MAPPING = {
    "WA_Spokane_17_SSW": "Spokane",
    "WA_Quinault_4_NE": "Quinault",
    "WA_Darrington_21_NNE": "Darrington",
}

# The 13 Washington Stations in scope
WA_STATIONS = [
    "Spokane",
    "Darrington",
    "Quinault",
    "Touchet_WA_824",
    "SourdoughGulch_WA_985",
    "CayusePass_WA",
    "Paradise_WA",
    "BurntMountain_WA",
    "BeaverPass_WA_990",
    "HartsPass_WA_515",
    "MartenRidge_WA_999",
    "MFNooksack_WA_1011",
    "RainyPass_WA_711"
]

def load_processed_stations(pipeline_root: str) -> pd.DataFrame:
    # Read config.yaml to get station processed paths
    config_path = os.path.join(pipeline_root, "src", "pipeline", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    stations_cfg = config.get("stations", {})
    frames = []
    
    for key, cfg in stations_cfg.items():
        # Only process if not commented out (YAML parsing naturally ignores commented sections)
        save_cfg = cfg.get("save", {})
        out_path = save_cfg.get("out_path")
        if not out_path:
            continue
            
        full_out_path = os.path.join(pipeline_root, out_path)
        if not os.path.exists(full_out_path):
            print(f"[WARN] Processed file missing for {key}: {full_out_path}")
            continue
            
        print(f"Reading processed data for {key} from {out_path}...")
        df = pd.read_csv(full_out_path, low_memory=False)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        
        # Standardise station_id column name
        if "station_id" in df.columns:
            df["station_id"] = df["station_id"].astype(str)
            df["station_id"] = df["station_id"].replace(STATION_NAME_MAPPING)
            
        # Keep only the Washington stations in scope
        unique_ids = df["station_id"].unique()
        if len(unique_ids) > 0 and unique_ids[0] in WA_STATIONS:
            frames.append(df)
            print(f"  Added station: {unique_ids[0]} ({len(df)} rows)")
        else:
            print(f"  Skipped non-WA or out-of-scope station: {unique_ids}")
            
    if not frames:
        raise ValueError("No processed WA station files found! Ensure pipeline was run first.")
        
    combined = pd.concat(frames, ignore_index=True)
    return combined

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing derived features (NDVI, NDMI, MSI, SAR, API)...")
    full = df.copy()
    
    # Sort dates for chronological calculations
    full = full.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    
    # Basic structural columns
    full[f"{PFX['SEA']}_sin_DOY"] = np.sin(2 * np.pi * full["DOY"] / 365.0)
    full[f"{PFX['SEA']}_cos_DOY"] = np.cos(2 * np.pi * full["DOY"] / 365.0)
    
    # Vegetation/Moisture Indices
    full[f"{PFX['OPT']}_NDVI"] = compute_ndvi(full, nir_col="s2_b8", red_col="s2_b4", eps=EPS)
    full[f"{PFX['OPT']}_NDMI"] = compute_ndmi(full, nir_col="s2_b8", swir_col="s2_b11", eps=EPS)
    full[f"{PFX['OPT']}_MSI"]  = compute_msi(full, swir_col="s2_b11", nir_col="s2_b8", eps=EPS)
    
    # SAR features
    full[f"{PFX['RAD']}_SAR_ratio"] = compute_sar_ratio(full, vv_col="s1_vv", vh_col="s1_vh", eps=EPS)
    full[f"{PFX['RAD']}_SAR_diff"]  = compute_sar_diff(full, vv_col="s1_vv", vh_col="s1_vh")
    
    # Weather metrics (Hydrological memory)
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
        
    # SMAP features
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
        
    print("Computing rolling stats, lags, and difference sequences (largo set)...")
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
        full = pd.concat([full, pd.DataFrame(new_cols, index=full.index)], axis=1)
        
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
    full = pd.concat([full, pd.DataFrame(rad_cols, index=full.index)], axis=1)
    
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
    full = pd.concat([full, pd.DataFrame(corr_cols, index=full.index)], axis=1)
    
    return full

def add_split_and_drift_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Splitting dataset and computing drift features...")
    df = df.copy()
    df["year"] = df[DATE_COL].dt.year.astype(int)
    
    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    val_df   = df[df["year"].isin(VAL_YEARS)].copy()
    test_df  = df[df["year"].isin(TEST_YEARS)].copy()
    
    # Check bounds
    print(f"Train years: {sorted(list(TRAIN_YEARS))}, rows: {len(train_df)}")
    print(f"Val years: {sorted(list(VAL_YEARS))}, rows: {len(val_df)}")
    print(f"Test years: {sorted(list(TEST_YEARS))}, rows: {len(test_df)}")
    
    # Calculate drift features based on train bounds
    ref_min_year = float(min(TRAIN_YEARS))
    ref_max_year = float(max(TRAIN_YEARS))
    denom = ref_max_year - ref_min_year
    
    def compute_drifts(split_df):
        split_df = split_df.copy()
        split_df["year_frac"] = (split_df["year"] - ref_min_year) / denom
        theta = 2 * np.pi * split_df["year_frac"]
        split_df["sin_year"] = np.sin(theta)
        split_df["cos_year"] = np.cos(theta)
        
        # API and SMAP drift interactions
        api_col = f"{PFX['MET']}_API"
        smap_col = "SMAP_sm_pm_interp_ema02"
        
        if api_col in split_df.columns:
            split_df["API_x_year"] = split_df[api_col] * split_df["year_frac"]
        else:
            split_df["API_x_year"] = np.nan
            
        if smap_col in split_df.columns:
            split_df["SMAP_x_year"] = split_df[smap_col] * split_df["year_frac"]
        else:
            split_df["SMAP_x_year"] = np.nan
            
        return split_df
        
    train_out = compute_drifts(train_df)
    val_out   = compute_drifts(val_df)
    test_out  = compute_drifts(test_df)
    
    # Compute monthly anomalies and zscores
    print("Computing seasonal anomalies (monthly z-score/anomaly)...")
    saz_cols = {}
    for col in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        saz_cols[f"{PFX['SEA']}_sa_{col}"] = pd.Series(np.nan, index=df.index, dtype=float)
        saz_cols[f"{PFX['SEA']}_z_{col}"] = pd.Series(np.nan, index=df.index, dtype=float)
        
    train_out = pd.concat([train_out, pd.DataFrame({k: saz_cols[k].loc[train_out.index] for k in saz_cols}, index=train_out.index)], axis=1)
    val_out   = pd.concat([val_out, pd.DataFrame({k: saz_cols[k].loc[val_out.index] for k in saz_cols}, index=val_out.index)], axis=1)
    test_out  = pd.concat([test_out, pd.DataFrame({k: saz_cols[k].loc[test_out.index] for k in saz_cols}, index=test_out.index)], axis=1)
    
    for col in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        train_out[f"{PFX['SEA']}_sa_{col}"] = train_only_monthly_anomaly_global(train_out, train_out, col=col, date_col=DATE_COL).values
        val_out[f"{PFX['SEA']}_sa_{col}"]   = train_only_monthly_anomaly_global(train_out, val_out, col=col, date_col=DATE_COL).values
        test_out[f"{PFX['SEA']}_sa_{col}"]  = train_only_monthly_anomaly_global(train_out, test_out, col=col, date_col=DATE_COL).values
        
        train_out[f"{PFX['SEA']}_z_{col}"] = train_only_monthly_zscore_global(train_out, train_out, col=col, date_col=DATE_COL, eps=EPS).values
        val_out[f"{PFX['SEA']}_z_{col}"]   = train_only_monthly_zscore_global(train_out, val_out, col=col, date_col=DATE_COL, eps=EPS).values
        test_out[f"{PFX['SEA']}_z_{col}"]  = train_only_monthly_zscore_global(train_out, test_out, col=col, date_col=DATE_COL, eps=EPS).values
        
    # FFT columns
    print("Computing rolling FFT features...")
    fft_cols_train = {}
    fft_cols_val = {}
    fft_cols_test = {}
    for sig in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        # We compute on combined and then extract, to avoid window edge effects at split boundaries
        combined_saz = pd.concat([train_out, val_out, test_out]).sort_values([GROUP_COL, DATE_COL])
        dom, ent = rolling_fft_dom_freq_and_entropy(
            combined_saz, col=sig, window=FFT_WIN, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, eps=1e-12
        )
        combined_saz[f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"] = dom
        combined_saz[f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"] = ent
        
        train_out[f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"] = combined_saz.loc[train_out.index, f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"].values
        train_out[f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"] = combined_saz.loc[train_out.index, f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"].values
        
        val_out[f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"] = combined_saz.loc[val_out.index, f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"].values
        val_out[f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"] = combined_saz.loc[val_out.index, f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"].values
        
        test_out[f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"] = combined_saz.loc[test_out.index, f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"].values
        test_out[f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"] = combined_saz.loc[test_out.index, f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"].values
        
    dslr_col = f"{PFX['MET']}_DSLR"
    train_out[f"{PFX['MET']}_DSLR_isnan"] = train_out[dslr_col].isna().astype(int).values
    val_out[f"{PFX['MET']}_DSLR_isnan"]   = val_out[dslr_col].isna().astype(int).values
    test_out[f"{PFX['MET']}_DSLR_isnan"]  = test_out[dslr_col].isna().astype(int).values
    
    return train_out, val_out, test_out

def merge_lia(df: pd.DataFrame, lia_path: str) -> pd.DataFrame:
    print("Merging Local Incident Angle (LIA) features...")
    if not os.path.exists(lia_path):
        raise FileNotFoundError(f"LIA CSV not found at: {lia_path}. Run GEE fetch_lia.py first.")
        
    lia = pd.read_csv(lia_path)
    lia_cols = ["station_id", "lia_mean_asc_deg", "lia_std_asc_deg", "lia_mean_desc_deg", "lia_std_desc_deg"]
    
    # Verify columns
    missing = [c for c in lia_cols if c not in lia.columns]
    if missing:
        raise ValueError(f"LIA CSV is missing columns: {missing}")
        
    lia = lia[lia_cols].copy()
    lia["station_id"] = lia["station_id"].astype(str)
    df["station_id"] = df["station_id"].astype(str)
    
    merged = df.merge(lia, on="station_id", how="left")
    
    # Check if any got NaNs after merge
    miss_rate = merged["lia_mean_asc_deg"].isna().mean()
    if miss_rate > 0:
        missing_stations = sorted(merged.loc[merged["lia_mean_asc_deg"].isna(), "station_id"].unique().tolist())
        raise ValueError(f"LIA merge produced missing values (rate={miss_rate:.4%}) for stations: {missing_stations}")
        
    print("  LIA merge successful.")
    return merged

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    print(f"Project root identified as: {project_root}")
    
    # 1. Load processed WA stations
    combined_df = load_processed_stations(project_root)
    print(f"Total initial rows loaded: {len(combined_df)}")
    
    # Filter columns to only what we need + target
    expected_cols = KEEP_META_COLS + BASE_COLS + [TARGET_COL]
    
    # Verify expected columns are present
    missing = sorted(list(set(expected_cols) - set(combined_df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in processed data: {missing}")
        
    combined_df = combined_df[expected_cols].copy()
    
    # Drop rows where target is NaN (following derived_6.0 policy)
    combined_df = combined_df.dropna(subset=[TARGET_COL]).copy()
    print(f"Rows after dropping empty targets ({TARGET_COL}): {len(combined_df)}")
    
    # 2. Add derived feature set
    df_derived = add_derived_features(combined_df)
    
    # 3. Merge LIA features
    lia_path = os.path.join(script_dir, "LIA", "stations_lia.csv")
    df_lia = merge_lia(df_derived, lia_path)
    
    # 4. Split and Add drifts/FFT
    train_df, val_df, test_df = add_split_and_drift_features(df_lia)
    
    # 5. Save outputs
    out_dir = script_dir
    train_path = os.path.join(out_dir, "train.csv")
    val_path   = os.path.join(out_dir, "val.csv")
    test_path  = os.path.join(out_dir, "test.csv")
    meta_path  = os.path.join(out_dir, "split_meta.json")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    meta = {
        "source": "processed station CSVs (WA-only)",
        "derived_features": "350+ lags, rolling means, and GEE remote-sensing features fully populated",
        "train_years": sorted(list(TRAIN_YEARS)),
        "val_years": sorted(list(VAL_YEARS)),
        "test_years": sorted(list(TEST_YEARS)),
        "lia_csv": os.path.relpath(lia_path, project_root),
        "stations": sorted(list(train_df["station_id"].unique())),
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        }
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\nSaved derived_8.1 splits successfully:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Val:   {val_path} ({len(val_df)} rows)")
    print(f"  Test:  {test_path} ({len(test_df)} rows)")
    print(f"  Meta:  {meta_path}")

if __name__ == "__main__":
    main()
