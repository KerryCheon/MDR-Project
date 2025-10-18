"""
step1_ground_cleaning.py

Usage:
    python step1_ground_cleaning.py /path/to/ground.csv --out_dir ./processed --tz America/Los_Angeles

This script:
- loads the provided comma-separated ground sensor CSV
- filters to station id 2021 (Lind #1)
- parses datetime, localizes timezone (default America/Los_Angeles)
- normalizes column names, maps target column to soil_moisture_2in
- handles missing values and optionally resamples to hourly
- writes cleaned CSV to out_dir/ground_cleaned_lind_2021.csv
"""

import os
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config / Column mapping
# -------------------------
STATION_ID = 2021
STATION_NAME = "Lind #1"
DEFAULT_TZ = "America/Los_Angeles"

# Map messy column names to friendly column names (expand if your CSV uses different labels)
COL_MAP = {
    "Date": "datetime",
    "Station Id": "station_id",
    "Station Name": "station_name",
    "Precipitation Increment (in)": "precip_in",
    "Air Temperature Observed (degF)": "air_temp_obs_f",
    "Air Temperature Average (degF)": "air_temp_avg_f",
    "Air Temperature Maximum (degF)": "air_temp_max_f",
    "Air Temperature Minimum (degF)": "air_temp_min_f",
    "Soil Moisture Percent -2in (pct)": "soil_moisture_2in",
    "Soil Moisture Percent -4in (pct)": "soil_moisture_4in",
    "Soil Moisture Percent -8in (pct)": "soil_moisture_8in",
    "Soil Moisture Percent -20in (pct)": "soil_moisture_20in",
    "Soil Moisture Percent -40in (pct)": "soil_moisture_40in",
    "Soil Temperature Observed -2in (degF)": "soil_temp_2in_f",
    "Soil Temperature Observed -4in (degF)": "soil_temp_4in_f",
    "Soil Temperature Observed -8in (degF)": "soil_temp_8in_f",
    "Soil Temperature Observed -20in (degF)": "soil_temp_20in_f",
    "Soil Temperature Observed -40in (degF)": "soil_temp_40in_f",
    "Relative Humidity (pct)": "rel_humidity",
    "Solar Radiation Average (watt/m2)": "solar_rad_wm2",
}

# Station metadata (will be added to output if desired)
STATION_META = {
    "station_id": STATION_ID,
    "station_name": STATION_NAME,
    "latitude": 47.00,
    "longitude": -118.57,
    "elevation_ft": 1650,
    "state": "WA",
    "network": "SCAN",
    "county": "Adams",
}

# -------------------------
# Helper functions
# -------------------------
def f_to_c(series: pd.Series) -> pd.Series:
    """Convert Fahrenheit to Celsius, handle NaNs."""
    return (series.astype(float) - 32.0) * 5.0 / 9.0

def safe_float(x):
    """Try convert to float; treat common placeholders as NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s in ["", "NA", "N/A", "M", "-", "--", "null", "None"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

# -------------------------
# Main cleaning pipeline
# -------------------------
def load_and_clean_ground(csv_path: str,
                          tz: str = DEFAULT_TZ,
                          resample_hourly: bool = True,
                          interp_method: Optional[str] = "time"):
    """
    Load CSV, filter to station, parse datetime, normalize columns, handle missing values.

    Parameters:
        csv_path: path to the raw CSV
        tz: timezone string for localization (pytz name)
        resample_hourly: if True, reindex to hourly frequency and interpolate gaps
        interp_method: interpolation method for numeric columns (e.g., 'time' or 'linear')

    Returns:
        df: cleaned DataFrame (datetime index localized), numeric columns converted
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)  # read as strings to handle odd tokens
    print(f"Raw rows: {len(df)}, columns: {list(df.columns)[:10]} ...")

    # Rename columns we know about
    rename_map = {col: COL_MAP[col] for col in COL_MAP if col in df.columns}
    df = df.rename(columns=rename_map)

    # Keep only the target station (Station Id may be string or numeric)
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].apply(lambda x: safe_float(x) if x is not None else np.nan)
        df = df[df["station_id"] == STATION_ID]
    else:
        print("Warning: 'Station Id' not found in CSV; assuming file contains only relevant station rows.")

    if df.empty:
        raise ValueError(f"No rows found for station_id {STATION_ID}. Check your CSV or station filter.")

    # Parse datetime
    if "datetime" not in df.columns:
        raise ValueError("CSV does not contain a 'Date' column (expected). Please check column headers.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    bad_dt = df["datetime"].isna().sum()
    if bad_dt > 0:
        print(f"Warning: {bad_dt} rows have invalid datetimes and will be dropped.")
        df = df.dropna(subset=["datetime"])

    # Localize timezone (assume naive timestamps are local station time)
    # If timestamps already include tz info, this will convert instead of localize
    try:
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize(tz)
        else:
            df["datetime"] = df["datetime"].dt.tz_convert(tz)
    except Exception as e:
        # fallback: set as naive
        print(f"Timezone localization failed: {e}. Leaving timestamps naive.")
    df = df.set_index("datetime")
    df = df.sort_index()

    # Convert numeric columns
    for col in list(df.columns):
        if col.startswith("soil_") or col.startswith("air_") or col.startswith("soil_temp") or col in ["precip_in", "rel_humidity", "solar_rad_wm2"]:
            df[col] = df[col].apply(safe_float)

    # Map soil moisture target name explicitly (if alternate name used)
    if "soil_moisture_2in" not in df.columns:
        # try to find variant names with substring match
        candidates = [c for c in df.columns if "Soil Moisture" in c or "soil" in c.lower() and "2" in c]
        print("soil_moisture_2in not found exactly; candidates:", candidates)

    # Convert soil temperature F -> C
    for st_col in ["soil_temp_2in_f", "soil_temp_4in_f", "soil_temp_8in_f", "soil_temp_20in_f", "soil_temp_40in_f",
                   "air_temp_obs_f", "air_temp_avg_f", "air_temp_max_f", "air_temp_min_f"]:
        if st_col in df.columns:
            df[st_col.replace("_f", "_c")] = f_to_c(df[st_col])
            # you may drop the old Fahrenheit column if you prefer
            # df = df.drop(columns=[st_col])

    # Add station metadata columns (constant)
    for k, v in STATION_META.items():
        df[k] = v

    # Resample / reindex to regular hourly grid if requested (useful for matching Sentinel)
    if resample_hourly:
        print("Resampling to hourly grid and interpolating numeric columns where appropriate.")
        start = df.index.min().floor("H")
        end = df.index.max().ceil("H")
        hourly_index = pd.date_range(start=start, end=end, freq="H", tz=df.index.tz)
        df = df.reindex(hourly_index.union(df.index)).sort_index()
        # keep station metadata constant (fill forward)
        meta_cols = list(STATION_META.keys())
        df[meta_cols] = df[meta_cols].ffill().bfill()
        # Interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if interp_method is not None:
            try:
                df[numeric_cols] = df[numeric_cols].interpolate(method=interp_method, limit_direction="both")
            except Exception:
                df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        # For still-missing numeric values, optionally forward-fill
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df

def summarize_and_plot(df: pd.DataFrame, out_dir: str):
    """Print summary and create a couple of quick plots saved to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== Data Summary ===")
    print("Time range:", df.index.min(), "to", df.index.max())
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nNumeric summary (first 10 numeric cols):\n", df.select_dtypes(include=[np.number]).iloc[:, :10].describe().T)

    # Plot soil_moisture_2in if present
    if "soil_moisture_2in" in df.columns:
        plt.figure(figsize=(10, 3))
        df["soil_moisture_2in"].plot(title="Soil Moisture -2in (pct) over time")
        plt.ylabel("pct")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "soil_moisture_2in_timeseries.png"))
        plt.close()

    # Plot scatter of soil moisture vs precipitation (if present)
    if ("precip_in" in df.columns) and ("soil_moisture_2in" in df.columns):
        plt.figure(figsize=(5, 4))
        plt.scatter(df["precip_in"], df["soil_moisture_2in"], s=8, alpha=0.6)
        plt.xlabel("Precipitation (in)")
        plt.ylabel("Soil Moisture -2in (pct)")
        plt.title("Soil moisture vs precipitation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "soil_vs_precip.png"))
        plt.close()

    # Save a small sample
    df.iloc[:200].to_csv(os.path.join(out_dir, "ground_sample_head.csv"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to ground sensor CSV (comma-separated)")
    parser.add_argument("--out_dir", default="./processed", help="directory to save cleaned output")
    parser.add_argument("--tz", default=DEFAULT_TZ, help="timezone to localize timestamps (default America/Los_Angeles)")
    parser.add_argument("--no_resample", dest="resample", action="store_false", help="disable hourly resampling/interpolation")
    args = parser.parse_args()

    df_clean = load_and_clean_ground(args.csv, tz=args.tz, resample_hourly=args.resample)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"ground_cleaned_lind_{STATION_ID}.csv")
    print(f"Saving cleaned CSV to {out_csv}")
    df_clean.to_csv(out_csv, index=True)
    summarize_and_plot(df_clean, args.out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
