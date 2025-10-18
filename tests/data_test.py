# Jakob Balkovec
# Data Validation Tests for final.csv

import pandas as pd
import numpy as np
from pathlib import Path

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "final.csv"

def status(tag, msg):
    color = {"PASS": GREEN, "FAIL": RED, "WARN": YELLOW}.get(tag, RESET)
    print(f"{BOLD}{color}[{tag}]\t{RESET} {msg}")

def load_data():
    if not DATA_PATH.exists():
        status("FAIL", f"File not found: {DATA_PATH}")
        return None
    try:
        df = pd.read_csv(DATA_PATH)
        status("PASS", f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        return df
    except Exception as e:
        status("FAIL", f"Could not read CSV: {e}")
        return None

def test_columns(df):
    expected = {"date","station_id","longitude","latitude",
                "air_temp_mean","precipitation","soil_moisture_5cm",
                "DOY","Rain_3d"}
    missing = expected - set(df.columns)
    if missing:
        status("FAIL", f"Missing columns: {missing}")
    else:
        status("PASS", "All expected columns present.")

def test_coordinates(df):
    if "longitude" not in df or "latitude" not in df:
        status("FAIL", "No coordinate columns found.")
        return
    lon, lat = df["longitude"].median(), df["latitude"].median()
    if 116 < abs(lon) < 119 and 46 < lat < 48:
        status("PASS", f"Coordinates plausible (lat≈{lat:.2f}, lon≈{lon:.2f}).")
    else:
        status("FAIL", f"Coordinates look wrong (lat={lat}, lon={lon}).")

def test_dates(df):
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        status("FAIL", f"Date parsing failed: {e}")
        return
    if df["date"].isna().sum() == 0:
        status("PASS", "All dates parsed successfully.")
    else:
        status("FAIL", "Some dates could not be parsed.")
    year_range = (df["date"].dt.year.min(), df["date"].dt.year.max())
    if 2006 < year_range[0] <= 2007 and year_range[1] >= 2024:
        status("PASS", f"Date range plausible: {year_range[0]}–{year_range[1]}.")
    else:
        status("FAIL", f"Unexpected date range: {year_range}.")

def test_ranges(df):
    checks = {
        "precipitation": (0, 200),
        "air_temp_mean": (-50, 60),
        "soil_moisture_5cm": (-99.0, 1.5) # as place holder
    }
    for col, (lo, hi) in checks.items():
        if col not in df:
            status("WARN", f"{col} missing, skipping range check.")
            continue
        valid = df[col].dropna().between(lo, hi)
        if valid.mean() > 0.95:
            status("PASS", f"{col} values mostly in expected range ({lo}, {hi}).")
        else:
            status("FAIL", f"{col} contains out-of-range values.")

def test_features(df):
    if "Rain_3d" not in df or "DOY" not in df:
        status("FAIL", "Derived features missing.")
        return
    if df["Rain_3d"].isna().sum() == 0 and df["DOY"].between(1, 366).all():
        status("PASS", "Derived features (Rain_3d, DOY) look valid.")
    else:
        status("FAIL", "Derived feature values look suspicious.")

if __name__ == "__main__":
    print(f"\n{BOLD}{CYAN}=== Running Data Validation for final.csv ==={RESET}\n\n")
    df = load_data()
    if df is not None:
        test_columns(df)
        test_coordinates(df)
        test_dates(df)
        test_ranges(df)
        test_features(df)
    print(f"\n\n{BOLD}{CYAN}=== Validation Complete ==={RESET}\n")
