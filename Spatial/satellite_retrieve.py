import ee
import pandas as pd
import numpy as np
import os
import argparse
from datetime import timedelta

# --------------------------
# Initialize Google Earth Engine
# --------------------------
try:
    ee.Initialize(project='mdr-research')
    print("✅ Earth Engine initialized with project 'mdr-research'")
except Exception as e:
    print(f"⚠️ Earth Engine init failed: {e}")
    print("Trying default initialization...")
    ee.Initialize()
    print("✅ Earth Engine initialized with default credentials")

# --------------------------
# Helper: get mean satellite values for a date/time
# --------------------------
def get_satellite_data(lat, lon, start_time, end_time):
    point = ee.Geometry.Point([lon, lat])

    # Example: Get precipitation estimate from CHIRPS and temperature from MODIS
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_time, end_time) \
        .filterBounds(point) \
        .select('precipitation')

    modis = ee.ImageCollection('MODIS/006/MOD11A1') \
        .filterDate(start_time, end_time) \
        .filterBounds(point) \
        .select('LST_Day_1km')

    # Reduce to mean over period and region
    chirps_mean = chirps.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=5000
    )

    modis_mean = modis.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    )

    precip_val = chirps_mean.getInfo().get('precipitation', np.nan)
    lst_val = modis_mean.getInfo().get('LST_Day_1km', np.nan)
    if lst_val is not np.nan:
        lst_val = float(lst_val) * 0.02 - 273.15  # Convert to Celsius

    return precip_val, lst_val

# --------------------------
# Main pipeline
# --------------------------
def main(args):
    print(f"Loading ground data from: {args.ground_csv}")
    df = pd.read_csv(args.ground_csv)

    # Handle datetime column properly
    if 'datetime' not in df.columns:
        print("No 'datetime' column found, using index as datetime...")
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.reset_index().rename(columns={'index': 'datetime'})
    else:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Get coordinates from metadata
    lat = df['latitude'].iloc[0]
    lon = df['longitude'].iloc[0]
    print(f"Using station coordinates: lat={lat}, lon={lon}")

    # Prepare output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect satellite data hourly (or every N hours for efficiency)
    results = []
    step_hours = 3  # change to 1 for full hourly resolution
    for i in range(0, len(df), step_hours):
        row_time = df['datetime'].iloc[i]
        start_time = (row_time - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
        end_time = (row_time + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

        try:
            precip, temp = get_satellite_data(lat, lon, start_time, end_time)
        except Exception as e:
            print(f"Error fetching data for {row_time}: {e}")
            precip, temp = np.nan, np.nan

        results.append({
            'datetime': row_time,
            'sat_precip_mm': precip,
            'sat_temp_c': temp
        })

    sat_df = pd.DataFrame(results)
    merged = pd.merge_asof(df.sort_values('datetime'),
                           sat_df.sort_values('datetime'),
                           on='datetime')

    out_path = os.path.join(args.out_dir, "satellite_lind_2021.csv")
    merged.to_csv(out_path, index=False)
    print(f"✅ Saved merged satellite + ground dataset to {out_path}")

# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve satellite data for ground station times")
    parser.add_argument("ground_csv", type=str, help="Path to cleaned ground data CSV")
    parser.add_argument("--out_dir", type=str, default="./satellite", help="Output directory")
    args = parser.parse_args()

    main(args)
