# Jakob Balkovec
# Satellite Pipe

# This module defines the SatellitePipe class, which retrieves satellite-derived
# features (LST, NDVI, Rain) from Google Earth Engine in batched requests
# and merges them into the input dataframe.

import ee
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import get_logger
from utils.config import load_config


class SatellitePipe:
    MODIS_LST = "MODIS/061/MOD11A1"
    MODIS_NDVI = "MODIS/061/MOD13Q1"
    GPM_RAIN = "NASA/GPM_L3/IMERG_V07"

    def __init__(self, config=None):
        self.config = config or load_config()
        self.logger = get_logger().getChild("satellite")

        try:
            ee.Initialize(project="mdr-project-475522")
            self.logger.info("Authenticated with Google Earth Engine (mdr-project-475522).")
        except Exception as e:
            self.logger.warning(f"EE init failed ({e}), trying to authenticate...")
            ee.Authenticate()
            ee.Initialize(project="mdr-project-475522")

    def fetch_satellite_batch(self, lat, lon, start_date, end_date):
        # pre: latitude, longitude, start_date, end_date
        # post: retrieves average satellite data for given location and date range
        # desc: Fetches LST, NDVI, and Rain data from GEE for given point.
        point = ee.Geometry.Point([lon, lat])

        try:
            # MODIS LST (Kelvin scaled ×0.02)
            lst = (
                ee.ImageCollection(self.MODIS_LST)
                .filterBounds(point)
                .filterDate(start_date, end_date)
                .select("LST_Day_1km")
                .mean()
                .multiply(0.02)
                .reduceRegion(ee.Reducer.mean(), point, 1000)
                .get("LST_Day_1km")
                .getInfo()
            )

            # MODIS NDVI (scale ×0.0001)
            ndvi = (
                ee.ImageCollection(self.MODIS_NDVI)
                .filterBounds(point)
                .filterDate(start_date, end_date)
                .select("NDVI")
                .mean()
                .multiply(0.0001)
                .reduceRegion(ee.Reducer.mean(), point, 250)
                .get("NDVI")
                .getInfo()
            )

            # GPM Rain (handle possible key variations)
            rain_img = (
                ee.ImageCollection(self.GPM_RAIN)
                .filterBounds(point)
                .filterDate(start_date, end_date)
                .select(["precipitationCal"], ["precipitation"])
                .mean()
            )
            rain = (
                rain_img.reduceRegion(ee.Reducer.mean(), point, 10000)
                .get("precipitation")
                .getInfo()
            )

            return {"LST": lst, "NDVI": ndvi, "Rain_sat": rain}

        except Exception as e:
            self.logger.warning(f"Batch satellite retrieval failed: {e}")
            return {"LST": None, "NDVI": None, "Rain_sat": None}

    def run(self, df):
        # pre:  dataframe with date, latitude, longitude columns
        # post: dataframe with satellite features merged
        # desc: Retrieves satellite data in monthly batches (parallelized) and merges with input dataframe.

        if df is None or df.empty:
            self.logger.warning("No data received in SatellitePipe.")
            return pd.DataFrame()

        cache_path = Path("data/processed/satellite_cache.json")
        cache = json.load(open(cache_path)) if cache_path.exists() else {}

        self.logger.info(f"Starting batched satellite retrieval for {len(df)} rows...")
        df["date"] = pd.to_datetime(df["date"])
        grouped = df.groupby(df["date"].dt.to_period("M"))

        results = []

        futures = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            for period, group in grouped:
                date_key = str(period)
                if date_key in cache:
                    continue

                start = group["date"].min().strftime("%Y-%m-%d")
                end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                lat, lon = group["latitude"].median(), group["longitude"].median()
                futures[executor.submit(self.fetch_satellite_batch, lat, lon, start, end)] = date_key

            for future in tqdm(as_completed(futures), total=len(futures), desc="Satellite batches"):
                date_key = futures[future]
                try:
                    cache[date_key] = future.result()
                except Exception as e:
                    self.logger.warning(f"Batch {date_key} failed: {e}")
                    cache[date_key] = {"LST": None, "NDVI": None, "Rain_sat": None}

        for period, group in grouped:
            date_key = str(period)
            res = cache.get(date_key, {"LST": None, "NDVI": None, "Rain_sat": None})
            for date in group["date"]:
                results.append({"date": date.strftime("%Y-%m-%d"), **res})

        # save cache
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        sat_df = pd.DataFrame(results).dropna(subset=["LST", "NDVI", "Rain_sat"], how="all")
        sat_df["date"] = pd.to_datetime(sat_df["date"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        merged = pd.merge(df, sat_df, on="date", how="left")

        # coverage summary
        coverage = {
            "LST": merged["LST"].notna().mean(),
            "NDVI": merged["NDVI"].notna().mean(),
            "Rain_sat": merged["Rain_sat"].notna().mean(),
        }
        self.logger.info(
            f"SatellitePipe complete — {len(merged)} records retrieved. "
            f"Coverage — LST: {coverage['LST']:.2%}, NDVI: {coverage['NDVI']:.2%}, Rain: {coverage['Rain_sat']:.2%}"
        )

        return merged
