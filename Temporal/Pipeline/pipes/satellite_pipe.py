# Jakob Balkovec & Kerry Cheon
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

    # NASA is a lot better for temporal data compared to Sentinel!

    def __init__(self, config=None):
        self.config = config or load_config()
        self.logger = get_logger().getChild("satellite")
        self.cache_path = Path(self.config["satellite"].get("cache_path", "data/cache/satellite_cache.json"))

        try:
            ee.Initialize(project="mdr-project-475522")
            self.logger.info("Authenticated with Google Earth Engine (mdr-project-475522).")
        except Exception as e:
            self.logger.warning(f"EE init failed ({e}), trying to authenticate...")
            ee.Authenticate()
            ee.Initialize(project="mdr-project-475522")

    def fetch_satellite_batch(self, lat, lon, start_date, end_date):
        point = ee.Geometry.Point([lon, lat])
        buffer_region = point.buffer(1000)  # 1km buffer for robust sampling

        # Add temporal padding - expand window by ±3 days
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)
        padded_start = start_dt.strftime("%Y-%m-%d")
        padded_end = end_dt.strftime("%Y-%m-%d")

        results = {"LST": None, "NDVI": None, "Rain_sat": None}

        try:
            # MODIS LST
            try:
                lst_collection = (
                    ee.ImageCollection(self.MODIS_LST)
                    .filterBounds(buffer_region)
                    .filterDate(padded_start, padded_end)
                    .select("LST_Day_1km")
                )

                count = lst_collection.size().getInfo()
                if count > 0:
                    raw_lst = (
                        lst_collection.mean()
                        .reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=buffer_region,
                            scale=1000,
                            bestEffort=True,
                            maxPixels=1e9
                        )
                    )

                    lst_val = raw_lst.get("LST_Day_1km").getInfo()
                    if lst_val is not None:
                        results["LST"] = float(lst_val) * 0.02

            except Exception as e:
                self.logger.debug(f"LST failed: {e}")

            # MODIS NDVI
            try:
                ndvi_collection = (
                    ee.ImageCollection(self.MODIS_NDVI)
                    .filterBounds(buffer_region)
                    .filterDate(padded_start, padded_end)
                    .select("NDVI")
                )

                count = ndvi_collection.size().getInfo()
                if count > 0:
                    raw_ndvi = (
                        ndvi_collection.mean()
                        .reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=buffer_region,
                            scale=250,
                            bestEffort=True,
                            maxPixels=1e9
                        )
                    )

                    ndvi_val = raw_ndvi.get("NDVI").getInfo()
                    if ndvi_val is not None:
                        results["NDVI"] = float(ndvi_val) * 0.0001

            except Exception as e:
                self.logger.debug(f"NDVI failed: {e}")

            # GPM Rain
            try:
                rain_collection = (
                    ee.ImageCollection(self.GPM_RAIN)
                    .filterBounds(buffer_region)
                    .filterDate(padded_start, padded_end)
                    .select("precipitation")
                )

                count = rain_collection.size().getInfo()
                if count > 0:
                    raw_rain = (
                        rain_collection.mean()
                        .reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=buffer_region,
                            scale=10000,
                            bestEffort=True,
                            maxPixels=1e9
                        )
                    )

                    rain_val = raw_rain.get("precipitation").getInfo()
                    if rain_val is not None:
                        results["Rain_sat"] = float(rain_val)

            except Exception as e:
                self.logger.debug(f"Rain failed: {e}")

            return results

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

        cache_path = self.cache_path
        if cache_path.exists():
            self.logger.info(f"Using satellite cache at {cache_path.resolve()}")
            with open(cache_path) as f:
                cache = json.load(f)
        else:
            cache = {}

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
        cache_path.parent.mkdir(parents=True, exist_ok=True)

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
