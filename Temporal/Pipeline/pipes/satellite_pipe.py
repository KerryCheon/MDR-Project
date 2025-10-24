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

    def __init__(self, config=None, station_name=None):
        # pre:  config is the global config (not per-station)
        # post: initializes SatellitePipe with GEE connection and caching settings
        # desc: Initializes Earth Engine and sets up cache path per station

        self.config = config or load_config()
        self.station_name = station_name or "global"
        self.logger = get_logger().getChild(f"satellite.{self.station_name}")

        # per-station cache template
        cache_template = self.config["satellite"].get(
            "cache_path",
            "Pipeline/data/cache/{station}_satellite_cache.json"
        )

        # resolve per-station cache file
        self.cache_path = Path(cache_template.format(station=self.station_name))

        self.logger.info(f"Satellite cache path set to: {self.cache_path}")

        try:
            ee.Initialize(project="mdr-project-475522")
            self.logger.info("Authenticated with Google Earth Engine (mdr-project-475522).")
        except Exception as e:
            self.logger.warning(f"EE init failed ({e}), trying to authenticate...")
            ee.Authenticate()
            ee.Initialize(project="mdr-project-475522")

    def fetch_satellite_batch(self, lat, lon, start_date, end_date):
        # pre: coordinates and date range provided
        # post: returns dictionary with LST, NDVI, and Rain_sat values

        point = ee.Geometry.Point([lon, lat])
        buffer_region = point.buffer(1000)  # 1 km buffer

        from datetime import datetime, timedelta
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)
        padded_start, padded_end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

        results = {"LST": None, "NDVI": None, "Rain_sat": None}

        try:
            # MODIS LST
            try:
                lst = (
                    ee.ImageCollection(self.MODIS_LST)
                    .filterBounds(buffer_region)
                    .filterDate(padded_start, padded_end)
                    .select("LST_Day_1km")
                )
                if lst.size().getInfo() > 0:
                    val = lst.mean().reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=buffer_region,
                        scale=1000,
                        bestEffort=True,
                        maxPixels=1e9,
                    ).get("LST_Day_1km").getInfo()
                    if val is not None:
                        results["LST"] = float(val) * 0.02
            except Exception as e:
                self.logger.debug(f"LST failed: {e}")

            # MODIS NDVI
            try:
                ndvi = (
                    ee.ImageCollection(self.MODIS_NDVI)
                    .filterBounds(buffer_region)
                    .filterDate(padded_start, padded_end)
                    .select("NDVI")
                )
                if ndvi.size().getInfo() > 0:
                    val = ndvi.mean().reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=buffer_region,
                        scale=250,
                        bestEffort=True,
                        maxPixels=1e9,
                    ).get("NDVI").getInfo()
                    if val is not None:
                        results["NDVI"] = float(val) * 0.0001
            except Exception as e:
                self.logger.debug(f"NDVI failed: {e}")

            # GPM Rain
            try:
                rain = (
                    ee.ImageCollection(self.GPM_RAIN)
                    .filterBounds(buffer_region)
                    .filterDate(padded_start, padded_end)
                    .select("precipitation")
                )
                if rain.size().getInfo() > 0:
                    val = rain.mean().reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=buffer_region,
                        scale=10000,
                        bestEffort=True,
                        maxPixels=1e9,
                    ).get("precipitation").getInfo()
                    if val is not None:
                        results["Rain_sat"] = float(val)
            except Exception as e:
                self.logger.debug(f"Rain failed: {e}")

            return results

        except Exception as e:
            self.logger.warning(f"Batch satellite retrieval failed: {e}")
            return results

    def run(self, df):
        if df is None or df.empty:
            self.logger.warning("No data received in SatellitePipe.")
            return pd.DataFrame()

        cache_path = self.cache_path
        cache = {}
        if cache_path.exists():
            self.logger.info(f"Using satellite cache at {cache_path.resolve()}")
            with open(cache_path) as f:
                cache = json.load(f)

        self.logger.info(f"[{self.station_name}] Starting batched satellite retrieval for {len(df)} rows...")
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

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Satellite ({self.station_name})"):
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

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        sat_df = pd.DataFrame(results).dropna(subset=["LST", "NDVI", "Rain_sat"], how="all")
        sat_df["date"] = pd.to_datetime(sat_df["date"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        merged = pd.merge(df, sat_df, on="date", how="left")

        coverage = {
            "LST": merged["LST"].notna().mean(),
            "NDVI": merged["NDVI"].notna().mean(),
            "Rain_sat": merged["Rain_sat"].notna().mean(),
        }

        self.logger.info(
            f"[{self.station_name}] SatellitePipe complete — {len(merged)} records retrieved. "
            f"Coverage — LST: {coverage['LST']:.2%}, NDVI: {coverage['NDVI']:.2%}, Rain: {coverage['Rain_sat']:.2%}"
        )
        return merged
