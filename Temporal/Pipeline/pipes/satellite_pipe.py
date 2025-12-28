# Jakob Balkovec & Kerry Cheon
# satellite_pipe.py
# Satellite Pipe (Sentinel-1, Sentinel-2, MODIS, GPM, DEM)

import ee
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.logger import get_logger
from ..utils.config import load_config


class SatellitePipe:
    MODIS_LST = "MODIS/061/MOD11A1"
    MODIS_NDVI = "MODIS/006/MOD13C2"
    # GPM_RAIN = "NASA/GPM_L3/IMERG_V07"  # deprecated: replaced by Open-Meteo rain/precip pipes

    S1_GRD = "COPERNICUS/S1_GRD"
    S2_L2A = "COPERNICUS/S2_SR_HARMONIZED"
    SRTM_DEM = "USGS/SRTMGL1_003"  # DEM dataset

    def __init__(self, config=None, station_name=None):
        self.config = config or load_config()
        self.station_name = station_name or "global"
        self.logger = get_logger().getChild(f"satellite.{self.station_name}")

        cache_template = self.config["satellite"].get(
            "cache_path",
            "Pipeline/data/cache/{station}_satellite_cache.json"
        )
        self.cache_path = Path(cache_template.format(station=self.station_name))
        self.logger.info(f"Satellite cache path set to: {self.cache_path}")

        try:
            ee.Initialize(project="mdr-project-475522")
        except Exception:
            ee.Authenticate()
            ee.Initialize(project="mdr-project-475522")

    def fetch_satellite_batch(self, lat, lon, start_date, end_date):
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)
        padded_start, padded_end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(1000)

        results = {
            "LST_modis": None,
            "NDVI_modis": None,
            # "Rain_sat": None,  # deprecated: replaced by Open-Meteo rain/precip pipes
            "s1_vv": None,
            "s1_vh": None,
            "s1_vv_dB": None,
            "s1_vh_dB": None,
            "s2_b2": None,
            "s2_b3": None,
            "s2_b4": None,
            "s2_b8": None,
            "s2_b11": None,
            "s2_b12": None,
            "elev": None,
            "slope": None,
            "aspect": None,
        }

        # ------- MODIS LST -------
        try:
            lst = (
                ee.ImageCollection(self.MODIS_LST)
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .select("LST_Day_1km")
            )
            if lst.size().getInfo() > 0:
                val = lst.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=1000,
                    bestEffort=True
                ).get("LST_Day_1km").getInfo()
                if val is not None:
                    results["LST_modis"] = float(val) * 0.02
        except Exception:
            pass

        # ------- MODIS NDVI -------
        try:
            ndvi = (
                ee.ImageCollection(self.MODIS_NDVI)
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .select("NDVI")
            )
            if ndvi.size().getInfo() > 0:
                val = ndvi.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=250,
                    bestEffort=True
                ).get("NDVI").getInfo()
                if val is not None:
                    results["NDVI_modis"] = float(val) * 0.0001
        except Exception:
            pass

        # ------- GPM Rain -------
        # try:
        #     rain = (
        #         ee.ImageCollection(self.GPM_RAIN)
        #         .filterBounds(buffer)
        #         .filterDate(padded_start, padded_end)
        #         .select("precipitation")
        #     )
        #     if rain.size().getInfo() > 0:
        #         val = rain.mean().reduceRegion(
        #             reducer=ee.Reducer.mean(),
        #             geometry=buffer,
        #             scale=10000,
        #             bestEffort=True
        #         ).get("precipitation").getInfo()
        #         if val is not None:
        #             results["Rain_sat"] = float(val)
        # except Exception:
        #     pass

        # ------- Sentinel-1 SAR -------
        try:
            s1 = (
                ee.ImageCollection(self.S1_GRD)
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .select(["VV", "VH"])
            )
            if s1.size().getInfo() > 0:
                stats = s1.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=30,
                    bestEffort=True
                )
                vv_db = stats.get("VV").getInfo()
                vh_db = stats.get("VH").getInfo()
                if vv_db is not None:
                    results["s1_vv_dB"] = float(vv_db)
                    results["s1_vv"] = float(10 ** (vv_db / 10))
                if vh_db is not None:
                    results["s1_vh_dB"] = float(vh_db)
                    results["s1_vh"] = float(10 ** (vh_db / 10))
        except Exception:
            pass

        # ------- Sentinel-2 Reflectance -------
        try:
            s2 = (
                ee.ImageCollection(self.S2_L2A)
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                .select(["B2","B3","B4","B8","B11","B12"])
            )
            if s2.size().getInfo() > 0:
                stats = s2.mean().divide(10000).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=20,
                    bestEffort=True
                )
                for band in ["B2","B3","B4","B8","B11","B12"]:
                    val = stats.get(band).getInfo()
                    if val is not None:
                        results[f"s2_{band.lower()}"] = float(val)
        except Exception:
            pass

        # ------- DEM Elevation, Slope, Aspect -------
        try:
            dem = ee.Image(self.SRTM_DEM).clip(buffer)

            elev = dem.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=30,
                bestEffort=True
            ).get("elevation").getInfo()
            if elev is not None:
                results["elev"] = float(elev)

            terrain = ee.Terrain.products(dem)
            terr_stats = terrain.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=30,
                bestEffort=True
            )
            slope = terr_stats.get("slope").getInfo()
            aspect = terr_stats.get("aspect").getInfo()

            if slope is not None:
                results["slope"] = float(slope)
            if aspect is not None:
                results["aspect"] = float(aspect)
        except Exception:
            pass

        return results

    def run(self, df):
        if df is None or df.empty:
            self.logger.warning("No data received in SatellitePipe.")
            return df

        cache = {}
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                cache = json.load(f)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        grouped = df.groupby(df["date"].dt.to_period("W"))

        futures = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            for period, group in grouped:
                start = group["date"].min().strftime("%Y-%m-%d")
                end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                date_key = f"{start}_{end}"
                if date_key not in cache:
                    lat = group["latitude"].median()
                    lon = group["longitude"].median()
                    futures[ex.submit(self.fetch_satellite_batch, lat, lon, start, end)] = date_key

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Satellite ({self.station_name})"):
                date_key = futures[future]
                try:
                    cache[date_key] = future.result()
                except Exception:
                    cache[date_key] = {}

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        sat = []
        for period, group in grouped:
            start = group["date"].min()
            end = group["date"].max()
            date_key = f"{start.strftime('%Y-%m-%d')}_{(end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
            res = cache.get(date_key, {})
            mid_date = start + (end - start) / 2
            sat.append({"date": mid_date, **res})

        sat_df = pd.DataFrame(sat)
        sat_df["date"] = pd.to_datetime(sat_df["date"])

        merged = pd.merge(df, sat_df, on="date", how="left")

        # hard guard: Rain_sat is deprecated and should not exist anymore
        if "Rain_sat" in merged.columns:
            merged = merged.drop(columns=["Rain_sat"])

        self.logger.info(f"[{self.station_name}] SatellitePipe complete — {len(merged)} rows")

        return merged
