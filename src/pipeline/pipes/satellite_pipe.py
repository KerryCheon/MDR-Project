# Jakob Balkovec & Kerry Cheon
# satellite_pipe.py
# Satellite Pipe (Sentinel-1, Sentinel-2, MODIS, GPM, DEM)

import ee
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime, timedelta
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

    SMAP_005 = "NASA/SMAP/SPL3SMP_E/005"
    SMAP_006 = "NASA/SMAP/SPL3SMP_E/006"

    def __init__(self, config=None, station_name=None):
        self.config = config or load_config()
        self.station_name = station_name or "global"
        self.logger = get_logger().getChild(f"satellite.{self.station_name}")

        cache_template = self.config["satellite"].get(
            "cache_path",
            "src/pipeline/data/cache/{station}_satellite_cache.json"
        )
        self.cache_path = Path(cache_template.format(station=self.station_name))
        self.logger.info(f"Satellite cache path set to: {self.cache_path}")

        try:
            ee.Initialize(project=self.config["satellite"].get("gee_project_id", "mdr-project-475522"))
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=self.config["satellite"].get("gee_project_id", "mdr-project-475522"))

    def fetch_smap_only(self, lat, lon, start_date, end_date):
        # added due to performance bottleneck
        # if cache is valid, it pulls from cache. If partial values in objects, it pulls from cache + satellite
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)
        padded_start, padded_end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(1000)

        out = {
            "SMAP_sm_am": None,
            "SMAP_sm_pm": None,
            "SMAP_qual_am": None,
            "SMAP_qual_pm": None,
        }

        try:
            smap = (
                ee.ImageCollection(self.SMAP_005)
                .merge(ee.ImageCollection(self.SMAP_006))
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
            )

            if smap.size().getInfo() > 0:
                stats = smap.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=9000,
                    bestEffort=True
                )

                sm_am = stats.get("soil_moisture_am").getInfo()
                sm_pm = stats.get("soil_moisture_pm").getInfo()
                q_am = stats.get("retrieval_qual_flag_am").getInfo()
                q_pm = stats.get("retrieval_qual_flag_pm").getInfo()

                if sm_am is not None:
                    out["SMAP_sm_am"] = float(sm_am)
                if sm_pm is not None:
                    out["SMAP_sm_pm"] = float(sm_pm)
                if q_am is not None:
                    out["SMAP_qual_am"] = float(q_am)
                if q_pm is not None:
                    out["SMAP_qual_pm"] = float(q_pm)

        except Exception:
            pass

        return out

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
            "SMAP_sm_am": None,
            "SMAP_sm_pm": None,
            "SMAP_qual_am": None,
            "SMAP_qual_pm": None,
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

        # ------- SMAP Soil Moisture (9 km) -------
        try:
            smap = (
                ee.ImageCollection(self.SMAP_005)
                .merge(ee.ImageCollection(self.SMAP_006))
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
            )

            if smap.size().getInfo() > 0:
                stats = smap.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=9000,
                    bestEffort=True
                )

                sm_am = stats.get("soil_moisture_am").getInfo()
                sm_pm = stats.get("soil_moisture_pm").getInfo()
                q_am = stats.get("retrieval_qual_flag_am").getInfo()
                q_pm = stats.get("retrieval_qual_flag_pm").getInfo()

                if sm_am is not None:
                    results["SMAP_sm_am"] = float(sm_am)
                if sm_pm is not None:
                    results["SMAP_sm_pm"] = float(sm_pm)
                if q_am is not None:
                    results["SMAP_qual_am"] = float(q_am)
                if q_pm is not None:
                    results["SMAP_qual_pm"] = float(q_pm)

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
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df["week"] = df["date"].dt.to_period("W-SUN").astype(str)
        grouped = df.groupby("week")

        required_smap_keys = {"SMAP_sm_am", "SMAP_sm_pm", "SMAP_qual_am", "SMAP_qual_pm"}

        default_res = {
            "LST_modis": None,
            "NDVI_modis": None,
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
            "SMAP_sm_am": None,
            "SMAP_sm_pm": None,
            "SMAP_qual_am": None,
            "SMAP_qual_pm": None,
        }

        futures = {}

        with ThreadPoolExecutor(max_workers=4) as ex:
            for week, group in grouped:
                start = group["date"].min().strftime("%Y-%m-%d")
                end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                date_key = f"{start}_{end}"

                lat = float(group["latitude"].median())
                lon = float(group["longitude"].median())

                if date_key not in cache:
                    futures[ex.submit(self.fetch_satellite_batch, lat, lon, start, end)] = ("full", date_key)
                else:
                    entry = cache.get(date_key, {})
                    if not required_smap_keys.issubset(set(entry.keys())):
                        futures[ex.submit(self.fetch_smap_only, lat, lon, start, end)] = ("smap", date_key)

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Satellite ({self.station_name})"):
                kind, date_key = futures[future]
                try:
                    new_data = future.result() or {}
                    if kind == "full":
                        cache[date_key] = new_data
                    else:
                        cache.setdefault(date_key, {})
                        cache[date_key].update(new_data)
                except Exception:
                    cache.setdefault(date_key, {})

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        sat = []
        for week, group in grouped:
            start = group["date"].min()
            end = group["date"].max()
            date_key = f"{start.strftime('%Y-%m-%d')}_{(end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"

            res = cache.get(date_key, {})
            sat.append({"week": str(week), **{**default_res, **res}})

        sat_df = pd.DataFrame(sat)
        if not sat_df.empty:
            sat_df["week"] = sat_df["week"].astype(str)

        self.logger.debug(f"[{self.station_name}] sat_df columns: {list(sat_df.columns)}")

        merged = pd.merge(df, sat_df, on="week", how="left").drop(columns=["week"])

        if "Rain_sat" in merged.columns:
            merged = merged.drop(columns=["Rain_sat"])

        self.logger.info(f"[{self.station_name}] SatellitePipe complete — {len(merged)} rows")
        return merged
