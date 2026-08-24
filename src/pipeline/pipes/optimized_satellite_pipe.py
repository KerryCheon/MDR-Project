# Jakob Balkovec & Kerry Cheon & Antigravity
# optimized_satellite_pipe.py
# Optimized Satellite Pipe (Sentinel-1, Sentinel-2, MODIS, SMAP, SRTM DEM)
# Features:
# 1. Single-shot static terrain extraction with zero GEE calls on full cache hits
# 2. Server-side multi-sensor dictionary unification (1 RPC per week in single-week mode)
# 3. Temporal batching via ee.FeatureCollection server-side reduction (1 RPC per 26-week chunk)
# 4. Strict dynamic data validation before caching (skips all-None failed reductions)
# 5. Per-week median coordinate semantics matching baseline SatellitePipe
# 6. Automatic fallback to single-week unified fetching on batch failure
# 7. Incremental periodic cache saving
# 8. 100% cache format and schema compatibility with baseline SatellitePipe

import ee
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.logger import get_logger
from ..utils.config import load_config
from ..utils.gee import initialize_ee


class OptimizedSatellitePipe:
    MODIS_LST = "MODIS/061/MOD11A1"
    MODIS_NDVI = "MODIS/061/MOD13A3"

    S1_GRD = "COPERNICUS/S1_GRD"
    S2_L2A = "COPERNICUS/S2_SR_HARMONIZED"
    SRTM_DEM = "USGS/SRTMGL1_003"

    SMAP_005 = "NASA/SMAP/SPL3SMP_E/005"
    SMAP_006 = "NASA/SMAP/SPL3SMP_E/006"

    REQUIRED_SMAP_KEYS = {"SMAP_sm_am", "SMAP_sm_pm", "SMAP_qual_am", "SMAP_qual_pm"}
    TERRAIN_KEYS = {"elev", "slope", "aspect"}

    DEFAULT_RES = {
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

    def __init__(self, config=None, station_name=None):
        self.config = config or load_config()
        self.station_name = station_name or "global"
        self.logger = get_logger().getChild(f"satellite_v2.{self.station_name}")

        cache_template = self.config.get("satellite", {}).get(
            "cache_path",
            "src/pipeline/data/cache/{station}_satellite_cache.json"
        )
        self.cache_path = Path(cache_template.format(station=self.station_name))
        self.logger.info(f"Satellite cache path set to: {self.cache_path}")

        # Batch chunk size (capped between 1 and 52 weeks, default 26)
        sat_cfg = self.config.get("satellite", {})
        configured_chunk = int(sat_cfg.get("batch_chunk_size", 26))
        self.batch_chunk_size = max(1, min(52, configured_chunk))
        self.use_server_batching = bool(sat_cfg.get("use_server_batching", True))

        initialize_ee(self.logger)

    def fetch_static_terrain(self, lat: float, lon: float) -> dict:
        """Fetches elevation, slope, and aspect once for a station coordinate buffer."""
        terrain_res = {"elev": None, "slope": None, "aspect": None}
        try:
            point = ee.Geometry.Point([lon, lat])
            buffer = point.buffer(1000)

            dem = ee.Image(self.SRTM_DEM)
            terrain = ee.Terrain.products(dem)
            combined = dem.select(["elevation"]).addBands(terrain.select(["slope", "aspect"]))

            stats = combined.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=30,
                bestEffort=True
            ).getInfo() or {}

            if stats.get("elevation") is not None:
                terrain_res["elev"] = float(stats["elevation"])
            if stats.get("slope") is not None:
                terrain_res["slope"] = float(stats["slope"])
            if stats.get("aspect") is not None:
                terrain_res["aspect"] = float(stats["aspect"])

            self.logger.debug(f"[{self.station_name}] Fetched static terrain: {terrain_res}")
        except Exception as e:
            self.logger.warning(f"[{self.station_name}] Failed to fetch static terrain: {e}")

        return terrain_res

    def _extract_terrain_from_cache(self, cache: dict) -> dict:
        """Attempts to extract cached static terrain from existing cache entries."""
        terrain = {"elev": None, "slope": None, "aspect": None}
        for entry in cache.values():
            if isinstance(entry, dict):
                for k in self.TERRAIN_KEYS:
                    if terrain[k] is None and entry.get(k) is not None:
                        terrain[k] = float(entry[k])
                if all(terrain[k] is not None for k in self.TERRAIN_KEYS):
                    break
        return terrain

    def _parse_dynamic_props(self, props: dict) -> dict:
        """Helper to convert raw GEE reduction properties to scaled satellite features."""
        out = {k: None for k in self.DEFAULT_RES if k not in ("elev", "slope", "aspect")}

        # MODIS LST (Kelvin scale factor 0.02)
        lst_val = props.get("lst_val")
        if lst_val is not None:
            out["LST_modis"] = float(lst_val) * 0.02

        # MODIS NDVI (scale factor 0.0001)
        ndvi_val = props.get("ndvi_val")
        if ndvi_val is not None:
            out["NDVI_modis"] = float(ndvi_val) * 0.0001

        # Sentinel-1 SAR (dB and linear scale)
        vv_db = props.get("s1_vv")
        if vv_db is not None:
            out["s1_vv_dB"] = float(vv_db)
            out["s1_vv"] = float(10 ** (float(vv_db) / 10))

        vh_db = props.get("s1_vh")
        if vh_db is not None:
            out["s1_vh_dB"] = float(vh_db)
            out["s1_vh"] = float(10 ** (float(vh_db) / 10))

        # Sentinel-2 Bands
        for band in ["b2", "b3", "b4", "b8", "b11", "b12"]:
            val = props.get(f"s2_{band}")
            if val is not None:
                out[f"s2_{band}"] = float(val)

        # SMAP Soil Moisture and Quality Flags
        sm_am = props.get("smap_sm_am")
        if sm_am is not None:
            out["SMAP_sm_am"] = float(sm_am)

        sm_pm = props.get("smap_sm_pm")
        if sm_pm is not None:
            out["SMAP_sm_pm"] = float(sm_pm)

        q_am = props.get("smap_qual_am")
        if q_am is not None:
            out["SMAP_qual_am"] = float(q_am)

        q_pm = props.get("smap_qual_pm")
        if q_pm is not None:
            out["SMAP_qual_pm"] = float(q_pm)

        return out

    def fetch_single_week_unified(self, lat: float, lon: float, start_date: str, end_date: str) -> dict:
        """Unified server-side dictionary reduction for a single weekly date range (1 GEE RPC)."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)
        padded_start, padded_end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(1000)

        out = {k: None for k in self.DEFAULT_RES if k not in ("elev", "slope", "aspect")}

        try:
            # 1. MODIS LST
            lst_stats = (
                ee.ImageCollection(self.MODIS_LST)
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .select(["LST_Day_1km"])
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=1000, bestEffort=True)
            )

            # 2. MODIS NDVI
            ndvi_stats = (
                ee.ImageCollection(self.MODIS_NDVI)
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .select(["NDVI"])
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=250, bestEffort=True)
            )

            # 3. Sentinel-1 SAR
            s1_stats = (
                ee.ImageCollection(self.S1_GRD)
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .select(["VV", "VH"])
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=30, bestEffort=True)
            )

            # 4. Sentinel-2
            s2_stats = (
                ee.ImageCollection(self.S2_L2A)
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                .select(["B2", "B3", "B4", "B8", "B11", "B12"])
                .map(lambda img: img.divide(10000))
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=20, bestEffort=True)
            )

            # 5. SMAP
            smap_stats = (
                ee.ImageCollection(self.SMAP_005)
                .merge(ee.ImageCollection(self.SMAP_006))
                .filterBounds(buffer)
                .filterDate(padded_start, padded_end)
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=9000, bestEffort=True)
            )

            # Combine into a single server-side dictionary
            combined_dict = ee.Dictionary({
                "lst_val": lst_stats.get("LST_Day_1km"),
                "ndvi_val": ndvi_stats.get("NDVI"),
                "s1_vv": s1_stats.get("VV"),
                "s1_vh": s1_stats.get("VH"),
                "s2_b2": s2_stats.get("B2"),
                "s2_b3": s2_stats.get("B3"),
                "s2_b4": s2_stats.get("B4"),
                "s2_b8": s2_stats.get("B8"),
                "s2_b11": s2_stats.get("B11"),
                "s2_b12": s2_stats.get("B12"),
                "smap_sm_am": smap_stats.get("soil_moisture_am"),
                "smap_sm_pm": smap_stats.get("soil_moisture_pm"),
                "smap_qual_am": smap_stats.get("retrieval_qual_flag_am"),
                "smap_qual_pm": smap_stats.get("retrieval_qual_flag_pm"),
            })

            props = combined_dict.getInfo() or {}
            out = self._parse_dynamic_props(props)

        except Exception as e:
            self.logger.debug(f"[{self.station_name}] fetch_single_week_unified failed for {start_date}_{end_date}: {e}")

        return out

    def fetch_smap_only(self, lat: float, lon: float, start_date: str, end_date: str) -> dict:
        """Pulls only SMAP features if partial cache is present (1 GEE RPC)."""
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

            stats = smap.mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=9000,
                bestEffort=True
            ).getInfo() or {}

            if stats.get("soil_moisture_am") is not None:
                out["SMAP_sm_am"] = float(stats["soil_moisture_am"])
            if stats.get("soil_moisture_pm") is not None:
                out["SMAP_sm_pm"] = float(stats["soil_moisture_pm"])
            if stats.get("retrieval_qual_flag_am") is not None:
                out["SMAP_qual_am"] = float(stats["retrieval_qual_flag_am"])
            if stats.get("retrieval_qual_flag_pm") is not None:
                out["SMAP_qual_pm"] = float(stats["retrieval_qual_flag_pm"])

        except Exception as e:
            self.logger.debug(f"[{self.station_name}] fetch_smap_only failed for {start_date}_{end_date}: {e}")

        return out

    def fetch_satellite_batch_collection(
        self,
        week_items: list[tuple[str, str, str, float, float]]
    ) -> dict[str, dict]:
        """Fetches dynamic satellite data for a batch of weeks in a single GEE FeatureCollection reduction.

        Args:
            week_items: List of tuples (date_key, start_date, end_date, lat, lon)

        Returns:
            Dictionary mapping date_key -> feature dictionary
        """
        features = []
        for date_key, start_date, end_date, lat, lon in week_items:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)
            padded_start = start_dt.strftime("%Y-%m-%d")
            padded_end = end_dt.strftime("%Y-%m-%d")

            point = ee.Geometry.Point([lon, lat])
            buffer = point.buffer(1000)

            feat = ee.Feature(
                buffer,
                {
                    "date_key": date_key,
                    "start": padded_start,
                    "end": padded_end,
                }
            )
            features.append(feat)

        fc = ee.FeatureCollection(features)

        def extract_week_features(feat):
            s = feat.get("start")
            e = feat.get("end")
            geom = feat.geometry()

            # 1. MODIS LST
            lst = (
                ee.ImageCollection(self.MODIS_LST)
                .filterBounds(geom)
                .filterDate(s, e)
                .select(["LST_Day_1km"])
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=1000, bestEffort=True)
            )

            # 2. MODIS NDVI
            ndvi = (
                ee.ImageCollection(self.MODIS_NDVI)
                .filterBounds(geom)
                .filterDate(s, e)
                .select(["NDVI"])
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=250, bestEffort=True)
            )

            # 3. Sentinel-1 SAR
            s1 = (
                ee.ImageCollection(self.S1_GRD)
                .filterBounds(geom)
                .filterDate(s, e)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .select(["VV", "VH"])
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=30, bestEffort=True)
            )

            # 4. Sentinel-2
            s2 = (
                ee.ImageCollection(self.S2_L2A)
                .filterBounds(geom)
                .filterDate(s, e)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                .select(["B2", "B3", "B4", "B8", "B11", "B12"])
                .map(lambda img: img.divide(10000))
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=20, bestEffort=True)
            )

            # 5. SMAP
            smap = (
                ee.ImageCollection(self.SMAP_005)
                .merge(ee.ImageCollection(self.SMAP_006))
                .filterBounds(geom)
                .filterDate(s, e)
                .mean()
                .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=9000, bestEffort=True)
            )

            return feat.set({
                "lst_val": ee.Dictionary(lst).get("LST_Day_1km", None),
                "ndvi_val": ee.Dictionary(ndvi).get("NDVI", None),
                "s1_vv": ee.Dictionary(s1).get("VV", None),
                "s1_vh": ee.Dictionary(s1).get("VH", None),
                "s2_b2": ee.Dictionary(s2).get("B2", None),
                "s2_b3": ee.Dictionary(s2).get("B3", None),
                "s2_b4": ee.Dictionary(s2).get("B4", None),
                "s2_b8": ee.Dictionary(s2).get("B8", None),
                "s2_b11": ee.Dictionary(s2).get("B11", None),
                "s2_b12": ee.Dictionary(s2).get("B12", None),
                "smap_sm_am": ee.Dictionary(smap).get("soil_moisture_am", None),
                "smap_sm_pm": ee.Dictionary(smap).get("soil_moisture_pm", None),
                "smap_qual_am": ee.Dictionary(smap).get("retrieval_qual_flag_am", None),
                "smap_qual_pm": ee.Dictionary(smap).get("retrieval_qual_flag_pm", None),
            })

        reduced_fc = fc.map(extract_week_features)
        fc_info = reduced_fc.getInfo() or {}

        results = {}
        returned_keys = set()
        for feature in fc_info.get("features", []):
            props = feature.get("properties", {})
            date_key = props.get("date_key")
            if date_key:
                returned_keys.add(date_key)
                parsed_features = self._parse_dynamic_props(props)
                results[date_key] = parsed_features

        # Log any missing date keys
        expected_keys = {item[0] for item in week_items}
        missing_keys = expected_keys - returned_keys
        if missing_keys:
            self.logger.debug(f"[{self.station_name}] Batch omitted {len(missing_keys)} date keys: {missing_keys}")

        return results

    def _save_cache_file(self, cache: dict):
        """Helper to write cache file to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"[{self.station_name}] Failed to save cache: {e}")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the optimized satellite feature pipeline."""
        if df is None or df.empty:
            self.logger.warning("No data received in OptimizedSatellitePipe.")
            return df

        cache = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    cache = json.load(f)
            except Exception as e:
                self.logger.warning(f"[{self.station_name}] Failed to load cache: {e}. Starting fresh.")
                cache = {}

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["week"] = df["date"].dt.to_period("W-SUN").astype(str)
        grouped = df.groupby("week")

        # Step 1: Categorize needed fetches with per-week median coordinates
        full_needed: list[tuple[str, str, str, float, float]] = []
        smap_needed: list[tuple[str, str, str, float, float]] = []

        for week, group in grouped:
            start = group["date"].min().strftime("%Y-%m-%d")
            end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            date_key = f"{start}_{end}"
            lat = float(group["latitude"].median())
            lon = float(group["longitude"].median())

            if date_key not in cache:
                full_needed.append((date_key, start, end, lat, lon))
            else:
                entry = cache.get(date_key, {})
                if not self.REQUIRED_SMAP_KEYS.issubset(set(entry.keys())):
                    smap_needed.append((date_key, start, end, lat, lon))

        # Step 2: Static terrain extraction (gated to zero GEE calls on full cache hit)
        terrain_stats = self._extract_terrain_from_cache(cache)
        terrain_complete = all(terrain_stats.get(k) is not None for k in self.TERRAIN_KEYS)

        if not terrain_complete and (full_needed or smap_needed or len(cache) == 0):
            # Fetch from GEE using median station coordinates
            rep_lat = float(df["latitude"].median())
            rep_lon = float(df["longitude"].median())
            fetched_terrain = self.fetch_static_terrain(rep_lat, rep_lon)
            for k in self.TERRAIN_KEYS:
                if terrain_stats.get(k) is None:
                    terrain_stats[k] = fetched_terrain.get(k)
        elif terrain_complete:
            self.logger.debug(f"[{self.station_name}] Reusing cached static terrain: {terrain_stats}")

        # Step 3: Fetch full dynamic satellite features
        successful_fetches = 0
        if full_needed:
            self.logger.info(
                f"[{self.station_name}] Fetching full satellite features for {len(full_needed)} uncached weeks..."
            )

            if self.use_server_batching:
                chunk_size = self.batch_chunk_size
                chunks = [full_needed[i:i + chunk_size] for i in range(0, len(full_needed), chunk_size)]

                with tqdm(total=len(full_needed), desc=f"SatelliteV2 ({self.station_name})") as pbar:
                    for chunk in chunks:
                        try:
                            batch_results = self.fetch_satellite_batch_collection(chunk)
                            for date_key, new_data in batch_results.items():
                                is_valid = any(v is not None for v in new_data.values()) if new_data else False
                                if is_valid:
                                    cache[date_key] = {**terrain_stats, **new_data}
                                    successful_fetches += 1
                            pbar.update(len(chunk))
                        except Exception as e:
                            self.logger.warning(
                                f"[{self.station_name}] Batch collection failed ({e}). Falling back to unified weekly fetches."
                            )
                            with ThreadPoolExecutor(max_workers=4) as ex:
                                futures = {
                                    ex.submit(self.fetch_single_week_unified, lat, lon, s, end_d): dk
                                    for dk, s, end_d, lat, lon in chunk
                                }
                                for future in as_completed(futures):
                                    dk = futures[future]
                                    try:
                                        new_data = future.result() or {}
                                        is_valid = any(v is not None for v in new_data.values()) if new_data else False
                                        if is_valid:
                                            cache[dk] = {**terrain_stats, **new_data}
                                            successful_fetches += 1
                                    except Exception as week_e:
                                        self.logger.warning(f"[{self.station_name}] Unified fetch failed for {dk}: {week_e}")
                                    pbar.update(1)

                        # Periodic cache flush after each chunk
                        if successful_fetches > 0:
                            self._save_cache_file(cache)
            else:
                # Direct thread pool of unified weekly calls
                with ThreadPoolExecutor(max_workers=4) as ex:
                    futures = {
                        ex.submit(self.fetch_single_week_unified, lat, lon, s, end_d): dk
                        for dk, s, end_d, lat, lon in full_needed
                    }
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"SatelliteV2 ({self.station_name})"):
                        dk = futures[future]
                        try:
                            new_data = future.result() or {}
                            is_valid = any(v is not None for v in new_data.values()) if new_data else False
                            if is_valid:
                                cache[dk] = {**terrain_stats, **new_data}
                                successful_fetches += 1
                                if successful_fetches % 20 == 0:
                                    self._save_cache_file(cache)
                        except Exception as e:
                            self.logger.warning(f"[{self.station_name}] Unified fetch failed for {dk}: {e}")

        # Step 4: Fetch partial SMAP features where needed
        if smap_needed:
            self.logger.info(
                f"[{self.station_name}] Fetching partial SMAP for {len(smap_needed)} weeks..."
            )
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = {
                    ex.submit(self.fetch_smap_only, lat, lon, s, end_d): dk
                    for dk, s, end_d, lat, lon in smap_needed
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"SMAP ({self.station_name})"):
                    dk = futures[future]
                    try:
                        smap_data = future.result() or {}
                        is_valid = any(v is not None for v in smap_data.values()) if smap_data else False
                        if is_valid:
                            cache.setdefault(dk, {})
                            cache[dk].update(smap_data)
                    except Exception as e:
                        self.logger.warning(f"[{self.station_name}] SMAP fetch failed for {dk}: {e}")

        # Ensure static terrain is populated across all cache entries
        for date_key in cache:
            for k in self.TERRAIN_KEYS:
                if cache[date_key].get(k) is None and terrain_stats.get(k) is not None:
                    cache[date_key][k] = terrain_stats[k]

        # Step 5: Final save of updated cache
        self._save_cache_file(cache)

        # Step 6: Assemble satellite dataframe
        sat = []
        for week, group in grouped:
            start = group["date"].min()
            end = group["date"].max()
            date_key = f"{start.strftime('%Y-%m-%d')}_{(end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"

            res = cache.get(date_key, {})
            # Fill missing keys from default_res
            merged_entry = {**self.DEFAULT_RES, **terrain_stats, **res}
            sat.append({"week": str(week), **merged_entry})

        sat_df = pd.DataFrame(sat)
        if not sat_df.empty:
            sat_df["week"] = sat_df["week"].astype(str)

        merged = pd.merge(df, sat_df, on="week", how="left").drop(columns=["week"])

        if "Rain_sat" in merged.columns:
            merged = merged.drop(columns=["Rain_sat"])

        self.logger.info(f"[{self.station_name}] OptimizedSatellitePipe complete — {len(merged)} rows")
        return merged


# Alias for backward compatibility
SatellitePipeV2 = OptimizedSatellitePipe
