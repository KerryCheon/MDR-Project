# Jakob Balkovec
# weather_pipe.py

# This module replaces the GPM_RAIN section of the satellite pipe using the OpenMeteo API
# this gives daily frequency of data and omits the need for interpolation

# Jakob Balkovec & Kerry Cheon
# Weather Pipe (Open-Meteo Historical: rain + precipitation)

import pandas as pd
from pathlib import Path
import json
import requests
from ..utils.logger import get_logger
from ..utils.config import load_config


class WeatherPipe:
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, config=None, station_name=None):
        self.config = config or load_config()
        self.station_name = station_name or "global"
        self.logger = get_logger().getChild(f"weather.{self.station_name}")

        cache_template = self.config.get(
            "weather", {}
        ).get(
            "cache_path",
            "Temporal/Pipeline/data/cache/{station}_weather_cache.pkl"
        )
        self.cache_path = Path(cache_template.format(station=self.station_name))
        self.logger.info(f"Weather cache path set to: {self.cache_path}")

        weather_cfg = self.config.get("weather", {})
        self.timezone = weather_cfg.get("timezone", "auto")
        self.chunk_years = int(weather_cfg.get("chunk_years", 1))  # 1-year chunks is safe for 20y hourly pulls
        self.timeout = int(weather_cfg.get("timeout_seconds", 60))

        # What we want from Open-Meteo:
        # rain = liquid-only (mm)
        # precipitation = total precip (rain + snow etc) (mm)
        self.hourly_vars = weather_cfg.get("hourly_vars", ["rain", "precipitation"])

        # Column names in your dataset
        self.out_cols = {
            "rain": weather_cfg.get("rain_col", "rain_mm"),
            "precipitation": weather_cfg.get("precip_col", "precip_mm"),
        }

    def _fetch_hourly(self, lat, lon, start_date, end_date) -> pd.DataFrame:
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": self.timezone,
            "hourly": ",".join(self.hourly_vars),
        }

        r = requests.get(self.ARCHIVE_URL, params=params, timeout=self.timeout)
        data = r.json()

        if r.status_code != 200 or data.get("error"):
            raise RuntimeError(f"Open-Meteo error: {data.get('reason', r.status_code)}")

        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        return df

    def _date_chunks(self, start: pd.Timestamp, end: pd.Timestamp):
        start = pd.to_datetime(start).normalize()
        end = pd.to_datetime(end).normalize()

        cur = start
        while cur <= end:
            chunk_end = (cur + pd.DateOffset(years=self.chunk_years) - pd.Timedelta(days=1)).normalize()
            if chunk_end > end:
                chunk_end = end
            yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
            cur = (chunk_end + pd.Timedelta(days=1)).normalize()

    def _load_cache(self):
        if not self.cache_path.exists():
            return {}
        try:
            return pd.read_pickle(self.cache_path)
        except Exception:
            # if cache got corrupted or format changed, start fresh
            return {}

    def _save_cache(self, cache: dict):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(cache, self.cache_path)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            self.logger.warning("No data received in WeatherPipe.")
            return df

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()

        lat = float(out["latitude"].median())
        lon = float(out["longitude"].median())

        start = out["date"].min()
        end = out["date"].max()

        cache_key = json.dumps({
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "tz": self.timezone,
            "hourly": self.hourly_vars,
        }, sort_keys=True)

        cache = self._load_cache()

        if cache_key in cache:
            self.logger.info("Weather cache hit.")
            daily_df = cache[cache_key]
        else:
            self.logger.info(f"Fetching Open-Meteo hourly weather in {self.chunk_years}y chunks: "
                             f"{start.strftime('%Y-%m-%d')} -- {end.strftime('%Y-%m-%d')}")

            frames = []
            for s, e in self._date_chunks(start, end):
                try:
                    hourly = self._fetch_hourly(lat, lon, s, e)
                    if not hourly.empty:
                        frames.append(hourly)
                except Exception as ex:
                    self.logger.warning(f"Open-Meteo chunk failed ({s}→{e}): {ex}")

            if frames:
                hourly_all = pd.concat(frames).sort_index()
                hourly_all = hourly_all[~hourly_all.index.duplicated(keep="last")]
            else:
                hourly_all = pd.DataFrame()

            # aggregate hourly -> daily sums (mm/day)
            if hourly_all.empty:
                daily_df = pd.DataFrame(columns=["date", self.out_cols["rain"], self.out_cols["precipitation"]])
            else:
                rename_map = {}
                if "rain" in hourly_all.columns:
                    rename_map["rain"] = self.out_cols["rain"]
                if "precipitation" in hourly_all.columns:
                    rename_map["precipitation"] = self.out_cols["precipitation"]

                hourly_all = hourly_all.rename(columns=rename_map)

                agg_cols = [c for c in [self.out_cols["rain"], self.out_cols["precipitation"]] if c in hourly_all.columns]

                daily = hourly_all[agg_cols].resample("D").sum()
                daily_df = daily.reset_index().rename(columns={"time": "date", "index": "date"})
                daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()

            cache[cache_key] = daily_df
            self._save_cache(cache)

        merged = pd.merge(out, daily_df, on="date", how="left")

        if "Rain_sat" in merged.columns:
            merged = merged.drop(columns=["Rain_sat"])

        self.logger.info(f"[{self.station_name}] WeatherPipe complete — {len(merged)} rows")
        return merged


