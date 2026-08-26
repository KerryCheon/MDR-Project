# ece_pipe.py
# In-situ ECE Soil Moisture Sensor Pipe
# Parses raw ECE CSV files and aggregates measurements to daily 24-hour windows.

from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.config import load_config
from ..utils.logger import get_logger


class ECEPipe:
    """
    Parses in-situ ECE sensor CSV files and aggregates measurements to 24-hour daily means.
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        parse_cfg = self.config

        self.in_dir = Path(parse_cfg.get("in_dir", "src/pipeline/data/raw/_ECE"))
        self.raw_file = parse_cfg.get("raw_file")
        self.station_name = parse_cfg.get("station", "unknown_station")
        self.latitude = parse_cfg.get("latitude")
        self.longitude = parse_cfg.get("longitude")
        self.elevation = parse_cfg.get("elevation")
        self.device_pattern = parse_cfg.get("device_pattern")

        self.logger = get_logger().getChild(f"ece.{self.station_name}")

    def _find_target_file(self) -> Path | None:
        """Locates the raw CSV file based on raw_file or device_pattern or station_name."""
        if self.raw_file:
            p = Path(self.raw_file)
            if p.exists():
                return p
            p_rel = Path("src/pipeline") / self.raw_file
            if p_rel.exists():
                return p_rel

        # Search in in_dir
        candidates = list(self.in_dir.glob("*.csv"))
        if self.device_pattern:
            for c in candidates:
                if self.device_pattern.lower() in c.name.lower():
                    return c

        for c in candidates:
            if self.station_name.lower() in c.name.lower():
                return c

        return None

    def parse_header_metadata(self, filepath: Path) -> dict:
        """Extracts Device info, DevEUI, Coordinates, and Soil Type from file header."""
        meta = {}
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(6)]

        header_str = "".join(lines[:4])

        dev_match = re.search(r"Device\s+(\d+)\s*-\s*([^\r\n,\"]+)", header_str)
        if dev_match:
            meta["device_id"] = int(dev_match.group(1))
            meta["location"] = dev_match.group(2).strip()

        deveui_match = re.search(r"DevEUI:\s*([a-fA-F0-9]+)", header_str)
        if deveui_match:
            meta["deveui"] = deveui_match.group(1).strip()

        coord_match = re.search(r"Coordinates:\s*\(([-\d.]+),\s*([-\d.]+)\)", header_str)
        if coord_match:
            meta["latitude"] = float(coord_match.group(1))
            meta["longitude"] = float(coord_match.group(2))

        soil_match = re.search(r"Soil Type:\s*([^\r\n,\"]+)", header_str)
        if soil_match:
            meta["soil_type"] = soil_match.group(1).strip()

        return meta

    def run(self, _=None) -> pd.DataFrame:
        filepath = self._find_target_file()
        if filepath is None or not filepath.exists():
            self.logger.error(f"[{self.station_name}] Could not find raw ECE file in {self.in_dir}")
            return pd.DataFrame()

        self.logger.info(f"[{self.station_name}] Parsing ECE raw file: {filepath.name}")
        header_meta = self.parse_header_metadata(filepath)

        lat = self.latitude if self.latitude is not None else header_meta.get("latitude")
        lon = self.longitude if self.longitude is not None else header_meta.get("longitude")

        # Read CSV skipping multiline header
        df_raw = pd.read_csv(filepath, skiprows=1)

        # Standardize column lookup
        col_map = {c.strip(): c for c in df_raw.columns}
        time_col = col_map.get("Timestamp (Seattle Time)") or col_map.get("Timestamp (UTC)")
        sm_col = col_map.get("Soil Moisture (%)")

        if not time_col or not sm_col:
            self.logger.error(f"[{self.station_name}] Missing timestamp or soil moisture columns in {filepath.name}")
            return pd.DataFrame()

        df = df_raw.dropna(subset=[time_col, sm_col]).copy()
        df["parsed_time"] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=["parsed_time"]).copy()

        df["date"] = df["parsed_time"].dt.floor("D")

        # Exclude partial start day (2026-07-19) and partial end day (2026-08-20)
        min_date = df["date"].min()
        max_date = df["date"].max()
        self.logger.info(
            f"[{self.station_name}] Raw recording span: {min_date.date()} to {max_date.date()} "
            f"({len(df)} total sub-minute samples)"
        )

        df_valid = df[(df["date"] > min_date) & (df["date"] < max_date)].copy()

        # 24-hour window aggregation (mean of Soil Moisture (%))
        daily = df_valid.groupby("date")[sm_col].agg(["count", "mean"]).reset_index()
        daily = daily.rename(columns={"count": "sample_count", "mean": "soil_moisture_pct"})

        # Convert percentage to volumetric water content fraction (m3/m3)
        daily["soil_moisture_5cm"] = daily["soil_moisture_pct"] / 100.0

        daily["station_id"] = self.station_name
        daily["latitude"] = float(lat) if lat is not None else np.nan
        daily["longitude"] = float(lon) if lon is not None else np.nan

        if self.elevation is not None:
            daily["elevation"] = float(self.elevation)

        daily["date"] = pd.to_datetime(daily["date"])

        # Sort chronologically
        daily = daily.sort_values("date").reset_index(drop=True)

        self.logger.info(
            f"[{self.station_name}] 24-hour aggregation complete: {len(daily)} valid days "
            f"from {daily['date'].min().date()} to {daily['date'].max().date()}"
        )
        return daily
