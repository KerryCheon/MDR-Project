# Jakob Balkovec & Kerry Cheon
# Feature Engineering Pipe

# This module defines the FeaturePipe class, which adds derived temporal features
# and placeholders (FOR-NOW) for satellite-derived features to the cleaned dataset.

import pandas as pd
from utils.logger import get_logger

from utils.math_utils import (
    compute_ndvi,
    compute_ndmi,
    compute_msi,
    compute_vv_vh_ratio,
    compute_backscatter_diff,
)

class FeaturePipe:
    def __init__(self, config=None, station_name=None):
        self.config = config or {}
        self.station_name = station_name or "global"
        self.logger = get_logger().getChild(f"feature.{self.station_name}")

    def run(self, df):
        if df is None or df.empty:
            self.logger.warning("FeaturePipe received empty DataFrame.")
            return df

        if "date" not in df.columns:
            self.logger.error("FeaturePipe missing date column.")
            return df

        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["DOY"] = df["date"].dt.dayofyear

        if "Rain_sat" in df.columns:
            df["Rain_3d"] = (
                pd.to_numeric(df["Rain_sat"], errors="coerce")
                .fillna(0)
                .rolling(window=3, min_periods=1)
                .sum()
            )

        if "soil_moisture_5cm" in df.columns:
            df["SM_prev"] = df["soil_moisture_5cm"].shift(1)

        if "s2_b4" in df.columns and "s2_b8" in df.columns:
            df["NDVI"] = compute_ndvi(df["s2_b4"], df["s2_b8"])

        if "s2_b8" in df.columns and "s2_b11" in df.columns:
            df["NDMI"] = compute_ndmi(df["s2_b8"], df["s2_b11"])

        if "s2_b8" in df.columns and "s2_b11" in df.columns:
            df["MSI"] = compute_msi(df["s2_b11"], df["s2_b8"])

        if "s1_vv" in df.columns and "s1_vh" in df.columns:
            df["SAR_ratio"] = compute_vv_vh_ratio(df["s1_vv"], df["s1_vh"])
            df["SAR_diff"] = compute_backscatter_diff(df["s1_vv"], df["s1_vh"])

        if "slope" in df.columns:
            df["slope_norm"] = df["slope"] / 90.0

        for col in ["NDVI", "NDMI", "MSI", "SAR_ratio", "SAR_diff"]:
            if col in df.columns:
                df[f"{col}_mask"] = df[col].notna().astype(int)

        self.logger.info(
            f"[{self.station_name}] FeaturePipe complete. "
            f"Rows: {len(df)}. Features added successfully."
        )

        return df
