# Jakob Balkovec
# Temporal Fill Pipe (Voting Ensemble Extended)

import numpy as np
import pandas as pd
import warnings

from ..utils.logger import get_logger
from ..utils.config import load_config
from ..imputers.api import transform_with_ensemble

# cuz I just cannot be bothered right now
warnings.filterwarnings(
    "ignore",
    message="Series.interpolate with object dtype is deprecated",
    category=FutureWarning
)
# cuz I just cannot be bothered right now

class TemporalFillPipe:
    def __init__(self, config=None, station_name=None):
        self.config = config or load_config()
        self.station_name = station_name or "global"

        # Features allowed to interpolate (fast or slow, but smooth enough)
        self.interpolate_cols = self.config.get("interpolate_cols", [
            "LST",
            "NDVI",
            "s1_vv",
            "s1_vh",
            "s2_b2",
            "s2_b3",
            "s2_b4",
            "s2_b8",
            "s2_b11",
            "s2_b12"
        ])

        # Features that are STATIC (no interpolation, no change)
        self.static_cols = [
            "elev",
            "slope",
            "aspect"
        ]

        # Features that SHOULD NOT be interpolated
        self.no_interpolate_cols = [
            "Rain_sat"   # fast-changing weather variable -> use masking, no guessing
        ]

        self.min_known_required = self.config.get("min_known_required", 20)
        self.logger = get_logger().getChild(f"temporal_fill.{self.station_name}")

    def _interpolate_single_col(self, df, col):
        try:
            work = df[["date", col]].drop_duplicates(subset="date").sort_values("date")

            known_count = work[col].notna().sum()
            if known_count < self.min_known_required:
                self.logger.info(f"[{self.station_name}] Skipping {col}: too few known points ({known_count})")
                df[f"{col}_mask"] = df[col].notna().astype(int)
                return df

            self.logger.info(f"[{self.station_name}] Interpolating {col} using ensemble voter")
            filled_df, _ = transform_with_ensemble(work.copy(), col=col, return_diag=True)

            df = df.merge(filled_df[["date", f"{col}_interp"]], on="date", how="left")
            df[f"{col}_mask"] = df[col].notna().astype(int)
            df[col] = df[col].combine_first(df[f"{col}_interp"])

            coverage = df[col].notna().mean()
            self.logger.info(f"[{self.station_name}] {col} coverage: {coverage:.2%}")

        except Exception as e:
            self.logger.warning(f"[{self.station_name}] FAILED {col}: {e}")
            df[f"{col}_mask"] = df[col].notna().astype(int)

        return df

    def run(self, df):
        if df is None or df.empty:
            self.logger.warning("Empty DF received into TemporalFillPipe.")
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        # Ensure masks exist even if interpolation fails
        for col in self.no_interpolate_cols + self.static_cols:
            if col in df.columns:
                df[f"{col}_mask"] = df[col].notna().astype(int)

        self.logger.info(
            f"[{self.station_name}] Running temporal fill on "
            f"{len(self.interpolate_cols)} columns"
        )

        for col in self.interpolate_cols:
            if col not in df.columns:
                self.logger.warning(f"[{self.station_name}] {col} missing in DF -> skip")
                continue

            df = self._interpolate_single_col(df, col)

        # DEM values should be stable, so forward/backfill if missing
        for col in self.static_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        self.logger.info(f"[{self.station_name}] TemporalFillPipe done -> {len(df)} rows")
        return df
