# Jakob Balkovec & Kerry Cheon
# Clean Pipe

# This module defines the CleanPipe class, which is responsible for cleaning
# and normalizing the parsed data before it is saved.

import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.config import load_config

class CleanPipe:
    # Multiple sentinel values used by USCRN
    SENTINEL_VALUES = [-9999.0, -99.0, -999.0]

    def __init__(self, config=None):
        # pre: config is a dictionary loaded from config.yaml or None
        # post: initializes CleanPipe with cleaning parameters
        # desc: sets up cleaning options like dropping missing values,
        #       filling NaNs, and selecting specific columns

        self.config = config or load_config()
        clean_cfg = self.config["clean"]

        self.drop_missing = clean_cfg.get("drop_missing", False)
        self.fillna_value = clean_cfg.get("fillna_value", None)
        self.keep_columns = clean_cfg.get("keep_columns", [])
        self.logger = get_logger().getChild("clean")

    def run(self, df):
        if df is None or df.empty:
            self.logger.warning("Received empty DataFrame in CleanPipe.")
            return pd.DataFrame()

        initial_rows = len(df)
        self.logger.info(f"Cleaning DataFrame with {initial_rows} rows.")

        # Replace ALL sentinel values with NaN
        df = df.replace(self.SENTINEL_VALUES, np.nan)

        # Validate coordinates (keep only valid ranges)
        if "longitude" in df.columns and "latitude" in df.columns:
            invalid_coords = (
                df["longitude"].isna()
                | df["latitude"].isna()
                | ~df["longitude"].between(-180, 180)
                | ~df["latitude"].between(-90, 90)
            )

            invalid_count = invalid_coords.sum()
            if invalid_count > 0:
                df = df[~invalid_coords]
                self.logger.info(f"Removed {invalid_count} rows with invalid coordinates.")
            else:
                self.logger.info("All coordinates are valid.")
        else:
            self.logger.warning("No longitude/latitude columns found — skipping coordinate validation.")

        # Count NaNs but don't drop them (preserve data)
        nan_count = df.isna().sum().sum()
        nan_pct = (nan_count / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
        self.logger.info(f"DataFrame contains {nan_count} NaN values ({nan_pct:.1f}% of data) — no dropping applied.")

        # Optional: Keep only specified columns
        if self.keep_columns:
            available_cols = [col for col in self.keep_columns if col in df.columns]
            missing_cols = [col for col in self.keep_columns if col not in df.columns]

            if missing_cols:
                self.logger.warning(f"Requested columns not found: {missing_cols}")

            df = df[available_cols]
            self.logger.info(f"Kept {len(available_cols)} specified columns.")

        # Ensure date column is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            invalid_dates = df["date"].isna().sum()
            if invalid_dates > 0:
                self.logger.warning(f"Found {invalid_dates} invalid dates (set to NaT).")

        final_rows = len(df)
        self.logger.info(f"CleanPipe complete — {final_rows} rows after cleaning ({initial_rows - final_rows} removed).")

        return df
