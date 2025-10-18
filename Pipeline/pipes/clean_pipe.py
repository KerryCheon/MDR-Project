# Jakob Balkovec
# Clean Pipe

# This module defines the CleanPipe class, which is responsible for cleaning
# and normalizing the parsed data before it is saved.

import pandas as pd
from utils.logger import get_logger
from utils.config import load_config

class CleanPipe:
    NA_VALUE = -9999.0

    def __init__(self, config=None):
        # pre: config is a dictionary loaded from config.yaml or None
        # post: initializes CleanPipe with cleaning parameters
        # desc: sets up cleaning options like dropping missing values,
        #       filling NaNs, and selecting specific columns

        self.config = config or load_config()
        clean_cfg = self.config["pipeline"]["clean"]

        self.drop_missing = clean_cfg.get("drop_missing", True)
        self.fillna_value = clean_cfg.get("fillna_value", None)
        self.keep_columns = clean_cfg.get("keep_columns", [])
        self.logger = get_logger().getChild("clean")

    def run(self, df):
        # pre: parsed DataFrame passed in from previous pipe
        # post: returns cleaned DataFrame with consistent types and normalized coordinates
        # desc: performs cleaning operations such as replacing NA values,
        #       validating coordinates, and handling missing data

        if df is None or df.empty:
            self.logger.warning("Received empty DataFrame in CleanPipe.")
            return pd.DataFrame()

        self.logger.info(f"Cleaning DataFrame with {len(df)} rows.")

        df = df.replace(self.NA_VALUE, pd.NA)

        # validate coordinates (keep only valid ranges)
        if "longitude" in df.columns and "latitude" in df.columns:
            before = len(df)
            df = df[
                df["longitude"].between(-180, 180) &
                df["latitude"].between(-90, 90)
            ]
            after = len(df)
            self.logger.info(f"Removed {before - after} rows with invalid coordinates.")
        else:
            self.logger.warning("No longitude/latitude columns found — skipping coordinate validation.")

        nan_count = df.isna().sum().sum()
        self.logger.info(f"DataFrame contains {nan_count} NaN values (no dropping applied).")

        if self.keep_columns:
            df = df[self.keep_columns]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        self.logger.info(f"CleanPipe complete — {len(df)} rows after cleaning.")
        return df
