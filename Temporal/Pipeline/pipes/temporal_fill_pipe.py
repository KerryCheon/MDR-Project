# Jakob Balkovec
# Temporal Fill Pipe

# This module defines the TemporalFillPipe class, which fills temporal gaps in the data
# for each station by interpolating missing timestamps.

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.config import load_config
from utils.impute_models import run_xgboost

# cuz I just couldn't be bothered to fix all the FutureWarnings right now
import warnings
warnings.filterwarnings(
    "ignore",
    message="Series.interpolate with object dtype is deprecated",
    category=FutureWarning
)
# cuz I just couldn't be bothered to fix all the FutureWarnings right now


class TemporalFillPipe:
    def __init__(self, config=None, station_name=None):
        self.config = config or load_config()
        self.station_name = station_name or "global"
        fill_cfg = self.config

        self.target_columns = fill_cfg.get("target_columns", [])
        self.max_gap_days = fill_cfg.get("max_gap_days", 5)
        self.switch_gap = fill_cfg.get("switch_gap", 4)
        self.regression_window = fill_cfg.get("regression_window", 7)

        self.logger = get_logger().getChild(f"temporal_fill.{self.station_name}")

    def run(self, df):
        if df is None or df.empty:
            self.logger.warning(f"[{self.station_name}] Received empty DataFrame in TemporalFillPipe.")
            return pd.DataFrame()

        if "date" not in df.columns:
            self.logger.error(f"[{self.station_name}] DataFrame missing required column: 'date'.")
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        satellite_cols = [c for c in ["LST", "NDVI", "Rain_sat"] if c in df.columns]
        if not satellite_cols:
            self.logger.info(f"[{self.station_name}] No satellite columns found for XGBoost imputation.")
            return df

        self.logger.info(
            f"[{self.station_name}] Running XGBoost imputation for satellite data: {', '.join(satellite_cols)}"
        )

        for col in satellite_cols:
            try:
                work = df[["date", col]].drop_duplicates(subset="date").sort_values("date")
                work = work.dropna(subset=["date"])

                n_known = work[col].notna().sum()
                if n_known < 2:
                    self.logger.warning(f"[{self.station_name}] Skipping {col}: insufficient known values ({n_known}).")
                    continue

                imputed = run_xgboost(work.copy(), col)

                # Merge predictions back by date
                merged = df.merge(
                    imputed[["date", col + "_interp"]],
                    on="date",
                    how="left"
                )

                # Replace original values with XGBoost predictions where missing
                df[col] = merged[col].combine_first(merged[col + "_interp"])
                coverage = df[col].notna().mean()
                self.logger.info(f"[{self.station_name}] {col}: XGBoost imputed coverage = {coverage:.2%}")

            except Exception as e:
                self.logger.warning(f"[{self.station_name}] XGBoost imputation failed for {col}: {e}")

        self.logger.info(f"[{self.station_name}] TemporalFillPipe complete — {len(df)} rows processed.")
        return df
