# Jakob Balkovec
# Temporal Fill Pipe

# This module defines the TemporalFillPipe class, which fills temporal gaps in the data
# for each station by interpolating missing timestamps.

import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils.config import load_config

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
        # pre: cleaned & time-aligned DataFrame with 'station_id' and 'date' columns
        # post: returns DataFrame with missing timestamps filled using hybrid strategy
        # desc: Fills small gaps via linear interpolation and large gaps via rolling regression.

        self.config = config or load_config()
        self.station_name = station_name or "global"
        fill_cfg = self.config

        self.target_columns = fill_cfg.get("target_columns", [])
        self.max_gap_days = fill_cfg.get("max_gap_days", 5)
        self.switch_gap = fill_cfg.get("switch_gap", 4)
        self.regression_window = fill_cfg.get("regression_window", 7)

        # Station-specific logger
        self.logger = get_logger().getChild(f"temporal_fill.{self.station_name}")

    def rolling_regression_fill(self, df, window=7):
        # pre: numeric time-series with missing values
        # post: fills large gaps (> switch_gap) using local regression
        # desc: Fits a short-term linear trend within a window to estimate missing values.

        df = df.sort_index()
        df = df.infer_objects(copy=False)

        for col in self.target_columns:
            mask = df[col].isna()
            for i in df.index[mask]:
                sub = df[col].loc[i - pd.Timedelta(days=window):i - pd.Timedelta(days=1)].dropna()
                if len(sub) >= 2:
                    try:
                        sub = sub.astype(float)
                        coeffs = np.polyfit(range(len(sub)), sub.values, deg=1)
                        df.loc[i, col] = coeffs[1] + coeffs[0] * len(sub)
                    except Exception as e:
                        self.logger.debug(f"[{self.station_name}] Regression fill failed for {col} at {i}: {e}")
        return df

    def run(self, df):
        # pre: cleaned DataFrame grouped by station and sorted by date
        # post: returns fully interpolated DataFrame
        # desc: Performs linear interpolation for small gaps and regression for larger ones.

        if df is None or df.empty:
            self.logger.warning(f"[{self.station_name}] Received empty DataFrame in TemporalFillPipe.")
            return pd.DataFrame()

        if not {"date", "station_id"}.issubset(df.columns):
            self.logger.error(f"[{self.station_name}] DataFrame missing required columns: 'date', 'station_id'.")
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        all_filled = []
        for station, group in df.groupby("station_id"):
            group = group.sort_values("date").set_index("date")
            full_index = pd.date_range(group.index.min(), group.index.max(), freq="D")
            group = group.reindex(full_index)
            group["station_id"] = station
            group = group.infer_objects(copy=False)

            for col in self.target_columns:
                if col not in group.columns:
                    continue

                is_na = group[col].isna()
                gap_groups = (is_na != is_na.shift()).cumsum()
                gaps = [
                    (g.index[0], g.index[-1], len(g))
                    for _, g in group[is_na].groupby(gap_groups[is_na])
                ]

                for start, end, length in gaps:
                    if length <= self.switch_gap:
                        try:
                            group.loc[start:end, col] = group[col].interpolate(
                                method="linear", limit=self.max_gap_days
                            ).loc[start:end]
                            self.logger.debug(f"[{self.station_name}] {station}: linear fill ({length}d) for {col}.")
                        except Exception as e:
                            self.logger.debug(f"[{self.station_name}] Linear fill failed for {col} ({length}d): {e}")
                    else:
                        group = self.rolling_regression_fill(group, window=self.regression_window)
                        self.logger.debug(f"[{self.station_name}] {station}: regression fill ({length}d) for {col}.")

            all_filled.append(group.reset_index().rename(columns={"index": "date"}))

        filled_df = pd.concat(all_filled, ignore_index=True)
        self.logger.info(
            f"[{self.station_name}] TemporalFillPipe complete — {len(filled_df)} rows (from {len(df)})."
        )
        return filled_df
