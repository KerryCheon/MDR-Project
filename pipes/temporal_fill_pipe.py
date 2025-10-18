# Jakob Balkovec
# Temporal Fill Pipe

# This modules defines the TemporalFillPipe class, which fills temporal gaps in the data
# for each station by interpolating missing timestamps.

import numpy as np
import pandas as pd
from utils.logger import get_logger
from utils.config import load_config

class TemporalFillPipe:
    def __init__(self, config=None):
        # pre: cleaned & time-aligned DataFrame with 'station_id' and 'date' columns
        # post: returns DataFrame with missing timestamps filled using hybrid strategy
        # desc: Fills small gaps via linear interpolation and large gaps via rolling regression.

        self.config = config or load_config()
        fill_cfg = self.config["pipeline"]["temporal_fill"]

        self.target_columns = fill_cfg.get("target_columns", ["soil_moisture_5cm"])
        self.max_gap_days = fill_cfg.get("max_gap_days", 5)
        self.switch_gap = fill_cfg.get("switch_gap", 4)  # threshold for switching to regression
        self.regression_window = fill_cfg.get("regression_window", 7)
        self.logger = get_logger().getChild("temporal_fill")

    def rolling_regression_fill(self, df, window=7):
        # pre: numeric time-series with missing values
        # post: fills large gaps (> switch_gap) using local regression
        # desc: Fits a short-term linear trend within a window to estimate missing values.

        df = df.sort_index()
        df = df.infer_objects(copy=False)  # ensures numeric dtype before interpolation

        for col in self.target_columns:
            mask = df[col].isna()
            for i in df.index[mask]:
                # look back a fixed-size window of valid values
                sub = df[col].loc[i - pd.Timedelta(days=window):i - pd.Timedelta(days=1)].dropna()
                if len(sub) >= 2:
                    try:
                        sub = sub.astype(float)
                        coeffs = np.polyfit(range(len(sub)), sub.values, deg=1)
                        df.loc[i, col] = coeffs[1] + coeffs[0] * len(sub)
                    except Exception as e:
                        self.logger.debug(f"Regression fill failed for {col} at {i}: {e}")
        return df

    def run(self, df):
        # pre: cleaned DataFrame grouped by station and sorted by date
        # post: returns fully interpolated DataFrame
        # desc: Performs linear interpolation for small gaps and regression for larger ones.

        if df is None or df.empty:
            self.logger.warning("Received empty DataFrame in TemporalFillPipe.")
            return pd.DataFrame()

        if not {"date", "station_id"}.issubset(df.columns):
            self.logger.error("DataFrame must contain 'date' and 'station_id' columns.")
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        all_filled = []
        for station, group in df.groupby("station_id"):
            group = group.sort_values("date").set_index("date")
            full_index = pd.date_range(group.index.min(), group.index.max(), freq="D")
            group = group.reindex(full_index)
            group["station_id"] = station
            group = group.infer_objects(copy=False)  # prevents FutureWarning from interpolate()

            for col in self.target_columns:
                if col not in group.columns:
                    continue

                # detect NaN groups (continuous gaps)
                is_na = group[col].isna()
                gap_groups = (is_na != is_na.shift()).cumsum()
                gaps = [
                    (g.index[0], g.index[-1], len(g))
                    for _, g in group[is_na].groupby(gap_groups[is_na])
                ]

                # fill gaps dynamically
                for start, end, length in gaps:
                    if length <= self.switch_gap:
                        # small gap = linear interpolate
                        try:
                            group.loc[start:end, col] = group[col].interpolate(
                                method="linear", limit=self.max_gap_days
                            ).loc[start:end]
                            self.logger.debug(f"{station}: linear fill ({length} days) for {col}.")
                        except Exception as e:
                            self.logger.debug(f"Linear fill failed for {col} ({length} days): {e}")
                    else:
                        # large gap = regression-based fill
                        group = self.rolling_regression_fill(group, window=self.regression_window)
                        self.logger.debug(f"{station}: regression fill ({length} days) for {col}.")
                    # FIXME: add max_gap_days enforcement if needed #1
                    # FIXME: use XGBoost or similar for more advanced regression if needed #2
                    # FIXME: 1 or 2 but not both

            all_filled.append(group.reset_index().rename(columns={"index": "date"}))

        filled_df = pd.concat(all_filled, ignore_index=True)
        self.logger.info(
            f"TemporalFillPipe complete — {len(filled_df)} rows (from {len(df)})."
        )
        return filled_df
