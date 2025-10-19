# Jakob Balkovec
# Feature Engineering Pipe

# This module defines the FeaturePipe class, which adds derived temporal features
# and placeholders (FOR-NOW) for satellite-derived features to the cleaned dataset.

import pandas as pd
from utils.logger import get_logger
from utils.config import load_config

class FeaturePipe:
    def __init__(self, config=None):
        # pre:  config is a dict or None
        # post: initializes FeaturePipe instance
        # desc: Adds derived temporal and environmental features to the dataset.

        self.config = config or load_config()
        self.logger = get_logger().getChild("feature")

    def run(self, df):
        # pre:  cleaned dataframe with date, precipitation, soil moisture columns
        # post: dataframe with new derived features added
        # desc: Computes rolling precipitation, DOY, lagged soil moisture, etc.

        if df.empty:
            self.logger.warning("FeaturePipe received empty DataFrame.")
            return df

        df = df.sort_values("date").reset_index(drop=True)

        # DOY
        df["DOY"] = df["date"].dt.dayofyear

        # rolling 3-day precipitation sum
        if "precipitation" in df.columns:
            df["Rain_3d"] = (
                pd.to_numeric(df["precipitation"], errors="coerce")  # ensure numeric
                .fillna(0)                                           # safe fill
                .rolling(window=3, min_periods=1)
                .sum()
            )

        # previous-day soil moisture
        if "soil_moisture_5cm" in df.columns:
            df["SM_prev"] = df["soil_moisture_5cm"].shift(1)

        # ECE
        df["SM_label"] = pd.NA

        self.logger.info(f"FeaturePipe complete — added derived temporal features.")
        return df
