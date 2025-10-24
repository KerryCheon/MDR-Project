# Jakob Balkovec & Kerry Cheon
# Feature Engineering Pipe

# This module defines the FeaturePipe class, which adds derived temporal features
# and placeholders (FOR-NOW) for satellite-derived features to the cleaned dataset.

import pandas as pd
from utils.logger import get_logger

class FeaturePipe:
    def __init__(self, config=None):
        # pre:  config is a dict or None
        # post: initializes FeaturePipe instance
        # desc: Adds derived temporal and environmental features to the dataset.

        self.config = config
        self.logger = get_logger().getChild("feature")

    def run(self, df):
        # pre:  cleaned dataframe with date, precipitation, soil moisture columns
        # post: dataframe with new derived features added
        # desc: Computes rolling precipitation, DOY, lagged soil moisture, etc.

        if df is None or df.empty:
            self.logger.warning("FeaturePipe received empty DataFrame.")
            return df

        df = df.sort_values("date").reset_index(drop=True)

        # Day of Year (DOY)
        if "date" in df.columns:
            df["DOY"] = df["date"].dt.dayofyear
        else:
            self.logger.warning("Missing 'date' column — cannot compute DOY.")

        # Rolling 3-day precipitation sum
        if "precipitation" in df.columns:
            df["Rain_3d"] = (
                pd.to_numeric(df["precipitation"], errors="coerce")
                .fillna(0)
                .rolling(window=3, min_periods=1)
                .sum()
            )

        # Previous-day soil moisture (lagged feature)
        if "soil_moisture_5cm" in df.columns:
            df["SM_prev"] = df["soil_moisture_5cm"].shift(1)

        # Placeholder for classification label (if needed later)
        df["SM_label"] = pd.NA

        self.logger.info("FeaturePipe complete — added derived temporal features.")
        return df
