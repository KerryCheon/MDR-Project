# Jakob Balkovec
# Whittaker Smoothing Pipe
#
# Applies Whittaker smoothing to selected time series columns
# using weights from temporal fill mask columns.

import pandas as pd
import numpy as np

from ..utils.logger import get_logger
from ..utils.config import load_config
from ..smoothing.whittaker import whittaker_smooth

class WhittakerPipe:
    def __init__(self, config=None, station_name=None):
        self.config = config or load_config()
        self.station_name = station_name or "global"
        ws_cfg = self.config.get("whittaker", {})

        # defaults if not defined
        self.target_columns = ws_cfg.get("target_columns", ["NDVI_modis"])
        self.lmbda = ws_cfg.get("lambda", 5000)

        self.logger = get_logger().getChild(f"whittaker.{self.station_name}")

    def run(self, df):
        if df is None or df.empty:
            self.logger.warning(f"[{self.station_name}] Empty DataFrame in WhittakerPipe")
            return df

        if "date" not in df.columns:
            self.logger.error(f"[{self.station_name}] No 'date' column found")
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        self.logger.info(
            f"[{self.station_name}] Whittaker smoothing on: "
            f"{', '.join(self.target_columns)} with lambda={self.lmbda}"
        )

        for col in self.target_columns:
            if col not in df.columns:
                self.logger.warning(f"[{self.station_name}] Skipping {col}: column missing")
                continue

            mask_col = f"{col}_mask"
            if mask_col not in df.columns:
                self.logger.warning(f"[{self.station_name}] No mask for {col}, constructing default")
                mask = df[col].notna().astype(int).values
            else:
                mask = df[mask_col].astype(int).values

            x = df[col].astype(float).values

            # run the smoother
            try:
                z = whittaker_smooth(x, lmbda=self.lmbda, mask=mask)
            except Exception as e:
                self.logger.error(f"[{self.station_name}] Smoothing failed for {col}: {e}")
                continue

            # only replace missing or previously-filled values (mask == 0)
            df[f"{col}_smooth"] = np.where(mask == 1, x, z)

        self.logger.info(f"[{self.station_name}] WhittakerPipe complete")
        return df
