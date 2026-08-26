# Jakob Balkovec
# Nov 16th 2025
# smoothing.py

import pandas as pd
from pipeline.imputers.base import BaseImputer

class RollingMeanImputer(BaseImputer):
    # desc: Rolling mean for smoothing and medium-size gaps.

    def __init__(self, **kwargs):
        super().__init__("rolling_mean", **kwargs)
        cfg = self.imputer_cfg.get("rolling_mean", {})
        window = cfg.get("window", 7)
        self.window = int(window)
        self.logger.debug(f"initialized with window={self.window}")

    def impute(self, dates, values, aux_df=None):
        self.logger.debug(f"starting rolling mean for feature '{values.name}' with {values.isna().sum()} missing")

        s = pd.Series(values.values, index=pd.to_datetime(dates))
        base = s.ffill().bfill()
        rolled = base.rolling(self.window, min_periods=1, center=True).mean()

        self.logger.debug("computed rolling mean values")

        counts = s.rolling(self.window, min_periods=1, center=True).count()
        conf = (counts / float(self.window)).clip(0.0, 1.0)
        conf[~s.isna()] = 1.0

        self.logger.debug("rolling mean imputation complete")
        return rolled, conf
