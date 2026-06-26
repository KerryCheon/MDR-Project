# Jakob Balkovec
# Nov 16th 2025
# fbfill.py

import pandas as pd
import numpy as np
from pipeline.Pipeline.imputers.base import BaseImputer

class ForwardBackwardImputer(BaseImputer):
    # desc: Fills very short gaps and edges using ffill + bfill.

    def __init__(self, **kwargs):
        super().__init__("ffill_bfill", **kwargs)
        cfg = self.imputer_cfg.get("ffill_bfill", {})
        tau = cfg.get("tau_days", 3.0)
        self.tau_days = float(tau)
        self.logger.debug(f"initialized with tau_days={self.tau_days}")

    def impute(self, dates, values, aux_df=None):
        self.logger.debug(f"starting ffill/bfill for feature '{values.name}' with {values.isna().sum()} missing")

        s = pd.Series(values.values, index=pd.to_datetime(dates))
        filled = s.ffill().bfill()

        mask_missing = s.isna()
        self.logger.debug(f"missing mask has {mask_missing.sum()} entries")

        valid_dates = pd.Series(s.index, index=s.index)
        valid_dates[mask_missing] = pd.NaT

        prev_valid = valid_dates.ffill()
        next_valid = valid_dates.bfill()

        dist_prev = (s.index - prev_valid).dt.days.astype(float)
        dist_next = (next_valid - s.index).dt.days.astype(float)

        nearest = np.minimum(dist_prev, dist_next)
        nearest.replace(np.inf, np.nan, inplace=True)
        self.logger.debug("computed nearest distances for confidence weighting")

        conf = np.exp(-nearest / self.tau_days)
        conf[~mask_missing] = 1.0

        self.logger.debug("ffill/bfill imputation complete")
        return filled, conf.fillna(0.0).clip(0.0, 1.0)
