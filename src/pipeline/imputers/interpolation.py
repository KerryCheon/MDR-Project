# Jakob Balkovec
# Nov 16th 2025
# interpolation.py

import pandas as pd
import numpy as np
from pipeline.Pipeline.imputers.base import BaseImputer

class LinearInterpolationImputer(BaseImputer):
    # desc: time-based linear interpolation for short gaps.

    def __init__(self, **kwargs):
        super().__init__("linear_interp", **kwargs)
        cfg = self.imputer_cfg.get("linear_interp", {})
        tau = cfg.get("tau_days", 7.0)
        self.tau_days = float(tau)
        self.logger.debug(f"initialized with tau_days={self.tau_days}")

    def impute(self, dates, values, aux_df=None):
        # pre:  expects date-aligned series
        # post: returns interpolated series with confidence

        self.logger.debug(f"starting interpolation for feature '{values.name}' with {values.isna().sum()} missing")

        s = pd.Series(values.values, index=pd.to_datetime(dates))
        s_interp = s.astype(float).interpolate(method="time")

        mask_missing = s.isna()
        self.logger.debug(f"missing mask computed: {mask_missing.sum()} missing points")

        valid_dates = pd.Series(s.index, index=s.index)
        valid_dates[mask_missing] = pd.NaT

        prev_valid = valid_dates.ffill()
        next_valid = valid_dates.bfill()

        gap_span = (next_valid - prev_valid).dt.days.astype(float)
        gap_span.replace(0, 0.1, inplace=True)

        self.logger.debug(f"computed gap spans (sample): {gap_span.head().tolist()}")

        conf = np.exp(-gap_span / self.tau_days)
        conf[~mask_missing] = 1.0

        self.logger.debug("Interpolation complete, generating confidence scores")

        return s_interp, conf.clip(0.0, 1.0)
