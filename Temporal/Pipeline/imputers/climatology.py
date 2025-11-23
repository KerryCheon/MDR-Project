# Jakob Balkovec
# Nov 16th 2025
# fbfill.py

import pandas as pd
import numpy as np
from Temporal.Pipeline.imputers.base import BaseImputer

class ClimatologyImputer(BaseImputer):
    # desc: Seasonal DOY filling. Strong for long gaps.

    def __init__(self):
        super().__init__("climatology")
        cfg = self.imputer_cfg.get("climatology", {})
        m = cfg.get("min_count_for_high_conf", 10)
        self.min_count_for_high_conf = int(m)
        self.clim_mean = None
        self.clim_count = None
        self.logger.debug(f"initialized with min_count_for_high_conf={self.min_count_for_high_conf}")

    def fit(self, dates, values, aux_df=None):
        s = pd.Series(values.values, index=pd.to_datetime(dates))
        df = pd.DataFrame({"doy": s.index.dayofyear, "val": s.values})

        g = df.dropna(subset=["val"]).groupby("doy")["val"]
        self.clim_mean = g.mean()
        self.clim_count = g.count()

        self.logger.debug(
            f"fit complete using {len(df.dropna(subset=['val']))} valid samples; "
            f"{self.clim_count.sum()} total climatology counts"
        )

        return self

    def impute(self, dates, values, aux_df=None):
        self.logger.debug(f"starting climatology imputation for feature '{values.name}'")

        s = pd.Series(values.values, index=pd.to_datetime(dates))

        if self.clim_mean is None:
            self.logger.debug("climatology not fit; returning fallback confidences")
            conf = pd.Series(0.0, index=s.index)
            conf[~s.isna()] = 1.0
            return s, conf

        doy = s.index.dayofyear
        clim_vals = self.clim_mean.reindex(doy).values
        clim_counts = self.clim_count.reindex(doy).fillna(0.0).values

        out = s.copy()
        mask_missing = s.isna()
        out[mask_missing] = clim_vals[mask_missing]

        raw_conf = clim_counts / float(self.min_count_for_high_conf)
        raw_conf = np.clip(raw_conf, 0.0, 1.0)

        self.logger.debug("assigned climatology values and computed raw confidences")

        conf = pd.Series(0.0, index=s.index)
        conf[mask_missing] = raw_conf[mask_missing]
        conf[~mask_missing] = 1.0

        self.logger.debug("climatology imputation complete")

        return out, conf
