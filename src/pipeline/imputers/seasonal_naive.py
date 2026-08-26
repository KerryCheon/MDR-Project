# Jakob Balkovec
# Nov 24th 2025
# seasonal_naive.py

# This model implements a Seasonal Naive imputer for temporal data.
# It fills missing values based on historical observations with the same
# day-of-year (DOY). This preserves seasonal cycles and is effective for
# long gaps where smoother models degrade.

import numpy as np
import pandas as pd

from pipeline.imputers.base import BaseImputer

class SeasonalNaiveImputer(BaseImputer):
    # desc: Day-of-Year historical lookup imputer (seasonal persistence)

    def __init__(self, **kwargs):
        super().__init__("seasonal_naive", **kwargs)

        cfg = self.imputer_cfg.get("seasonal_naive", {})
        self.min_count_for_high_conf = int(cfg.get("min_count_for_high_conf", 10))
        self.use_median = bool(cfg.get("use_median", True))

        self.doy_map = None
        self.count_map = None
        self.fitted = False

        self.logger.debug(
            f"initialized with min_count_for_high_conf={self.min_count_for_high_conf}, "
            f"use_median={self.use_median}"
        )

    def fit(self, dates, values, aux_df=None):
        # pre: dates and values contain at least some real samples
        # post: computes mapping DOY -> historical values

        s_dates = pd.to_datetime(dates)
        s_vals = values

        mask = ~s_vals.isna()
        valid_dates = s_dates[mask]
        valid_vals = s_vals[mask]

        if len(valid_vals) < 3:
            self.fitted = False
            self.logger.warning(
                f"seasonal naive disabled; insufficient real samples ({len(valid_vals)})"
            )
            return self

        doy = valid_dates.dt.dayofyear
        df = pd.DataFrame({"doy": doy, "val": valid_vals.values})

        grouped = df.groupby("doy")["val"]

        if self.use_median:
            self.doy_map = grouped.median()
        else:
            self.doy_map = grouped.mean()

        self.count_map = grouped.count()

        self.fitted = True

        self.logger.debug(
            f"fit complete using {len(valid_vals)} samples, "
            f"{self.count_map.sum()} total DOY entries"
        )

        return self

    def impute(self, dates, values, aux_df=None):
        # pre: dates and values provided
        # post: fills missing values using DOY history
        # desc: seasonal naive DOY lookup

        self.logger.debug(f"starting seasonal naive imputation for feature '{values.name}'")

        s_dates = pd.to_datetime(dates)
        out = values.copy()
        conf = pd.Series(0.0, index=values.index)

        if not self.fitted:
            self.logger.debug("model not fitted; returning fallback confidences")
            conf[~values.isna()] = 1.0
            return out, conf

        doy = s_dates.dt.dayofyear
        missing_mask = values.isna()

        for i in np.where(missing_mask)[0]:
            d = doy.iloc[i]

            if d not in self.doy_map.index:
                # fallback: nearest DOY in history
                nearest = self._nearest_doy(d)
                if nearest is None:
                    out.iloc[i] = np.nan
                    conf.iloc[i] = 0.0
                    continue

                val = self.doy_map.loc[nearest]
                count = self.count_map.loc[nearest]
            else:
                val = self.doy_map.loc[d]
                count = self.count_map.loc[d]

            out.iloc[i] = val

            # confidence proportional to how often this DOY was observed historically
            raw_conf = count / float(self.min_count_for_high_conf)
            conf.iloc[i] = float(np.clip(raw_conf, 0.0, 1.0))

        # observed always confidence 1
        conf[~missing_mask] = 1.0

        self.logger.debug(
            "seasonal naive imputation complete "
            f"(value range=[{np.nanmin(out):.3f}, {np.nanmax(out):.3f}])"
        )

        out.index  = pd.to_datetime(dates)
        conf.index = pd.to_datetime(dates)
        return out, conf

    def _nearest_doy(self, d):
        # helper: find nearest recorded day-of-year in history
        if self.doy_map is None or len(self.doy_map) == 0:
            return None
        idx = self.doy_map.index
        diffs = np.abs(idx - d)
        return int(idx[np.argmin(diffs)])
