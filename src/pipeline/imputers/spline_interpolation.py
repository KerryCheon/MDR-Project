# Jakob Balkovec
# Nov 24th 2025
# spline_interpolation.py

# This model implements a Spline Interpolation imputer for temporal data.
# It uses a smoothed univariate spline (cubic) to infer missing values,
# which is ideal for smooth remote-sensing signals (NDVI, LST, etc).

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from pipeline.Pipeline.imputers.base import BaseImputer

class SplineImputer(BaseImputer):
    # desc: Smooth cubic-spline imputer for continuous temporal signals.

    def __init__(self, **kwargs):
        super().__init__("spline", **kwargs)

        cfg = self.imputer_cfg.get("spline", {})

        # smoothing factor (s). Smaller = more wiggly. Larger = smoother.
        self.smooth = float(cfg.get("smooth", 0.5))

        # penalty for extrapolation
        self.extrap_penalty = float(cfg.get("extrap_penalty", 0.3))

        # numerical stability settings
        self.k = int(cfg.get("degree", 3))  # cubic spline

        self.model = None
        self.fitted = False

        self.logger.debug(
            f"initialized with smooth={self.smooth}, degree={self.k}, "
            f"extrap_penalty={self.extrap_penalty}"
        )


    def fit(self, dates, values, aux_df=None):
        # convert timeline
        mask = ~values.isna()
        t = pd.to_datetime(dates[mask]).astype("int64").to_numpy()
        y = values[mask].to_numpy()

        if len(y) < self.k + 1:
            self.fitted = False
            self.logger.warning(
                f"insufficient samples ({len(y)} < {self.k+1}); "
                "disabling SplineImputer for this feature."
            )
            return self

        # normalize t for numeric stability
        self.t0 = t.min()
        t_norm = (t - self.t0).astype(float)

        try:
            # s = smoothing factor
            self.model = UnivariateSpline(
                t_norm,
                y.astype(float),
                k=self.k,
                s=self.smooth
            )
            self.fitted = True

            self.logger.debug(
                f"fit complete on {len(y)} samples; "
                f"t_range=[{t_norm.min()}, {t_norm.max()}]"
            )

        except Exception as e:
            self.fitted = False
            self.logger.error(f"spline fit failed: {e}")

        return self


    def impute(self, dates, values, aux_df=None):
        self.logger.debug(f"starting spline imputation for feature '{values.name}'")

        out = values.copy()
        conf = pd.Series(0.0, index=values.index)

        if not self.fitted:
            self.logger.debug("model not fitted; returning fallback confidences")
            conf[~values.isna()] = 1.0
            return out, conf

        t_test = pd.to_datetime(dates).astype("int64").to_numpy()
        t_norm = (t_test - self.t0).astype(float)

        try:
            preds = self.model(t_norm)
            # second derivative for curvature
            curvature = np.abs(self.model.derivative(n=2)(t_norm))

            # baseline confidence
            base_conf = np.exp(-curvature)

            # penalize extrapolation
            t_min = t_norm.min()
            t_max = t_norm.max()
            train_min = self.model.get_knots()[0]
            train_max = self.model.get_knots()[-1]

            extrap_mask = (t_norm < train_min) | (t_norm > train_max)
            base_conf[extrap_mask] *= self.extrap_penalty

            # clip and convert
            base_conf = np.clip(base_conf, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"spline predict failed: {e}")
            preds = np.full(len(values), np.nan)
            base_conf = np.zeros(len(values))

        mask_missing = values.isna()
        out[mask_missing] = preds[mask_missing]
        conf[mask_missing] = base_conf[mask_missing]

        # real observations always confidence=1
        conf[~mask_missing] = 1.0

        self.logger.debug(
            "spline imputation complete "
            f"(pred range=[{np.nanmin(preds):.3f}, {np.nanmax(preds):.3f}], "
            f"conf range=[{np.nanmin(base_conf):.3f}, {np.nanmax(base_conf):.3f}])"
        )

        out.index  = pd.to_datetime(dates)
        conf.index = pd.to_datetime(dates)
        return out, conf
