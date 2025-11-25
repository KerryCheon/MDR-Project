# Jakob Balkovec
# Nov 24th 2025
# gaussian_regression.py

# This model implements a Gaussian Regression imputer for temporal data.
# It uses Gaussian Process Regression to fill in missing values based on the
# temporal patterns in the dataset.

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from Temporal.Pipeline.imputers.base import BaseImputer

class GaussianProcessImputer(BaseImputer):
    # desc: Smooth temporal model using RBF kernel + noise term.

    def __init__(self):
        super().__init__("gaussian_process")

        cfg = self.imputer_cfg.get("gaussian_process", {})
        self.length_scale = float(cfg.get("length_scale", 30.0))
        self.noise_level = float(cfg.get("noise_level", 1.0))

        self.model = None
        self.fitted = False

        self.logger.debug(
            f"initialized with length_scale={self.length_scale}, "
            f"noise_level={self.noise_level}"
        )

    def fit(self, dates, values, aux_df=None):
        # Convert to numeric timeline
        mask = ~values.isna()
        t = pd.to_datetime(dates[mask]).astype("int64").to_numpy()
        y = values[mask].to_numpy()

        if len(y) < 3:
            self.fitted = False
            self.logger.warning(
                f"insufficient training samples ({len(y)}). "
                "Gaussian Process disabled for this feature."
            )
            return self

        X = t.reshape(-1, 1)

        kernel = RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=self.noise_level)

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            optimizer=None       # stable + deterministic
        )

        self.model.fit(X, y)
        self.fitted = True

        self.logger.debug(
            f"fit complete on {len(y)} samples; kernel={self.model.kernel_}"
        )

        return self

    def impute(self, dates, values, aux_df=None):
        self.logger.debug(f"starting GP imputation for feature '{values.name}'")

        s_dates = pd.to_datetime(dates)
        out = values.copy()
        conf = pd.Series(0.0, index=values.index)

        if not self.fitted:
            self.logger.debug("model not fitted; returning fallback confidences")
            conf[~values.isna()] = 1.0
            return out, conf

        # prepare prediction points
        t_test = s_dates.astype("int64").to_numpy().reshape(-1, 1)

        mean, std = self.model.predict(t_test, return_std=True)

        # gaussian confidence rule
        gp_conf = np.exp(-std)

        # only fill NaNs
        mask_missing = values.isna()
        out[mask_missing] = mean[mask_missing]
        conf[mask_missing] = gp_conf[mask_missing]

        # real observations always confidence=1
        conf[~mask_missing] = 1.0

        self.logger.debug(
            "GP imputation complete "
            f"(mean range=[{np.nanmin(mean):.3f}, {np.nanmax(mean):.3f}], "
            f"std range=[{np.nanmin(std):.3f}, {np.nanmax(std):.3f}])"
        )

        out.index  = pd.to_datetime(dates)
        conf.index = pd.to_datetime(dates)
        return out, conf
