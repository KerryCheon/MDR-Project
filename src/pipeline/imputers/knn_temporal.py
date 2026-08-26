# Jakob Balkovec
# Nov 24th 2025
# knn_temporal.py

# This model implements a K-Nearest Neighbors imputer for temporal data.
# It uses the KNN algorithm to fill in missing values based on the nearest neighbors
# in the dataset, considering the temporal aspect of the data.

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from pipeline.imputers.base import BaseImputer

class KNNImputer(BaseImputer):
    # desc: K-Nearest Neighbors imputer for temporal data

    def __init__(self, **kwargs):
        super().__init__("knn", **kwargs)

        cfg = self.imputer_cfg.get("knn", {})

        self.n_neighbors = int(cfg.get("n_neighbors", 5))
        self.distance_weighting = bool(cfg.get("distance_weighting", True))

        self.model = None
        self.fitted = False

        self.logger.debug(
            f"initialized with n_neighbors={self.n_neighbors}, "
            f"distance_weighting={self.distance_weighting}"
        )

    def fit(self, dates, values, aux_df=None):
        # dates: pd.Series or array of actual datetime
        # values: real values only

        mask = ~values.isna()

        t = pd.to_datetime(dates[mask]).astype("int64") / 86400_000_000_000
        self.train_dates = t.astype("float64").to_numpy()

        self.train_vals = values[mask].to_numpy()

        if len(self.train_vals) < 2:
            self.fitted = False
            self.logger.warning(
                f"insufficient training samples ({len(self.train_vals)}); "
                "disabling KNN for this feature."
            )
            return self

        # sklearn needs 2D
        X = np.asarray(self.train_dates, dtype="float64").reshape(-1, 1)
        y = self.train_vals

        weights = "distance" if self.distance_weighting else "uniform"

        self.model = KNeighborsRegressor(
            n_neighbors=min(self.n_neighbors, len(X)),
            weights=weights
        )

        self.model.fit(X, y)
        self.fitted = True

        self.logger.debug(
            f"fit complete using {len(self.train_vals)} samples; "
            f"weights={weights}"
        )

        return self

    def impute(self, dates, values, aux_df=None):
        # pre: target_dates: pd.Series or array of actual datetime
        # post: returns predicted values and confidence scores for target_dates
        # desc: predicts missing values using the fitted KNN model

        self.logger.debug(f"starting KNN imputation for feature '{values.name}'")

        s_dates = pd.to_datetime(dates)
        out = values.copy()
        conf = pd.Series(0.0, index=values.index)

        if not self.fitted:
            self.logger.debug("model not fitted; returning fallback confidences")
            conf[~values.isna()] = 1.0
            return out, conf

        t_test = s_dates.astype("int64") / 86400_000_000_000
        X_test = np.asarray(t_test, dtype="float64").reshape(-1, 1)

        preds = self.model.predict(X_test)

        dists, _ = self.model.kneighbors(X_test)
        knn_conf = 1 / (1 + dists.mean(axis=1))

        mask_missing = values.isna()
        out[mask_missing] = preds[mask_missing]
        conf[mask_missing] = knn_conf[mask_missing]

        conf[~mask_missing] = 1.0

        self.logger.debug(
            "KNN imputation complete "
            f"(pred range=[{np.nanmin(preds):.3f}, {np.nanmax(preds):.3f}], "
            f"conf range=[{np.nanmin(knn_conf):.3f}, {np.nanmax(knn_conf):.3f}])"
        )

        out.index  = pd.to_datetime(dates)
        conf.index = pd.to_datetime(dates)
        return out, conf
