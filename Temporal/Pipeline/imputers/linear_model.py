# Jakob Balkovec
# Nov 16th 2025
# linear_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Temporal.Pipeline.imputers.base import BaseImputer

class LinearModelImputer(BaseImputer):
    # desc: Linear regression imputer using temporal encodings and optional cross-feature predictors.

    def __init__(self):
        super().__init__("linear_model")
        cfg = self.imputer_cfg.get("linear_model", {})
        self.min_known = int(cfg.get("min_known", 15))
        self.use_cross = cfg.get("use_cross_features", True)
        self.model = None
        self.features = []
        self.logger.debug(
            f"initialized with min_known={self.min_known}, use_cross_features={self.use_cross}"
        )

    def fit(self, dates, values, aux_df=None):
        if aux_df is None or "date" not in aux_df.columns:
            self.logger.debug("aux_df missing or has no 'date' column; skipping fit")
            self.active = False
            return self

        self.logger.debug(f"starting fit for feature '{values.name}'")

        df = aux_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        col = values.name
        df[col] = values.values

        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
        df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

        feats = ["DOY_sin", "DOY_cos", "year"]

        # optional cross-features (only if present)
        if self.use_cross:
            for c in ["LST", "NDVI", "Rain_sat"]:
                if c != col and c in df.columns:
                    feats.append(c)

        self.features = feats
        self.logger.debug(f"using features: {self.features}")

        known = df.dropna(subset=[col])

        if len(known) < self.min_known:
            self.active = False
            self.logger.warning(
                f"not enough samples ({len(known)}) to train linear model -- DISABLING LM"
            )
            self.model = None
            return self

        X_train = known[self.features]
        mask = X_train.notna().all(axis=1)
        X_train = X_train[mask]

        y_train = known.loc[mask, col]

        if len(X_train) < self.min_known:
            self.active = False
            self.logger.warning(
                f"not enough clean samples after feature mask ({len(X_train)}) -- DISABLING LM"
            )
            self.model = None
            return self

        model = LinearRegression()
        model.fit(X_train, y_train)
        self.model = model

        self.logger.debug(f"trained linear model with {len(X_train)} clean samples")
        return self

    def impute(self, dates, values, aux_df=None):
        if not self.active:
            self.logger.debug("imputer inactive; returning None")
            return None, None

        self.logger.debug(f"starting imputation for feature '{values.name}'")

        s = pd.Series(values.values, index=pd.to_datetime(dates))

        if self.model is None or aux_df is None:
            self.logger.debug("model not trained or aux_df missing; falling back to original values")
            conf = pd.Series(0.0, index=s.index)
            conf[~s.isna()] = 1.0
            return s, conf

        df = aux_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df[values.name] = s.reindex(df["date"].values).values

        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        X_pred = df[self.features].fillna(method="ffill").fillna(method="bfill")
        preds = self.model.predict(X_pred)

        self.logger.debug("generated predictions from linear model")

        s_pred = pd.Series(preds, index=df["date"]).reindex(s.index)

        raw_conf = min(1.0, len(df.dropna(subset=[values.name])) / float(self.min_known))
        conf = pd.Series(raw_conf, index=s.index)
        conf[~s.isna()] = 1.0

        self.logger.debug("linear regression imputation complete")
        return s_pred, conf.clip(0.0, 1.0)
