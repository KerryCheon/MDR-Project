# Jakob Balkovec
# Nov 16th 2025
# fbfill.py

import pandas as pd
import numpy as np
from pipeline.Pipeline.imputers.base import BaseImputer
from xgboost import XGBRegressor

class XGBImputer(BaseImputer):
    # desc: Context-based imputer using temporal encodings + cross-feature signals.

    def __init__(self):
        super().__init__("xgboost")
        cfg = self.imputer_cfg.get("xgboost", {})
        min_known = cfg.get("min_known", 30)
        self.min_known = int(min_known)
        self.model = None
        self.features = []
        self.logger.debug(f"initialized with min_known={self.min_known}")

    def fit(self, dates, values, aux_df=None):
        if aux_df is None or "date" not in aux_df.columns:
            self.logger.debug("aux_df missing or has no 'date' column; skipping xgb fit")
            return self

        self.logger.debug(f"starting xgb fit for feature '{values.name}'")

        df = aux_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        col = values.name
        feats = ["DOY_sin", "DOY_cos", "year"]
        feats += [c for c in ["LST", "NDVI", "Rain_sat"] if c != col and c in df.columns]
        self.features = feats

        self.logger.debug(f"xgb features: {self.features}")

        df[col] = values.values
        known = df.dropna(subset=[col])

        if len(known) < self.min_known:
            self.active = False
            self.logger.warning(
                f"not enough samples ({len(known)}) to train imputer -- DISABLING XGB"
            )
            self.model = None
            return self

        X_all = aux_df.loc[known.index, self.features]
        mask_real_feats = X_all.notna().all(axis=1)

        X_train = X_all[mask_real_feats]
        y_train = known[col].loc[X_train.index]

        if len(X_train) < self.min_known:
            self.active = False
            self.logger.warning(
                f"not enough clean samples ({len(X_train)}) -- DISABLING XGB"
            )
            self.model = None
            return self

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)
        self.model = model

        self.logger.debug(f"trained xgb model with {len(known)} samples")
        return self

    def impute(self, dates, values, aux_df=None):
        if not self.active:
            self.logger.debug("imputer inactive; returning None")
            return None, None

        self.logger.debug(f"starting xgb imputation for feature '{values.name}'")

        s = pd.Series(values.values, index=pd.to_datetime(dates))

        if self.model is None or aux_df is None:
            self.logger.debug("xgb model not trained or aux_df missing; falling back to original values")
            conf = pd.Series(0.0, index=s.index)
            conf[~s.isna()] = 1.0
            return s, conf

        df = aux_df.copy()
        df["date"] = pd.to_datetime(df["date"]).sort_values()

        df[values.name] = s.reindex(df["date"].values).values

        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["DOY_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["DOY_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        X_pred = df[self.features].fillna(method="ffill").fillna(method="bfill")
        preds = self.model.predict(X_pred)

        self.logger.debug("generated xgb predictions")

        s_pred = pd.Series(preds, index=df["date"]).reindex(s.index)

        conf = pd.Series(0.0, index=s.index)
        conf[s.isna()] = 0.7
        conf[~s.isna()] = 1.0

        self.logger.debug("xgb imputation complete")

        return s_pred, conf
