# Jakob Balkovec
# Gradient Boosting Model

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Models.base import BaseModel


class XGBModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.log = get_logger("models.xgb")

        cfg = (config or {}).get("params", {}) if isinstance(config, dict) else {}

        max_depth = int(cfg.get("max_depth", 6))
        learning_rate = float(cfg.get("learning_rate", 0.05))
        max_iter = int(cfg.get("max_iter", 800))
        min_samples_leaf = int(cfg.get("min_samples_leaf", 20))
        l2_regularization = float(cfg.get("l2_regularization", 0.0))
        random_state = int(cfg.get("random_state", 42))

        self.model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("hgb", HistGradientBoostingRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=max_iter,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=random_state,
            ))
        ])

    def fit(self, X, y):
        y_num = pd.to_numeric(y, errors="coerce").to_numpy()
        if np.isnan(y_num).any():
            raise ValueError("XGBModel.fit: y has NaNs after coercion")

        self.model.fit(X, y_num)
        self.log.info("fit: done (X=%s)", getattr(X, "shape", None))
        return self

    def predict(self, X):
        return self.model.predict(X)
