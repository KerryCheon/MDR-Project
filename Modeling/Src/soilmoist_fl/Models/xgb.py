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

        max_depth = int(cfg.get("max_depth", 4))
        learning_rate = float(cfg.get("learning_rate", 0.03))
        max_iter = int(cfg.get("max_iter", 1200))
        min_samples_leaf = int(cfg.get("min_samples_leaf", 30))
        l2_regularization = float(cfg.get("l2_regularization", 1.0))
        random_state = int(cfg.get("random_state", 42))

        early_stopping = bool(cfg.get("early_stopping", True))
        validation_fraction = float(cfg.get("validation_fraction", 0.1))
        n_iter_no_change = int(cfg.get("n_iter_no_change", 30))
        tol = float(cfg.get("tol", 1e-7))

        self.model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("hgb", HistGradientBoostingRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=max_iter,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                tol=tol,
                random_state=random_state,
            ))
        ])

        self.log.info(
            "init: max_depth=%d lr=%.4f max_iter=%d min_samples_leaf=%d l2=%.3f early_stopping=%s",
            max_depth, learning_rate, max_iter, min_samples_leaf, l2_regularization, str(early_stopping)
        )

    def fit(self, X, y):
        y_num = pd.to_numeric(y, errors="coerce").to_numpy()
        if np.isnan(y_num).any():
            raise ValueError("XGBModel.fit: y has NaNs after coercion")

        self.model.fit(X, y_num)
        self.log.info("fit: done (X=%s)", getattr(X, "shape", None))
        return self

    def predict(self, X):
        return self.model.predict(X)
