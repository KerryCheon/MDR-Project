# Jakob Balkovec
# Random Forest Model

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from Utils.logging import get_logger
from Src.soilmoist_fl.Models.base import BaseModel


class RFModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.log = get_logger("models.rf")

        cfg = (config or {}).get("params", {}) if isinstance(config, dict) else {}

        n_estimators = int(cfg.get("n_estimators", 500))
        max_depth = cfg.get("max_depth", None)
        max_depth = None if (max_depth in (None, "null", "None")) else int(max_depth)

        min_samples_leaf = int(cfg.get("min_samples_leaf", 1))
        max_features = cfg.get("max_features", "auto")
        bootstrap = bool(cfg.get("bootstrap", True))
        random_state = int(cfg.get("random_state", 42))
        n_jobs = int(cfg.get("n_jobs", -1))

        self.model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=n_jobs,
            ))
        ])

    def fit(self, X, y):
        y_num = pd.to_numeric(y, errors="coerce").to_numpy()
        if np.isnan(y_num).any():
            raise ValueError("RFModel.fit: y has NaNs after coercion")

        self.model.fit(X, y_num)
        self.log.info("fit: done (X=%s)", getattr(X, "shape", None))
        return self

    def predict(self, X):
        return self.model.predict(X)
