# Jakob Balkovec
# Linear Model

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Models.base import BaseModel


class LinearModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.log = get_logger("models.linear")

        cfg = (config or {}).get("params", {}) if isinstance(config, dict) else {}

        # Mild regularization by default (correlated features)
        alpha = float(cfg.get("alpha", 3.0))
        fit_intercept = bool(cfg.get("fit_intercept", True))

        self.model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(
                alpha=alpha,
                fit_intercept=fit_intercept,
                random_state=42,
            )),
        ])

        self.log.info(
            "init: alpha=%.3f fit_intercept=%s",
            alpha,
            str(fit_intercept),
        )

    def fit(self, X, y):
        y_num = pd.to_numeric(y, errors="coerce").to_numpy()
        if np.isnan(y_num).any():
            raise ValueError("LinearModel.fit: y has NaNs after coercion")

        self.model.fit(X, y_num)
        self.log.info("fit: done (X=%s)", getattr(X, "shape", None))
        return self

    def predict(self, X):
        return self.model.predict(X)
