"""Routing / Gating strategy implementations for derived_8.4-eval-1.1."""

from __future__ import annotations

from typing import Protocol
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class Router(Protocol):
    def fit(self, train: pd.DataFrame) -> Router:
        ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ...


class GlobalSingleRouter:
    """1-regime global baseline router returning regime 0 for all samples."""

    def fit(self, train: pd.DataFrame) -> GlobalSingleRouter:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame), dtype=int)


class V0FullRouter:
    """KMeans(k=2) router fitted on 50 OVERALL_SELECTED_FEATURES_V0."""

    def __init__(self, features: list[str], seed: int = 42) -> None:
        self.features = list(features)
        self.seed = int(seed)
        self.means: pd.Series | None = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2, random_state=self.seed, n_init=10)

    def fit(self, train: pd.DataFrame) -> V0FullRouter:
        values = train.loc[:, self.features].copy()
        self.means = values.mean()
        values = values.fillna(self.means)
        self.kmeans.fit(self.scaler.fit_transform(values))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("V0FullRouter must be fitted before predict.")
        values = frame.loc[:, self.features].copy().fillna(self.means)
        return self.kmeans.predict(self.scaler.transform(values))


class DynamicClusterRouter:
    """KMeans(k=2) router fitted on 3 dynamic features."""

    def __init__(self, features: list[str] | None = None, seed: int = 42) -> None:
        self.features = features or ["SMAP_sm_pm_interp_lag1", "G_API", "LST_modis"]
        self.seed = int(seed)
        self.means: pd.Series | None = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2, random_state=self.seed, n_init=10)

    def fit(self, train: pd.DataFrame) -> DynamicClusterRouter:
        values = train.loc[:, self.features].copy()
        self.means = values.mean()
        values = values.fillna(self.means)
        self.kmeans.fit(self.scaler.fit_transform(values))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("DynamicClusterRouter must be fitted before predict.")
        values = frame.loc[:, self.features].copy().fillna(self.means)
        return self.kmeans.predict(self.scaler.transform(values))


class UnivariateGAPIRouter:
    """Quantile median threshold router on G_API (low -> 0, high -> 1)."""

    def __init__(self, col: str = "G_API") -> None:
        self.col = col
        self.threshold: float | None = None

    def fit(self, train: pd.DataFrame) -> UnivariateGAPIRouter:
        series = train[self.col].fillna(train[self.col].mean())
        self.threshold = float(series.median())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("UnivariateGAPIRouter must be fitted before predict.")
        series = frame[self.col].fillna(self.threshold)
        return np.where(series < self.threshold, 0, 1)


class SeasonalBinaryRouter:
    """Calendar month router (Dry May-Oct -> 0, Wet Nov-Apr -> 1)."""

    def fit(self, train: pd.DataFrame) -> SeasonalBinaryRouter:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        months = frame["month"].to_numpy(dtype=int)
        is_dry = np.isin(months, [5, 6, 7, 8, 9, 10])
        return np.where(is_dry, 0, 1)


class TrainedGatingRouter:
    """XGBClassifier-based binary threshold router (trains on y < 0.16, predicts without target leak)."""

    def __init__(
        self,
        gate_features: list[str],
        target: str = "soil_moisture_5cm",
        threshold: float = 0.16,
        seed: int = 42,
        device: str = "cuda",
    ) -> None:
        self.gate_features = list(gate_features)
        self.target = target
        self.threshold = threshold
        self.seed = int(seed)
        self.device = device
        self.clf: XGBClassifier | None = None

    def fit(self, train: pd.DataFrame) -> TrainedGatingRouter:
        y = train[self.target].to_numpy(dtype=float)
        y_binary = np.where(y < self.threshold, 0, 1)
        params = {
            "n_estimators": 2500,
            "learning_rate": 0.005,
            "max_depth": 9,
            "min_child_weight": 8,
            "gamma": 0.0,
            "reg_lambda": 0.75,
            "reg_alpha": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": self.device,
            "random_state": self.seed,
            "n_jobs": 1,
        }
        self.clf = XGBClassifier(**params)
        self.clf.fit(train.loc[:, self.gate_features], y_binary, verbose=False)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("TrainedGatingRouter must be fitted before predict.")
        return np.asarray(self.clf.predict(frame.loc[:, self.gate_features])).ravel()


def get_router(strategy_name: str, v0_features: list[str], seed: int = 42):
    if strategy_name == "Global_Single":
        return GlobalSingleRouter()
    elif strategy_name == "Clustering_V0_Full_k2":
        return V0FullRouter(v0_features, seed=seed)
    elif strategy_name == "Clustering_Dynamic_k2":
        return DynamicClusterRouter(seed=seed)
    elif strategy_name == "Univariate_G_API_k2":
        return UnivariateGAPIRouter()
    elif strategy_name == "Seasonal_Binary_k2":
        return SeasonalBinaryRouter()
    elif strategy_name == "Trained_Gating_k2":
        return TrainedGatingRouter(v0_features, seed=seed)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
