"""Routing / Gating strategy implementations for derived_8.4-eval-1.4 (LOSO).

Same 6 routers as derived_8.4-eval-1.1 / eval-1.2 / eval-1.3, plus the NEW
gating-analysis-1.0 K-sweep strategies: `Clustering_Backbone54_k3/_k4`,
`Clustering_Static_k2/_k3/_k4`, `Clustering_Weather_k2/_k3/_k4`,
`Clustering_Dynamic_k3/_k4` and `Clustering_V0_Full_k3/_k4` — KMeans routers
(K in {2,3,4}, seed 42, n_init 10) fitted on the strategy's clustering feature
set (54 backbone / 58 static / 16 weather / 3 dynamic / 50 V0) with the
gating-analysis-1.0 recipe (mean-impute -> StandardScaler -> KMeans). Every
router exposes a `fit(train)` method so it can be refitted per LOSO fold on the
fold's trainval only (no held-out-station leakage into routing).
10→"""

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


class _KMeansRouter:
    """Shared KMeans router machinery: mean-impute -> StandardScaler -> KMeans.

    Same recipe as eval-1.1's winning router and derived_8.4-gating-analysis-1.0
    (KMeans(random_state=42, n_init=10); the clustering audit doc confirms this
    is the protocol exported by the gating analysis). ``n_clusters`` = K regimes.
    """

    def __init__(self, features: list[str], n_clusters: int = 2, seed: int = 42) -> None:
        self.features = list(features)
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        self.means: pd.Series | None = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)

    def fit(self, train: pd.DataFrame) -> "_KMeansRouter":
        values = train.loc[:, self.features].copy()
        self.means = values.mean()
        values = values.fillna(self.means)
        self.kmeans.fit(self.scaler.fit_transform(values))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predict.")
        values = frame.loc[:, self.features].copy().fillna(self.means)
        return self.kmeans.predict(self.scaler.transform(values))


class V0FullRouter(_KMeansRouter):
    """KMeans router fitted on the 50 OVERALL_SELECTED_FEATURES_V0 (K=2 by default)."""


class Backbone54Router(_KMeansRouter):
    """KMeans router fitted on the 54 shared-backbone features.

    NEW in derived_8.4-eval-1.3: the routing features are the SAME 54 features
    used by the single-regime global model (``shared_backbone_54``), so the
    multi-regime model is a direct development of the single-regime model.
    """


class DynamicClusterRouter(_KMeansRouter):
    """KMeans router fitted on the 3 dynamic features."""

    def __init__(self, features: list[str] | None = None, n_clusters: int = 2, seed: int = 42) -> None:
        super().__init__(
            features or ["SMAP_sm_pm_interp_lag1", "G_API", "LST_modis"],
            n_clusters=n_clusters,
            seed=seed,
        )


class StaticClusterRouter(_KMeansRouter):
    """KMeans router fitted on the 58 static/environmental attributes.

    NEW in eval-1.4 (from derived_8.4-gating-analysis-1.0): columns constant
    within each station (coordinates, terrain, soil, land cover, bioclimatic
    normals); feature list pinned in config.yaml from the gating exports.
    """


class WeatherClusterRouter(_KMeansRouter):
    """KMeans router fitted on the 16 curated dynamic weather/land-surface drivers.

    NEW in eval-1.4 (from derived_8.4-gating-analysis-1.0): precipitation, LST,
    SMAP products, vegetation indices, SAR ratio/diff, G_API/G_DSLR, rain sums.
    """


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


def get_router(strategy_name: str, v0_features: list[str], backbone_54: list[str] | None = None,
               seed: int = 42, device: str = "cuda", n_clusters: int = 2,
               feature_sets: dict[str, list[str]] | None = None):
    """Build a router for a strategy name.

    ``feature_sets`` maps the gating-analysis clustering aliases
    (static / weather / dynamic) to their pinned feature lists (config.yaml
    `gating_clustering_strategies.router_feature_sets`); ``n_clusters`` = K.
    """
    if strategy_name == "Global_Single":
        return GlobalSingleRouter()
    elif strategy_name == "Clustering_V0_Full_k2":
        return V0FullRouter(v0_features, n_clusters=2, seed=seed)
    elif strategy_name.startswith("Clustering_V0_Full_k"):
        return V0FullRouter(v0_features, n_clusters=n_clusters, seed=seed)
    elif strategy_name == "Clustering_Backbone54_k2":
        return Backbone54Router(backbone_54 or [], n_clusters=2, seed=seed)
    elif strategy_name.startswith("Clustering_Backbone54_k"):
        return Backbone54Router(backbone_54 or [], n_clusters=n_clusters, seed=seed)
    elif strategy_name.startswith("Clustering_Static_k"):
        feats = list((feature_sets or {}).get("static", []))
        if not feats:
            raise ValueError("Clustering_Static_* requires feature_sets['static'] (config.yaml).")
        return StaticClusterRouter(feats, n_clusters=n_clusters, seed=seed)
    elif strategy_name.startswith("Clustering_Weather_k"):
        feats = list((feature_sets or {}).get("weather", []))
        if not feats:
            raise ValueError("Clustering_Weather_* requires feature_sets['weather'] (config.yaml).")
        return WeatherClusterRouter(feats, n_clusters=n_clusters, seed=seed)
    elif strategy_name.startswith("Clustering_Dynamic_k"):
        return DynamicClusterRouter(n_clusters=n_clusters, seed=seed)
    elif strategy_name == "Univariate_G_API_k2":
        return UnivariateGAPIRouter()
    elif strategy_name == "Seasonal_Binary_k2":
        return SeasonalBinaryRouter()
    elif strategy_name == "Trained_Gating_k2":
        return TrainedGatingRouter(v0_features, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
