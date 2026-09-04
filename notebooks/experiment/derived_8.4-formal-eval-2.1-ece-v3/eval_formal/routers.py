"""Routing / Gating strategy implementations for derived_8.4-formal-eval-2.1-ece-v3.

Applies the missingness-aware MoE routing fix from derived_8.4-ece-router-salvage-2.0:
  - Input-only availability gate: triggers when full SMAP block is missing or miss rate > 0.10.
  - Auxiliary fallback: routes gated samples using SMAP-free UnivariateGAPIRouter.
  - Soft ambiguity blending: provides convex combo weights w0*E0 + w1*E1 for ambiguous rows.
On clean WA data, the gate never triggers, preserving historical benchmarks.
"""

from __future__ import annotations

from typing import Protocol
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def smap_router_features(router_features: list[str]) -> list[str]:
    return [f for f in router_features if "SMAP" in f]


def availability_gate(frame: pd.DataFrame, router_features: list[str],
                      tau: float = 0.10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Input-only reliability flag. No target used.

    Returns (gated, miss_rate, smap_miss_rate). Gated when the full SMAP
    router-feature block is native-missing OR the overall router miss rate
    exceeds tau.
    """
    feats = list(router_features)
    miss = frame.loc[:, feats].isna().to_numpy(dtype=float)
    miss_rate = miss.mean(axis=1)
    smap = smap_router_features(feats)
    if smap:
        smap_miss = frame.loc[:, smap].isna().to_numpy(dtype=float).mean(axis=1)
        smap_block_missing = smap_miss >= 1.0 - 1e-12
    else:
        smap_miss = np.zeros(len(frame))
        smap_block_missing = np.zeros(len(frame), dtype=bool)
    gated = smap_block_missing | (miss_rate > float(tau))
    return gated.astype(bool), miss_rate, smap_miss


def softmax_neg_dists(dists: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    assert t > 0, "temperature must be positive"
    z = -np.asarray(dists, dtype=float) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


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

    def predict_weights(self, frame: pd.DataFrame) -> np.ndarray:
        n = len(frame)
        return np.column_stack([np.ones(n), np.zeros(n)])


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

    def predict_weights(self, frame: pd.DataFrame) -> np.ndarray:
        labels = self.predict(frame)
        return np.column_stack([labels == 0, labels == 1]).astype(float)


class SeasonalBinaryRouter:
    """Calendar month router (Dry May-Oct -> 0, Wet Nov-Apr -> 1)."""

    def fit(self, train: pd.DataFrame) -> SeasonalBinaryRouter:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        months = frame["month"].to_numpy(dtype=int)
        is_dry = np.isin(months, [5, 6, 7, 8, 9, 10])
        return np.where(is_dry, 0, 1)

    def predict_weights(self, frame: pd.DataFrame) -> np.ndarray:
        labels = self.predict(frame)
        return np.column_stack([labels == 0, labels == 1]).astype(float)


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
        # NOTE: gate_features are passed with native NaNs — XGBoost handles
        # missing natively (unlike the KMeans routers above, which mean-impute
        # via StandardScaler). Do NOT fillna here; it would shift split points.
        y = train[self.target].to_numpy(dtype=float)
        y_binary = np.where(y < self.threshold, 0, 1)
        params = {            "n_estimators": 2500,
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

    def predict_weights(self, frame: pd.DataFrame) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("TrainedGatingRouter must be fitted before predict.")
        proba = self.clf.predict_proba(frame.loc[:, self.gate_features])
        return np.asarray(proba, dtype=float)


class SalvagedKMeansRouter:
    """KMeans router enhanced with the missingness-aware availability gate from salvage-2.0.

    When routing features contain missing SMAP features or miss rate > tau:
      - Uses auxiliary UnivariateGAPIRouter to avoid the SMAP missingness routing trap.
    """

    def __init__(
        self,
        features: list[str],
        seed: int = 42,
        tau: float = 0.10,
        temperature: float = 0.25,
        apply_fix: bool = True,
    ) -> None:
        self.features = list(features)
        self.seed = int(seed)
        self.tau = float(tau)
        self.temperature = float(temperature)
        self.apply_fix = bool(apply_fix)

        self.means: pd.Series | None = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2, random_state=self.seed, n_init=10)
        self.aux_router = UnivariateGAPIRouter()
        self.wa_threshold: float = 1.55
        self.wa_median: float = 3.65

    def fit(self, train: pd.DataFrame) -> SalvagedKMeansRouter:
        values = train.loc[:, self.features].copy()
        self.means = values.mean()
        values = values.fillna(self.means)
        X_scaled = self.scaler.fit_transform(values)
        self.kmeans.fit(X_scaled)

        if self.apply_fix:
            self.aux_router.fit(train)
            dists = self.kmeans.transform(X_scaled)
            margins = np.abs(dists[:, 0] - dists[:, 1])
            self.wa_threshold = float(np.percentile(margins, 5))
            self.wa_median = float(np.median(margins))
        return self

    def static_dists(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("Router must be fitted before static_dists.")
        values = frame.loc[:, self.features].copy().fillna(self.means)
        return np.asarray(self.kmeans.transform(self.scaler.transform(values)), dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("Router must be fitted before predict.")
        values = frame.loc[:, self.features].copy().fillna(self.means)
        static_labels = self.kmeans.predict(self.scaler.transform(values))

        if not self.apply_fix:
            return static_labels

        gated, _, _ = availability_gate(frame, self.features, self.tau)
        if not gated.any():
            return static_labels

        aux_labels = np.asarray(self.aux_router.predict(frame)).ravel().astype(int)
        return np.where(gated, aux_labels, static_labels)

    def predict_weights(self, frame: pd.DataFrame) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("Router must be fitted before predict_weights.")
        dists = self.static_dists(frame)
        static_labels = np.argmin(dists, axis=1)

        if not self.apply_fix:
            return np.column_stack([static_labels == 0, static_labels == 1]).astype(float)

        gated, _, _ = availability_gate(frame, self.features, self.tau)
        margins = np.abs(dists[:, 0] - dists[:, 1])
        ambiguous = margins < self.wa_threshold
        w_static_soft = softmax_neg_dists(dists, self.temperature)

        aux_labels = np.asarray(self.aux_router.predict(frame)).ravel().astype(int)
        pseudo = np.where(
            (aux_labels == 0)[:, None],
            np.column_stack([np.zeros(len(frame)), np.full(len(frame), self.wa_median)]),
            np.column_stack([np.full(len(frame), self.wa_median), np.zeros(len(frame))]),
        )
        w_aux_soft = softmax_neg_dists(pseudo, self.temperature)

        hard_onehot = np.column_stack([static_labels == 0, static_labels == 1]).astype(float)
        w_auto_soft = np.where(
            gated[:, None],
            w_aux_soft,
            np.where(ambiguous[:, None], w_static_soft, hard_onehot),
        )
        return w_auto_soft


class V0FullRouter(SalvagedKMeansRouter):
    """KMeans(k=2) router fitted on 50 OVERALL_SELECTED_FEATURES_V0 with availability gate."""

    def __init__(self, features: list[str], seed: int = 42, tau: float = 0.10,
                 temperature: float = 0.25, apply_fix: bool = True) -> None:
        super().__init__(features=features, seed=seed, tau=tau,
                         temperature=temperature, apply_fix=apply_fix)


class Backbone54Router(SalvagedKMeansRouter):
    """KMeans(k=2) router fitted on the 54 shared-backbone features with availability gate."""

    def __init__(self, features: list[str], seed: int = 42, tau: float = 0.10,
                 temperature: float = 0.25, apply_fix: bool = True) -> None:
        super().__init__(features=features, seed=seed, tau=tau,
                         temperature=temperature, apply_fix=apply_fix)


class DynamicClusterRouter(SalvagedKMeansRouter):
    """KMeans(k=2) router fitted on 3 dynamic features with availability gate."""

    def __init__(self, features: list[str] | None = None, seed: int = 42, tau: float = 0.10,
                 temperature: float = 0.25, apply_fix: bool = True) -> None:
        feats = features or ["SMAP_sm_pm_interp_lag1", "G_API", "LST_modis"]
        super().__init__(features=feats, seed=seed, tau=tau,
                         temperature=temperature, apply_fix=apply_fix)


def get_router(strategy_name: str, v0_features: list[str], backbone_54: list[str] | None = None,
               seed: int = 42, device: str = "cuda", apply_fix: bool = True,
               tau: float = 0.10, temperature: float = 0.25):
    if strategy_name == "Global_Single":
        return GlobalSingleRouter()
    elif strategy_name == "Clustering_V0_Full_k2":
        return V0FullRouter(v0_features, seed=seed, tau=tau, temperature=temperature, apply_fix=apply_fix)
    elif strategy_name == "Clustering_Backbone54_k2":
        return Backbone54Router(backbone_54 or [], seed=seed, tau=tau, temperature=temperature, apply_fix=apply_fix)
    elif strategy_name == "Clustering_Dynamic_k2":
        return DynamicClusterRouter(seed=seed, tau=tau, temperature=temperature, apply_fix=apply_fix)
    elif strategy_name == "Univariate_G_API_k2":
        return UnivariateGAPIRouter()
    elif strategy_name == "Seasonal_Binary_k2":
        return SeasonalBinaryRouter()
    elif strategy_name == "Trained_Gating_k2":
        return TrainedGatingRouter(v0_features, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
