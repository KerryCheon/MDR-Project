"""Guarded backbone routers for derived_8.4-routing-optimize-1.0.

W2 spec (see README §Design):
  - Routing features are ALWAYS the shared 54-feature backbone (full-covariate
    routing narrative preserved; no V0-50 import, no hand-picked subset).
  - V-A GuardedBackbone54Router: per-sample KMeans fit (gating-analysis recipe:
    mean-impute -> StandardScaler -> KMeans(2, seed 42, n_init 10)) + per-station
    majority label from the FIT frame only, broadcast at predict time.
  - V-B StationMeanBackbone54Router: same fit machinery, but the station label
    comes from a single KMeans predict on the station-mean feature vector.
  - Unseen stations (held-out LOSO fold, ECE, new region), missing station_id,
    or availability-gated rows fall back to per-sample static prediction with a
    trainval-calibrated margin fallback to the global expert (salvage pattern
    from formal-eval-2.1-ece-v3/eval_formal/routers.py, thresholds fit on the
    fit frame only, never on test).
  - Cluster ids are canonicalized after each fit so c1 is the drier regime
    (lower mean SMAP_sm_pm_interp_rollmean30 on the fit frame).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CANONICAL_FEATURE = "SMAP_sm_pm_interp_rollmean30"
HOLDOUT_COLUMN = "station_id"


def smap_router_features(router_features: list[str]) -> list[str]:
    return [f for f in router_features if "SMAP" in f]


def availability_gate(
    frame: pd.DataFrame, router_features: list[str], tau: float = 0.10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Input-only reliability flag (no target used)."""
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


class UnivariateGAPIRouter:
    """SMAP-free auxiliary router (median threshold on G_API, fit frame only)."""

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


class GlobalSingleRouter:
    """1-regime global baseline router returning regime 0 for all samples."""

    def fit(self, train: pd.DataFrame) -> GlobalSingleRouter:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame), dtype=int)


class _BackboneKMeansBase:
    """Shared KMeans machinery + canonicalization + margin calibration."""

    def __init__(self, features: list[str], seed: int = 42, tau: float = 0.10) -> None:
        self.features = list(features)
        self.seed = int(seed)
        self.tau = float(tau)
        self.means: pd.Series | None = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2, random_state=self.seed, n_init=10)
        self.aux_router = UnivariateGAPIRouter()
        self.margin_p5: float | None = None
        self.margin_median: float | None = None
        self.label_flip_applied = False

    def _fit_core(self, train: pd.DataFrame) -> np.ndarray:
        values = train.loc[:, self.features].copy()
        self.means = values.mean()
        filled = values.fillna(self.means)
        # NOTE: ascontiguousarray guards against pandas handing sklearn an
        # F-order block layout (boolean row slices under LOSO), which
        # KMeans.predict rejects with "ndarray is not C-contiguous".
        x_scaled = np.ascontiguousarray(self.scaler.fit_transform(filled))
        self.kmeans.fit(x_scaled)
        self.aux_router.fit(train)
        dists = self.kmeans.transform(x_scaled)
        margins = np.abs(dists[:, 0] - dists[:, 1])
        self.margin_p5 = float(np.percentile(margins, 5))
        self.margin_median = float(np.median(margins))
        labels = self.kmeans.predict(x_scaled)
        self._canonicalize(train, labels)
        return np.asarray(self.kmeans.predict(x_scaled)).ravel().astype(int)

    def _canonicalize(self, train: pd.DataFrame, labels: np.ndarray) -> None:
        """Ensure c1 = drier regime (lower fit-frame mean of CANONICAL_FEATURE)."""
        if CANONICAL_FEATURE not in train.columns:
            self.label_flip_applied = False
            return
        frame = train.copy()
        frame["_cl"] = np.asarray(labels).ravel().astype(int)
        means = frame.groupby("_cl")[CANONICAL_FEATURE].mean()
        if len(means) == 2 and bool(means.loc[0] < means.loc[1]):
            # c0 already drier than c1 -> swap so c1 is drier
            self._swap_labels()
            self.label_flip_applied = True
        else:
            self.label_flip_applied = False

    def _swap_labels(self) -> None:
        centers = np.asarray(self.kmeans.cluster_centers_)
        # NOTE: .copy() keeps the swapped centers C-contiguous; a bare
        # [::-1, :] view breaks KMeans.predict ("ndarray is not C-contiguous").
        self.kmeans.cluster_centers_ = np.ascontiguousarray(centers[::-1, :])

    def _static_predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.means is None or self.margin_p5 is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predict.")
        values = frame.loc[:, self.features].copy().fillna(self.means)
        x_scaled = np.ascontiguousarray(self.scaler.transform(values))
        dists = np.asarray(self.kmeans.transform(x_scaled), dtype=float)
        labels = np.asarray(self.kmeans.predict(x_scaled)).ravel().astype(int)
        margins = np.abs(dists[:, 0] - dists[:, 1])
        return labels, margins

    def _fallback_flags(self, frame: pd.DataFrame, margins: np.ndarray) -> np.ndarray:
        gated, _, _ = availability_gate(frame, self.features, self.tau)
        ambiguous = margins < float(self.margin_p5)
        return gated | ambiguous


class Backbone54Router(_BackboneKMeansBase):
    """Unguarded baseline: per-sample KMeans on the 54 backbone (reference arm)."""

    def fit(self, train: pd.DataFrame) -> Backbone54Router:
        self._fit_core(train)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        labels, _ = self._static_predict(frame)
        return labels


class V0FullRouter(_BackboneKMeansBase):
    """Legacy reference: per-sample KMeans on the 50 V0 features (sensitivity only)."""

    def fit(self, train: pd.DataFrame) -> V0FullRouter:
        self._fit_core(train)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        labels, _ = self._static_predict(frame)
        return labels


class GuardedBackbone54Router(_BackboneKMeansBase):
    """V-A: majority-vote guard over the 54-backbone KMeans (primary candidate).

    Known stations (present in the fit frame, not gated) get their fit-frame
    majority label; unseen/missing/gated rows fall back to the per-sample
    static prediction.
    """

    def __init__(self, features: list[str], seed: int = 42, tau: float = 0.10) -> None:
        super().__init__(features, seed=seed, tau=tau)
        self.station_majority: dict[str, int] = {}
        self.majority_strength: dict[str, float] = {}

    def fit(self, train: pd.DataFrame) -> GuardedBackbone54Router:
        labels = self._fit_core(train)
        frame = train.copy()
        frame["_cl"] = labels
        for station, group in frame.groupby(HOLDOUT_COLUMN):
            counts = group["_cl"].value_counts()
            winner = int(counts.index[0])
            # Deterministic tie-break: smaller canonical cluster id wins.
            if len(counts) == 2 and counts.iloc[0] == counts.iloc[1]:
                winner = int(min(counts.index.tolist()))
            self.station_majority[str(station)] = winner
            self.majority_strength[str(station)] = float(counts.iloc[0] / len(group))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        static_labels, margins = self._static_predict(frame)
        gated, _, _ = availability_gate(frame, self.features, self.tau)
        out = static_labels.copy()
        if HOLDOUT_COLUMN in frame.columns:
            stations = frame[HOLDOUT_COLUMN].astype(str).to_numpy()
            for i, station in enumerate(stations):
                if gated[i]:
                    continue  # keep aux-independent static label; expert fallback handled downstream
                if station in self.station_majority:
                    out[i] = self.station_majority[station]
        return out.astype(int)

    def fallback_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows needing global-expert fallback: unseen station or gated/ambiguous."""
        _, margins = self._static_predict(frame)
        gated, _, _ = availability_gate(frame, self.features, self.tau)
        ambiguous = margins < float(self.margin_p5)
        if HOLDOUT_COLUMN in frame.columns:
            stations = frame[HOLDOUT_COLUMN].astype(str).to_numpy()
            unseen = np.array([s not in self.station_majority for s in stations])
        else:
            unseen = np.ones(len(frame), dtype=bool)
        return (gated | ambiguous | unseen).astype(bool)


class StationMeanBackbone54Router(_BackboneKMeansBase):
    """V-B: station-mean guard challenger (route the site, not the sample)."""

    def __init__(self, features: list[str], seed: int = 42, tau: float = 0.10) -> None:
        super().__init__(features, seed=seed, tau=tau)
        self.station_label: dict[str, int] = {}

    def fit(self, train: pd.DataFrame) -> StationMeanBackbone54Router:
        self._fit_core(train)
        assert self.means is not None
        station_means = train.groupby(HOLDOUT_COLUMN)[self.features].mean()
        filled = station_means.fillna(self.means)
        x_scaled = np.ascontiguousarray(self.scaler.transform(filled))
        labels = np.asarray(self.kmeans.predict(x_scaled)).ravel().astype(int)
        self.station_label = {
            str(station): int(label) for station, label in zip(station_means.index, labels)
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        static_labels, _ = self._static_predict(frame)
        gated, _, _ = availability_gate(frame, self.features, self.tau)
        out = static_labels.copy()
        if HOLDOUT_COLUMN in frame.columns:
            stations = frame[HOLDOUT_COLUMN].astype(str).to_numpy()
            for i, station in enumerate(stations):
                if gated[i]:
                    continue
                if station in self.station_label:
                    out[i] = self.station_label[station]
        return out.astype(int)


def get_router(
    strategy_name: str,
    backbone_54: list[str] | None = None,
    v0_features: list[str] | None = None,
    seed: int = 42,
    tau: float = 0.10,
):
    if strategy_name == "Global_Single":
        return GlobalSingleRouter()
    if strategy_name == "Guarded_Backbone54_k2":
        return GuardedBackbone54Router(backbone_54 or [], seed=seed, tau=tau)
    if strategy_name == "StationMean_Backbone54_k2":
        return StationMeanBackbone54Router(backbone_54 or [], seed=seed, tau=tau)
    if strategy_name == "Clustering_Backbone54_k2":
        return Backbone54Router(backbone_54 or [], seed=seed, tau=tau)
    if strategy_name == "Clustering_V0_Full_k2":
        return V0FullRouter(v0_features or [], seed=seed, tau=tau)
    raise ValueError(f"Unknown strategy name: {strategy_name}")
