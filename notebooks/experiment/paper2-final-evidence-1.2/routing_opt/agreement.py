"""Agreement / purity / K-sweep diagnostics (trainval-fit only, no test fitting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def corrected_agreement(a: np.ndarray, b: np.ndarray) -> float:
    agree = float((np.asarray(a).ravel() == np.asarray(b).ravel()).mean())
    return max(agree, 1.0 - agree)


def station_purity(frame: pd.DataFrame, labels: np.ndarray, col: str = "station_id") -> pd.DataFrame:
    labels = np.asarray(labels).ravel().astype(int)
    rows = []
    for station, idx in frame.groupby(col).groups.items():
        lab = labels[np.asarray(list(idx), dtype=int)]
        counts = np.bincount(lab, minlength=int(lab.max()) + 1)
        rows.append({
            "station_id": station,
            "n": int(len(lab)),
            "dominant_cluster": int(np.argmax(counts)),
            "purity": float(counts.max() / len(lab)),
        })
    out = pd.DataFrame(rows)
    out["mean_purity"] = out["purity"].mean()
    return out


def scaled_matrix(train: pd.DataFrame, features: list[str]):
    values = train.loc[:, features].copy()
    means = values.mean()
    filled = values.fillna(means)
    scaler = StandardScaler()
    x_scaled = np.ascontiguousarray(scaler.fit_transform(filled))
    return x_scaled, scaler, means


def k_sweep_quality(
    train: pd.DataFrame, features: list[str], ks: tuple[int, ...] = (2, 3, 4), seed: int = 42
) -> pd.DataFrame:
    """Trainval-only internal indices + purity + balance for automatic K-selection."""
    x_scaled, _, _ = scaled_matrix(train, features)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
        labels = km.fit_predict(x_scaled)
        pur = station_purity(train.reset_index(drop=True), labels)["purity"].mean()
        sizes = np.bincount(labels).tolist()
        rows.append({
            "K": int(k),
            "silhouette": float(silhouette_score(x_scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
            "inertia": float(km.inertia_),
            "mean_station_purity_trainval": float(pur),
            "min_cluster_share": float(min(sizes) / len(labels)),
            "cluster_sizes": sizes,
        })
    return pd.DataFrame(rows)


def pairwise_ari(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(np.asarray(a).ravel(), np.asarray(b).ravel()))
