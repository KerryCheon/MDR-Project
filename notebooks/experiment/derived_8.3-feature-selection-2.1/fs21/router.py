"""Train-only V0 K=2 routing with target-free reference alignment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .artifacts import stable_json_hash


@dataclass
class FrozenRouter:
    columns: list[str]
    means: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    centers: np.ndarray
    label_mapping: dict[int, int]
    n_init: int
    random_state: int

    def _values(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.columns).difference(frame.columns))
        if missing:
            raise ValueError(f"router features are missing: {missing[:10]}")
        values = (
            frame[self.columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=float)
        )
        fill = np.broadcast_to(self.means, values.shape)
        return np.where(np.isfinite(values), values, fill)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = self._values(frame)
        return (values - self.scaler_mean) / self.scaler_scale

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scaled = self.transform(frame)
        distance_matrix = np.linalg.norm(
            scaled[:, None, :] - self.centers[None, :, :], axis=2
        )
        raw = np.argmin(distance_matrix, axis=1).astype(int)
        labels = np.asarray([self.label_mapping[int(value)] for value in raw], dtype=int)
        distances = distance_matrix[np.arange(len(raw)), raw]
        return labels, distances

    def to_dict(self) -> dict:
        payload = {
            "columns": list(self.columns),
            "means": self.means.tolist(),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "centers": self.centers.tolist(),
            "label_mapping": {
                str(key): int(value) for key, value in self.label_mapping.items()
            },
            "n_init": int(self.n_init),
            "random_state": int(self.random_state),
            "target_free_alignment": True,
        }
        payload["router_hash"] = stable_json_hash(payload)
        return payload


def _router_values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"router features are missing: {missing[:10]}")
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _best_centroid_mapping(
    centers_in_reference_space: np.ndarray,
    reference_centers: np.ndarray,
) -> dict[int, int]:
    n_clusters = len(reference_centers)
    if centers_in_reference_space.shape != reference_centers.shape:
        raise ValueError("router centroid shapes do not match for alignment")
    best = None
    for assignment in permutations(range(n_clusters)):
        cost = sum(
            float(
                np.linalg.norm(
                    centers_in_reference_space[source]
                    - reference_centers[assignment[source]]
                )
            )
            for source in range(n_clusters)
        )
        candidate = (cost, tuple(assignment))
        if best is None or candidate < best:
            best = candidate
    return {source: int(best[1][source]) for source in range(n_clusters)}


def fit_router(
    frame: pd.DataFrame,
    router_config: Mapping[str, object],
    columns: list[str],
    *,
    reference: FrozenRouter | None = None,
) -> FrozenRouter:
    expected = {
        "kind": "clustering_v0_full_k2",
        "feature_count": 50,
        "imputation": "mean",
        "scaler": "StandardScaler",
        "n_clusters": 2,
        "n_init": 10,
        "random_state": 42,
        "alignment_target_free": True,
    }
    mismatches = {
        key: (router_config.get(key), value)
        for key, value in expected.items()
        if router_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid frozen router configuration: {mismatches}")
    if len(columns) != 50 or len(columns) != len(set(columns)):
        raise ValueError("router must use the exact ordered 50-feature V0 list")
    values = _router_values(frame, columns)
    means = values.mean(axis=0)
    if means.isna().any():
        raise ValueError(
            f"router has all-missing fit features: {means.index[means.isna()].tolist()}"
        )
    filled = values.fillna(means).to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)
    kmeans = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=42,
    ).fit(scaled)
    if reference is None:
        mapping = {0: 0, 1: 1}
    else:
        raw_centers = kmeans.cluster_centers_ * scaler.scale_ + scaler.mean_
        centers_in_reference_space = (
            raw_centers - reference.scaler_mean
        ) / reference.scaler_scale
        mapping = _best_centroid_mapping(
            centers_in_reference_space,
            reference.centers,
        )
    return FrozenRouter(
        columns=list(columns),
        means=means.to_numpy(dtype=float),
        scaler_mean=scaler.mean_.astype(float),
        scaler_scale=scaler.scale_.astype(float),
        centers=kmeans.cluster_centers_.astype(float),
        label_mapping=mapping,
        n_init=10,
        random_state=42,
    )


def router_diagnostics(
    frame: pd.DataFrame,
    router: FrozenRouter,
    *,
    station_col: str,
) -> dict[str, pd.DataFrame]:
    labels, distances = router.predict(frame)
    routed = pd.DataFrame(
        {
            "regime": labels,
            "route_distance": distances,
            "year": frame["_year"].to_numpy(dtype=int),
            "month": frame["_month"].to_numpy(dtype=int),
            "station": frame[station_col].astype(str).to_numpy(),
        }
    )
    populations = []
    for group_name, columns in (
        ("year", ["year", "regime"]),
        ("month", ["month", "regime"]),
        ("station", ["station", "regime"]),
    ):
        table = routed.groupby(columns, sort=True).size().rename("row_count").reset_index()
        table["grouping"] = group_name
        populations.append(table)
    missing = _router_values(frame, router.columns).isna().mean().rename("missing_rate")
    distance = (
        routed.groupby("regime", sort=True)["route_distance"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    return {
        "populations": pd.concat(populations, ignore_index=True),
        "route_distance": distance,
        "feature_missingness": missing.rename_axis("feature").reset_index(),
    }

