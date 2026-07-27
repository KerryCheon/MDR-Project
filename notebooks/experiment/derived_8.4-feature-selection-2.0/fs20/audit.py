"""Selector-collapse audit across the historic routes and profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import ExperimentData
from .selection import SelectionResult, select_features


@dataclass
class RouteLabels:
    """Train-only-fitted labels for all three split partitions."""

    route: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _kmeans_labels(
    data: ExperimentData, columns: list[str], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = data.train.loc[:, columns].mean()
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
    train_values = data.train.loc[:, columns].fillna(means)
    kmeans.fit(scaler.fit_transform(train_values))

    def predict(frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, columns].fillna(means)
        return kmeans.predict(scaler.transform(values))

    return predict(data.train), predict(data.val), predict(data.test)


def make_routes(data: ExperimentData, seed: int) -> dict[str, RouteLabels]:
    """Build audit routes with exactly train-only route fitting."""
    dynamic_columns = ["SMAP_sm_pm_interp_lag1", "G_API", "LST_modis"]
    routes: dict[str, RouteLabels] = {
        "global": RouteLabels(
            route="global",
            train=np.zeros(len(data.train), dtype=int),
            val=np.zeros(len(data.val), dtype=int),
            test=np.zeros(len(data.test), dtype=int),
        )
    }
    for route_name, columns in {
        "Clustering_Dynamic_k2": dynamic_columns,
        "Clustering_V0_Full_k2": data.v0_features,
    }.items():
        train, val, test = _kmeans_labels(data, columns, seed)
        routes[route_name] = RouteLabels(route_name, train, val, test)

    train_api = data.train["G_API"]
    threshold = float(train_api.fillna(train_api.mean()).quantile(0.5))

    def binner(frame: pd.DataFrame) -> np.ndarray:
        values = frame["G_API"].fillna(train_api.mean())
        return np.where(values < threshold, 0, 1)

    routes["Univariate_G_API_k2"] = RouteLabels(
        route="Univariate_G_API_k2",
        train=binner(data.train),
        val=binner(data.val),
        test=binner(data.test),
    )
    return routes


def classify_collapse(stable_count: int) -> str:
    """Give the audit a human-readable collapse status."""
    if stable_count < 20:
        return "hard_collapsed"
    if stable_count < 50:
        return "truncated"
    return "healthy"


def _result_payload(
    route: str,
    cluster: str,
    profile: str,
    result: SelectionResult,
    n_train: int,
    n_val: int,
    n_test: int,
    target_std: float,
) -> dict[str, Any]:
    payload = result.as_dict()
    payload.update(
        {
            "route": route,
            "cluster": cluster,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "target_std": target_std,
            "collapse_status": classify_collapse(len(result.stable_selected)),
        }
    )
    return payload


def run_audit(
    data: ExperimentData, config: dict[str, Any], artifact_dir: Path
) -> tuple[dict[str, dict[str, dict[str, SelectionResult]]], pd.DataFrame]:
    """Run the audited profiles and persist one readable artifact per partition."""
    audit_cfg = config["audit"]
    selection_cfg = config["selection"]
    routes = make_routes(data, int(config["model"]["seed"]))
    audit_root = artifact_dir / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, dict[str, SelectionResult]]] = {}
    rows: list[dict[str, Any]] = []

    for route_name, labels in routes.items():
        cluster_ids = ["global"] if route_name == "global" else ["0", "1"]
        all_results[route_name] = {}
        for cluster_id in cluster_ids:
            cluster = 0 if cluster_id == "global" else int(cluster_id)
            train_mask = np.ones(len(data.train), dtype=bool) if cluster_id == "global" else labels.train == cluster
            val_mask = np.ones(len(data.val), dtype=bool) if cluster_id == "global" else labels.val == cluster
            test_mask = np.ones(len(data.test), dtype=bool) if cluster_id == "global" else labels.test == cluster
            X_train = data.train.loc[train_mask, data.feature_columns].reset_index(drop=True)
            y_train = data.train.loc[train_mask, data.target].reset_index(drop=True)
            all_results[route_name][cluster_id] = {}
            for profile_name in audit_cfg["profiles"]:
                print(
                    f"[audit] route={route_name} cluster={cluster_id} profile={profile_name} "
                    f"train_rows={len(X_train)}",
                    flush=True,
                )
                result = select_features(
                    X_train,
                    y_train,
                    profile_name,
                    top_k=int(selection_cfg["top_k"]),
                    elasticnet_k=int(selection_cfg["elasticnet_k"]),
                    bootstrap_k=int(selection_cfg["bootstrap_k"]),
                    n_boot=int(audit_cfg["stability_n_boot"]),
                    sample_fraction=float(audit_cfg["stability_sample_fraction"]),
                    min_freq=float(selection_cfg["stability_min_freq"]),
                    min_keep=int(selection_cfg["stability_min_keep"]),
                    random_state=int(config["model"]["seed"]),
                )
                all_results[route_name][cluster_id][profile_name] = result
                payload = _result_payload(
                    route_name,
                    cluster_id,
                    profile_name,
                    result,
                    int(train_mask.sum()),
                    int(val_mask.sum()),
                    int(test_mask.sum()),
                    float(y_train.std(ddof=0)),
                )
                out_path = audit_root / route_name / f"cluster_{cluster_id}" / f"{profile_name}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                rows.append(
                    {
                        "route": route_name,
                        "cluster": cluster_id,
                        "profile": profile_name,
                        "n_train": int(train_mask.sum()),
                        "n_val": int(val_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "target_std": float(y_train.std(ddof=0)),
                        "raw_features": result.stage_counts["raw"],
                        "mi_features": result.stage_counts["mi"],
                        "candidate_features": result.stage_counts["candidate"],
                        "elasticnet_nonzero": result.enet_nonzero,
                        "stability_features": result.stage_counts["stability"],
                        "repaired_features": result.stage_counts["repaired"],
                        "alpha": result.alpha,
                        "l1_ratio": result.l1_ratio,
                        "fallback_applied": result.fallback_applied,
                        "collapse_status": classify_collapse(len(result.stable_selected)),
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["route", "cluster", "profile"])
    summary.to_csv(audit_root / "collapse_audit.csv", index=False)
    return all_results, summary
