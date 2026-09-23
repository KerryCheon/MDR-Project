#!/usr/bin/env python3
"""CLI diagnostic: V0 vs Backbone vs GuardedV-A vs StationMeanV-B agreement.

Fits all routers on trainval (or per-LOSO-fold trainval) and reports ARI /
corrected agreement / group sizes / purity. No XGBoost training, CPU-only.
Fitting never touches the test split except via predict (out-of-sample confirmation).

Usage (from notebooks/, uv env):
    uv run python experiment/derived_8.4-routing-optimize-1.0/analyze_agreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from routing_opt.agreement import corrected_agreement, k_sweep_quality, station_purity
from routing_opt.data import load_experiment_data
from routing_opt.routers import get_router

PROJECT_ROOT = EXP_DIR.parents[2]


def _labels(router, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(router.predict(frame)).ravel().astype(int)


def main() -> None:
    with open(EXP_DIR / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data = load_experiment_data(PROJECT_ROOT, config)
    b54 = data.shared_backbone_54
    v0 = data.v0_features
    seed = int(config["router"]["seed"])
    tau = float(config["router"]["tau"])

    print(f"V0 features: {len(v0)} | backbone-54: {len(b54)}")
    print(f"overlap: {len(set(v0) & set(b54))}")
    print(f"in V0, not in backbone-54: {sorted(set(v0) - set(b54))}")
    print(f"in backbone-54, not in V0: {sorted(set(b54) - set(v0))}")
    print()

    routers = {
        "V0_Full": get_router("Clustering_V0_Full_k2", backbone_54=b54, v0_features=v0, seed=seed, tau=tau),
        "Backbone": get_router("Clustering_Backbone54_k2", backbone_54=b54, seed=seed, tau=tau),
        "GuardedV-A": get_router("Guarded_Backbone54_k2", backbone_54=b54, seed=seed, tau=tau),
        "StationMeanV-B": get_router("StationMean_Backbone54_k2", backbone_54=b54, seed=seed, tau=tau),
    }
    for name, router in routers.items():
        router.fit(data.trainval)

    frames = {"trainval": data.trainval, "test": data.test}
    names = list(routers.keys())
    for fname, frame in frames.items():
        labs = {name: _labels(routers[name], frame) for name in names}
        print(f"### Partition sizes + purity ({fname}, n={len(frame)})")
        for name in names:
            sizes = np.bincount(labs[name]).tolist()
            pur = station_purity(frame.reset_index(drop=True), labs[name])["purity"].mean()
            print(f"  {name:14s} sizes={sizes} mean_purity={pur:.4f}")
        print(f"### Pairwise ARI / corrected agreement ({fname})")
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                rows.append({
                    "A": a,
                    "B": b,
                    "ARI": round(float(adjusted_rand_score(labs[a], labs[b])), 4),
                    "agree": round(corrected_agreement(labs[a], labs[b]), 4),
                })
        print(pd.DataFrame(rows).to_markdown(index=False))
        print()

    print("### Per-LOSO-fold agreement (router refit on 6-station fold trainval)")
    fold_rows = []
    for station in sorted(data.test["station_id"].unique()):
        tr = data.trainval[data.trainval["station_id"] != station].reset_index(drop=True)
        te = data.test[data.test["station_id"] == station].reset_index(drop=True)
        fold_routers = {
            name: get_router(
                {"V0_Full": "Clustering_V0_Full_k2", "Backbone": "Clustering_Backbone54_k2",
                 "GuardedV-A": "Guarded_Backbone54_k2",
                 "StationMeanV-B": "StationMean_Backbone54_k2"}[name],
                backbone_54=b54, v0_features=v0, seed=seed, tau=tau,
            ).fit(tr)
            for name in names
        }
        fold_labs_tr = {n: _labels(fold_routers[n], tr) for n in names}
        fold_labs_te = {n: _labels(fold_routers[n], te) for n in names}
        row: dict[str, object] = {"held_out": station}
        for pair in [("V0_Full", "Backbone"), ("Backbone", "GuardedV-A"),
                     ("Backbone", "StationMeanV-B"), ("GuardedV-A", "StationMeanV-B")]:
            a, b = pair
            row[f"ARI_tr_{a}_vs_{b}"] = round(
                float(adjusted_rand_score(fold_labs_tr[a], fold_labs_tr[b])), 4)
            row[f"ARI_te_{a}_vs_{b}"] = round(
                float(adjusted_rand_score(fold_labs_te[a], fold_labs_te[b])), 4)
        guarded = fold_routers["GuardedV-A"]
        assert hasattr(guarded, "fallback_mask")
        row["guarded_fallback_rate_te"] = round(float(guarded.fallback_mask(te).mean()), 4)
        fold_rows.append(row)
    print(pd.DataFrame(fold_rows).to_markdown(index=False))
    print()

    print("### K-sweep quality (Backbone54, trainval-only)")
    print(k_sweep_quality(data.trainval, b54, tuple(config["k_sweep"]), seed).to_markdown(index=False))


if __name__ == "__main__":
    main()
