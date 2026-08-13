#!/usr/bin/env python3
"""Diagnostic: do Clustering_V0_Full_k2 and Clustering_Backbone54_k2 route to the same regimes?

Both strategies are KMeans(k=2, random_state=42, n_init=10) but fitted on DIFFERENT
feature sets — the 50 OVERALL_SELECTED_FEATURES_V0 vs the 54 shared-backbone features —
so the resulting partitions (and therefore each specialist's training data) are not
expected to be identical. This script quantifies the difference per LOSO fold:

  1. feature-set overlap between V0-50 and backbone-54,
  2. cluster-assignment agreement between the two routers on the fold trainval and the
     held-out station's test rows (Adjusted Rand Index + label-permutation-corrected
     raw agreement),
  3. per-cluster sizes on the fold trainval (the specialists' training data),
  4. regime characterization: cluster means of key dynamic features for each router.

Usage:
    uv run python analyze_router_agreement.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

from eval13.data import load_experiment_data
from eval13.routers import Backbone54Router, V0FullRouter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent

KEY_FEATURES = [
    "SMAP_sm_pm_interp", "G_API", "LST_modis", "F_NDVI", "precip_mm", "soil_moisture_5cm",
]


def corrected_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Raw label agreement with the best 2-cluster permutation (flip)."""
    agree = float((a == b).mean())
    return max(agree, 1.0 - agree)


def main() -> None:
    with open(EXP_DIR / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data = load_experiment_data(PROJECT_ROOT, config)

    v0 = data.v0_features
    b54 = data.shared_backbone_54
    print(f"V0 features: {len(v0)} | backbone-54: {len(b54)}")
    print(f"overlap: {len(set(v0) & set(b54))}")
    print(f"in V0, not in backbone-54: {sorted(set(v0) - set(b54))}")
    print(f"in backbone-54, not in V0: {sorted(set(b54) - set(v0))}")
    print()

    rows = []
    for station in sorted(data.test["station_id"].unique()):
        tr = data.trainval[data.trainval["station_id"] != station].reset_index(drop=True)
        te = data.test[data.test["station_id"] == station].reset_index(drop=True)

        r_v0 = V0FullRouter(v0, seed=42).fit(tr)
        r_b54 = Backbone54Router(b54, seed=42).fit(tr)
        l1_tr = np.asarray(r_v0.predict(tr)).ravel().astype(int)
        l2_tr = np.asarray(r_b54.predict(tr)).ravel().astype(int)
        l1_te = np.asarray(r_v0.predict(te)).ravel().astype(int)
        l2_te = np.asarray(r_b54.predict(te)).ravel().astype(int)

        rows.append({
            "station": station,
            "ari_trainval": adjusted_rand_score(l1_tr, l2_tr),
            "ari_test": adjusted_rand_score(l1_te, l2_te),
            "agree_trainval": corrected_agreement(l1_tr, l2_tr),
            "agree_test": corrected_agreement(l1_te, l2_te),
            "v0_c0_n": int((l1_tr == 0).sum()),
            "v0_c1_n": int((l1_tr == 1).sum()),
            "b54_c0_n": int((l2_tr == 0).sum()),
            "b54_c1_n": int((l2_tr == 1).sum()),
        })
    df = pd.DataFrame(rows)
    print("### Per-fold router agreement (KMeans k=2, seed 42: V0-50 vs backbone-54)")
    print(df.to_markdown(index=False))
    print()
    print("ARI: 1.0 = identical partition, 0.0 = random. `agree` = raw label agreement "
          "corrected for the 2-cluster label flip.")

    # Regime characterization on the FULL trainval (all 7 stations, the full-baseline
    # router fit): cluster means of key dynamic features per router.
    print("\n### Regime characterization (router fit on the full trainval, all 7 stations)")
    r_v0 = V0FullRouter(v0, seed=42).fit(data.trainval)
    r_b54 = Backbone54Router(b54, seed=42).fit(data.trainval)
    l_v0 = np.asarray(r_v0.predict(data.trainval)).ravel().astype(int)
    l_b54 = np.asarray(r_b54.predict(data.trainval)).ravel().astype(int)
    for name, labels in (("V0_Full", l_v0), ("Backbone54", l_b54)):
        frame = data.trainval.copy()
        frame["_cl"] = labels
        print(f"\n{name}: cluster sizes = {np.bincount(labels).tolist()}")
        for feat in KEY_FEATURES:
            means = frame.groupby("_cl")[feat].mean().round(4)
            print(f"  {feat:24s} c0={means.get(0, float('nan')):8.4f}  c1={means.get(1, float('nan')):8.4f}")
    # Cross-tabulation of the two partitions (full trainval).
    ct = pd.crosstab(l_b54, l_v0)
    print("\n### Cross-tabulation Backbone54-cluster x V0-cluster (full trainval rows)")
    print(ct.to_string())


if __name__ == "__main__":
    main()
