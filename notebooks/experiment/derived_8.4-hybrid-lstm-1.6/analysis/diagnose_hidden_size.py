"""
Diagnostic for derived_8.4-hybrid-lstm-1.6: why hybrids don't beat tabular-only,
and why H20 is the best hidden size.

Computes, per hidden size H and representation (ctx/hh/hp):
  1. LSTM-only test metrics (from artifacts/h{H}/lstm_metrics.json)
  2. |corr| of each repr feature with the target (signal)
  3. max |corr| of each repr feature with any of the 54 backbone features
     (redundancy proxy)
  4. mean R^2 of regressing each repr feature on the 54 backbone features
     (linear redundancy: how much of the repr is already in the tabular set)
  5. effective rank (n components for 95% variance) of the repr matrix
  6. XGBoost gain-based importance share by feature type (tabular vs repr)
     for the Clustering_V0_k2 hybrid models (cluster-0 and cluster-1 boosters)

Run from notebooks/:  uv run --project . python experiment/derived_8.4-hybrid-lstm-1.6/analysis/diagnose_hidden_size.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP_DIR))

from eval_hybrid.data import load_hybrid_experiment_data  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

with open(EXP_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)
BACKBONE = list(config["shared_backbone_54"])
TARGET = str(config["data"]["target"])
HIDDEN_SIZES = [40, 20, 16, 8, 4]
REPRS = ["ctx", "hh", "hp"]


def lin_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R^2 of ordinary least squares y ~ X (with intercept), complete cases only."""
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(y) < X.shape[1] + 5:
        return float("nan")
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def effective_rank95(X: np.ndarray) -> int:
    X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s**2 / np.sum(s**2)
    return int(np.searchsorted(np.cumsum(var), 0.95) + 1)


def gain_share_by_type(model_paths: list[Path], repr_prefix: str) -> dict[str, float]:
    """Aggregate XGBoost gain importance by feature-type (tabular vs repr) across boosters."""
    gain = {}
    for path in model_paths:
        if not path.exists():
            continue
        m = XGBRegressor()
        m.load_model(str(path))
        m.get_booster().set_param({"tree_method": "hist", "device": "cpu", "n_jobs": 1})
        for feat, g in m.get_booster().get_score(importance_type="gain").items():
            gain[feat] = gain.get(feat, 0.0) + g
    tab = sum(g for f, g in gain.items() if not f.startswith(repr_prefix))
    rep = sum(g for f, g in gain.items() if f.startswith(repr_prefix))
    tot = tab + rep
    return {"tabular": tab / tot, "repr": rep / tot} if tot > 0 else {"tabular": 0.0, "repr": 0.0}


def main():
    out = []
    print("=" * 100)
    print("LSTM-only quality + representation redundancy per hidden size")
    print("=" * 100)
    print(f"{'H':>3} {'repr':>4} {'lstm_test_r2':>13} {'mean|corr_y|':>13} "
          f"{'max|corr_y|':>11} {'mean max|corr_b|':>15} {'mean linR2_b':>12} {'dead_frac':>10} {'rank95':>7} {'n_cols':>6}")
    for h in HIDDEN_SIZES:
        with open(EXP_DIR / f"artifacts/h{h}/lstm_metrics.json") as f:
            lstm = json.load(f)
        lstm_r2 = lstm["test"]["r2"]
        for rtype in REPRS:
            data = load_hybrid_experiment_data(PROJECT_ROOT, EXP_DIR, config, repr_type=rtype, hidden_size=h)
            tv = data.trainval
            rcols = data.repr_feature_cols
            y = tv[TARGET].to_numpy(dtype=float)
            R = tv[rcols].to_numpy(dtype=float)
            B = tv[BACKBONE].to_numpy(dtype=float)

            corr_y = pd.DataFrame(R, columns=rcols).corrwith(pd.Series(y, index=tv.index)).abs()
            corr_b = tv[rcols].apply(
                lambda c: tv[BACKBONE].corrwith(c).abs().max(), axis=0
            ).to_numpy(dtype=float)
            dead_frac = float(np.mean(np.nanstd(R, axis=0) < 1e-6))
            lin_r2s = np.array([lin_r2(B, R[:, i]) for i in range(R.shape[1])])
            rank95 = effective_rank95(R)

            out.append({
                "H": h, "repr": rtype, "n_cols": R.shape[1],
                "lstm_test_r2": lstm_r2,
                "mean_abs_corr_y": float(np.mean(np.abs(corr_y))),
                "max_abs_corr_y": float(np.max(np.abs(corr_y))),
                "mean_max_abs_corr_backbone": float(np.nanmean(corr_b)),
                "mean_linR2_backbone": float(np.nanmean(lin_r2s)),
                "dead_unit_frac": dead_frac,
                "rank95": rank95,
            })
            print(f"{h:>3} {rtype:>4} {lstm_r2:>13.4f} {np.mean(np.abs(corr_y)):>13.4f} "
                  f"{np.max(np.abs(corr_y)):>11.4f} {np.nanmean(corr_b):>15.3f} {np.nanmean(lin_r2s):>12.3f} "
                  f"{dead_frac:>11.2f} {rank95:>7} {R.shape[1]:>6}")

    print()
    print("=" * 100)
    print("XGBoost gain-importance share by feature type (Clustering_V0_k2 models)")
    print("=" * 100)
    print(f"{'H':>3} {'repr':>4} {'gain% tabular':>14} {'gain% repr':>11}   (cluster boosters)")
    for rtype in REPRS:
        for h in HIDDEN_SIZES:
            paths = [
                EXP_DIR / "models" / f"Clustering_V0_k2_h{h}_{rtype}_cluster_0.json",
                EXP_DIR / "models" / f"Clustering_V0_k2_h{h}_{rtype}_cluster_1.json",
            ]
            share = gain_share_by_type(paths, rtype + "_")
            tab = share.get("tabular", 0.0) * 100
            rep = 100.0 - tab
            print(f"{h:>3} {rtype:>4} {tab:>14.1f} {rep:>11.1f}")

    df = pd.DataFrame(out)
    df.to_csv(EXP_DIR / "artifacts" / "hidden_size_diagnostic.csv", index=False)
    print(f"\n[Saved] {EXP_DIR / 'artifacts' / 'hidden_size_diagnostic.csv'}")


if __name__ == "__main__":
    main()
