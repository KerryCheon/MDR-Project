#!/usr/bin/env python3
"""Early-stopping-rule simulation on saved per-epoch curves (offline, no GPU).

1.2's overfitting analysis found the 2-regime-96 winner's TEST error bottoms
out at ~epoch 90 (rmse 0.0451) while VAL stays flat to epoch 260 (best-val
0.0531) — patience-60 selects a late epoch and the final test rmse is 0.0498.
This script replays alternative HONEST early-stopping / epoch-selection rules on
the saved per-epoch val/test RMSE curves (curves.npy: [val, aux, test]) of every
1.2 (or 1.3) job, and reports the pooled test RMSE each rule would have
achieved, vs the current patience-60 rule and an ORACLE (argmin test) bound.

Rules (all use only val/aux curves — test is never touched for selection):
  - patience60 (baseline): best epoch = argmin(val).  [current 1.2 behavior]
  - patienceN (N in {20, 40}): best epoch = argmin(val) but stopping at N; since
    best_model.pt is always the argmin-val checkpoint, the chosen epoch equals
    argmin(val) — reported for reference.
  - plateau(W, eps): stop at the FIRST epoch e >= W where the best val inside
    [e-W+1, e] has not beaten the running best by >= eps; chosen epoch = argmin
    val in [1, e]. Grid: W in {20, 40, 60}, eps in {1e-4, 3e-4}.
  - val_aux: chosen epoch = argmin((val+aux)/2).
  - oracle (reference only): argmin(test).

Output: stopping_<tag>_summary.csv (per config + per family-rule aggregates).

Usage:
  python analyze_stopping.py [--exp-dir <1.2|1.3 dir>] [--out <1.3 dir>] [--tag 12]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR_12 = Path(__file__).resolve().parent.parent / "derived_8.4-eval-mlp-1.2"
EXP_DIR_13 = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR_13))

CLUSTERS = ("0", "1")


def load_curves(spec_dir: Path):
    """curves.npy = 3-row stack [val, aux, test]; per-specialist (2-regime)."""
    p = spec_dir / "curves.npy"
    if not p.exists():
        return None
    return np.load(p)


def pooled_rmse(rmse_by_cluster: list[float], n_by_cluster: list[int]) -> float:
    total = sum(n_by_cluster)
    if total <= 0:
        return float("nan")
    return float(np.sqrt(sum((n / total) * r**2 for n, r in zip(n_by_cluster, rmse_by_cluster))))


def choose_epoch_plateau(val: np.ndarray, window: int, eps: float, min_epochs: int = 30) -> int:
    """First epoch e >= max(window, min_epochs) where the last `window` vals have
    not beaten the running best by >= eps. Returns chosen epoch = argmin val in [1, e]."""
    best = float("inf")
    for e in range(len(val)):
        v = val[e]
        if v < best:
            best = v
        if e + 1 < max(window, min_epochs):
            continue
        window_min = float(np.min(val[e + 1 - window : e + 1]))
        if best - window_min < eps:
            chosen = int(np.argmin(val[: e + 1])) + 1
            return chosen
    return int(np.argmin(val)) + 1


def analyze_exp(exp_dir: Path, out: Path, tag: str) -> pd.DataFrame:
    models_root = exp_dir / "models"
    rows = []
    for family_dir in sorted([p for p in models_root.iterdir() if p.is_dir()]):
        family = family_dir.name
        if not family.startswith("2regime"):
            continue
        for cdir in sorted([p for p in family_dir.iterdir() if p.is_dir()]):
            config_id = cdir.name
            agg_path = cdir / "meta.json"
            if not agg_path.exists():
                continue
            agg = json.loads(agg_path.read_text(encoding="utf-8"))
            if agg.get("config", {}).get("architecture", "mlp") != "mlp":
                continue
            # skip the broken EMA configs (inherited EMA trainer never converges
            # — documented negative; their curves are garbage and would skew the
            # rule aggregates).
            if agg.get("config", {}).get("ema", False):
                continue
            seeds = sorted(
                [int(p.name.split("_")[1]) for p in cdir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
            )
            for s in seeds:
                sdir = cdir / f"seed_{s}"
                seed_meta_path = sdir / "meta.json"
                if not seed_meta_path.exists():
                    continue
                seed_meta = json.loads(seed_meta_path.read_text(encoding="utf-8"))
                ns_test = [seed_meta["per_cluster"][cl]["n_test"] for cl in CLUSTERS]
                curves = []
                ok = True
                for cl in CLUSTERS:
                    c = load_curves(sdir / f"spec_{cl}")
                    if c is None:
                        ok = False
                        break
                    curves.append(c)
                if not ok:
                    continue
                # test rmse per cluster at each cluster's own chosen epoch (1-based)
                def test_rmse_at(epoch: int, curve: np.ndarray) -> float:
                    if epoch < 1 or epoch > curve.shape[1]:
                        return float("nan")
                    return float(curve[2][epoch - 1])

                rules = {"patience60": None, "patience20": None, "patience40": None,
                         "val_aux": None, "plateau_w20e1e-4": (20, 1e-4),
                         "plateau_w40e1e-4": (40, 1e-4), "plateau_w40e3e-4": (40, 3e-4),
                         "plateau_w60e1e-4": (60, 1e-4)}
                row = {"family": family, "config_id": config_id, "seed": s}
                for rule, params in rules.items():
                    epochs = []
                    for c in curves:
                        val = c[0]
                        if rule == "patience60" or rule.startswith("patience"):
                            epochs.append(int(np.argmin(val)) + 1)
                        elif rule == "val_aux":
                            aux = c[1]
                            if np.all(np.isnan(aux)):
                                epochs.append(int(np.argmin(val)) + 1)
                            else:
                                epochs.append(int(np.argmin(val + aux)) + 1)
                        else:  # plateau
                            w, eps = params
                            epochs.append(choose_epoch_plateau(val, w, eps))
                    rms = [test_rmse_at(e, c) for e, c in zip(epochs, curves)]
                    pooled = pooled_rmse(rms, ns_test)
                    row[f"{rule}_epoch"] = int(np.mean(epochs))
                    row[f"{rule}_test_rmse"] = pooled
                # oracle
                epochs_or = [int(np.argmin(c[2])) + 1 for c in curves]
                row["oracle_test_rmse"] = pooled_rmse(
                    [float(c[2][e - 1]) for c, e in zip(curves, epochs_or)], ns_test)
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / f"stopping_{tag}_summary.csv", index=False)
        print(f"[stop] wrote {out / f'stopping_{tag}_summary.csv'} ({len(df)} rows)", flush=True)
        # aggregates per family
        print("\n=== Stopping-rule aggregates (mean pooled test RMSE across configs/seeds) ===", flush=True)
        agg_rows = []
        for family in ["2regime_96", "2regime_54"]:
            sub = df[df["family"] == family]
            if sub.empty:
                continue
            for col in df.columns:
                if col.endswith("_test_rmse"):
                    vals = sub[col].dropna()
                    if vals.empty:
                        continue
                    agg_rows.append({
                        "family": family, "rule": col.replace("_test_rmse", ""),
                        "mean_test_rmse": float(vals.mean()),
                        "median_test_rmse": float(vals.median()),
                        "n": int(vals.shape[0]),
                    })
            print(f"\n--- {family} ---", flush=True)
            sub_agg = pd.DataFrame([r for r in agg_rows if r["family"] == family]).sort_values("mean_test_rmse")
            print(sub_agg.to_string(index=False), flush=True)
        pd.DataFrame(agg_rows).to_csv(out / f"stopping_{tag}_aggregate.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR_12,
                        help="experiment dir whose models/curves to analyze (default: 1.2)")
    parser.add_argument("--out", type=Path, default=EXP_DIR_13)
    parser.add_argument("--tag", default="12")
    args = parser.parse_args()
    analyze_exp(args.exp_dir, args.out, args.tag)


if __name__ == "__main__":
    main()
