#!/usr/bin/env python3
"""Early-stopping-rule simulation on saved per-epoch curves (offline, no GPU).

1.2's overfitting analysis found the 2-regime-96 winner's TEST error bottoms
out at ~epoch 90 (rmse 0.0451) while VAL stays flat to epoch 260 (best-val
0.0531) — patience-60 selects a late epoch and the final test rmse is 0.0498.
1.3 replayed alternative HONEST early-stopping / epoch-selection rules on the
saved per-epoch val/test RMSE curves (curves.npy: [val, aux, test]) and
confirmed no honest rule beats patience-60 (plateau rules stop too early).
This 2.0 version additionally replays the SWA rule on the SWA snapshot curves
(curves_swa.npy: [swa_val, swa_aux, swa_test], written by the mlp23 trainer (byte-identical to mlp22)
when swa=true) — SWA is the 2.0 mechanism for smoothing the flat-val region,
so the replay checks it against patience-60 honestly.

Rules (all use only val/aux curves — test is never touched for selection):
  - patience60 (baseline): best epoch = argmin(val).  [current behavior]
  - patienceN (N in {20, 40}): reported for reference (same checkpoint).
  - plateau(W, eps): stop at the FIRST epoch e >= W where the best val inside
    [e-W+1, e] has not beaten the running best by >= eps; chosen epoch = argmin
    val in [1, e]. Grid: W in {20, 40, 60}, eps in {1e-4, 3e-4}.
  - val_aux: chosen epoch = argmin((val+aux)/2).
  - swa_val: chosen epoch = argmin(swa_val curve) (uses curves_swa.npy when
    present; falls back to the live curve otherwise). NEW in 2.0.
  - oracle (reference only): argmin(test).

Output: stopping_<tag>_summary.csv (per config + per family-rule aggregates).

Usage:
  python analyze_stopping.py [--exp-dir <dir>] [--out <dir>] [--tag 22]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

CLUSTERS = ("0", "1")
FAMILIES = ("2regime_96", "2regime_54", "2regime_mixed")
HONEST_ARCHS = ("mlp", "fg", "plr")


def load_curves(spec_dir: Path, swa: bool = False):
    """curves.npy = 3-row stack [val, aux, test]; curves_swa.npy same layout for
    the SWA snapshot. Per-specialist (2-regime)."""
    p = spec_dir / ("curves_swa.npy" if swa else "curves.npy")
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


def _rule_epoch(rule: str, params, curve: np.ndarray, swa_curve) -> int:
    """Chosen epoch (1-based) for one specialist under `rule`."""
    val = curve[0]
    if rule.startswith("patience"):
        return int(np.argmin(val)) + 1
    if rule == "val_aux":
        aux = curve[1]
        if np.all(np.isnan(aux)):
            return int(np.argmin(val)) + 1
        return int(np.argmin(val + aux)) + 1
    if rule == "swa_val":
        if swa_curve is not None and swa_curve.shape[0] >= 1:
            sval = swa_curve[0]
            if np.all(np.isnan(sval)):
                return int(np.argmin(val)) + 1
            return int(np.argmin(sval)) + 1
        return int(np.argmin(val)) + 1
    # plateau
    w, eps = params
    return choose_epoch_plateau(val, w, eps)


def _test_rmse_at(epoch: int, curve: np.ndarray, swa_curve) -> float:
    """Test RMSE of the curve family at `epoch` (1-based); SWA test curve when available."""
    if epoch < 1:
        return float("nan")
    c = swa_curve if (swa_curve is not None and swa_curve.shape[0] >= 3 and epoch <= swa_curve.shape[1]) else curve
    if epoch > c.shape[1]:
        return float("nan")
    return float(c[2][epoch - 1])


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
            if agg.get("config", {}).get("architecture", "mlp") not in HONEST_ARCHS:
                continue
            # skip the broken legacy EMA configs (documented negative; their
            # curves are garbage and would skew the rule aggregates).
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
                curves, swa_curves = [], []
                ok = True
                for cl in CLUSTERS:
                    c = load_curves(sdir / f"spec_{cl}")
                    sc = load_curves(sdir / f"spec_{cl}", swa=True)
                    if c is None:
                        ok = False
                        break
                    curves.append(c)
                    swa_curves.append(sc)
                if not ok:
                    continue

                rules = {"patience60": None, "patience20": None, "patience40": None,
                         "val_aux": None, "swa_val": None,
                         "plateau_w20e1e-4": (20, 1e-4),
                         "plateau_w40e1e-4": (40, 1e-4), "plateau_w40e3e-4": (40, 3e-4),
                         "plateau_w60e1e-4": (60, 1e-4)}
                row = {"family": family, "config_id": config_id, "seed": s,
                       "swa": bool(agg.get("config", {}).get("swa", False))}
                for rule, params in rules.items():
                    epochs = [_rule_epoch(rule, params, c, sc)
                              for c, sc in zip(curves, swa_curves)]
                    rms = [_test_rmse_at(e, c, sc) for e, c, sc in zip(epochs, curves, swa_curves)]
                    row[f"{rule}_epoch"] = float(np.mean(epochs))
                    row[f"{rule}_test_rmse"] = pooled_rmse(rms, ns_test)
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
        for family in FAMILIES:
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
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR,
                        help="experiment dir whose models/curves to analyze (default: this dir)")
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    parser.add_argument("--tag", default="22")
    args = parser.parse_args()
    analyze_exp(args.exp_dir, args.out, args.tag)


if __name__ == "__main__":
    main()
