#!/usr/bin/env python3
"""Selection-reliability diagnostic for derived_8.4-eval-mlp-2.3 (offline, no GPU).

2.0's selection section documented that the mixed family's val ranking is
noisy: Spearman(val_rmse, test_r2) = -0.455 (p=0.187) vs +0.549 (54) and
-0.042 (96), and the val top-2 configs (fg) underperformed the val-3rd on
test. 2.2's structural mitigations: no fg/plr in the pool (the main 2.0
source of val/test disagreement) and 3-seed winner selection (phases 1-2-3).

This script quantifies how much the extra seeds actually bought:

  1. Spearman(val_rmse, test_r2) per family at 1-seed, 2-seed and 3-seed
     aggregation — does the val ranking become more predictive of test as
     seeds accumulate? (The depths cover different config subsets — phases
     2/3 ran only the val-selected top-N/M — so depth-to-depth comparisons
     are indicative, not controlled; the n_configs column makes the subsets
     auditable.)
  2. Seed consistency: per-config |val(s42) - val(s7)| and |val(s42) -
     val(s123)| (per-seed val RMSE columns in sweep_results.csv). Large
     single-seed swings are the noise floor the multi-seed selection removes.
  3. Phase stability: the winner by 1-seed val vs 2-seed val vs 3-seed val —
     does the selection flip between phases, and does the 3-seed winner
     differ from the 2-seed one 2.0 would have picked?

Outputs: selection_summary.csv + printed report.

Usage:
    python analyze_selection.py [--out .]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXP_DIR = Path(__file__).resolve().parent

FAMILIES = ["2regime_96", "2regime_54", "2regime_mixed"]
HONEST_ARCHS = ("mlp", "fg", "plr")
SEEDS = (42, 7, 123)


def _spearman(sub: pd.DataFrame, metric_col: str) -> tuple[float, float, int]:
    valid = sub.dropna(subset=[metric_col, "test_r2"])
    if len(valid) < 8:
        return float("nan"), float("nan"), int(len(valid))
    rho, p = spearmanr(valid[metric_col], valid["test_r2"])
    return float(rho), float(p), int(len(valid))


def analyze_selection(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        sub = sweep[(sweep["family"] == fam) & sweep["architecture"].isin(HONEST_ARCHS)].copy()
        if sub.empty:
            continue
        # 1-seed / 2-seed / 3-seed val RMSE columns
        sub["val_rmse_1s"] = sub["seed42_val_rmse"]
        # 2-seed mean where both seeds exist, else NaN
        two = sub[["seed42_val_rmse", "seed7_val_rmse"]].dropna()
        sub["val_rmse_2s"] = two.mean(axis=1).reindex(sub.index)
        three = sub[["seed42_val_rmse", "seed7_val_rmse", "seed123_val_rmse"]].dropna()
        sub["val_rmse_3s"] = three.mean(axis=1).reindex(sub.index)

        for label, col in [("1-seed (42)", "val_rmse_1s"), ("2-seed (42,7)", "val_rmse_2s"),
                           ("3-seed (42,7,123)", "val_rmse_3s")]:
            rho, p, n = _spearman(sub, col)
            rows.append({"family": fam, "aggregation": label, "n_configs": n,
                         "spearman_val_test": rho, "p_value": p})

        # seed consistency (median |delta| val RMSE between seeds, per config)
        for a, b in [(42, 7), (42, 123), (7, 123)]:
            ca, cb = f"seed{a}_val_rmse", f"seed{b}_val_rmse"
            if ca not in sub.columns or cb not in sub.columns:
                continue
            d = (sub[ca] - sub[cb]).abs().dropna()
            if d.empty:
                continue
            rows.append({"family": fam, "aggregation": f"seed_consistency|{a}-{b}",
                         "n_configs": int(d.shape[0]), "median_abs_delta_val": float(d.median()),
                         "mean_abs_delta_val": float(d.mean())})

        # phase stability: winner at each aggregation depth
        for label, col in [("1-seed (42)", "val_rmse_1s"), ("2-seed (42,7)", "val_rmse_2s"),
                           ("3-seed (42,7,123)", "val_rmse_3s")]:
            d = sub.dropna(subset=[col])
            if d.empty:
                continue
            w = d.sort_values(col).iloc[0]
            rows.append({"family": fam, "aggregation": f"winner|{label}",
                         "config_id": w["config_id"], "val_rmse": w[col],
                         "test_r2": w["test_r2"]})
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SELECTION-RELIABILITY DIAGNOSTIC — derived_8.4-eval-mlp-2.3 (3-seed selection)")
    print("=" * 78)
    print("\n### Spearman(val_rmse, test_r2) by aggregation depth")
    print("| family | aggregation | n_configs | spearman | p |")
    print("|:---|---:|---:|---:|---:|")
    for _, r in df[df["aggregation"].str.startswith(("1-seed (", "2-seed (", "3-seed ("))].iterrows():
        print(f"| {r['family']} | {r['aggregation']} | {r['n_configs']} | {r['spearman_val_test']:+.3f} | {r['p_value']:.3f} |")
    print("(n_configs differs across depths — the 2/3-seed sets are the val-selected top-N/M subsets, so")
    print("depth-to-depth Spearman comparisons are indicative, not controlled.)")

    print("\n### Seed consistency (median |delta| per-config val RMSE between seeds)")
    print("| family | pair | n_configs | median | mean |")
    print("|:---|---:|---:|---:|---:|")
    for _, r in df[df["aggregation"].str.startswith("seed_consistency")].iterrows():
        print(f"| {r['family']} | {r['aggregation'].split('|')[1]} | {r['n_configs']} | "
              f"{r['median_abs_delta_val']:.5f} | {r['mean_abs_delta_val']:.5f} |")

    print("\n### Phase stability — winner at each seed depth (test R2 for reference)")
    print("| family | depth | config_id | val_rmse | test_r2 |")
    print("|:---|---:|---|---:|---:|")
    for _, r in df[df["aggregation"].str.startswith("winner|")].iterrows():
        depth = r["aggregation"].split("|")[1]
        print(f"| {r['family']} | {depth} | {r['config_id']} | {r['val_rmse']:.5f} | {r['test_r2']:.4f} |")
    print("\n(If the 3-seed winner differs from the 1-seed one, the extra seeds changed the "
          "selection — the diagnostic makes that explicit rather than hiding it.)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    sweep = pd.read_csv(args.out / "sweep_results.csv")
    df = analyze_selection(sweep)
    df.to_csv(args.out / "selection_summary.csv", index=False)
    print(f"[selection] wrote selection_summary.csv ({len(df)} rows)")
    print_report(df)


if __name__ == "__main__":
    main()
