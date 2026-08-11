#!/usr/bin/env python3
"""Val-year selection-reliability diagnostic for derived_8.4-eval-mlp-2.2.

NEW in 2.2 — the honest answer to 2.1's headline negative finding: the
mixed/54 families' val ranking is noisy (Spearman(val, test) = -0.309 /
-0.555 even at 3-seed aggregation), and more seeds on the same val split did
not fix it. This diagnostic splits the official val set (2021-2022) by YEAR
and asks which val year is the better proxy for test, and whether the
val-selected winner is stable under leave-one-val-year-out selection.

The selection RULE is NOT changed (protocol): winners stay 3-seed mean val
RMSE on the FULL official val. Everything here is diagnostic.

Inputs (all saved by the sweep):
  - artifacts/val_meta.npz       y_val / year / station of the full val set
  - artifacts/labels_val.npy     cluster label per full-val row
  - models/<fam>/<cid>/seed_<s>/spec_<c>/val_preds.npy   best-val preds of
    specialist c (NEW in mlp22: post-training eval-mode forward, same
    deployed weights as preds.npy). spec preds[i] <-> the i-th val row of
    cluster c (the per-cluster feature set's val frame is
    val.loc[labels==c].reset_index(drop=True)).
  - sweep_results.csv            3-seed mean test R2 per config

Output:
  - val_year_summary.csv         per-config per-year val RMSE (3-seed mean)
  - printed tables (top-10 per family, Spearman per year, winner stability)

Usage:
    uv run --no-sync python analyze_val_years.py [--exp-dir <dir>] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXP_DIR = Path(__file__).resolve().parent

FAMILIES = ["2regime_96", "2regime_54", "2regime_mixed"]
SEEDS = [42, 7, 123]
VAL_YEARS = [2021, 2022]


def _pooled_rmse(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def analyze_exp(exp_dir: Path, out: Path) -> pd.DataFrame:
    artifacts = exp_dir / "artifacts"
    val_meta = np.load(artifacts / "val_meta.npz", allow_pickle=True)
    y_val = val_meta["y_val"].astype(np.float64)
    year = val_meta["year"].astype(np.int64)
    labels_val = np.load(artifacts / "labels_val.npy").astype(np.int64)

    sweep = pd.read_csv(exp_dir / "sweep_results.csv")
    rows = []
    for family in FAMILIES:
        for _, r in sweep[sweep["family"] == family].iterrows():
            cid = r["config_id"]
            n_seeds = int(r["n_seeds"])
            per_year_preds: dict[int, list[np.ndarray]] = {yy: [] for yy in VAL_YEARS}
            for s in SEEDS[:n_seeds]:
                full_preds = np.full(len(y_val), np.nan, dtype=np.float64)
                ok = True
                for cl in ("0", "1"):
                    p = exp_dir / "models" / family / cid / f"seed_{s}" / f"spec_{cl}" / "val_preds.npy"
                    if not p.exists():
                        ok = False
                        break
                    spec_preds = np.load(p).astype(np.float64)
                    pos = np.where(labels_val == int(cl))[0]
                    if len(spec_preds) != len(pos):
                        ok = False
                        break
                    full_preds[pos] = spec_preds
                if not ok or np.isnan(full_preds).any():
                    continue
                for yy in VAL_YEARS:
                    mask = year == yy
                    per_year_preds[yy].append(full_preds[mask])
            if not per_year_preds[VAL_YEARS[0]]:
                continue
            row = {
                "family": family,
                "config_id": cid,
                "n_seeds": n_seeds,
                "val_rmse": r["val_rmse"],
                "test_r2": r["test_r2"],
            }
            for yy in VAL_YEARS:
                preds_list = per_year_preds[yy]
                row[f"val_{yy}_rmse"] = float(np.mean(
                    [_pooled_rmse(y_val[year == yy], p) for p in preds_list]))
                row[f"val_{yy}_n"] = int((year == yy).sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "val_year_summary.csv", index=False)
    return df


def print_tables(df: pd.DataFrame, out: Path) -> None:
    print("VAL-YEAR SELECTION-RELIABILITY DIAGNOSTIC — derived_8.4-eval-mlp-2.2")
    print("(diagnostic only; the deployed selection rule stays 3-seed mean val RMSE on the full official val)")
    print()

    # 1) top-10 by full-val RMSE with the per-year RMSEs
    for family in FAMILIES:
        sub = df[df["family"] == family].sort_values("val_rmse").head(10)
        print(f"#### {family} — top-10 by full-val RMSE (with per-year val RMSE)")
        cols = ["config_id", "n_seeds", "val_rmse", "val_2021_rmse", "val_2022_rmse", "test_r2"]
        print(sub[cols].to_markdown(index=False))
        print()

    # 2) Spearman of each val signal vs test (over the phase-1 pool)
    print("#### Spearman(val signal, test R2) per family (phase-1 pool, 3-seed aggregation)")
    rows = []
    for family in FAMILIES:
        sub = df[df["family"] == family].dropna(subset=["val_rmse", "test_r2"])
        for sig in ("val_rmse", "val_2021_rmse", "val_2022_rmse"):
            s = sub.dropna(subset=[sig])
            if len(s) < 4:
                continue
            rho, p = spearmanr(s[sig], s["test_r2"])
            rows.append({"family": family, "signal": sig, "n_configs": len(s),
                         "spearman": rho, "p_value": p})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()

    # 3) winner stability under leave-one-val-year-out selection
    print("#### Winner stability under leave-one-val-year-out selection (3-seed means)")
    rows = []
    for family in FAMILIES:
        sub = df[df["family"] == family].dropna(subset=["val_rmse"])
        for sig in ("val_rmse", "val_2021_rmse", "val_2022_rmse"):
            s = sub.dropna(subset=[sig]).sort_values(sig)
            if s.empty:
                continue
            w = s.iloc[0]
            rows.append({"family": family, "selected_by": sig,
                         "winner": w["config_id"], "winner_val": w[sig],
                         "winner_test_r2": w["test_r2"]})
    print(pd.DataFrame(rows).to_markdown(index=False))
    print()

    # 4) agreement: does the full-val winner match the test-best?
    print("#### Full-val winner vs test-best (reporting only — test-best is leakage)")
    rows = []
    for family in FAMILIES:
        sub = df[df["family"] == family].dropna(subset=["val_rmse", "test_r2"])
        if sub.empty:
            continue
        w = sub.sort_values("val_rmse").iloc[0]
        tb = sub.sort_values("test_r2", ascending=False).iloc[0]
        rows.append({"family": family,
                     "val_winner": w["config_id"], "val_winner_test_r2": w["test_r2"],
                     "test_best": tb["config_id"], "test_best_r2": tb["test_r2"],
                     "match": w["config_id"] == tb["config_id"]})
    print(pd.DataFrame(rows).to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    df = analyze_exp(args.exp_dir, args.out)
    print(f"[val-years] wrote {args.out / 'val_year_summary.csv'} ({len(df)} rows)", flush=True)
    print_tables(df, args.out)


if __name__ == "__main__":
    main()
