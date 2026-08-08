#!/usr/bin/env python3
"""Overfitting-symptom analysis for derived_8.4-eval-mlp-1.3 (no retraining).

Quantifies the generalization failure modes of the MLP sweep directly from the
saved artifacts:

  1. train-fit vs held-out gap   — aux2020 RMSE (the 2020 slice of TRAIN,
     n=2519, seen by the model) vs val RMSE vs test RMSE, per family.
  2. capacity vs transfer        — n_params buckets vs median val/test R²:
     does extra capacity buy in-sample fit that does not transfer?
  3. val-overfitters             — residual nets: best val AND best train-fit,
     worst test (the 1.1 selection failure mode).
  4. per-epoch curve shape       — for each family's 2-seed val winner: test
     bottoms out early then RISES while train-fit (aux) keeps improving, and
     val stays flat -> early stopping on val cannot catch the overfitting.
  5. systematic bias on test     — MLP avg test bias vs XGBoost references.

Outputs: overfitting_summary.csv (per-family summary) + printed report.

Usage:
    python analyze_overfitting.py [--out .]

The report notebook (derived_8.4-eval-mlp-1.3.ipynb) imports compute_overfitting
and renders the same tables; README.md tables are copied from its stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]

FAMILIES = ["2regime_96", "2regime_54"]
XGB_REF_BASE = {  # pooled_bias of the eval-1.1 references (from its metrics_summary.csv)
    "2regime": 0.00648567,
    "global": 0.0105484,
}


def _load_sweep(exp_dir: Path) -> pd.DataFrame:
    return pd.read_csv(exp_dir / "sweep_results.csv")


def compute_overfitting(sweep: pd.DataFrame, exp_dir: Path) -> dict:
    """Return a dict of the symptom metrics (used by both CLI and notebook)."""
    out: dict = {}
    curve_rows: list[dict] = []

    for fam in FAMILIES:
        sub = sweep[(sweep["family"] == fam) & (sweep["architecture"] == "mlp")].dropna(subset=["test_r2"])
        res = sweep[(sweep["family"] == fam) & (sweep["architecture"] == "residual")].dropna(subset=["test_r2"])

        # --- 1. train-fit vs held-out gap ---
        out[f"{fam}_med_aux"] = float(sub["aux_rmse"].median())
        out[f"{fam}_med_val"] = float(sub["val_rmse"].median())
        out[f"{fam}_med_test"] = float(sub["test_rmse"].median())
        out[f"{fam}_train_val_ratio"] = float(sub["val_rmse"].median() / sub["aux_rmse"].median())

        # --- 2. capacity vs transfer ---
        sub2 = sub.copy()
        sub2["cap"] = pd.cut(sub2["n_params"], bins=[0, 2e5, 5e5, 1e6, 2e6],
                             labels=["<200k", "200-500k", "500k-1M", "1M+"])
        cap_rows = []
        for b, g in sub2.groupby("cap", observed=True):
            cap_rows.append({
                "family": fam, "capacity": str(b), "n_configs": len(g),
                "med_val_rmse": round(float(g["val_rmse"].median()), 4),
                "med_test_r2": round(float(g["test_r2"].median()), 4),
                "med_test_bias": round(float(g["test_bias"].median()), 4),
            })
        out[f"{fam}_capacity"] = cap_rows

        # --- 3. residual nets (val-overfitters) ---
        if not res.empty:
            out[f"{fam}_residuals"] = res[["config_id", "n_params", "val_rmse", "aux_rmse", "test_r2", "test_bias"]].to_dict("records")

        # --- 4. per-epoch curve shape for the 2-seed val winner (cluster 0) ---
        winner = sub.sort_values("val_rmse").iloc[0]["config_id"]
        curve_path = exp_dir / "models" / fam / winner / "seed_42" / "spec_0" / "curves.npy"
        if curve_path.exists():
            c = np.load(curve_path)  # [val, aux, test]
            val, aux, test = c[0], c[1], c[2]
            best_val_ep = int(np.nanargmin(val)) + 1
            best_test_ep = int(np.nanargmin(test)) + 1
            curve_rows.append({
                "family": fam, "config_id": winner,
                "aux_ep260": round(float(aux[min(len(aux), 260) - 1]), 4),
                "aux_ep100": round(float(aux[min(len(aux), 100) - 1]), 4),
                "val_plateau": round(float(val[min(len(val), 260) - 1]), 4),
                "test_min": round(float(np.nanmin(test)), 4),
                "test_min_epoch": best_test_ep,
                "test_at_best_val": round(float(test[best_val_ep - 1]), 4),
                "test_final": round(float(test[-1]), 4),
                "test_rise_after_min": round(float(test[-1] - np.nanmin(test)), 4),
            })
        out[f"{fam}_winner_curves"] = curve_rows

        # --- 5. systematic bias on test ---
        out[f"{fam}_med_test_bias"] = float(sub["test_bias"].median())

    out["curve_rows"] = curve_rows
    return out


def print_report(r: dict) -> None:
    """Render the symptom tables (used by both the CLI and the report notebook)."""
    print("=" * 78)
    print("OVERFITTING-SYMPTOM ANALYSIS — derived_8.4-eval-mlp-1.3 (from sweep artifacts)")
    print("=" * 78)

    print("\n### 1. Train-fit vs held-out gap (median RMSE over 2-regime MLP configs)")
    print("| family     |   aux2020 (train-fit) |   val |   test |   val/train ratio |")
    print("|:-----------|----------------------:|------:|-------:|------------------:|")
    for fam in FAMILIES:
        print(f"| {fam} | {r[f'{fam}_med_aux']:.4f} | {r[f'{fam}_med_val']:.4f} | "
              f"{r[f'{fam}_med_test']:.4f} | {r[f'{fam}_train_val_ratio']:.1f}x |")

    print("\n### 2. Capacity vs test transfer (median by n_params bucket)")
    print("| family     | capacity   |   n_configs |   med_val_rmse |   med_test_r2 |   med_test_bias |")
    print("|:-----------|:-----------|------------:|---------------:|--------------:|----------------:|")
    for fam in FAMILIES:
        for row in r[f"{fam}_capacity"]:
            print(f"| {row['family']} | {row['capacity']} | {row['n_configs']} | "
                  f"{row['med_val_rmse']:.4f} | {row['med_test_r2']:.4f} | {row['med_test_bias']:.4f} |")

    print("\n### 3. Residual nets — best in-sample, worst test (val-overfitters)")
    print("| family     | config_id |   n_params |   val_rmse |   aux_rmse |   test_r2 |   test_bias |")
    print("|:-----------|:----------|-----------:|-----------:|-----------:|----------:|------------:|")
    for fam in FAMILIES:
        for row in r.get(f"{fam}_residuals", []):
            print(f"| {fam} | {row['config_id']} | {row['n_params']} | {row['val_rmse']:.4f} | "
                  f"{row['aux_rmse']:.4f} | {row['test_r2']:.4f} | {row['test_bias']:.4f} |")

    print("\n### 4. Per-epoch curve shape for the 2-seed val winner (cluster-0 specialist)")
    print("| family     | config_id |   aux_ep100 |   aux_ep260 |   val_plateau |   test_min |   test_min_epoch |   test_at_best_val |   test_final |   test_rise_after_min |")
    print("|:-----------|:----------|------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|-------------:|----------------------:|")
    for row in r["curve_rows"]:
        print(f"| {row['family']} | {row['config_id']} | {row['aux_ep100']:.4f} | {row['aux_ep260']:.4f} | "
              f"{row['val_plateau']:.4f} | {row['test_min']:.4f} | {row['test_min_epoch']} | "
              f"{row['test_at_best_val']:.4f} | {row['test_final']:.4f} | {row['test_rise_after_min']:.4f} |")

    print("\n### 5. Systematic bias on test (MLP vs XGBoost references)")
    print(f"MLP median test bias — 2regime_96: {r['2regime_96_med_test_bias']:.4f}, "
          f"2regime_54: {r['2regime_54_med_test_bias']:.4f}")
    print(f"XGBoost references (eval-1.1) — 2-regime: {XGB_REF_BASE['2regime']:.4f}, "
          f"global: {XGB_REF_BASE['global']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    sweep = _load_sweep(args.out)
    r = compute_overfitting(sweep, args.out)
    print_report(r)

    # persist per-family summary
    rows = []
    for fam in FAMILIES:
        rows.append({
            "family": fam,
            "med_aux_rmse": r[f"{fam}_med_aux"],
            "med_val_rmse": r[f"{fam}_med_val"],
            "med_test_rmse": r[f"{fam}_med_test"],
            "val_over_train_ratio": r[f"{fam}_train_val_ratio"],
            "med_test_bias": r[f"{fam}_med_test_bias"],
        })
    pd.DataFrame(rows).to_csv(args.out / "overfitting_summary.csv", index=False)
    print(f"\n[overfit] wrote overfitting_summary.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
