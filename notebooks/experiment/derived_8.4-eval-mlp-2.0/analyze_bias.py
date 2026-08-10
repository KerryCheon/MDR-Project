#!/usr/bin/env python3
"""Systematic-bias diagnostic for derived_8.4-eval-mlp-2.0 (offline, no GPU).

mlp-1.2/1.3 documented that the 2-regime-96 MLPs carry a systematic positive
test bias with bias^2 ~ 10-17% of MSE (vs ~0.5% for the XGBoost 2-regime
reference), and that the bias scales with capacity (<200k params -> bias
~0.005, 1M+ -> ~0.020). Post-hoc val-fit calibration does NOT transfer
(documentd negative), so the 2.0 debias levers are architecture + training
(capacity control, SWA, mixup, grouped towers).

This script reports, per config and per family, the decomposition
    MSE = bias^2 + ubRMSE^2        (ubRMSE = unbiased RMSE)
    bias2_mse_share = bias^2 / MSE
from the sweep results and per-cluster test metrics, plus the mlp-1.3 median
for reference. The headline success criterion is a per-family median
bias2_mse_share < 5% (from 10-17% in 1.3's 96-family).

Outputs: bias_summary.csv + bias_by_cluster.csv + printed report.

Usage:
    python analyze_bias.py [--out .]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent

FAMILIES = ["2regime_96", "2regime_54", "2regime_mixed"]
HONEST_ARCHS = ("mlp", "fg", "plr")


def compute_bias_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """bias^2/MSE share per config from sweep_results.csv rows."""
    rows = []
    for _, r in sweep.iterrows():
        rmse = r.get("test_rmse")
        bias = r.get("test_bias")
        if rmse is None or bias is None or not np.isfinite(rmse) or not np.isfinite(bias):
            continue
        mse = float(rmse) ** 2
        bias2 = float(bias) ** 2
        rows.append({
            "family": r["family"],
            "config_id": r["config_id"],
            "architecture": r["architecture"],
            "n_seeds": r.get("n_seeds"),
            "test_r2": r.get("test_r2"),
            "test_rmse": rmse,
            "test_bias": bias,
            "bias2": bias2,
            "mse": mse,
            "bias2_mse_share": bias2 / mse if mse > 0 else float("nan"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family", "bias2_mse_share"]).reset_index(drop=True)
    return df


def compute_bias_by_cluster(exp_dir: Path, sweep: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster bias metrics from each config's aggregated meta.json."""
    rows = []
    for _, r in sweep.iterrows():
        meta_path = exp_dir / "models" / r["family"] / r["config_id"] / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for cl in ("0", "1"):
            cm = meta.get("per_cluster", {}).get(cl, {}).get("test")
            if cm is None:
                continue
            rmse, bias = float(cm["rmse"]), float(cm["bias"])
            mse = rmse ** 2
            rows.append({
                "family": r["family"],
                "config_id": r["config_id"],
                "architecture": r["architecture"],
                "cluster": int(cl),
                "n_test": int(meta["per_cluster"][cl].get("n_test", 0)),
                "test_r2": float(cm["r2"]),
                "test_rmse": rmse,
                "test_bias": bias,
                "bias2_mse_share": (bias ** 2) / mse if mse > 0 else float("nan"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family", "cluster", "bias2_mse_share"]).reset_index(drop=True)
    return df


def print_report(bias_df: pd.DataFrame, cluster_df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SYSTEMATIC-BIAS DIAGNOSTIC — derived_8.4-eval-mlp-2.0")
    print("=" * 78)
    print("\nbias^2/MSE share = squared pooled test bias / MSE (MSE = bias^2 + ubRMSE^2).")
    print("1.3 reference medians: 2regime_96 ~10-17%, 2regime_54 ~1%.")
    print("2.0 success criterion: per-family median < 5%.\n")

    print("### Per-family median bias^2/MSE share (honest architectures)")
    print("| family     | n_configs |   med_bias2_mse_share |   med_test_bias |   med_test_r2 |")
    print("|:-----------|----------:|----------------------:|----------------:|--------------:|")
    for fam in FAMILIES:
        sub = bias_df[(bias_df["family"] == fam) & bias_df["architecture"].isin(HONEST_ARCHS)]
        if sub.empty:
            continue
        print(f"| {fam} | {len(sub)} | {sub['bias2_mse_share'].median():.4f} | "
              f"{sub['test_bias'].median():.4f} | {sub['test_r2'].median():.4f} |")

    print("\n### Worst 8 configs by bias^2/MSE share (all architectures)")
    cols = ["family", "config_id", "architecture", "test_r2", "test_rmse", "test_bias", "bias2_mse_share"]
    print(bias_df.sort_values("bias2_mse_share", ascending=False).head(8)[cols].to_string(index=False), flush=True)

    print("\n### Best 8 configs by bias^2/MSE share (all architectures)")
    print(bias_df.sort_values("bias2_mse_share").head(8)[cols].to_string(index=False), flush=True)

    if not cluster_df.empty:
        print("\n### Per-cluster median bias^2/MSE share (honest architectures)")
        print("| family     | cluster |   med_bias2_mse_share |   med_test_bias |   med_test_r2 |")
        print("|:-----------|--------:|----------------------:|----------------:|--------------:|")
        for fam in FAMILIES:
            for cl in (0, 1):
                sub = cluster_df[(cluster_df["family"] == fam) & (cluster_df["cluster"] == cl)
                                 & cluster_df["architecture"].isin(HONEST_ARCHS)]
                if sub.empty:
                    continue
                print(f"| {fam} | {cl} | {sub['bias2_mse_share'].median():.4f} | "
                      f"{sub['test_bias'].median():.4f} | {sub['test_r2'].median():.4f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    sweep = pd.read_csv(args.out / "sweep_results.csv")
    bias_df = compute_bias_summary(sweep)
    cluster_df = compute_bias_by_cluster(args.out, sweep)

    bias_df.to_csv(args.out / "bias_summary.csv", index=False)
    cluster_df.to_csv(args.out / "bias_by_cluster.csv", index=False)
    print(f"[bias] wrote bias_summary.csv ({len(bias_df)} rows) + bias_by_cluster.csv ({len(cluster_df)} rows)")

    print_report(bias_df, cluster_df)


if __name__ == "__main__":
    main()
