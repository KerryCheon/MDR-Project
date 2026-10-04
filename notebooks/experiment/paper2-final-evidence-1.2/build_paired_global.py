#!/usr/bin/env python3
"""Aggregate paper2-final-evidence-1.2 Global_Single paired outputs.

Reads driver outputs (temporal_seed_summary.csv, loso_seed_station.csv) and
writes provenance-stamped paired tables that REPLACE the sensitivity-anchored
Global rows from derived_8.4-formal-eval-1.0:

    global_paired_temporal_seed.csv      30 rows, + evidence_role/source_harness
    global_paired_temporal_summary.csv   1 row (mean/sd/rmse over 30 seeds)
    global_paired_loso_seed_station.csv  35 rows, + evidence_role/source_harness
    global_paired_loso_summary.csv       1 row (station-mean over 5x7)

Also prints a cross-harness seed-paired check vs 1.0 Guarded/Backbone means
(read-only on ../paper2-final-evidence-1.0) for a quick sanity look.

Usage:
    uv run --project <repo>/notebooks --no-sync python build_paired_global.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
HARNESS = "paper2-final-evidence-1.2"
ROLE = "paired_in_harness"
EVIDENCE_10 = EXP_DIR.parent / "paper2-final-evidence-1.0"


def main() -> None:
    temporal = pd.read_csv(EXP_DIR / "temporal_seed_summary.csv")
    loso = pd.read_csv(EXP_DIR / "loso_seed_station.csv")

    g_temporal = temporal[temporal["strategy_name"] == "Global_Single"].copy()
    if len(g_temporal) != 30 or g_temporal["seed"].nunique() != 30:
        raise SystemExit(
            f"expected 30 distinct Global_Single temporal seeds, found {len(g_temporal)} rows"
        )
    g_temporal["evidence_role"] = ROLE
    g_temporal["source_harness"] = HARNESS
    g_temporal.to_csv(EXP_DIR / "global_paired_temporal_seed.csv", index=False)

    t_summary = pd.DataFrame([{
        "strategy_name": "Global_Single",
        "display_name": "Global (paired)",
        "evidence_role": ROLE,
        "source_harness": HARNESS,
        "n_seeds": 30,
        "mean_r2": float(g_temporal["r2"].mean()),
        "seed_sd_r2": float(g_temporal["r2"].std(ddof=1)),
        "mean_rmse": float(g_temporal["rmse"].mean()),
    }])
    t_summary.to_csv(EXP_DIR / "global_paired_temporal_summary.csv", index=False)

    g_loso = loso[loso["strategy_name"] == "Global_Single"].copy()
    if len(g_loso) != 35 or g_loso["seed"].nunique() != 5 or g_loso["station"].nunique() != 7:
        raise SystemExit(
            f"expected 35 Global_Single LOSO rows (5 seeds x 7 stations), found {len(g_loso)}"
        )
    g_loso["evidence_role"] = ROLE
    g_loso["source_harness"] = HARNESS
    g_loso.to_csv(EXP_DIR / "global_paired_loso_seed_station.csv", index=False)

    l_summary = pd.DataFrame([{
        "strategy_name": "Global_Single",
        "evidence_role": ROLE,
        "source_harness": HARNESS,
        "loso_mean_r2": float(g_loso["r2"].mean()),
        "loso_mean_rmse": float(g_loso["rmse"].mean()),
    }])
    l_summary.to_csv(EXP_DIR / "global_paired_loso_summary.csv", index=False)

    print(f"[1.2] temporal: n=30 mean_r2={t_summary.iloc[0]['mean_r2']:.6f} "
          f"sd={t_summary.iloc[0]['seed_sd_r2']:.6f} rmse={t_summary.iloc[0]['mean_rmse']:.6f}")
    print(f"[1.2] loso: 35 rows station-mean_r2={l_summary.iloc[0]['loso_mean_r2']:.6f} "
          f"rmse={l_summary.iloc[0]['loso_mean_rmse']:.6f}")

    # Cross-harness sanity check vs 1.0 Guarded/Backbone (same seeds, same protocol).
    seed_cmp = pd.read_csv(EVIDENCE_10 / "main_temporal_seed_comparison.csv")
    guard = seed_cmp[seed_cmp["strategy_name"] == "Guarded_Backbone54_k2"].set_index("seed")["r2"]
    back = seed_cmp[seed_cmp["strategy_name"] == "Clustering_Backbone54_k2"].set_index("seed")["r2"]
    g_by_seed = g_temporal.set_index("seed")["r2"]
    common = g_by_seed.index.intersection(guard.index)
    if len(common) == 30:
        d_guard = [(guard[s] - g_by_seed[s]) for s in common]
        d_back = [(back[s] - g_by_seed[s]) for s in common]
        print(f"[paired vs 1.0 Guarded] mean Guarded-Global R2 diff={statistics.mean(d_guard):+.6f} "
              f"(all {sum(d > 0 for d in d_guard)}/30 seeds Guarded higher: "
              f"{all(d > 0 for d in d_guard)})")
        print(f"[paired vs 1.0 Backbone] mean Backbone-Global R2 diff={statistics.mean(d_back):+.6f}")
    else:
        print(f"[warn] seed overlap with 1.0 is {len(common)}/30; skipping paired check", file=sys.stderr)

    print("[done] wrote global_paired_*.csv")


if __name__ == "__main__":
    main()
