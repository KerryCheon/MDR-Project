#!/usr/bin/env python3
"""SWA re-test diagnostic for derived_8.4-eval-mlp-2.1 (offline, no GPU).

2.0's SWA section documented a negative: with the 2.0 recipe (swa_start_frac
0.6, equal-weight average over epochs 240-400, BN recalibrated before each
SWA-val eval), no SWA snapshot ever beat the live best on val, and the
BN-recalibration pass leaked RNG into the live trajectory (the "gains are
live-trajectory artifacts" caveat). 2.1 applies the two prescribed fixes:

  (a) swa_start_frac is a swept knob {0.7, 0.75, 0.8, 0.85};
  (b) the BN recalibration runs inside mlp21._rng_guard(), so a swa job's
      LIVE trajectory is bit-identical to its swa=false anchor.

This script reads the per-seed per-cluster metas (the aggregated meta.json
does not carry the SWA fields) and:

  1. per (family, config, seed, cluster): live best val RMSE vs SWA-snapshot
     best val RMSE, and which one was deployed (val_rmse_live / val_rmse_swa /
     deployed in the seed meta);
  2. deployment counts: how many (seed, specialist) jobs deployed SWA, and on
     which configs — the headline "did the fair re-test deploy SWA?" answer;
  3. bit-identity stack check: for every swa config whose anchor (same id
     without the `_swa*` suffix) exists in the same family, max |diff| between
     the swa config's LIVE val curve and the anchor's val curve (both
     curves.npy[0]). With the RNG guard this must be 0 (or < 1e-12); any
     nonzero diff means the guard is broken and swa gains are again
     un-attributable.

Outputs: swa_summary.csv (per-config deployment table) + printed report.

Usage:
    python analyze_swa.py [--out .]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent

FAMILIES = ["2regime_96", "2regime_54", "2regime_mixed"]
CLUSTERS = ("0", "1")

_SWA_SUFFIX = re.compile(r"_swa(\d{3}|)$")


def anchor_id(config_id: str) -> str:
    """The id of the non-SWA anchor: strip the trailing `_swa...` suffix."""
    return _SWA_SUFFIX.sub("", config_id)


def load_seed_metas(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        fam_dir = exp_dir / "models" / fam
        if not fam_dir.exists():
            continue
        for cdir in sorted(fam_dir.iterdir()):
            if not cdir.is_dir():
                continue
            for sdir in sorted([p for p in cdir.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
                seed = int(sdir.name.split("_")[1])
                smeta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
                is_swa = bool(smeta.get("config", {}).get("swa", False))
                for cl in CLUSTERS:
                    cm = smeta.get("per_cluster", {}).get(cl)
                    if cm is None:
                        continue
                    rows.append({
                        "family": fam,
                        "config_id": cdir.name,
                        "seed": seed,
                        "cluster": int(cl),
                        "swa": is_swa,
                        "swa_start_frac": smeta.get("config", {}).get("swa_start_frac", None),
                        "val_rmse_live": cm.get("val_rmse_live"),
                        "val_rmse_swa": cm.get("val_rmse_swa"),
                        "deployed": cm.get("deployed", "live"),
                        "swa_best_epoch": cm.get("swa_best_epoch", 0),
                        "best_epoch": cm.get("best_epoch", 0),
                        "test_r2_cl": cm.get("test", {}).get("r2"),
                    })
    df = pd.DataFrame(rows)
    return df


def bit_identity_check(exp_dir: Path, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Compare each swa config's LIVE val curve to its non-SWA anchor's val curve."""
    rows = []
    for _, r in meta_df[meta_df["swa"]].drop_duplicates(["family", "config_id", "seed"]).iterrows():
        fam, cid, seed = r["family"], r["config_id"], int(r["seed"])
        anc = anchor_id(cid)
        if anc == cid:
            continue  # no anchor to compare against
        for cl in CLUSTERS:
            swa_curve_path = exp_dir / "models" / fam / cid / f"seed_{seed}" / f"spec_{cl}" / "curves.npy"
            anc_curve_path = exp_dir / "models" / fam / anc / f"seed_{seed}" / f"spec_{cl}" / "curves.npy"
            if not (swa_curve_path.exists() and anc_curve_path.exists()):
                continue
            swa_live = np.load(swa_curve_path)[0]     # live val curve of the swa job
            anc_live = np.load(anc_curve_path)[0]     # anchor val curve
            n = min(len(swa_live), len(anc_live))
            if n == 0:
                continue
            diff = float(np.max(np.abs(swa_live[:n] - anc_live[:n])))
            rows.append({
                "family": fam, "config_id": cid, "anchor": anc, "seed": seed,
                "cluster": int(cl), "n_epochs_compared": n, "max_abs_diff": diff,
                "bit_identical": bool(diff < 1e-12),
            })
    return pd.DataFrame(rows)


def print_report(meta_df: pd.DataFrame, ident_df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SWA RE-TEST DIAGNOSTIC — derived_8.4-eval-mlp-2.1 (RNG-guarded, swept start frac)")
    print("=" * 78)
    swa = meta_df[meta_df["swa"]]
    if swa.empty:
        print("No SWA configs found in the sweep artifacts.")
        return

    print("\n### Per-config SWA deployment (per-seed per-cluster live vs SWA val)")
    def _swa_mean(s: pd.Series) -> float:
        # specialists that early-stopped before swa_start_epoch keep inf in
        # their per-seed meta; mean over the started ones only, else NaN.
        vals = [float(v) for v in s if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    agg = (swa.groupby(["family", "config_id"])
              .agg(n_seeds=("seed", "nunique"),
                   n_specs=("cluster", "size"),
                   n_deployed_swa=("deployed", lambda s: int((s == "swa").sum())),
                   val_rmse_live=("val_rmse_live", "mean"),
                   val_rmse_swa=("val_rmse_swa", _swa_mean),
                   swa_start_frac=("swa_start_frac", "first"))
              .reset_index())

    def _fmt(v: float) -> str:
        return "n/a" if not np.isfinite(v) else f"{v:.5f}"

    print("| family | config_id | swa_start_frac | n_seeds | n_specs | specs_deployed_swa | val_rmse_live | val_rmse_swa |")
    print("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in agg.sort_values(["family", "config_id"]).iterrows():
        print(f"| {r['family']} | {r['config_id']} | {r['swa_start_frac']} | {r['n_seeds']} | {r['n_specs']} | "
              f"{r['n_deployed_swa']} | {_fmt(r['val_rmse_live'])} | {_fmt(r['val_rmse_swa'])} |")
    total_specs = int(swa.shape[0])
    n_deployed = int((swa["deployed"] == "swa").sum())
    print(f"\n**Deployment verdict:** {n_deployed}/{total_specs} (seed, specialist) jobs deployed the SWA "
          f"snapshot. {'SWA beats the live best on val for at least one job — the fair re-test is positive.' if n_deployed else 'No SWA snapshot beat the live best on val — the equal-weight recipe remains a documented negative even with the two 2.1 fixes.'}")

    if ident_df.empty:
        print("\n### Bit-identity stack check: no (swa config, anchor) pairs with curves found.")
    else:
        n_ident = int(ident_df["bit_identical"].sum())
        print(f"\n### Bit-identity stack check (RNG guard proof)")
        print(f"Live val curve of a swa job vs its non-SWA anchor's val curve (same seed, same cluster): "
              f"{n_ident}/{len(ident_df)} pairs bit-identical (max|diff| < 1e-12).")
        bad = ident_df[~ident_df["bit_identical"]]
        if not bad.empty:
            print("NON-IDENTICAL pairs (the RNG guard is broken for these):")
            print(bad.to_string(index=False))
        else:
            print("All compared pairs are bit-identical — the live trajectory is untouched by SWA "
                  "bookkeeping, so any `_swa*` gain is attributable to SWA, not RNG drift.")
        worst = ident_df.sort_values("max_abs_diff", ascending=False).head(5)
        print("\nWorst 5 pairs by max|diff|:")
        print(worst.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    meta_df = load_seed_metas(args.out)
    if meta_df.empty:
        print("[swa] no seed metas found — run the sweep first")
        return
    ident_df = bit_identity_check(args.out, meta_df)

    meta_df.to_csv(args.out / "swa_seed_meta.csv", index=False)
    ident_df.to_csv(args.out / "swa_bit_identity.csv", index=False)
    print(f"[swa] wrote swa_seed_meta.csv ({len(meta_df)} rows) + swa_bit_identity.csv ({len(ident_df)} rows)")
    print_report(meta_df, ident_df)


if __name__ == "__main__":
    main()
