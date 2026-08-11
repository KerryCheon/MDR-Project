#!/usr/bin/env python3
"""Cross-version anchor check: 2.3 (v10) vs 2.2 — bit-identical curves?

Offline comparison of the shared non-SWA anchor configs' saved per-epoch
curves between derived_8.4-eval-mlp-2.2 (v9, its own H100 node) and this
experiment (v10). The two runs used identical hyperparameters, data and — by
construction — the identical training path (the mlp23 trainer is byte-
identical to mlp22: 2.3 makes NO training-path change). If max|diff| = 0 the
v10 run reproduced 2.2's curves bit-identically on a (different) node — the
same evidence chain as 2.1 vs 2.0 and 2.2 vs 2.1 (both were exact).

One anchor per family (seed 42, cluster-0 specialist), all in-grid as
regular 2.3 configs (the 2.2 val-selected winners):
  - 2regime_54  / w448x448x448_d0.3_huber0.1_gelu_lr1e-3 (2.2 54 val winner)
  - 2regime_mixed / w512x512x512_d0.3_huber0.03_lr1e-3 (2.2 mixed val winner)
  - 2regime_96  / w512x512x512_d0.3_lr1e-3          (2.2 96 val winner)

Result is written to artifacts/anchor_vs_22_comparison.json (committed
evidence for the README claim).

Usage:
    uv run --no-sync python compare_anchor_vs_2.2.py [--out .]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent

ANCHORS = [
    {"family": "2regime_54", "config_id": "w448x448x448_d0.3_huber0.1_gelu_lr1e-3", "seed": 42, "spec": 0},
    {"family": "2regime_mixed", "config_id": "w512x512x512_d0.3_huber0.03_lr1e-3", "seed": 42, "spec": 0},
    {"family": "2regime_96", "config_id": "w512x512x512_d0.3_lr1e-3", "seed": 42, "spec": 0},
]
ROWS = {0: "val", 1: "aux", 2: "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    results = {}
    all_identical = True
    for anchor in ANCHORS:
        f, cid, seed, spec = anchor["family"], anchor["config_id"], anchor["seed"], anchor["spec"]
        p23 = EXP_DIR / "models" / f / cid / f"seed_{seed}" / f"spec_{spec}" / "curves.npy"
        p22 = EXP_DIR.parent / "derived_8.4-eval-mlp-2.2" / "models" / f / cid / f"seed_{seed}" / f"spec_{spec}" / "curves.npy"
        if not p23.exists() or not p22.exists():
            raise SystemExit(f"curves not found:\n  2.3: {p23}\n  2.2: {p22}")

        c23, c22 = np.load(p23), np.load(p22)
        if c23.shape != c22.shape:
            raise SystemExit(f"shape mismatch: 2.3 {c23.shape} vs 2.2 {c22.shape}")

        per_row = {}
        for i, name in ROWS.items():
            a, b = c23[i], c22[i]
            d = np.abs(a - b)
            per_row[name] = {
                "n_epochs": int(len(a)),
                "max_abs_diff": float(d.max()),
                "n_nonzero": int((d > 0).sum()),
            }

        best23 = int(np.nanargmin(c23[0]))
        best22 = int(np.nanargmin(c22[0]))
        verdict = "bit-identical" if all(r["max_abs_diff"] == 0.0 for r in per_row.values()) else "differs"
        all_identical = all_identical and verdict == "bit-identical"
        results[f"{f}/{cid}"] = {
            "config": {**anchor,
                       "paths": {"2.3": str(p23.relative_to(EXP_DIR.parents[1])),
                                 "2.2": str(p22.relative_to(EXP_DIR.parents[1]))}},
            "per_row_max_abs_diff": per_row,
            "best_val_epoch_1idx": {"2.3": best23 + 1, "2.2": best22 + 1},
            "best_val_rmse": {"2.3": float(c23[0][best23]), "2.2": float(c22[0][best22])},
            "verdict": verdict,
        }
        print(f"[compare-anchor] {f}/{cid}: {verdict} — max|diff| per row = "
              f"{ {k: v['max_abs_diff'] for k, v in per_row.items()} } "
              f"(best val 2.3 ep {best23 + 1} = {results[f'{f}/{cid}']['best_val_rmse']['2.3']:.8f} vs "
              f"2.2 ep {best22 + 1} = {results[f'{f}/{cid}']['best_val_rmse']['2.2']:.8f})")

    out = args.out / "artifacts" / "anchor_vs_22_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "overall_verdict": "bit-identical" if all_identical else "differs",
        "anchors": results,
    }, indent=2), encoding="utf-8")
    print(f"[compare-anchor] overall: {'bit-identical' if all_identical else 'differs'}; "
          f"wrote {out.relative_to(args.out)}")


if __name__ == "__main__":
    main()
