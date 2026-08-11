#!/usr/bin/env python3
"""Cross-version anchor check: 2.1 (v8) vs 2.0 — bit-identical curves?

Offline comparison of the shared non-SWA anchor config's saved per-epoch
curves between derived_8.4-eval-mlp-2.0 (its own H100 node) and this
experiment (ac096, job 2032849). The two runs used identical hyperparameters
and data (the v8 hidden_sizes fix does not touch this config, whose
hidden_sizes were always explicit). If max|diff| = 0 the v8 run reproduced
2.0's curves bit-identically on a different node — evidence that the earlier
~4-5% relative anchor mismatch was the v7 hidden_sizes bug, not cross-node
nondeterminism.

Result is written to artifacts/anchor_vs_20_comparison.json (committed
evidence for the README claim).

Usage:
    uv run --no-sync python compare_anchor_vs_2.0.py [--out .]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent

ANCHOR = {
    "family": "2regime_54",
    "config_id": "w384x384_d0.3_gelu",
    "seed": 42,
    "spec": 0,
}
ROWS = {0: "val", 1: "aux", 2: "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    f, cid, seed, spec = ANCHOR["family"], ANCHOR["config_id"], ANCHOR["seed"], ANCHOR["spec"]
    p21 = EXP_DIR / "models" / f / cid / f"seed_{seed}" / f"spec_{spec}" / "curves.npy"
    p20 = EXP_DIR.parent / "derived_8.4-eval-mlp-2.0" / "models" / f / cid / f"seed_{seed}" / f"spec_{spec}" / "curves.npy"
    if not p21.exists() or not p20.exists():
        raise SystemExit(f"curves not found:\n  2.1: {p21}\n  2.0: {p20}")

    c21, c20 = np.load(p21), np.load(p20)
    if c21.shape != c20.shape:
        raise SystemExit(f"shape mismatch: 2.1 {c21.shape} vs 2.0 {c20.shape}")

    per_row = {}
    for i, name in ROWS.items():
        a, b = c21[i], c20[i]
        d = np.abs(a - b)
        per_row[name] = {
            "n_epochs": int(len(a)),
            "max_abs_diff": float(d.max()),
            "n_nonzero": int((d > 0).sum()),
        }

    best21 = int(np.nanargmin(c21[0]))
    best20 = int(np.nanargmin(c20[0]))
    result = {
        "config": {**ANCHOR, "paths": {"2.1": str(p21.relative_to(EXP_DIR.parents[1])), "2.0": str(p20.relative_to(EXP_DIR.parents[1]))}},
        "per_row_max_abs_diff": per_row,
        "best_val_epoch_1idx": {"2.1": best21 + 1, "2.0": best20 + 1},
        "best_val_rmse": {"2.1": float(c21[0][best21]), "2.0": float(c20[0][best20])},
        "verdict": "bit-identical" if all(r["max_abs_diff"] == 0.0 for r in per_row.values()) else "differs",
    }
    out = args.out / "artifacts" / "anchor_vs_20_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[compare-anchor] {result['verdict']}: max|diff| per row = "
          f"{ {k: v['max_abs_diff'] for k, v in per_row.items()} } (best val 2.1 ep {best21 + 1} = "
          f"{result['best_val_rmse']['2.1']:.8f} vs 2.0 ep {best20 + 1} = {result['best_val_rmse']['2.0']:.8f})")
    print(f"[compare-anchor] wrote {out.relative_to(args.out)}")


if __name__ == "__main__":
    main()
