#!/usr/bin/env python3
"""Refresh the 6 val-selected-winner configurations after a fixed val selection.

The main GPU run's val-selection phase (select_deltas_val.py) used the proxy
hyperparameters (500 trees) for the ranking backbone and the 9-point val grid; the
protocol requires the SAME exact hyperparameters as the final evaluation (2500
trees). After re-running select_deltas_val.py (fixed), this script:

  --check: compares the new val_selected_deltas.json against the currently pinned
           val-winner configurations; prints "changed" if any winner count or
           addition list differs, "unchanged" otherwise.
  --purge: deletes the stale per-seed artifacts (job metas, predictions, weights)
           of the 6 val-winner configs so the drivers re-compute them under the
           same data_version (the 14 test-selected/none configs are untouched and
           resume as completed).

Usage:
    uv run python refresh_val_winners.py --check
    uv run python refresh_val_winners.py --purge
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

VAL_CONFIGS = [
    "Clustering_V0_Full_k2_val_winner",
    "Clustering_Backbone54_k2_val_winner",
    "Clustering_Dynamic_k2_val_winner",
    "Univariate_G_API_k2_val_winner",
    "Seasonal_Binary_k2_val_winner",
    "Trained_Gating_k2_val_winner",
]


def new_val_entries() -> dict[str, dict]:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    val_path = EXP_DIR / Path(config["val_selection"]["output_json"])
    val = json.loads(val_path.read_text(encoding="utf-8"))
    entries = {}
    for strategy in config["val_selection"]["strategies"]:
        entry = val["strategies"][strategy]
        winner = entry["winner"]
        entries[f"{strategy}_val_winner"] = {
            "c0": int(winner[0]),
            "c1": int(winner[1]),
            "additions": {
                "0": list(entry["additions"]["0"][: int(winner[0])]),
                "1": list(entry["additions"]["1"][: int(winner[1])]),
            },
        }
    return entries


def pinned_val_entries() -> dict[str, dict]:
    pin_path = EXP_DIR / "pinned_configurations.json"
    if not pin_path.exists():
        return {}
    pinned = json.loads(pin_path.read_text(encoding="utf-8"))
    return {
        cfg["config_id"]: {
            "c0": cfg["cluster_0_count"],
            "c1": cfg["cluster_1_count"],
            "additions": cfg["cluster_additions"],
        }
        for cfg in pinned
        if cfg["config_id"] in VAL_CONFIGS
    }


def changed() -> bool:
    new_vals = new_val_entries()
    old_vals = pinned_val_entries()
    if set(new_vals) != set(old_vals):
        return True
    for cid in VAL_CONFIGS:
        if new_vals[cid] != old_vals[cid]:
            print(f"[check] {cid} changed: "
                  f"old={old_vals[cid]['c0']},{old_vals[cid]['c1']} "
                  f"new={new_vals[cid]['c0']},{new_vals[cid]['c1']}", flush=True)
            return True
    return False


def purge() -> None:
    config = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
    pred_dirs = [EXP_DIR / Path(config["temporal"]["predictions_dir"]),
                 EXP_DIR / Path(config["loso"]["predictions_dir"])]
    models_dir = EXP_DIR / Path(config["temporal"]["models_dir"])
    jobs_dir = EXP_DIR / "artifacts" / "jobs"
    removed = 0
    for cid in VAL_CONFIGS:
        for d in pred_dirs:
            if d.exists():
                for f in d.glob(f"{cid}__s*__*"):
                    f.unlink()
                    removed += 1
        if models_dir.exists():
            for f in models_dir.glob(f"{cid}__s*__*"):
                f.unlink()
                removed += 1
        if jobs_dir.exists():
            for jd in jobs_dir.glob(f"{cid}__s*__*"):
                shutil.rmtree(jd, ignore_errors=True)
                removed += 1
    print(f"[purge] removed {removed} stale artifacts for the {len(VAL_CONFIGS)} "
          f"val-winner configs (14 core configs untouched)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Print changed/unchanged.")
    parser.add_argument("--purge", action="store_true", help="Delete stale val-winner artifacts.")
    args = parser.parse_args()
    if args.check:
        print("changed" if changed() else "unchanged")
    elif args.purge:
        purge()
    else:
        raise SystemExit("pass --check or --purge")
