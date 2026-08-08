#!/usr/bin/env python3
"""Extrapolation (OOD) diagnostic for derived_8.4-eval-mlp-1.3.

Same definition as mlp-1.0: test rows whose top-k gain backbone features fall
outside the trainval [min, max] range are flagged OOD; compares RMSE / R² of
the best neural models per family (and the 5-seed retrain ensembles) vs the
XGBoost references on in-distribution vs OOD slices.

Outputs: ood_summary.csv + ood_extrapolation.png.

Usage:
    python analyze_extrapolation.py [--config config.yaml] [--out .] [--top-k 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
sys.path.insert(0, str(EVAL11_DIR))
sys.path.insert(0, str(EXP_DIR))

from eval11.data import load_experiment_data  # noqa: E402
from eval11.evaluator import compute_metrics  # noqa: E402
from mlp13.plots import matplotlib, plt  # noqa: E402

FAMILY_LABELS = {
    "2regime_54": "2regime-54",
    "2regime_96": "2regime-96",
}


def top_gain_features(config: dict, k: int = 10) -> list[str]:
    pool_path = PROJECT_ROOT / Path(config["candidate_pool_file"])
    df = pd.read_csv(pool_path)
    df = df[df["feature"].isin(config["shared_backbone_54"])].copy()
    df = df.sort_values("gain", ascending=False)
    return df["feature"].head(k).tolist()


def ood_mask(frame: pd.DataFrame, feats: list[str], train_frame: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(frame), dtype=bool)
    for f in feats:
        lo = float(train_frame[f].min())
        hi = float(train_frame[f].max())
        vals = frame[f].to_numpy(dtype=float)
        outside = (vals < lo - 1e-9) | (vals > hi + 1e-9)
        outside = outside & np.isfinite(vals)
        mask |= outside
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_experiment_data(PROJECT_ROOT, config)
    feats = top_gain_features(config, args.top_k)
    print(f"[ood] top-{len(feats)} gain features: {feats}", flush=True)

    ood = ood_mask(data.test, feats, data.trainval)
    n_ood = int(ood.sum())
    print(f"[ood] {n_ood}/{len(data.test)} test rows OOD on >=1 top-gain feature "
          f"({100 * n_ood / len(data.test):.1f}%)", flush=True)
    y_test = data.test[data.target].to_numpy(dtype=float)

    selected = json.loads((args.out / "selected_features.json").read_text(encoding="utf-8"))
    sources: list[tuple[str, np.ndarray]] = []

    def _load_preds(path: Path) -> np.ndarray | None:
        return np.load(path) if path.exists() else None

    for family in [f["id"] for f in config["families"]]:
        cid = selected.get("mlp_winners", {}).get(family)
        if cid:
            p = _load_preds(args.out / "models" / family / cid / "preds.npy")
            if p is not None:
                sources.append((f"MLP {FAMILY_LABELS[family]} ({cid})", p))
        # retrain ensemble if present
        ep = args.out / "models" / "retrain" / f"{family}__{cid}" / "ens_preds.npy" if cid else None
        if ep is not None and ep.exists():
            sources.append((f"MLP {FAMILY_LABELS[family]} (retrain-ens)", np.load(ep)))

    for xgb_pred_file, xgb_label in [
        (EVAL11_DIR / "models" / "Global_Single_backbone_0_0_preds.npy", "XGBoost Global (54)"),
        (EVAL11_DIR / "models" / "Clustering_V0_Full_k2_winner_preds.npy", "XGBoost 2-Regime (Winner)"),
    ]:
        p = _load_preds(xgb_pred_file)
        if p is not None:
            sources.append((xgb_label, p))

    rows = []
    for name, preds in sources:
        for slice_name, mask in [("all", np.ones(len(y_test), dtype=bool)),
                                 ("in_distribution", ~ood),
                                 ("ood", ood)]:
            m = compute_metrics(y_test[mask], preds[mask])
            rows.append({"model": name, "slice": slice_name, "n": int(mask.sum()),
                         "r2": m["r2"], "rmse": m["rmse"], "bias": m["bias"], "mae": m["mae"]})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "ood_summary.csv", index=False)
    print(df.to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    width = 0.22
    models = df["model"].unique()
    slices = ["all", "in_distribution", "ood"]
    for i, mname in enumerate(models):
        sub = df[df["model"] == mname].set_index("slice")
        vals = [sub.loc[s, "rmse"] if s in sub.index else float("nan") for s in slices]
        ax.bar(np.arange(len(slices)) + i * width, vals, width, label=mname)
    ax.set_xticks(np.arange(len(slices)) + width)
    ax.set_xticklabels(slices)
    ax.set_ylabel("RMSE")
    ax.set_title("Extrapolation check: neural tabular vs XGBoost on OOD test slices "
                 f"(top-{len(feats)} gain features outside train range)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.out / "ood_extrapolation.png", dpi=150)
    plt.close()
    print("[ood] wrote ood_summary.csv + ood_extrapolation.png", flush=True)


if __name__ == "__main__":
    main()
