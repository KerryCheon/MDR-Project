"""
Window-size sweep for LSTM v20.

Run train_v20 first so outputs_v20/metrics.json contains the selected pruned
feature set, then sweep:
    python -m Models.Temporal.lstm.train_v20_window_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.train_v20 import (
    ALL_FEATURES,
    OUT_DIR,
    PATIENCE,
    REPORT_FIG_DIR,
    SEED,
    train_model,
)

DEFAULT_SEQ_LENS = [5, 7, 10, 14, 20, 30]


def load_feature_set(metrics_path: Path, feature_set: str):
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run python -m Models.Temporal.lstm.train_v20 first "
            "so the sweep can use the pruned feature sets."
        )
    with open(metrics_path) as f:
        metrics = json.load(f)

    if feature_set == "full38":
        return ALL_FEATURES, "full38"

    if feature_set == "selected":
        feature_cols = metrics.get("selected_pruned_features")
        source = metrics.get("selected_pruned_run")
    else:
        run = metrics.get("pruned_candidates", {}).get(feature_set)
        feature_cols = run.get("feature_cols") if run else None
        source = feature_set

    if not feature_cols:
        raise ValueError(
            f"{metrics_path} does not contain feature set {feature_set!r}. "
            "Expected one of: selected, top15, top20, top25, full38."
        )
    return feature_cols, source


def save_window_sweep_plot(summary: pd.DataFrame, out_path: Path, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(summary["seq_len"], summary["val_r2"], marker="o", label="val")
    axes[0].plot(summary["seq_len"], summary["test_r2"], marker="o", label="test")
    axes[0].set_ylabel("R2")
    axes[0].legend()

    axes[1].plot(summary["seq_len"], summary["val_rmse"], marker="o", label="val")
    axes[1].plot(summary["seq_len"], summary["test_rmse"], marker="o", label="test")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("Sequence length")
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep v20 seq_len values.")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--metrics-path", type=Path, default=OUT_DIR / "metrics.json")
    parser.add_argument(
        "--feature-set",
        choices=["selected", "top15", "top20", "top25", "full38"],
        default="selected",
        help="Feature set to sweep. 'selected' uses selected_pruned_features from metrics.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    feature_cols, selected_run = load_feature_set(args.metrics_path, args.feature_set)
    out_name = selected_run or args.feature_set
    out_name = out_name.replace("/", "_")
    print(
        f"[window sweep] feature set={selected_run} "
        f"({len(feature_cols)} features); seq_lens={args.seq_lens}"
    )
    print(
        "[caveat] windows are built after year-filtering in dataset.py:_build_sequences, "
        "so larger seq_len values drop more rows at split boundaries."
    )

    rows = []
    for seq_len in args.seq_lens:
        run = train_model(
            feature_cols=feature_cols,
            seq_len=seq_len,
            variant=f"v20_window_{out_name}_seq{seq_len}",
            out_dir=OUT_DIR / "window_sweep" / out_name / f"seq{seq_len}",
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=SEED,
        )
        results = run["results"]
        rows.append(
            {
                "seq_len": seq_len,
                "feature_count": len(feature_cols),
                "best_epoch": results["config"]["best_epoch"],
                "epochs_run": results["config"]["epochs_run"],
                "val_r2": results["val"]["r2"],
                "val_rmse": results["val"]["rmse"],
                "test_r2": results["test"]["r2"],
                "test_rmse": results["test"]["rmse"],
                "train_n": results["train"]["n"],
                "val_n": results["val"]["n"],
                "test_n": results["test"]["n"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("seq_len")
    best = summary.loc[summary["val_rmse"].idxmin()].to_dict()
    summary_path = OUT_DIR / f"window_sweep_{out_name}_summary.csv"
    json_path = OUT_DIR / f"window_sweep_{out_name}_metrics.json"
    plot_path = REPORT_FIG_DIR / f"window_sweep_{out_name}.png"

    summary.to_csv(summary_path, index=False)
    payload = {
        "selected_feature_source": selected_run,
        "feature_cols": feature_cols,
        "best_by_val_rmse": best,
        "rows": rows,
        "caveat": "Windows are built after year-filtering, so larger windows drop more rows at split boundaries.",
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    save_window_sweep_plot(summary, plot_path, f"v20 window-size sweep ({out_name})")

    print("\n[summary]")
    print(summary.to_string(index=False))
    print(
        f"\n[best] seq_len={int(best['seq_len'])} by val RMSE "
        f"(val R2={best['val_r2']:.4f}, test R2={best['test_r2']:.4f})"
    )
    print(f"[saved] {summary_path}")
    print(f"[saved] {json_path}")
    print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
