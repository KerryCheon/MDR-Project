"""
Longer-training audit for v20 pruned feature sets.

This checks whether the selected pruned LSTM candidates are genuinely converged
or still improving under a longer training budget.

Usage:
    MPLCONFIGDIR=/private/tmp/mpl .venv/bin/python -m Models.Temporal.lstm.train_v20_training_audit
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

from Models.Temporal.lstm.train_v20 import OUT_DIR, REPORT_FIG_DIR, SEED, train_model
from Models.Temporal.lstm.train_v20_window_sweep import load_feature_set


def save_long_training_plot(rows: list[dict], out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for row in rows:
        history = row["history"]
        epochs = [h["epoch"] for h in history]
        label = row["feature_set"]
        axes[0].plot(epochs, [h["val_rmse"] for h in history], label=label)
        axes[1].plot(epochs, [h["val_r2"] for h in history], label=label)
        axes[0].axvline(row["best_epoch"], linestyle="--", alpha=0.35)
        axes[1].axvline(row["best_epoch"], linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Val RMSE")
    axes[1].set_ylabel("Val R2")
    axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("v20 longer-training audit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run longer-training audit for v20.")
    parser.add_argument("--feature-sets", nargs="+", default=["top20", "top25"], choices=["top15", "top20", "top25"])
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--metrics-path", type=Path, default=OUT_DIR / "metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    for feature_set in args.feature_sets:
        feature_cols, source = load_feature_set(args.metrics_path, feature_set)
        run = train_model(
            feature_cols=feature_cols,
            seq_len=args.seq_len,
            variant=f"v20_longtrain_{source}",
            out_dir=OUT_DIR / "training_audit" / source,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=SEED,
            save_report_curve=REPORT_FIG_DIR / f"training_curve_long_{source}.png",
        )
        results = run["results"]
        cfg = results["config"]
        history = results["history"]
        old_best_before_13 = min(h["val_rmse"] for h in history[:13]) if len(history) >= 13 else None
        later_best = min(h["val_rmse"] for h in history[13:]) if len(history) > 13 else None
        best_val_rmse = min(h["val_rmse"] for h in history)
        row = {
            "feature_set": source,
            "feature_count": len(feature_cols),
            "seq_len": args.seq_len,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "best_epoch": cfg["best_epoch"],
            "epochs_run": cfg["epochs_run"],
            "stopped_by_patience": cfg["stopped_by_patience"],
            "improved_after_epoch_13": later_best is not None and old_best_before_13 is not None and later_best < old_best_before_13,
            "best_val_rmse": best_val_rmse,
            "val_r2": results["val"]["r2"],
            "val_rmse": results["val"]["rmse"],
            "test_r2": results["test"]["r2"],
            "test_rmse": results["test"]["rmse"],
            "history": history,
        }
        rows.append(row)

    summary = pd.DataFrame([{k: v for k, v in row.items() if k != "history"} for row in rows])
    summary_path = OUT_DIR / "training_audit_summary.csv"
    json_path = OUT_DIR / "training_audit_metrics.json"
    plot_path = REPORT_FIG_DIR / "training_audit_longer.png"

    summary.to_csv(summary_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    save_long_training_plot(rows, plot_path)

    print("\n[training audit summary]")
    print(summary.to_string(index=False))
    print(f"[saved] {summary_path}")
    print(f"[saved] {json_path}")
    print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
