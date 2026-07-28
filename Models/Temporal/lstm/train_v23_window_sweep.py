"""
LSTM Track A - seq_len re-sweep on the coverage-fixed dataset builder.

Re-runs the seq_len sweep (5, 10, 20, 30) on the top25 feature set using
build_datasets_v2, across 3 seeds, to check whether last session's
"seq30 beats seq5" conclusion survives now that every seq_len sees the
identical val/test row set (4016 test / 2720 val), rather than shrinking
val/test coverage as seq_len grows (the old build_datasets bug).

Usage (after train_v23_baseline.py, whose train_one_seed this reuses):
    python -m Models.Temporal.lstm.train_v23_window_sweep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.train_v20 import MAX_EPOCHS, PATIENCE, REPORT_FIG_DIR, _json_safe, load_splits, DATA_DIR
from Models.Temporal.lstm.train_v23_baseline import OUT_DIR as BASELINE_OUT_DIR
from Models.Temporal.lstm.train_v23_baseline import TOP25_FEATURES, train_one_seed

OUT_DIR = BASELINE_OUT_DIR / "window_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LENS = [5, 10, 20, 30]
SEEDS = [42, 7, 123]

EXPECTED_TEST_N = 4016
EXPECTED_VAL_N = 2720


def save_window_sweep_plot(summary: pd.DataFrame, out_path: Path, title: str):
    agg = summary.groupby("seq_len").agg(
        val_r2_mean=("val_r2", "mean"), test_r2_mean=("test_r2", "mean"),
        val_rmse_mean=("val_rmse", "mean"), test_rmse_mean=("test_rmse", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(agg["seq_len"], agg["val_r2_mean"], marker="o", label="val (mean of 3 seeds)")
    axes[0].plot(agg["seq_len"], agg["test_r2_mean"], marker="o", label="test (mean of 3 seeds)")
    axes[0].set_ylabel("R2")
    axes[0].legend()

    axes[1].plot(agg["seq_len"], agg["val_rmse_mean"], marker="o", label="val")
    axes[1].plot(agg["seq_len"], agg["test_rmse_mean"], marker="o", label="test")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("Sequence length")
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    train_df, val_df, test_df = load_splits(DATA_DIR)

    rows = []
    for seq_len in SEQ_LENS:
        for seed in SEEDS:
            run = train_one_seed(
                feature_cols=TOP25_FEATURES,
                seq_len=seq_len,
                variant=f"v23_windowsweep_seq{seq_len}_seed{seed}",
                out_dir=OUT_DIR / f"seq{seq_len}" / f"seed{seed}",
                seed=seed,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )
            r = run["results"]
            test_n = r["test"]["n"]
            val_n = r["val"]["n"]
            assert test_n == EXPECTED_TEST_N, f"seq_len={seq_len} seed={seed}: test_n={test_n} != {EXPECTED_TEST_N}"
            assert val_n == EXPECTED_VAL_N, f"seq_len={seq_len} seed={seed}: val_n={val_n} != {EXPECTED_VAL_N}"
            rows.append(
                {
                    "seq_len": seq_len,
                    "seed": seed,
                    "val_r2": r["val"]["r2"],
                    "val_rmse": r["val"]["rmse"],
                    "test_r2": r["test"]["r2"],
                    "test_rmse": r["test"]["rmse"],
                    "test_n": test_n,
                    "val_n": val_n,
                    "best_epoch": r["config"]["best_epoch"],
                }
            )

    summary = pd.DataFrame(rows).sort_values(["seq_len", "seed"])
    print("\n[coverage check] test_n/val_n identical across every seq_len (bit-exact):")
    print(summary.groupby("seq_len")[["test_n", "val_n"]].first())

    agg = summary.groupby("seq_len").agg(
        val_r2_mean=("val_r2", "mean"), val_r2_std=("val_r2", "std"),
        test_r2_mean=("test_r2", "mean"), test_r2_std=("test_r2", "std"),
        val_rmse_mean=("val_rmse", "mean"), test_rmse_mean=("test_rmse", "mean"),
    ).reset_index()
    val_selected = agg.loc[agg["val_rmse_mean"].idxmin(), "seq_len"]
    test_best = agg.loc[agg["test_r2_mean"].idxmax(), "seq_len"]

    summary_path = OUT_DIR / "summary.csv"
    json_path = OUT_DIR / "metrics.json"
    plot_path = REPORT_FIG_DIR / "window_sweep_v23_top25.png"

    summary.to_csv(summary_path, index=False)
    payload = {
        "feature_cols": TOP25_FEATURES,
        "seq_lens": SEQ_LENS,
        "seeds": SEEDS,
        "expected_test_n": EXPECTED_TEST_N,
        "expected_val_n": EXPECTED_VAL_N,
        "rows": rows,
        "aggregate_by_seq_len": agg.to_dict(orient="records"),
        "val_selected_seq_len": int(val_selected),
        "test_best_seq_len": int(test_best),
        "val_test_rank_agrees": bool(val_selected == test_best),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    save_window_sweep_plot(summary, plot_path, "v23 window-size re-sweep (top25, coverage-fixed, 3 seeds)")

    print("\n[aggregate by seq_len]")
    print(agg.to_string(index=False))
    print(f"\n[val-selected] seq_len={val_selected} (lowest mean val RMSE)")
    print(f"[test-best]    seq_len={test_best} (highest mean test R2)")
    print(f"[agreement]    val selection matches test-best: {val_selected == test_best}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {json_path}")
    print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
