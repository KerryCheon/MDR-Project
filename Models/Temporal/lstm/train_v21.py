"""
LSTM v21 - Jakob 38 features plus v9/LSTM-unique feature union.

This follow-up keeps the v20 training setup fixed, but expands the feature
surface to:

    Jakob MDR-v25 38 features + v9 features not already in Jakob's list

The goal is to test whether Experiment 1 was too strict by replacing the old
LSTM feature set entirely instead of adding Jakob's aligned tabular features
on top of the LSTM-specific raw/static inputs.

Default usage runs the full union audit, prunes by validation permutation
importance, and sweeps window sizes for the validation-selected seq_len=10
candidate:

    MPLCONFIGDIR=/private/tmp/mpl .venv/bin/python -m Models.Temporal.lstm.train_v21
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.train_v9 import ALL_FEATURES as V9_FEATURES
from Models.Temporal.lstm.train_v20 import (
    ALL_FEATURES as JAKOB_FEATURES,
    ANCHORS as V20_ANCHORS,
    MAX_EPOCHS,
    PATIENCE,
    REPORT_FIG_DIR,
    SEED,
    _json_safe,
    _strip_runtime,
    permutation_importance,
    top_k_features,
    train_model,
)

LSTM_DIR = Path(__file__).parent
OUT_DIR = LSTM_DIR / "outputs_v21"
OUT_DIR.mkdir(exist_ok=True)
REPORT_FIG_DIR.mkdir(exist_ok=True)

LSTM_UNIQUE_FEATURES = [c for c in V9_FEATURES if c not in JAKOB_FEATURES]
ALL_FEATURES = JAKOB_FEATURES + LSTM_UNIQUE_FEATURES

DEFAULT_SEQ_LENS = [5, 7, 10, 14, 20, 30]

ANCHORS = {
    **V20_ANCHORS,
    "v20_full38_test_r2": 0.6963052749633789,
    "v20_top25_seq30_test_r2": 0.7900784611701965,
    "v20_top25_seq30_test_rmse": 0.04334788769483566,
    "v21_union_feature_count": len(ALL_FEATURES),
    "v21_lstm_unique_feature_count": len(LSTM_UNIQUE_FEATURES),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LSTM v21 union-feature audit.")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--keep-ks", type=int, nargs="+", default=[20, 25, 30, 35, 40])
    parser.add_argument("--no-prune", action="store_true", help="Run only full58 + importance.")
    parser.add_argument("--skip-importance", action="store_true", help="Skip permutation importance.")
    parser.add_argument("--skip-window-sweep", action="store_true", help="Skip window sweep.")
    return parser.parse_args()


def save_audit_comparison_plot(payload: dict, out_path: Path):
    rows = [
        ("v9 baseline", ANCHORS["v9_test_r2"]),
        ("v20 full38", ANCHORS["v20_full38_test_r2"]),
        ("v20 top25 seq30", ANCHORS["v20_top25_seq30_test_r2"]),
        ("Jakob XGBoost", ANCHORS["jakob_xgboost_test_r2"]),
    ]

    full = payload.get("full58")
    if full:
        rows.append(("v21 full58", full["results"]["test"]["r2"]))

    for name, run in payload.get("pruned_candidates", {}).items():
        rows.append((f"v21 {name}", run["results"]["test"]["r2"]))

    window = payload.get("window_sweep")
    if window and window.get("best_by_val_rmse"):
        best = window["best_by_val_rmse"]
        rows.append((f"v21 {window['selected_seq10_run']} seq{int(best['seq_len'])}", best["test_r2"]))

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["0.65" if not label.startswith("v21") else "tab:blue" for label in labels]
    colors = ["tab:green" if label == "Jakob XGBoost" else color for label, color in zip(labels, colors)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Test R2")
    ax.set_ylim(0.55, 0.85)
    ax.set_title("v21 union-feature audit")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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


def run_window_sweep(feature_cols: list[str], selected_seq10_run: str, args) -> dict:
    rows = []
    out_name = selected_seq10_run.replace("/", "_")
    print(
        f"[v21 window sweep] feature set={selected_seq10_run} "
        f"({len(feature_cols)} features); seq_lens={args.seq_lens}"
    )
    print(
        "[caveat] windows are built after year-filtering in dataset.py:_build_sequences, "
        "so larger seq_len values drop more rows at split boundaries."
    )

    for seq_len in args.seq_lens:
        run = train_model(
            feature_cols=feature_cols,
            seq_len=seq_len,
            variant=f"v21_window_{out_name}_seq{seq_len}",
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
    plot_path = REPORT_FIG_DIR / f"window_sweep_v21_{out_name}.png"

    summary.to_csv(summary_path, index=False)
    payload = {
        "selected_seq10_run": selected_seq10_run,
        "feature_cols": feature_cols,
        "best_by_val_rmse": best,
        "rows": rows,
        "summary_csv": str(summary_path),
        "figure": str(plot_path),
        "caveat": "Windows are built after year-filtering, so larger windows drop more rows at split boundaries.",
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    save_window_sweep_plot(summary, plot_path, f"v21 window-size sweep ({out_name})")

    print("\n[v21 window summary]")
    print(summary.to_string(index=False))
    print(
        f"\n[v21 window best] seq_len={int(best['seq_len'])} by val RMSE "
        f"(val R2={best['val_r2']:.4f}, test R2={best['test_r2']:.4f})"
    )
    print(f"[saved] {summary_path}")
    print(f"[saved] {json_path}")
    print(f"[saved] {plot_path}")
    return payload


def write_metrics(payload: dict):
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    print(f"[saved] {metrics_path}")


def main():
    args = parse_args()
    if len(set(ALL_FEATURES)) != len(ALL_FEATURES):
        dupes = sorted({c for c in ALL_FEATURES if ALL_FEATURES.count(c) > 1})
        raise ValueError(f"Duplicate v21 features: {dupes}")

    print(
        f"[v21] Jakob 38 + LSTM-unique {len(LSTM_UNIQUE_FEATURES)} "
        f"= {len(ALL_FEATURES)} features on derived_8.0"
    )
    print(f"[v21] LSTM-unique additions: {', '.join(LSTM_UNIQUE_FEATURES)}")
    print(
        f"[anchors] v9 test R2={ANCHORS['v9_test_r2']:.3f} | "
        f"v20 top25 seq30 test R2={ANCHORS['v20_top25_seq30_test_r2']:.3f} | "
        f"XGBoost test R2={ANCHORS['jakob_xgboost_test_r2']:.3f}"
    )

    full = train_model(
        feature_cols=ALL_FEATURES,
        seq_len=args.seq_len,
        variant="v21_full58",
        out_dir=OUT_DIR / "full58",
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_report_curve=REPORT_FIG_DIR / "training_curve_v21_full58.png",
    )

    payload = {
        "anchors": ANCHORS,
        "jakob_features": JAKOB_FEATURES,
        "v9_features": V9_FEATURES,
        "lstm_unique_features": LSTM_UNIQUE_FEATURES,
        "union_features": ALL_FEATURES,
        "full58": _strip_runtime(full),
        "importance": None,
        "pruned_candidates": {},
        "selected_seq10_run": "full58",
        "selected_seq10_features": ALL_FEATURES,
        "selected_seq10_feature_count": len(ALL_FEATURES),
        "window_sweep": None,
    }

    importance_payload = None
    if not args.skip_importance:
        importance_payload = permutation_importance(
            model=full["model"],
            loader=full["loaders"]["val"],
            device=full["device"],
            feature_cols=full["feature_cols"],
            train_feature_means=full["train_feature_means"],
            out_json=OUT_DIR / "feature_importance.json",
            out_png=REPORT_FIG_DIR / "lstm_feature_importance_v21_union.png",
        )
        payload["importance"] = {
            "json": str(OUT_DIR / "feature_importance.json"),
            "figure": str(REPORT_FIG_DIR / "lstm_feature_importance_v21_union.png"),
            "ranking": importance_payload["ranking"],
        }

    if importance_payload is not None and not args.no_prune:
        for k in args.keep_ks:
            if k >= len(ALL_FEATURES):
                raise ValueError(f"Pruned keep-K must be < {len(ALL_FEATURES)}; got {k}")
            feature_cols = top_k_features(importance_payload, k)
            run = train_model(
                feature_cols=feature_cols,
                seq_len=args.seq_len,
                variant=f"v21_top{k}",
                out_dir=OUT_DIR / f"pruned_top{k}",
                max_epochs=args.max_epochs,
                patience=args.patience,
                save_report_curve=REPORT_FIG_DIR / f"training_curve_v21_top{k}.png",
            )
            payload["pruned_candidates"][f"top{k}"] = _strip_runtime(run)

        seq10_candidates = {"full58": payload["full58"], **payload["pruned_candidates"]}
        selected_name, selected = min(
            seq10_candidates.items(),
            key=lambda item: item[1]["results"]["val"]["rmse"],
        )
        payload["selected_seq10_run"] = selected_name
        payload["selected_seq10_features"] = selected["feature_cols"]
        payload["selected_seq10_feature_count"] = len(selected["feature_cols"])
        print(
            f"[v21 selected seq10] {selected_name}: {len(selected['feature_cols'])} features, "
            f"val RMSE={selected['results']['val']['rmse']:.5f}, "
            f"test R2={selected['results']['test']['r2']:.4f}"
        )

    if not args.skip_window_sweep:
        payload["window_sweep"] = run_window_sweep(
            payload["selected_seq10_features"],
            payload["selected_seq10_run"],
            args,
        )

    comparison_path = REPORT_FIG_DIR / "v21_union_test_r2_comparison.png"
    save_audit_comparison_plot(payload, comparison_path)
    payload["comparison_figure"] = str(comparison_path)
    write_metrics(payload)


if __name__ == "__main__":
    main()
