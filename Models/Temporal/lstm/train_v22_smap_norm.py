"""
LSTM v22 - SMAP-specific rolling per-station input normalization.

This experiment addresses the 06-30 future-work item: SMAP readings were the
dominant input OOD source in the 2023-2025 test years, so test whether replacing
absolute SMAP values with trailing per-station anomalies improves the temporal
LSTM.

The normalization is causal with respect to each station's timeline: for each
continuous SMAP feature used by a run, the value at date t is transformed using
only that station's current and past values in a trailing row window. The
transformed splits then go through the normal train-only imputer/scaler used by
the v20 trainer.

Default usage runs two targeted checks:
    1. v20 top25 + seq_len=30, the current best temporal LSTM.
    2. v21 full58 + seq_len=10, the best v21 test-performing union run.

    MPLCONFIGDIR=/private/tmp/mpl .venv/bin/python -m Models.Temporal.lstm.train_v22_smap_norm
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
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.train_v20 import (
    OUT_DIR as V20_OUT_DIR,
    REPORT_FIG_DIR,
    SEED,
    _json_safe,
    train_model,
)
from Models.Temporal.lstm.train_v21 import ALL_FEATURES as V21_FULL58_FEATURES

LSTM_DIR = Path(__file__).parent
OUT_DIR = LSTM_DIR / "outputs_v22_smap_norm"
OUT_DIR.mkdir(exist_ok=True)
REPORT_FIG_DIR.mkdir(exist_ok=True)

DEFAULT_ROLLING_WINDOW = 30
DEFAULT_MIN_PERIODS = 5


def load_v20_feature_set(feature_set: str) -> list[str]:
    metrics_path = V20_OUT_DIR / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    if feature_set == "selected":
        feature_cols = metrics.get("selected_pruned_features")
    else:
        run = metrics.get("pruned_candidates", {}).get(feature_set)
        feature_cols = run.get("feature_cols") if run else None

    if not feature_cols:
        raise ValueError(f"Missing v20 feature set {feature_set!r} in {metrics_path}")
    return feature_cols


def continuous_smap_features(feature_cols: list[str]) -> list[str]:
    """SMAP-derived continuous inputs to normalize, excluding binary masks."""
    return [
        col
        for col in feature_cols
        if "SMAP" in col and "mask" not in col.lower() and not col.lower().endswith("_isnan")
    ]


def _rolling_station_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = series.astype(float)
    roll = values.rolling(window=window, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std(ddof=0)

    expanding_mean = values.expanding(min_periods=1).mean()
    expanding_std = values.expanding(min_periods=2).std(ddof=0)
    mean = mean.fillna(expanding_mean)
    std = std.fillna(expanding_std)
    std = std.mask(std <= 1e-6)

    z = (values - mean) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def smap_rolling_normalize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    window: int,
    min_periods: int,
):
    smap_cols = continuous_smap_features(feature_cols)
    if not smap_cols:
        return train_df, val_df, test_df, {
            "name": "smap_rolling_station_zscore",
            "applied": False,
            "reason": "no continuous SMAP columns in feature set",
        }

    pieces = []
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part = df.copy()
        part["_split"] = split_name
        part["_row_id"] = np.arange(len(part))
        pieces.append(part)

    combined = pd.concat(pieces, ignore_index=True)
    combined["_date_sort"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["station_id", "_date_sort", "_split", "_row_id"]).copy()

    for col in smap_cols:
        combined[col] = combined.groupby("station_id", sort=False)[col].transform(
            lambda s: _rolling_station_zscore(s, window=window, min_periods=min_periods)
        )

    out = {}
    for split_name in ["train", "val", "test"]:
        split = combined.loc[combined["_split"] == split_name].sort_values("_row_id")
        out[split_name] = split.drop(columns=["_split", "_row_id", "_date_sort"]).reset_index(drop=True)

    return out["train"], out["val"], out["test"], {
        "name": "smap_rolling_station_zscore",
        "applied": True,
        "window": window,
        "min_periods": min_periods,
        "smap_columns": smap_cols,
        "note": (
            "Computed per station over the chronological train+val+test timeline "
            "using rolling current/past values only, then split back by original rows."
        ),
    }


def make_smap_transform(window: int, min_periods: int):
    def _transform(train_df, val_df, test_df, feature_cols):
        return smap_rolling_normalize_splits(
            train_df,
            val_df,
            test_df,
            feature_cols,
            window=window,
            min_periods=min_periods,
        )

    return _transform


def feature_ood_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict]:
    rows = []
    for col in continuous_smap_features(feature_cols):
        train_values = pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float)
        test_values = pd.to_numeric(test_df[col], errors="coerce").to_numpy(dtype=float)
        train_values = train_values[np.isfinite(train_values)]
        test_values = test_values[np.isfinite(test_values)]
        if train_values.size == 0 or test_values.size == 0:
            rows.append({"feature": col, "ood_fraction": np.nan, "train_p1": np.nan, "train_p99": np.nan})
            continue
        p1, p99 = np.percentile(train_values, [1, 99])
        frac = np.mean((test_values < p1) | (test_values > p99))
        rows.append(
            {
                "feature": col,
                "ood_fraction": float(frac),
                "train_p1": float(p1),
                "train_p99": float(p99),
            }
        )
    return rows


def compute_ood_before_after(feature_cols: list[str], window: int, min_periods: int) -> dict:
    from Models.Temporal.lstm.train_v20 import load_splits

    train_df, val_df, test_df = load_splits()
    before = feature_ood_summary(train_df, test_df, feature_cols)
    train_norm, _, test_norm, metadata = smap_rolling_normalize_splits(
        train_df,
        val_df,
        test_df,
        feature_cols,
        window=window,
        min_periods=min_periods,
    )
    after = feature_ood_summary(train_norm, test_norm, feature_cols)
    return {"before": before, "after": after, "metadata": metadata}


def save_ood_plot(payload: dict, out_path: Path, title: str):
    before = {row["feature"]: row["ood_fraction"] for row in payload["before"]}
    after = {row["feature"]: row["ood_fraction"] for row in payload["after"]}
    features = list(before.keys())
    if not features:
        return

    x = np.arange(len(features))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(features) * 1.1), 4.8))
    ax.bar(x - width / 2, [before[f] for f in features], width, label="before")
    ax.bar(x + width / 2, [after[f] for f in features], width, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=35, ha="right")
    ax.set_ylabel("Test fraction outside train [p1, p99]")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def strip_run(run: dict, ood_payload: dict, ood_figure: Path):
    return {
        "variant": run["variant"],
        "feature_cols": run["feature_cols"],
        "feature_count": len(run["feature_cols"]),
        "smap_columns": continuous_smap_features(run["feature_cols"]),
        "results": run["results"],
        "out_dir": str(run["out_dir"]),
        "ood": ood_payload,
        "ood_figure": str(ood_figure),
    }


def run_variant(
    *,
    name: str,
    feature_cols: list[str],
    seq_len: int,
    window: int,
    min_periods: int,
    max_epochs: int,
    patience: int,
):
    ood_payload = compute_ood_before_after(feature_cols, window, min_periods)
    ood_figure = REPORT_FIG_DIR / f"smap_norm_ood_{name}.png"
    save_ood_plot(ood_payload, ood_figure, f"SMAP OOD before/after ({name})")

    run = train_model(
        feature_cols=feature_cols,
        seq_len=seq_len,
        variant=f"v22_smapnorm_{name}",
        out_dir=OUT_DIR / name,
        max_epochs=max_epochs,
        patience=patience,
        seed=SEED,
        save_report_curve=REPORT_FIG_DIR / f"training_curve_v22_smapnorm_{name}.png",
        split_transform=make_smap_transform(window, min_periods),
    )
    return strip_run(run, ood_payload, ood_figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SMAP-specific normalization LSTM checks.")
    parser.add_argument("--window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--min-periods", type=int, default=DEFAULT_MIN_PERIODS)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["top25_seq30", "full58_seq10"],
        default=["top25_seq30", "full58_seq10"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {
        "method": "SMAP continuous feature rolling per-station z-score before normal v20 preprocessing",
        "window": args.window,
        "min_periods": args.min_periods,
        "variants": {},
        "anchors": {
            "v20_top25_seq30_test_r2": 0.7900784611701965,
            "v20_top25_seq30_test_rmse": 0.04334788769483566,
            "v21_full58_seq10_test_r2": 0.7647864818572998,
            "v21_full58_seq10_test_rmse": 0.045691292732954025,
        },
    }

    if "top25_seq30" in args.variants:
        top25 = load_v20_feature_set("top25")
        payload["variants"]["top25_seq30"] = run_variant(
            name="top25_seq30",
            feature_cols=top25,
            seq_len=30,
            window=args.window,
            min_periods=args.min_periods,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

    if "full58_seq10" in args.variants:
        payload["variants"]["full58_seq10"] = run_variant(
            name="full58_seq10",
            feature_cols=V21_FULL58_FEATURES,
            seq_len=10,
            window=args.window,
            min_periods=args.min_periods,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
