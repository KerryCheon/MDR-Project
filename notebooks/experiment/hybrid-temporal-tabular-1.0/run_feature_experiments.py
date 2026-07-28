from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import (
    ARTIFACT_DIR,
    TARGET,
    add_soil_texture_one_hot,
    evaluate_predictions,
    feature_variants,
    fit_xgboost,
    load_splits,
    make_prediction_frame,
    save_json,
)

LEGACY_TEMPORAL_STATIONS = [
    "Darrington",
    "Quinault",
    "SourdoughGulch_WA_985",
    "Spokane",
    "Touchet_WA_824",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["all_stations", "legacy_five"],
        default="all_stations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ARTIFACT_DIR / "feature_experiments" / args.scope
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    splits, texture_columns = add_soil_texture_one_hot(load_splits())
    if args.scope == "legacy_five":
        splits = {
            name: frame[frame["station_id"].isin(LEGACY_TEMPORAL_STATIONS)].copy()
            for name, frame in splits.items()
        }
    variants = feature_variants(texture_columns)
    results = {}
    summary_rows = []

    for name, features in variants.items():
        print(f"[feature] {name}: {len(features)} features")
        imputer, model = fit_xgboost(splits["train"], splits["val"], features)
        variant_payload = {
            "feature_count": len(features),
            "features": features,
            "best_iteration": int(model.best_iteration),
        }
        for split_name in ("train", "val", "test"):
            frame = splits[split_name]
            predictions = model.predict(imputer.transform(frame[features]))
            prediction_frame = make_prediction_frame(frame, predictions)
            prediction_frame.to_csv(
                prediction_dir / f"{name}_{split_name}.csv", index=False
            )
            evaluation = evaluate_predictions(prediction_frame)
            variant_payload[split_name] = evaluation
            row = {
                "variant": name,
                "feature_count": len(features),
                "best_iteration": int(model.best_iteration),
                "split": split_name,
                **evaluation["overall"],
                "macro_station_r2": evaluation["macro_station_r2"],
            }
            summary_rows.append(row)
        results[name] = variant_payload

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    save_json(results, output_dir / "metrics.json")

    plot_frame = summary[summary["split"].isin(["val", "test"])].copy()
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for split_name, color in (("val", "tab:blue"), ("test", "tab:orange")):
        group = plot_frame[plot_frame["split"] == split_name]
        x = range(len(group))
        offset = -0.18 if split_name == "val" else 0.18
        axes[0].bar(
            [value + offset for value in x],
            group["r2"],
            width=0.36,
            label=split_name,
            color=color,
        )
        axes[1].bar(
            [value + offset for value in x],
            group["rmse"],
            width=0.36,
            label=split_name,
            color=color,
        )
    labels = variants.keys()
    axes[0].set_ylabel("R2")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(range(len(variants)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("SMAP, NDVI/LST, and Soil Feature Experiments")
    figure.tight_layout()
    figure.savefig(output_dir / "feature_variant_comparison.png", dpi=160)
    plt.close(figure)

    val_rows = summary[summary["split"] == "val"].sort_values("rmse")
    print("\nValidation ranking")
    print(val_rows[["variant", "feature_count", "r2", "rmse", "mae"]].to_string(index=False))
    print("\nTest results")
    print(
        summary[summary["split"] == "test"]
        [["variant", "feature_count", "r2", "rmse", "mae"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
