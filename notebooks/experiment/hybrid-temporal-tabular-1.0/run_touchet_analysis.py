from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from common import (
    ARTIFACT_DIR,
    DATA_DIR,
    TARGET,
    regression_metrics,
    save_json,
)

TOUCHET = "Touchet_WA_824"
FEATURES_TO_AUDIT = [
    "SMAP_sm_interp",
    "SMAP_x_year",
    "V_rollstd_SMAP_sm_interp_kobs30",
    "F_NDVI",
    "LST_modis",
    "G_API",
    "G_rain_sum_7d",
    "J_clay_wfrac_b0",
    "J_sand_wfrac_b0",
    "J_soil_texture_usda_b0",
]


def load_raw_splits():
    splits = {
        name: pd.read_csv(DATA_DIR / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    for frame in splits.values():
        frame["date"] = pd.to_datetime(frame["date"])
    return splits


def load_prediction(path, label):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["station_id"] == TOUCHET].copy()
    return frame.rename(columns={"y_pred": label})


def prediction_metrics(label, frame):
    metrics = regression_metrics(frame["y_true"], frame[label])
    return {
        "model": label,
        "coverage_n": len(frame),
        "target_mean": float(frame["y_true"].mean()),
        "target_std": float(frame["y_true"].std(ddof=0)),
        **metrics,
    }


def main() -> None:
    output_dir = ARTIFACT_DIR / "touchet_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_raw_splits()
    touchet = {
        name: frame[frame["station_id"] == TOUCHET].copy()
        for name, frame in splits.items()
    }

    last_train_date = touchet["train"]["date"].max()
    first_test_date = touchet["test"]["date"].min()
    long_gap_days = int((first_test_date - last_train_date).days)

    feature_root = ARTIFACT_DIR / "feature_experiments" / "legacy_five" / "predictions"
    hybrid_root = ARTIFACT_DIR / "hybrid"
    model_frames = {
        "tabular_baseline_38": load_prediction(
            feature_root / "baseline_38_test.csv", "tabular_baseline_38"
        ),
        "tabular_minus_smap_x_year": load_prediction(
            feature_root / "minus_smap_x_year_test.csv",
            "tabular_minus_smap_x_year",
        ),
        "temporal_current": load_prediction(
            hybrid_root
            / "encoders"
            / "current_with_smap_x_year"
            / "test_predictions.csv",
            "temporal_current",
        ),
        "temporal_alternative": load_prediction(
            hybrid_root
            / "encoders"
            / "alternative_without_smap_x_year"
            / "test_predictions.csv",
            "temporal_alternative",
        ),
        "strict_tabular": load_prediction(
            hybrid_root
            / "strict_frozen"
            / "predictions"
            / "T0_tabular_38_test.csv",
            "strict_tabular",
        ),
        "strict_hybrid": load_prediction(
            hybrid_root
            / "strict_frozen"
            / "predictions"
            / "T3_selected_tabular_plus_embedding_test.csv",
            "strict_hybrid",
        ),
    }

    metric_rows = [
        prediction_metrics(label, frame)
        for label, frame in model_frames.items()
    ]

    test = touchet["test"].sort_values("date").copy()
    train = touchet["train"].sort_values("date").copy()
    test["station_mean"] = float(train[TARGET].mean())
    monthly_means = train.groupby(train["date"].dt.month)[TARGET].mean()
    test["monthly_climatology"] = test["date"].dt.month.map(monthly_means)
    test["smap_only"] = test["SMAP_sm_interp"]
    previous = pd.concat([train[["date", TARGET]], test[["date", TARGET]]]).sort_values("date")
    previous["persistence"] = previous[TARGET].shift(1)
    test = test.merge(previous[["date", "persistence"]], on="date", how="left")
    for label in ("station_mean", "monthly_climatology", "smap_only", "persistence"):
        frame = test[["station_id", "date", TARGET, label]].rename(
            columns={TARGET: "y_true"}
        )
        metric_rows.append(prediction_metrics(label, frame))

    metrics_frame = pd.DataFrame(metric_rows).sort_values("rmse")
    metrics_frame.to_csv(output_dir / "touchet_metrics.csv", index=False)

    shift_rows = []
    for feature in FEATURES_TO_AUDIT:
        train_values = pd.to_numeric(train[feature], errors="coerce").dropna().to_numpy()
        test_values = pd.to_numeric(test[feature], errors="coerce").dropna().to_numpy()
        if len(train_values) == 0 or len(test_values) == 0:
            shift_rows.append(
                {
                    "feature": feature,
                    "train_mean": np.nan,
                    "train_std": np.nan,
                    "test_mean": np.nan,
                    "test_std": np.nan,
                    "standardized_mean_difference": np.nan,
                    "wasserstein_distance": np.nan,
                    "test_outside_train_p1_p99": np.nan,
                    "train_missing_fraction": float(train[feature].isna().mean()),
                    "test_missing_fraction": float(test[feature].isna().mean()),
                }
            )
            continue
        train_mean = float(np.mean(train_values))
        test_mean = float(np.mean(test_values))
        train_std = float(np.std(train_values))
        percentile_1, percentile_99 = np.quantile(train_values, [0.01, 0.99])
        shift_rows.append(
            {
                "feature": feature,
                "train_mean": train_mean,
                "train_std": train_std,
                "test_mean": test_mean,
                "test_std": float(np.std(test_values)),
                "standardized_mean_difference": (
                    (test_mean - train_mean) / train_std if train_std > 0 else np.nan
                ),
                "wasserstein_distance": float(
                    wasserstein_distance(train_values, test_values)
                ),
                "test_outside_train_p1_p99": float(
                    np.mean((test_values < percentile_1) | (test_values > percentile_99))
                ),
                "train_missing_fraction": float(train[feature].isna().mean()),
                "test_missing_fraction": float(test[feature].isna().mean()),
            }
        )
    shift_frame = pd.DataFrame(shift_rows).sort_values(
        "standardized_mean_difference", key=lambda values: values.abs(), ascending=False
    )
    shift_frame.to_csv(output_dir / "touchet_feature_shift.csv", index=False)

    full_prediction_parts = []
    for split_name in ("train", "test"):
        prediction = pd.read_csv(
            feature_root / f"minus_smap_x_year_{split_name}.csv"
        )
        prediction["date"] = pd.to_datetime(prediction["date"])
        prediction = prediction[prediction["station_id"] == TOUCHET]
        full_prediction_parts.append(prediction)
    full_predictions = pd.concat(full_prediction_parts).sort_values("date")
    full_truth = pd.concat([train, test]).sort_values("date")

    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for index, split_frame in enumerate((train, test)):
        axes[0].plot(
            split_frame["date"],
            split_frame[TARGET],
            color="black",
            linewidth=1.3,
            label="ground truth" if index == 0 else None,
        )
    for index, split_frame in enumerate(full_prediction_parts):
        axes[0].plot(
            split_frame["date"],
            split_frame["y_pred"],
            color="tab:blue",
            linewidth=1.0,
            alpha=0.85,
            label="tabular minus SMAP_x_year" if index == 0 else None,
        )
    axes[0].axvspan(last_train_date, first_test_date, color="0.8", alpha=0.5, label="data gap")
    axes[0].set_ylabel("Soil moisture")
    axes[0].set_title("Touchet soil moisture over time: ground truth vs model")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.20)
    for split_frame in full_prediction_parts:
        residual = split_frame["y_pred"] - split_frame["y_true"]
        axes[1].plot(split_frame["date"], residual, color="tab:red", linewidth=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axvspan(last_train_date, first_test_date, color="0.8", alpha=0.5)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.20)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    figure.tight_layout()
    figure.savefig(output_dir / "touchet_full_history_ground_truth_vs_model.png", dpi=170)
    plt.close(figure)

    shared = test[["date", TARGET]].rename(columns={TARGET: "y_true"})
    for label, frame in model_frames.items():
        shared = shared.merge(frame[["date", label]], on="date", how="left")
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(shared["date"], shared["y_true"], color="black", linewidth=2.0, label="ground truth")
    for label, color in (
        ("tabular_minus_smap_x_year", "tab:blue"),
        ("temporal_current", "tab:green"),
        ("temporal_alternative", "tab:olive"),
        ("strict_hybrid", "tab:red"),
    ):
        axes[0].plot(shared["date"], shared[label], linewidth=1.0, alpha=0.85, label=label)
        axes[1].plot(shared["date"], shared[label] - shared["y_true"], linewidth=1.0, label=label)
    warmup_end = first_test_date + pd.Timedelta(days=20)
    axes[0].axvspan(first_test_date, warmup_end, color="tab:orange", alpha=0.15, label="post-gap warm-up")
    axes[1].axvspan(first_test_date, warmup_end, color="tab:orange", alpha=0.15)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Soil moisture")
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Date")
    axes[0].set_title("Touchet test-period model comparison")
    axes[0].legend(ncol=2)
    axes[1].legend(ncol=2)
    axes[0].grid(alpha=0.20)
    axes[1].grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_dir / "touchet_test_model_comparison.png", dpi=170)
    plt.close(figure)

    distribution_features = [
        "SMAP_sm_interp",
        "SMAP_x_year",
        "V_rollstd_SMAP_sm_interp_kobs30",
        "F_NDVI",
        "LST_modis",
        "G_API",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, feature in zip(axes.flat, distribution_features):
        train_values = train[feature].dropna()
        test_values = test[feature].dropna()
        if len(train_values) and len(test_values):
            axis.hist(train_values, bins=30, density=True, alpha=0.50, label="train")
            axis.hist(test_values, bins=30, density=True, alpha=0.50, label="test")
        else:
            axis.text(
                0.5,
                0.5,
                "100% missing\nfor Touchet",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_title(feature)
        axis.grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend()
    figure.suptitle("Touchet feature distributions: train versus test")
    figure.tight_layout()
    figure.savefig(output_dir / "touchet_feature_distributions.png", dpi=160)
    plt.close(figure)

    legacy_stations = [
        "Darrington",
        "Quinault",
        "SourdoughGulch_WA_985",
        "Spokane",
        TOUCHET,
    ]
    test_five = splits["test"][splits["test"]["station_id"].isin(legacy_stations)]
    station_groups = [
        test_five.loc[test_five["station_id"] == station, TARGET].dropna().to_numpy()
        for station in legacy_stations
    ]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(station_groups, tick_labels=legacy_stations, showfliers=False)
    axis.set_ylabel("Soil moisture")
    axis.set_title("Test target distribution by original temporal station")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_dir / "target_distribution_by_station.png", dpi=160)
    plt.close(figure)

    hidden_summary_path = ARTIFACT_DIR / "hybrid" / "hidden_trace" / "hidden_trace_by_station.csv"
    hidden_summary = pd.read_csv(hidden_summary_path)
    hidden_summary.to_csv(output_dir / "hidden_trace_by_station.csv", index=False)

    summary = {
        "station": TOUCHET,
        "train_rows": len(train),
        "val_rows": len(touchet["val"]),
        "test_rows": len(test),
        "train_date_range": [train["date"].min(), train["date"].max()],
        "test_date_range": [test["date"].min(), test["date"].max()],
        "long_gap_days": long_gap_days,
        "train_target_mean": float(train[TARGET].mean()),
        "train_target_std": float(train[TARGET].std(ddof=0)),
        "test_target_mean": float(test[TARGET].mean()),
        "test_target_std": float(test[TARGET].std(ddof=0)),
        "metrics": metric_rows,
        "largest_feature_shifts": shift_frame.head(5).to_dict(orient="records"),
        "fully_missing_features": shift_frame.loc[
            (shift_frame["train_missing_fraction"] == 1.0)
            & (shift_frame["test_missing_fraction"] == 1.0),
            "feature",
        ].tolist(),
        "hidden_trace": hidden_summary[
            hidden_summary["station_id"] == TOUCHET
        ].to_dict(orient="records"),
    }
    save_json(summary, output_dir / "summary.json")
    print("\nTouchet metrics")
    print(metrics_frame[["model", "coverage_n", "r2", "rmse", "mae", "bias"]].to_string(index=False))
    print("\nLargest feature shifts")
    print(
        shift_frame[
            [
                "feature",
                "standardized_mean_difference",
                "wasserstein_distance",
                "test_outside_train_p1_p99",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    print(f"\nTouchet gap: {long_gap_days} days")


if __name__ == "__main__":
    main()
