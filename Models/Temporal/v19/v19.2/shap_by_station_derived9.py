#!/usr/bin/env python3
"""Per-station SHAP feature importance for the v19.2 weighted drift model.

This follows the v19.2 notebook setup, but reads the derived_9.0 split and
writes one mean(|SHAP|) bar plot per station from the test split.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor


SEED = 42
TARGET_COL = "soil_moisture_5cm"
KEEP_META_COLS = ["station_id", "date", "longitude", "latitude"]

FEATURE_COLS = [
    "SMAP_sm_pm_interp_ema02",
    "V_rollmin_LST_modis_kobs30",
    "D_sin_DOY",
    "G_rain_sum_3d",
    "V_ema_G_API_kobs7",
    "V_rollmin_G_API_kobs30",
    "G_rain_sum_7d",
    "C_lag_LST_modis_kobs30",
    "C_lag_G_API_kobs1",
    "V_ema_G_API_kobs14",
    "V_rollmean_G_API_kobs14",
    "G_API",
    "G_DSLR",
    "SMAP_ampm_diff_interp",
    "V_rollmax_G_API_kobs30",
    "V_ema_G_API_kobs30",
    "V_rollmean_s2_b11_kobs7",
    "V_ema_LST_modis_kobs7",
    "V_rollmean_G_API_kobs7",
    "C_lag_s2_b11_kobs30",
    "A_d_E_SAR_diff_kobs14",
    "C_lag_LST_modis_kobs6",
    "A_d_LST_modis_kobs14",
    "A_d_SMAP_sm_interp_kobs14",
    "V_rollstd_SMAP_sm_interp_kobs30",
    "SMAP_sm_interp_grad7",
    "year_frac",
    "sin_year",
    "cos_year",
    "API_x_year",
    "SMAP_x_year",
    "slope",
    "elev",
    "K_slope_sin",
    "K_slope_cos",
    "K_aspect_cos",
    "J_clay_wfrac_b0",
    "J_sand_wfrac_b0",
]

XGB_PARAMS_DRIFT_W = dict(
    objective="reg:pseudohubererror",
    random_state=SEED,
    n_jobs=-1,
    subsample=0.9,
    colsample_bytree=0.8,
    max_depth=8,
    min_child_weight=2,
    n_estimators=5500,
    learning_rate=0.04,
    reg_lambda=1.5,
    reg_alpha=0.03,
    gamma=0.0,
)


def find_project_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for cand in candidates:
        if (cand / "Temporal/Pipeline/data").exists() and (cand / "Models/Temporal").exists():
            return cand
    raise FileNotFoundError("Could not locate repo root containing Temporal/Pipeline/data")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def read_split(split_dir: Path, name: str) -> pd.DataFrame:
    cols = KEEP_META_COLS + FEATURE_COLS + [TARGET_COL]
    df = pd.read_csv(split_dir / f"{name}.csv", usecols=cols)
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{name}.csv is missing columns: {missing}")
    return df


def recency_weights(trainval_df: pd.DataFrame, beta: float) -> np.ndarray:
    years = pd.to_datetime(trainval_df["date"], errors="coerce").dt.year.astype(float)
    max_year = years.max()
    weights = np.exp(beta * (years - max_year))
    return np.asarray(weights / weights.mean())


def mean_abs_shap(booster: xgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    dmatrix = xgb.DMatrix(df[FEATURE_COLS], feature_names=FEATURE_COLS)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    shap_values = contribs[:, :-1]
    values = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": values})
    out["normalized_mean_abs_shap"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_station_importance(importance: pd.DataFrame, station: str, n_rows: int, top_k: int, out_path: Path) -> None:
    plot_df = importance.head(top_k).iloc[::-1]

    height = max(5.0, 0.34 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#3f7f93")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"{station} - Top {len(plot_df)} SHAP Features (n={n_rows})")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="derived_9.0", help="Split folder under Temporal/Pipeline/data/splits")
    parser.add_argument("--top-k", type=int, default=20, help="Number of features to show per station plot")
    parser.add_argument("--max-rows-per-station", type=int, default=1000, help="Sample cap for SHAP rows per station")
    parser.add_argument("--beta", type=float, default=0.2, help="v19.2 recency weight beta")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    project_root = find_project_root()
    split_dir = project_root / "Temporal/Pipeline/data/splits" / args.split
    output_dir = project_root / "Models/Temporal/v19/v19.2/shap_by_station_derived9"
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading split: {split_dir}")
    train_df = read_split(split_dir, "train")
    val_df = read_split(split_dir, "val")
    test_df = read_split(split_dir, "test")

    trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    weights = recency_weights(trainval_df, beta=args.beta)

    print(f"Training weighted v19.2 model on {len(trainval_df):,} rows and {len(FEATURE_COLS)} features")
    model = XGBRegressor(**XGB_PARAMS_DRIFT_W)
    model.fit(
        trainval_df[FEATURE_COLS],
        trainval_df[TARGET_COL],
        sample_weight=weights,
        verbose=0,
    )

    booster = model.get_booster()
    summaries = []

    for station in sorted(test_df["station_id"].dropna().unique()):
        station_df = test_df[test_df["station_id"] == station].copy()
        n_total = len(station_df)
        if args.max_rows_per_station and n_total > args.max_rows_per_station:
            station_df = station_df.sample(n=args.max_rows_per_station, random_state=SEED)

        importance = mean_abs_shap(booster, station_df)
        safe_station = slugify(station)
        csv_path = tables_dir / f"{safe_station}_shap_importance.csv"
        png_path = plots_dir / f"{safe_station}_shap_top{args.top_k}.png"

        importance.to_csv(csv_path, index=False)
        plot_station_importance(importance, station, len(station_df), args.top_k, png_path)

        top = importance.iloc[0]
        summaries.append(
            {
                "station_id": station,
                "rows_total": n_total,
                "rows_shap": len(station_df),
                "top_feature": top["feature"],
                "top_mean_abs_shap": top["mean_abs_shap"],
                "plot_path": str(png_path.relative_to(project_root)),
                "table_path": str(csv_path.relative_to(project_root)),
            }
        )
        print(f"Saved {station}: {png_path}")

    summary = pd.DataFrame(summaries)
    summary_path = output_dir / "station_shap_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
