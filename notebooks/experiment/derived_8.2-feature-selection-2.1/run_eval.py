#!/usr/bin/env python3
"""CLI re-eval for derived_8.2-feature-selection-2.1 (locked protocol, GPU when available).

Uses V6 feature lists under this experiment's artifacts/ (copied from 2.0; re-run run_selection.py to regenerate).
Does not re-run feature selection.

Usage (from repo root, notebooks uv/venv):
  PYTHONPATH=. notebooks/.venv/bin/python \\
    notebooks/experiment/derived_8.2-feature-selection-2.1/run_eval.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = Path(__file__).resolve().parent
OPT_1_0 = EXP_DIR.parent / "derived_8.0-optimization-1.0"
OUT_DIR = EXP_DIR / "artifacts" / "eval"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    dummy = xgb.XGBRegressor(n_estimators=1, device="cuda")
    dummy.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    XGB_DEVICE = "cuda"
except Exception as e:
    XGB_DEVICE = "cpu"
    print(f"XGBoost CUDA probe failed ({e}); falling back to CPU.")

XGB_PARAMS_LITE = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "min_child_weight": 10,
    "reg_lambda": 1.5,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 1500,
    "learning_rate": 0.01,
    "tree_method": "hist",
    "device": XGB_DEVICE,
    "n_jobs": -1,
    "random_state": SEED,
    "verbosity": 0,
}
BETA = 0.2
TARGET = "soil_moisture_5cm"
OPT10_HAND_DRIFT_R2 = 0.8253479076167946

HAND_MDR_V25 = [
    "SMAP_sm_pm_interp_ema02", "V_rollmin_LST_modis_kobs30", "D_sin_DOY", "G_rain_sum_3d",
    "V_ema_G_API_kobs7", "V_rollmin_G_API_kobs30", "G_rain_sum_7d", "C_lag_LST_modis_kobs30",
    "C_lag_G_API_kobs1", "V_ema_G_API_kobs14", "V_rollmean_G_API_kobs14", "G_API", "G_DSLR",
    "SMAP_ampm_diff_interp", "V_rollmax_G_API_kobs30", "V_ema_G_API_kobs30", "V_rollmean_s2_b11_kobs7",
    "V_ema_LST_modis_kobs7", "V_rollmean_G_API_kobs7", "C_lag_s2_b11_kobs30", "A_d_E_SAR_diff_kobs14",
    "C_lag_LST_modis_kobs6", "A_d_LST_modis_kobs14", "A_d_SMAP_sm_interp_kobs14",
    "V_rollstd_SMAP_sm_interp_kobs30", "SMAP_sm_interp_grad7", "year_frac", "sin_year", "cos_year",
    "API_x_year", "SMAP_x_year", "slope", "elev", "K_slope_sin", "K_slope_cos", "K_aspect_cos",
    "J_clay_wfrac_b0", "J_sand_wfrac_b0",
]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred
    ae = np.abs(err)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "ubRMSE": float(np.std(err)),
        "Bias": float(np.mean(err)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Med|Err|": float(np.median(ae)),
        "Pearson": float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1 and np.std(y_pred) > 0
        else 0.0,
    }


def temporal_weights(dates, beta=BETA):
    years = pd.to_datetime(dates).dt.year.to_numpy().astype(float)
    t_max = years.max()
    w = np.exp(beta * (years - t_max))
    w = w / w.mean()
    return np.asarray(w).ravel()


def load_splits(dataset):
    base = PROJECT_ROOT / "data" / "splits" / dataset
    train = pd.read_csv(base / "train.csv", parse_dates=["date"])
    val = pd.read_csv(base / "val.csv", parse_dates=["date"])
    test = pd.read_csv(base / "test.csv", parse_dates=["date"])
    return train, val, test


def load_feature_sets_for_dataset(dataset):
    sets = {}
    if dataset == "derived_8.0":
        sets["hand_mdr_v25"] = list(HAND_MDR_V25)
        opt = OPT_1_0 / "selected_features.json"
        if opt.exists():
            sets["opt1.0_pipeline"] = json.loads(opt.read_text())
    elif dataset == "derived_8.2":
        spec = importlib.util.spec_from_file_location(
            "dm82", PROJECT_ROOT / "data/splits/derived_8.2/dataset_metadata.py"
        )
        dm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dm)
        sets["V3_sota"] = list(dm.OVERALL_SELECTED_FEATURES_V3)
        sets["V5_bad"] = list(dm.OVERALL_SELECTED_FEATURES_V5)

    art = ARTIFACTS_DIR / dataset
    if art.is_dir():
        for path in sorted(art.glob("*/selected_features.json")):
            payload = json.loads(path.read_text())
            variant = payload.get("variant") or path.parent.name
            sets[f"v6_{variant}"] = list(payload["features"])
    return sets


def train_eval(train_df, val_df, test_df, features, weighted=True):
    tv = pd.concat([train_df, val_df], ignore_index=True)
    missing = [f for f in features if f not in tv.columns]
    if missing:
        raise ValueError(f"Missing features: {missing[:8]}...")

    X_tv = tv[features].apply(pd.to_numeric, errors="coerce")
    y_tv = pd.to_numeric(tv[TARGET], errors="coerce")
    X_te = test_df[features].apply(pd.to_numeric, errors="coerce")
    y_te = pd.to_numeric(test_df[TARGET], errors="coerce")

    tv_ok = y_tv.notna()
    te_ok = y_te.notna()
    X_tv, y_tv = X_tv.loc[tv_ok], y_tv.loc[tv_ok]
    X_te, y_te = X_te.loc[te_ok], y_te.loc[te_ok]
    dates_tv = tv.loc[tv_ok, "date"]

    w = temporal_weights(dates_tv, BETA) if weighted else None
    model = xgb.XGBRegressor(**XGB_PARAMS_LITE)
    model.fit(X_tv, y_tv, sample_weight=w)
    pred = np.asarray(model.predict(X_te)).ravel()
    metrics = compute_metrics(y_te, pred)

    years = pd.to_datetime(test_df.loc[te_ok, "date"]).dt.year.to_numpy()
    y_te_np = np.asarray(y_te, dtype=float).ravel()
    by_year = {}
    for yr in sorted(np.unique(years)):
        m = years == yr
        by_year[int(yr)] = compute_metrics(y_te_np[m], pred[m])
    return metrics, by_year, pred


def best_v6(df, dataset):
    sub = df[(df.dataset == dataset) & (df.feature_set.str.startswith("v6_"))]
    if sub.empty:
        return None
    return sub.loc[sub["R2"].idxmax()]


def gate_report(summary_slice: pd.DataFrame) -> dict:
    report = {}
    s80 = summary_slice[summary_slice.dataset == "derived_8.0"]
    hand = s80[s80.feature_set == "hand_mdr_v25"]
    v6 = best_v6(summary_slice, "derived_8.0")
    if len(hand) and v6 is not None:
        hand_r2 = float(hand.iloc[0]["R2"])
        v6_r2 = float(v6["R2"])
        report["8.0"] = {
            "hand_r2": hand_r2,
            "best_v6": v6["feature_set"],
            "best_v6_r2": v6_r2,
            "delta": v6_r2 - hand_r2,
            "pass": bool(v6_r2 >= hand_r2 - 0.01),
        }

    s82 = summary_slice[summary_slice.dataset == "derived_8.2"]
    v3 = s82[s82.feature_set == "V3_sota"]
    v6b = best_v6(summary_slice, "derived_8.2")
    if len(v3) and v6b is not None:
        v3_r2 = float(v3.iloc[0]["R2"])
        v6_r2 = float(v6b["R2"])
        report["8.2"] = {
            "v3_r2": v3_r2,
            "best_v6": v6b["feature_set"],
            "best_v6_r2": v6_r2,
            "delta": v6_r2 - v3_r2,
            "pass": bool(v6_r2 >= v3_r2 - 0.01),
        }
    return report


def main() -> None:
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"OUT_DIR={OUT_DIR}")
    print(f"SEED={SEED} XGB_DEVICE={XGB_DEVICE} xgb={xgb.__version__}")

    all_rows = []
    year_rows = []

    for dataset in ["derived_8.0", "derived_8.2"]:
        train_df, val_df, test_df = load_splits(dataset)
        fsets = load_feature_sets_for_dataset(dataset)
        print(f"\n=== {dataset}: {len(fsets)} feature sets × 2 protocols ===")
        for name, feats in fsets.items():
            for weighted in (True, False):
                label = "drift" if weighted else "no-drift"
                print(f"  training {name} (n={len(feats)}, {label}) ...", flush=True)
                try:
                    metrics, by_year, _ = train_eval(
                        train_df, val_df, test_df, feats, weighted=weighted
                    )
                except Exception as e:
                    print(f"    FAILED: {e}")
                    continue
                all_rows.append(
                    {
                        "dataset": dataset,
                        "feature_set": name,
                        "n_features": len(feats),
                        "weighted": weighted,
                        **metrics,
                    }
                )
                for yr, m in by_year.items():
                    year_rows.append(
                        {
                            "dataset": dataset,
                            "feature_set": name,
                            "weighted": weighted,
                            "year": yr,
                            **m,
                        }
                    )
                print(
                    f"    R2={metrics['R2']:.4f} RMSE={metrics['RMSE']:.4f} "
                    f"MAE={metrics['MAE']:.4f} Pearson={metrics['Pearson']:.4f}"
                )

    summary = pd.DataFrame(all_rows)
    by_year_df = pd.DataFrame(year_rows)
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    by_year_df.to_csv(OUT_DIR / "metrics_by_year.csv", index=False)

    gates = {
        "with_drift": gate_report(summary[summary.weighted]),
        "no_drift": gate_report(summary[~summary.weighted]),
        "meta": {
            "protocol": "2.1",
            "device": XGB_DEVICE,
            "seed": SEED,
            "beta": BETA,
            "weights": "mean_normalized_exp",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "opt1.0_hand_drift_r2_ref": OPT10_HAND_DRIFT_R2,
        },
    }
    (OUT_DIR / "success_gates.json").write_text(json.dumps(gates, indent=2))
    print(json.dumps(gates, indent=2))

    hand_drift = summary[
        (summary.dataset == "derived_8.0")
        & (summary.feature_set == "hand_mdr_v25")
        & (summary.weighted)
    ]
    if len(hand_drift):
        hr2 = float(hand_drift.iloc[0]["R2"])
        delta = hr2 - OPT10_HAND_DRIFT_R2
        print(
            f"\nSanity hand+drift R2={hr2:.4f} vs opt-1.0 Model5={OPT10_HAND_DRIFT_R2:.4f} "
            f"(Δ={delta:+.4f})"
        )

    protocols = [(True, "With drift (β=0.2, mean-norm)"), (False, "No drift (unweighted)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row_i, (weighted, prot_title) in enumerate(protocols):
        for col_i, ds in enumerate(["derived_8.0", "derived_8.2"]):
            ax = axes[row_i, col_i]
            sub = summary[(summary.dataset == ds) & (summary.weighted == weighted)].sort_values(
                "R2"
            )
            if sub.empty:
                ax.set_title(f"{ds}: no data")
                continue
            ax.barh(sub["feature_set"], sub["R2"])
            ax.set_title(f"{ds} — {prot_title}")
            ax.set_xlabel("Test R²")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "r2_comparison.png", dpi=140)
    plt.close(fig)

    for weighted, tag in [(True, "weighted"), (False, "unweighted")]:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        for ax, ds in zip(axes2, ["derived_8.0", "derived_8.2"]):
            sub = summary[(summary.dataset == ds) & (summary.weighted == weighted)].sort_values(
                "R2"
            )
            if sub.empty:
                continue
            ax.barh(sub["feature_set"], sub["R2"])
            ax.set_title(f"{ds} test R² ({tag})")
            ax.set_xlabel("R²")
        plt.tight_layout()
        fig2.savefig(OUT_DIR / f"r2_comparison_{tag}.png", dpi=140)
        plt.close(fig2)

    print(f"\nWrote metrics + gates + figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
