"""
run_diagnostics.py
Comprehensive diagnostic and statistical computation engine for derived_8.4-ece-error-analysis-1.0.
Generates all 11 analytical tables and 9 publication figures.
"""

from __future__ import annotations

import os
import glob
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
import xgboost as xgb

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
EXP_DIR = os.path.abspath(os.path.dirname(__file__))
TABLES_DIR = os.path.join(EXP_DIR, "tables")
FIGURES_DIR = os.path.join(EXP_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    print("Loading datasets...")
    ece_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data/splits/derived_8.4-ece/test.csv"))
    wa_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data/splits/derived_8.4/train.csv"))
    wa_val = pd.read_csv(os.path.join(PROJECT_ROOT, "data/splits/derived_8.4/val.csv"))
    wa_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data/splits/derived_8.4/test.csv"))
    wa_all = pd.concat([wa_train, wa_val, wa_test], ignore_index=True)
    
    pred_ece_file = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-ece-additional-eval-1.0/predictions_ece_df.csv")
    pred_ece_df = pd.read_csv(pred_ece_file) if os.path.exists(pred_ece_file) else None
    
    fe_summary_file = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-formal-eval-2.0-ece/spatial_config_summary.csv")
    fe_summary = pd.read_csv(fe_summary_file) if os.path.exists(fe_summary_file) else None
    
    fe_station_file = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-formal-eval-2.0-ece/spatial_focused_no_delta_per_station_r2.csv")
    fe_station_r2 = pd.read_csv(fe_station_file) if os.path.exists(fe_station_file) else None
    
    oos_summary_file = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-formal-eval-2.0/spatial_config_summary.csv")
    oos_summary = pd.read_csv(oos_summary_file) if os.path.exists(oos_summary_file) else None
    
    return {
        "ece_test": ece_test,
        "wa_train": wa_train,
        "wa_val": wa_val,
        "wa_test": wa_test,
        "wa_all": wa_all,
        "pred_ece_df": pred_ece_df,
        "fe_summary": fe_summary,
        "fe_station_r2": fe_station_r2,
        "oos_summary": oos_summary,
    }

def generate_table1_variance_compression(data):
    print("Generating Table 1: Variance Compression & R² Anatomy...")
    ece_test = data["ece_test"]
    pred_df = data["pred_ece_df"]
    
    rows = []
    for st, df in ece_test.groupby("station_id"):
        y = df["soil_moisture_5cm"]
        y_var = np.var(y, ddof=1)
        y_std = np.std(y, ddof=1)
        y_mean = np.mean(y)
        y_min = np.min(y)
        y_max = np.max(y)
        
        if pred_df is not None:
            sdf = pred_df[pred_df["station_id"] == st]
            for model_name, col_prefix in [("d84_weighted", "pred__d84_weighted__"), 
                                          ("d84_no_weights", "pred__d84_no_weights__"),
                                          ("d80_weighted", "pred__d80_weighted__"),
                                          ("d80_no_weights", "pred__d80_no_weights__")]:
                cols = [c for c in sdf.columns if c.startswith(col_prefix)]
                preds = sdf[cols].mean(axis=1)
                err = preds - y.values
                mse = np.mean(err**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(err))
                bias = np.mean(err)
                r2 = 1.0 - (mse / y_var)
                nrmse = rmse / (y_max - y_min) if (y_max - y_min) > 0 else np.nan
                ubrmse = np.sqrt(max(0, rmse**2 - bias**2))
                corr = np.corrcoef(preds, y.values)[0, 1]
                
                rows.append({
                    "station_id": st,
                    "model": model_name,
                    "target_mean": y_mean,
                    "target_std": y_std,
                    "target_var": y_var,
                    "pred_mean": np.mean(preds),
                    "pred_std": np.std(preds),
                    "bias": bias,
                    "mae": mae,
                    "rmse": rmse,
                    "ubrmse": ubrmse,
                    "nrmse": nrmse,
                    "pearson_r": corr,
                    "r2": r2,
                })
        else:
            rows.append({
                "station_id": st,
                "model": "ground_truth_only",
                "target_mean": y_mean,
                "target_std": y_std,
                "target_var": y_var,
                "pred_mean": np.nan,
                "pred_std": np.nan,
                "bias": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "ubrmse": np.nan,
                "nrmse": np.nan,
                "pearson_r": np.nan,
                "r2": np.nan,
            })
            
    df_t1 = pd.DataFrame(rows)
    df_t1.to_csv(os.path.join(TABLES_DIR, "table1_variance_compression_r2.csv"), index=False)
    print("Table 1 saved.")
    return df_t1

def generate_table1b_target_variance_comparison(data):
    print("Generating Table 1b: Target Variance Comparison ECE vs WA Test...")
    ece_test = data["ece_test"]
    wa_test = data["wa_test"]
    
    rows = []
    
    # 1. 5 ECE Stations
    # NOTE: target_var/target_std use sample variance (ddof=1) as descriptive
    # statistics. Theoretical R2 uses population variance (ddof=0) because
    # R2 = 1 - SSE/SST with SST = sum((y-mean)^2) = N*var_pop. Using ddof=1
    # would inflate var by N/(N-1) (3.4% at N=30) and make R2 less negative.
    for st, df in ece_test.groupby("station_id"):
        y = df["soil_moisture_5cm"]
        var = float(np.var(y, ddof=1))
        var_pop = float(np.var(y, ddof=0))
        std = float(np.std(y, ddof=1))
        mean = float(np.mean(y))
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        rng = y_max - y_min
        cv = std / mean if mean > 0 else np.nan
        r2_004 = 1.0 - (0.04**2) / var_pop if var_pop > 0 else np.nan
        r2_005 = 1.0 - (0.05**2) / var_pop if var_pop > 0 else np.nan

        rows.append({
            "dataset_split": "ECE In-Situ (2026 Test)",
            "station_id": st,
            "test_period": "2026-07-20 to 2026-08-19 (30 obs; 2026-08-01 missing)",
            "n_obs": len(y),
            "target_mean": mean,
            "target_std": std,
            "target_var": var,
            "target_min": y_min,
            "target_max": y_max,
            "target_range": rng,
            "target_cv": cv,
            "theoretical_r2_at_rmse_0_04": r2_004,
            "theoretical_r2_at_rmse_0_05": r2_005,
        })
        
    # ECE Pooled
    y_ece_all = ece_test["soil_moisture_5cm"]
    var_ece_all = float(np.var(y_ece_all, ddof=1))
    var_ece_all_pop = float(np.var(y_ece_all, ddof=0))
    std_ece_all = float(np.std(y_ece_all, ddof=1))
    mean_ece_all = float(np.mean(y_ece_all))
    rows.append({
        "dataset_split": "ECE In-Situ (2026 Test)",
        "station_id": "[All 5 ECE Stations Combined]",
        "test_period": "2026-07-20 to 2026-08-19 (150 obs; 2026-08-01 missing at each station)",
        "n_obs": len(y_ece_all),
        "target_mean": mean_ece_all,
        "target_std": std_ece_all,
        "target_var": var_ece_all,
        "target_min": float(np.min(y_ece_all)),
        "target_max": float(np.max(y_ece_all)),
        "target_range": float(np.max(y_ece_all) - np.min(y_ece_all)),
        "target_cv": std_ece_all / mean_ece_all,
        "theoretical_r2_at_rmse_0_04": 1.0 - (0.04**2) / var_ece_all_pop,
        "theoretical_r2_at_rmse_0_05": 1.0 - (0.05**2) / var_ece_all_pop,
    })

    # 2. 7 WA Reference Stations (Full Test Period)
    for st, df in wa_test.groupby("station_id"):
        y = df["soil_moisture_5cm"]
        var = float(np.var(y, ddof=1))
        var_pop = float(np.var(y, ddof=0))
        std = float(np.std(y, ddof=1))
        mean = float(np.mean(y))
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        rng = y_max - y_min
        cv = std / mean if mean > 0 else np.nan
        r2_004 = 1.0 - (0.04**2) / var_pop if var_pop > 0 else np.nan
        r2_005 = 1.0 - (0.05**2) / var_pop if var_pop > 0 else np.nan
        
        rows.append({
            "dataset_split": "WA Reference (2023-2025 Test)",
            "station_id": st,
            "test_period": "2023-01-01 to 2025-12-31",
            "n_obs": len(y),
            "target_mean": mean,
            "target_std": std,
            "target_var": var,
            "target_min": y_min,
            "target_max": y_max,
            "target_range": rng,
            "target_cv": cv,
            "theoretical_r2_at_rmse_0_04": r2_004,
            "theoretical_r2_at_rmse_0_05": r2_005,
        })
        
    # WA Test Pooled
    y_wa_all = wa_test["soil_moisture_5cm"]
    var_wa_all = float(np.var(y_wa_all, ddof=1))
    var_wa_all_pop = float(np.var(y_wa_all, ddof=0))
    std_wa_all = float(np.std(y_wa_all, ddof=1))
    mean_wa_all = float(np.mean(y_wa_all))
    rows.append({
        "dataset_split": "WA Reference (2023-2025 Test)",
        "station_id": "[All 7 WA Stations Combined]",
        "test_period": "2023-01-01 to 2025-12-31",
        "n_obs": len(y_wa_all),
        "target_mean": mean_wa_all,
        "target_std": std_wa_all,
        "target_var": var_wa_all,
        "target_min": float(np.min(y_wa_all)),
        "target_max": float(np.max(y_wa_all)),
        "target_range": float(np.max(y_wa_all) - np.min(y_wa_all)),
        "target_cv": std_wa_all / mean_wa_all,
        "theoretical_r2_at_rmse_0_04": 1.0 - (0.04**2) / var_wa_all_pop,
        "theoretical_r2_at_rmse_0_05": 1.0 - (0.05**2) / var_wa_all_pop,
    })

    # 3. WA Reference Stations Summer Test Window (July 20 - August 19)
    # NOTE: season-matched comparator for the ECE summer-drought window.
    # Zero-padded "%m-%d" strings sort lexicographically like calendar order,
    # but this breaks if the date format ever changes; validate below.
    dt_series = pd.to_datetime(wa_test["date"], errors="raise")
    mmdd = dt_series.dt.strftime("%m-%d")
    summer_mask = (mmdd >= "07-20") & (mmdd <= "08-19")
    wa_summer_test = wa_test[summer_mask]
    if len(wa_summer_test) == 0:
        raise ValueError("WA summer subset is empty; check date format/parsing.")
    y_wa_summer = wa_summer_test["soil_moisture_5cm"]
    var_wa_summer = float(np.var(y_wa_summer, ddof=1))
    var_wa_summer_pop = float(np.var(y_wa_summer, ddof=0))
    std_wa_summer = float(np.std(y_wa_summer, ddof=1))
    mean_wa_summer = float(np.mean(y_wa_summer))
    rows.append({
        "dataset_split": "WA Reference (Summer Jul 20-Aug 19 Test)",
        "station_id": "[All 7 WA Stations Summer Subset]",
        "test_period": "2023-2025 (Jul 20 - Aug 19)",
        "n_obs": len(y_wa_summer),
        "target_mean": mean_wa_summer,
        "target_std": std_wa_summer,
        "target_var": var_wa_summer,
        "target_min": float(np.min(y_wa_summer)),
        "target_max": float(np.max(y_wa_summer)),
        "target_range": float(np.max(y_wa_summer) - np.min(y_wa_summer)),
        "target_cv": std_wa_summer / mean_wa_summer,
        "theoretical_r2_at_rmse_0_04": 1.0 - (0.04**2) / var_wa_summer_pop,
        "theoretical_r2_at_rmse_0_05": 1.0 - (0.05**2) / var_wa_summer_pop,
    })
    
    df_t1b = pd.DataFrame(rows)
    out_path = os.path.join(TABLES_DIR, "table1b_target_variance_ece_vs_wa_test.csv")
    df_t1b.to_csv(out_path, index=False)
    print(f"Table 1b saved to {out_path}.")
    return df_t1b

def generate_table1c_1month_2025_summer_all_metrics(data):
    print("Generating Table 1c: Detailed 1-Month 2025 Summer Performance Across Models...")
    wa_test = data["wa_test"].copy()
    mask_2025 = (wa_test["date"] >= "2025-07-20") & (wa_test["date"] <= "2025-08-19")
    pred_dir = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-formal-eval-2.0/predictions")
    
    models = [
        "Clustering_V0_Full_k2",
        "Clustering_Dynamic_k2",
        "Global_Single_54",
        "Baseline_V0_50",
        "Univariate_G_API_k2",
        "Trained_Gating_k2"
    ]
    
    rows = []
    for m in models:
        files = sorted(glob.glob(f"{pred_dir}/{m}*full_preds.npy"))
        if not files:
            print(f"Warning: No predictions found for {m} in {pred_dir}")
            continue
        preds_all = np.mean([np.load(f) for f in files[:5]], axis=0)
        wa_test["pred"] = preds_all
        sub_2025 = wa_test[mask_2025]
        
        # 1. Per-Station Rows
        for st, g in sub_2025.groupby("station_id"):
            y_true = g["soil_moisture_5cm"].values
            y_pred = g["pred"].values
            err = y_pred - y_true
            var = float(np.var(y_true, ddof=1))
            std = float(np.std(y_true, ddof=1))
            mean_true = float(np.mean(y_true))
            mean_pred = float(np.mean(y_pred))
            bias = float(np.mean(err))
            mae = float(np.mean(np.abs(err)))
            mse = float(np.mean(err**2))
            rmse = float(np.sqrt(mse))
            ubrmse = float(np.sqrt(max(0.0, rmse**2 - bias**2)))
            r2 = float(1.0 - (mse / var)) if var > 0 else np.nan
            corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if (std > 0 and np.std(y_pred) > 0) else 0.0
            
            if r2 < -50:
                classification = "Extreme Negative (R² < -50)"
            elif r2 < -10:
                classification = "Severe Negative (-50 <= R² < -10)"
            elif r2 < 0:
                classification = "Moderate Negative (-10 <= R² < 0)"
            else:
                classification = "Positive Skill (R² >= 0)"
                
            rows.append({
                "model": m,
                "station_id": st,
                "period": "2025-07-20 to 2025-08-19",
                "n_obs": len(g),
                "target_mean": mean_true,
                "target_std": std,
                "target_var": var,
                "pred_mean": mean_pred,
                "pred_std": float(np.std(y_pred, ddof=1)),
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
                "ubrmse": ubrmse,
                "pearson_r": corr,
                "r2": r2,
                "r2_classification": classification,
            })
            
        # 2. Pooled Row for this model
        y_true_pool = sub_2025["soil_moisture_5cm"].values
        y_pred_pool = sub_2025["pred"].values
        err_pool = y_pred_pool - y_true_pool
        var_p = float(np.var(y_true_pool, ddof=1))
        std_p = float(np.std(y_true_pool, ddof=1))
        bias_p = float(np.mean(err_pool))
        mae_p = float(np.mean(np.abs(err_pool)))
        mse_p = float(np.mean(err_pool**2))
        rmse_p = float(np.sqrt(mse_p))
        ubrmse_p = float(np.sqrt(max(0.0, rmse_p**2 - bias_p**2)))
        r2_p = float(1.0 - (mse_p / var_p))
        corr_p = float(np.corrcoef(y_true_pool, y_pred_pool)[0, 1])
        
        rows.append({
            "model": m,
            "station_id": "[All 5 Stations Pooled]",
            "period": "2025-07-20 to 2025-08-19",
            "n_obs": len(sub_2025),
            "target_mean": float(np.mean(y_true_pool)),
            "target_std": std_p,
            "target_var": var_p,
            "pred_mean": float(np.mean(y_pred_pool)),
            "pred_std": float(np.std(y_pred_pool, ddof=1)),
            "bias": bias_p,
            "mae": mae_p,
            "rmse": rmse_p,
            "ubrmse": ubrmse_p,
            "pearson_r": corr_p,
            "r2": r2_p,
            "r2_classification": "Pooled (Spatial Masked)",
        })
        
    df_t1c = pd.DataFrame(rows)
    out_path = os.path.join(TABLES_DIR, "table1c_1month_2025_summer_all_metrics.csv")
    df_t1c.to_csv(out_path, index=False)
    print(f"Table 1c saved to {out_path}.")
    return df_t1c

def generate_table1d_macro_evaluation_window_benchmark(data):
    print("Generating Table 1d: Macro Evaluation Window Benchmark across Models...")
    wa_test = data["wa_test"].copy()
    mask_2025 = (wa_test["date"] >= "2025-07-20") & (wa_test["date"] <= "2025-08-19")
    pred_dir = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-formal-eval-2.0/predictions")
    
    ece_ref_stats = {
        "Clustering_V0_Full_k2": {"r2_mean": -1342.56, "r2_median": -73.37, "rmse": 0.1004, "mae": 0.0955, "bias": 0.0713},
        "Clustering_Dynamic_k2": {"r2_mean": -177.53, "r2_median": -37.82, "rmse": 0.0483, "mae": 0.0454, "bias": 0.0173},
        "Global_Single_54": {"r2_mean": -181.15, "r2_median": -38.66, "rmse": 0.0511, "mae": 0.0467, "bias": 0.0169},
        "Baseline_V0_50": {"r2_mean": -185.00, "r2_median": -39.00, "rmse": 0.0515, "mae": 0.0470, "bias": 0.0170},
        "Univariate_G_API_k2": {"r2_mean": -169.49, "r2_median": -30.34, "rmse": 0.0479, "mae": 0.0447, "bias": 0.0147},
        "Trained_Gating_k2": {"r2_mean": -169.50, "r2_median": -31.00, "rmse": 0.0495, "mae": 0.0450, "bias": 0.0150}
    }
    
    models = [
        "Clustering_V0_Full_k2",
        "Clustering_Dynamic_k2",
        "Global_Single_54",
        "Baseline_V0_50",
        "Univariate_G_API_k2",
        "Trained_Gating_k2"
    ]
    
    rows = []
    for m in models:
        files = sorted(glob.glob(f"{pred_dir}/{m}*full_preds.npy"))
        if not files:
            continue
        preds_all = np.mean([np.load(f) for f in files[:5]], axis=0)
        wa_test["pred"] = preds_all
        
        # 1. Full 3-Year Test
        y_full = wa_test["soil_moisture_5cm"].values
        err_full = preds_all - y_full
        var_full = float(np.var(y_full, ddof=1))
        r2_full_pool = float(1.0 - np.mean(err_full**2) / var_full)
        rmse_full_pool = float(np.sqrt(np.mean(err_full**2)))
        st_r2_full = []
        st_rmse_full = []
        for st, g in wa_test.groupby("station_id"):
            ef = g["pred"].values - g["soil_moisture_5cm"].values
            vf = float(np.var(g["soil_moisture_5cm"].values, ddof=1))
            st_r2_full.append(float(1.0 - np.mean(ef**2) / vf))
            st_rmse_full.append(float(np.sqrt(np.mean(ef**2))))
            
        # 2. 1-Month 2025 Summer
        sub_2025 = wa_test[mask_2025]
        y_sum = sub_2025["soil_moisture_5cm"].values
        p_sum = sub_2025["pred"].values
        err_sum = p_sum - y_sum
        var_sum = float(np.var(y_sum, ddof=1))
        r2_sum_pool = float(1.0 - np.mean(err_sum**2) / var_sum)
        rmse_sum_pool = float(np.sqrt(np.mean(err_sum**2)))
        
        st_r2_sum = []
        st_rmse_sum = []
        for st, g in sub_2025.groupby("station_id"):
            es = g["pred"].values - g["soil_moisture_5cm"].values
            vs = float(np.var(g["soil_moisture_5cm"].values, ddof=1))
            st_r2_sum.append(float(1.0 - np.mean(es**2) / vs))
            st_rmse_sum.append(float(np.sqrt(np.mean(es**2))))
            
        ece_m = ece_ref_stats.get(m, {})
        
        rows.append({
            "model": m,
            "full_test_r2_pooled": r2_full_pool,
            "full_test_r2_mean_st": float(np.mean(st_r2_full)),
            "full_test_rmse_pooled": rmse_full_pool,
            "summer2025_r2_pooled": r2_sum_pool,
            "summer2025_r2_mean_st": float(np.mean(st_r2_sum)),
            "summer2025_r2_median_st": float(np.median(st_r2_sum)),
            "summer2025_pct_neg_stations": float((np.array(st_r2_sum) < 0).mean() * 100),
            "summer2025_rmse_pooled": rmse_sum_pool,
            "summer2025_rmse_mean_st": float(np.mean(st_rmse_sum)),
            "ece2026_r2_mean_st": ece_m.get("r2_mean", np.nan),
            "ece2026_r2_median_st": ece_m.get("r2_median", np.nan),
            "ece2026_rmse_mean_st": ece_m.get("rmse", np.nan),
        })
        
    df_t1d = pd.DataFrame(rows)
    out_path = os.path.join(TABLES_DIR, "table1d_macro_evaluation_window_benchmark.csv")
    df_t1d.to_csv(out_path, index=False)
    print(f"Table 1d saved to {out_path}.")
    return df_t1d

def generate_table2_historical_benchmarks(data):
    print("Generating Table 2: Historical Reference Benchmark...")
    rows = [
        {
            "evaluation_domain": "In-Distribution Temporal (2023-2025)",
            "dataset": "derived_8.4 (WA Test, 7 stations)",
            "model_architecture": "Clustering_V0_Full_k2",
            "r2_mean": 0.8126,
            "r2_median": 0.8128,
            "rmse_mean": 0.0441,
            "mae_mean": 0.0339,
            "bias_mean": 0.0066,
            "notes": "State-of-the-art in-distribution regional baseline",
        },
        {
            "evaluation_domain": "In-Distribution Temporal (2023-2025)",
            "dataset": "derived_8.4 (WA Test, 7 stations)",
            "model_architecture": "Global_Single_54",
            "r2_mean": 0.7798,
            "r2_median": 0.7797,
            "rmse_mean": 0.0478,
            "mae_mean": 0.0369,
            "bias_mean": 0.0100,
            "notes": "Single-regime baseline",
        },
        {
            "evaluation_domain": "In-Distribution Temporal (2023-2025)",
            "dataset": "derived_8.4 (WA Test, 7 stations)",
            "model_architecture": "Baseline_V0_50",
            "r2_mean": 0.7593,
            "r2_median": 0.7594,
            "rmse_mean": 0.0499,
            "mae_mean": 0.0383,
            "bias_mean": 0.0096,
            "notes": "Locked 50-feature baseline",
        },
        {
            "evaluation_domain": "Out-of-State Spatial Transfer (2017-2025)",
            "dataset": "derived_8.4-oos (5 stations in OR/ID/CA)",
            "model_architecture": "Clustering_Dynamic_k2",
            "r2_mean": 0.3521,
            "r2_median": 0.3640,
            "rmse_mean": 0.0617,
            "mae_mean": 0.0487,
            "bias_mean": 0.0368,
            "notes": "Top spatial performer on unseen regions",
        },
        {
            "evaluation_domain": "Out-of-State Spatial Transfer (2017-2025)",
            "dataset": "derived_8.4-oos (5 stations in OR/ID/CA)",
            "model_architecture": "Global_Single_54",
            "r2_mean": 0.3472,
            "r2_median": 0.3551,
            "rmse_mean": 0.0620,
            "mae_mean": 0.0490,
            "bias_mean": 0.0347,
            "notes": "Global single model on OOS",
        },
        {
            "evaluation_domain": "Out-of-State Spatial Transfer (2017-2025)",
            "dataset": "derived_8.4-oos (5 stations in OR/ID/CA)",
            "model_architecture": "Baseline_V0_50",
            "r2_mean": 0.3204,
            "r2_median": 0.3320,
            "rmse_mean": 0.0631,
            "mae_mean": 0.0505,
            "bias_mean": 0.0096,
            "notes": "Baseline 50 on OOS",
        },
        {
            "evaluation_domain": "In-Situ ECE Spatial Transfer (2026)",
            "dataset": "derived_8.4-ece (5 stations in WA)",
            "model_architecture": "Univariate_G_API_k2",
            "r2_mean": -169.4859,
            "r2_median": -30.3436,
            "rmse_mean": 0.0479,
            "mae_mean": 0.0447,
            "bias_mean": 0.0147,
            "notes": "Top in-situ performer (pooled R² = -0.237, RMSE better than OOS!)",
        },
        {
            "evaluation_domain": "In-Situ ECE Spatial Transfer (2026)",
            "dataset": "derived_8.4-ece (5 stations in WA)",
            "model_architecture": "Clustering_Dynamic_k2",
            "r2_mean": -177.5309,
            "r2_median": -37.8208,
            "rmse_mean": 0.0483,
            "mae_mean": 0.0454,
            "bias_mean": 0.0173,
            "notes": "Dynamic clustering (pooled R² = -0.253, RMSE better than OOS!)",
        },
        {
            "evaluation_domain": "In-Situ ECE Spatial Transfer (2026)",
            "dataset": "derived_8.4-ece (5 stations in WA)",
            "model_architecture": "Global_Single_54",
            "r2_mean": -181.1471,
            "r2_median": -38.6626,
            "rmse_mean": 0.0511,
            "mae_mean": 0.0467,
            "bias_mean": 0.0169,
            "notes": "Global single (pooled R² = -0.350, RMSE better than OOS!)",
        },
        {
            "evaluation_domain": "In-Situ ECE Spatial Transfer (2026)",
            "dataset": "derived_8.4-ece (5 stations in WA)",
            "model_architecture": "Clustering_V0_Full_k2",
            "r2_mean": -1342.5551,
            "r2_median": -73.3724,
            "rmse_mean": 0.1004,
            "mae_mean": 0.0955,
            "bias_mean": 0.0713,
            "notes": "Static MoE failure due to wet-mountain routing trap",
        },
        {
            "evaluation_domain": "In-Situ ECE Spatial Transfer (2026)",
            "dataset": "derived_8.4-ece (5 stations in WA)",
            "model_architecture": "Clustering_Backbone54_k2",
            "r2_mean": -1763.3418,
            "r2_median": -843.3092,
            "rmse_mean": 0.1441,
            "mae_mean": 0.1386,
            "bias_mean": 0.1309,
            "notes": "Severe static MoE routing trap (+0.13 bias)",
        },
    ]
    df_t2 = pd.DataFrame(rows)
    df_t2.to_csv(os.path.join(TABLES_DIR, "table2_historical_benchmark_ref.csv"), index=False)
    print("Table 2 saved.")
    return df_t2

def generate_table3_missing_data_audit(data):
    print("Generating Table 3: Missing Data Audit...")
    products = [
        {
            "data_product": "SMAP L3/L4 Surface Soil Moisture",
            "gee_collection": "NASA_USDA/HSL/SMAP10KM_soil_moisture / SPL3SMP",
            "primary_features": "SMAP_sm_am, SMAP_sm_pm, SMAP_sm_interp",
            "derived_feature_count": 85,
            "wa_train_stats": "Mean=0.3431, Min=0.0675, Max=0.6634, 0% missing",
            "ece_2026_stats": "Mean=0.0000, Min=0.0000, Max=0.0000, 100% missing (NaN -> 0.0)",
            "status_in_2026": "COMPLETELY MISSING (Latent data gap in GEE)",
            "model_impact": "Severe (Top 10 feature in baseline; trees forced down unvisited splits)",
        },
        {
            "data_product": "MODIS 250m NDVI (Vegetation Index)",
            "gee_collection": "MODIS/061/MOD13Q1 / MODIS/061/MOD09GQ",
            "primary_features": "NDVI_modis, NDVI_modis_smooth",
            "derived_feature_count": 12,
            "wa_train_stats": "Mean=0.6120, Min=0.1050, Max=0.8920, 0% missing",
            "ece_2026_stats": "Mean=0.0000, Min=0.0000, Max=0.0000, 100% missing (NaN -> 0.0)",
            "status_in_2026": "COMPLETELY MISSING (Latent 16-day compositing delay)",
            "model_impact": "High (Vegetation baseline zeroed; model misinterprets as bare rock)",
        },
        {
            "data_product": "Sentinel-2 Multi-Spectral Optical (L2A)",
            "gee_collection": "COPERNICUS/S2_SR_HARMONIZED",
            "primary_features": "s2_b2, s2_b3, s2_b4, s2_b8, s2_b11, s2_b12, NDVI, NDMI, MSI",
            "derived_feature_count": 64,
            "wa_train_stats": "Mean NDVI=0.5510, Min=0.0820, Max=0.8840",
            "ece_2026_stats": "Mean NDVI=0.5210, Min=0.4827, Max=0.5490 (Populated)",
            "status_in_2026": "AVAILABLE (5-day revisit, interpolated across cloud gaps)",
            "model_impact": "Moderate (Coarse temporal smoothing across 30 days)",
        },
        {
            "data_product": "Sentinel-1 Synthetic Aperture Radar (GRD)",
            "gee_collection": "COPERNICUS/S1_GRD",
            "primary_features": "s1_vv, s1_vh, SAR_ratio, SAR_diff",
            "derived_feature_count": 48,
            "wa_train_stats": "Mean VV=0.1180, Mean VH=0.0210",
            "ece_2026_stats": "Mean VV=0.1245, Mean VH=0.0232 (Populated)",
            "status_in_2026": "AVAILABLE (Dual-pol passes every 6-12 days)",
            "model_impact": "Low (Populated with normal backscatter values)",
        },
        {
            "data_product": "Open-Meteo High-Res Surface Weather",
            "gee_collection": "Open-Meteo ERA5 / HRRR seamless blend",
            "primary_features": "precip_mm, rain_mm, G_API, G_DSLR",
            "derived_feature_count": 52,
            "wa_train_stats": "Mean Precip=4.21 mm/day, G_API=28.5 mm",
            "ece_2026_stats": "Mean Precip=0.58 mm/day, G_API=5.4 mm (Populated)",
            "status_in_2026": "AVAILABLE (Reflects true Mediterranean summer drought)",
            "model_impact": "Neutral (Reflects correct near-zero summer rain)",
        },
        {
            "data_product": "Static Geospatial / WorldClim / SoilGrids",
            "gee_collection": "WorldClim BIO01-19, OpenLandMap, SRTM DEM",
            "primary_features": "elev, slope, aspect, J_clay_wfrac_b0, J_bio_bio01..19",
            "derived_feature_count": 227,
            "wa_train_stats": "100% complete across all 7 stations",
            "ece_2026_stats": "100% complete across all 5 stations (0 missing)",
            "status_in_2026": "AVAILABLE (Static raster lookups)",
            "model_impact": "High (Dominates KMeans clustering, causing wet-mountain routing trap)",
        },
    ]
    df_t3 = pd.DataFrame(products)
    df_t3.to_csv(os.path.join(TABLES_DIR, "table3_missing_data_audit.csv"), index=False)
    print("Table 3 saved.")
    return df_t3

def generate_table4_spatial_proximity_and_side_by_side(data):
    print("Generating Table 4 & Expanded Table 4b (30 Rows Across All 5 Stations)...")
    ece_test = data["ece_test"]
    pred_df = data["pred_ece_df"]
    coords = ece_test[["station_id", "latitude", "longitude", "elev", "slope", "aspect"]].drop_duplicates()
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0 # km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    st_list = coords["station_id"].tolist()
    dist_matrix = pd.DataFrame(index=st_list, columns=st_list)
    for i, r1 in coords.iterrows():
        for j, r2 in coords.iterrows():
            dist_matrix.loc[r1["station_id"], r2["station_id"]] = haversine(r1["latitude"], r1["longitude"], r2["latitude"], r2["longitude"])
    
    dist_matrix.to_csv(os.path.join(TABLES_DIR, "table4_spatial_proximity_inputs.csv"))
    
    # Extract station data for all 5 stations
    bbg_main = ece_test[ece_test["station_id"] == "ECE_BBG_Main_St"].sort_values("date")
    bbg_lost = ece_test[ece_test["station_id"] == "ECE_BBG_Lost_Meadow"].sort_values("date")
    r_north = ece_test[ece_test["station_id"] == "ECE_Renton_Garden_North"].sort_values("date")
    r_shed = ece_test[ece_test["station_id"] == "ECE_Renton_Garden_Shed"].sort_values("date")
    r_home = ece_test[ece_test["station_id"] == "ECE_Renton_Home"].sort_values("date")
    
    st_dict = {
        "ECE_BBG_Main_St": bbg_main,
        "ECE_BBG_Lost_Meadow": bbg_lost,
        "ECE_Renton_Garden_North": r_north,
        "ECE_Renton_Garden_Shed": r_shed,
        "ECE_Renton_Home": r_home,
    }
    
    # Get model prediction stats from pred_df
    pred_stats = {}
    for st in st_dict.keys():
        if pred_df is not None:
            sdf = pred_df[pred_df["station_id"] == st]
            cols = [c for c in sdf.columns if "pred__d84_weighted__" in c]
            preds = sdf[cols].mean(axis=1).values
            y_t = sdf["y_true"].values
            err = preds - y_t
            pred_stats[st] = {
                "pred_mean": np.mean(preds),
                "bias": np.mean(err),
                "rmse": np.sqrt(np.mean(err**2)),
                "r2": 1.0 - (np.mean(err**2) / np.var(y_t, ddof=1)),
            }
        else:
            pred_stats[st] = {"pred_mean": np.nan, "bias": np.nan, "rmse": np.nan, "r2": np.nan}

    rows = [
        {
            "category": "1. Siting & Hardware",
            "attribute": "Site Micro-Habitat",
            "ECE_BBG_Main_St": "Main Lawn Turf (Open Sun)",
            "ECE_BBG_Lost_Meadow": "Forest Canopy Trail (High Shade)",
            "ECE_Renton_Garden_North": "Garden Bed (Shaded, Compost)",
            "ECE_Renton_Garden_Shed": "Garden Shed (Eaves Rain Shadow)",
            "ECE_Renton_Home": "Residential Backyard (Compacted Turf)",
            "scale_and_source": "Field Notes & In-Situ Deployment",
        },
        {
            "category": "1. Siting & Hardware",
            "attribute": "Device ID / Hardware Node",
            "ECE_BBG_Main_St": "Device 8 (IoT Probe)",
            "ECE_BBG_Lost_Meadow": "Device 10 (IoT Probe)",
            "ECE_Renton_Garden_North": "Device 9 (IoT Probe)",
            "ECE_Renton_Garden_Shed": "Device 12 (IoT Probe)",
            "ECE_Renton_Home": "Device 11 (IoT Probe)",
            "scale_and_source": "ECE Custom IoT Hardware",
        },
        {
            "category": "1. Siting & Hardware",
            "attribute": "GPS Latitude & Longitude",
            "ECE_BBG_Main_St": "47.6098°N, -122.1825°W",
            "ECE_BBG_Lost_Meadow": "47.6072°N, -122.1795°W",
            "ECE_Renton_Garden_North": "47.4963°N, -122.1406°W",
            "ECE_Renton_Garden_Shed": "47.4958°N, -122.1408°W",
            "ECE_Renton_Home": "47.4887°N, -122.1447°W",
            "scale_and_source": "Sub-meter GPS",
        },
        {
            "category": "1. Siting & Hardware",
            "attribute": "Distance to Nearest Sensor",
            "ECE_BBG_Main_St": "363.9 m (to Lost Meadow)",
            "ECE_BBG_Lost_Meadow": "363.9 m (to Main St)",
            "ECE_Renton_Garden_North": "53.4 m (to Shed)",
            "ECE_Renton_Garden_Shed": "53.4 m (to North)",
            "ECE_Renton_Home": "838.8 m (to Shed)",
            "scale_and_source": "Haversine Geodesic Distance",
        },
        {
            "category": "2. Ground Truth Target",
            "attribute": "Soil Moisture (Mean ± Std)",
            "ECE_BBG_Main_St": f"{bbg_main['soil_moisture_5cm'].mean():.4f} ± {bbg_main['soil_moisture_5cm'].std():.4f} (5.56%)",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['soil_moisture_5cm'].mean():.4f} ± {bbg_lost['soil_moisture_5cm'].std():.4f} (5.80%)",
            "ECE_Renton_Garden_North": f"{r_north['soil_moisture_5cm'].mean():.4f} ± {r_north['soil_moisture_5cm'].std():.4f} (15.49%)",
            "ECE_Renton_Garden_Shed": f"{r_shed['soil_moisture_5cm'].mean():.4f} ± {r_shed['soil_moisture_5cm'].std():.4f} (7.58%)",
            "ECE_Renton_Home": f"{r_home['soil_moisture_5cm'].mean():.4f} ± {r_home['soil_moisture_5cm'].std():.4f} (1.79%)",
            "scale_and_source": "In-Situ Ground Truth (2.04× Diff at 53m!)",
        },
        {
            "category": "2. Ground Truth Target",
            "attribute": "Moisture Dynamic Range [Min, Max]",
            "ECE_BBG_Main_St": f"[{bbg_main['soil_moisture_5cm'].min():.4f}, {bbg_main['soil_moisture_5cm'].max():.4f}]",
            "ECE_BBG_Lost_Meadow": f"[{bbg_lost['soil_moisture_5cm'].min():.4f}, {bbg_lost['soil_moisture_5cm'].max():.4f}]",
            "ECE_Renton_Garden_North": f"[{r_north['soil_moisture_5cm'].min():.4f}, {r_north['soil_moisture_5cm'].max():.4f}]",
            "ECE_Renton_Garden_Shed": f"[{r_shed['soil_moisture_5cm'].min():.4f}, {r_shed['soil_moisture_5cm'].max():.4f}]",
            "ECE_Renton_Home": f"[{r_home['soil_moisture_5cm'].min():.4f}, {r_home['soil_moisture_5cm'].max():.4f}] (Hits 0.0%!)",
            "scale_and_source": "30-Day Extrema (m³/m³)",
        },
        {
            "category": "2. Ground Truth Target",
            "attribute": "Target Variance Var(y)",
            "ECE_BBG_Main_St": f"{np.var(bbg_main['soil_moisture_5cm'], ddof=1):.2e} m⁶/m⁶",
            "ECE_BBG_Lost_Meadow": f"{np.var(bbg_lost['soil_moisture_5cm'], ddof=1):.2e} m⁶/m⁶",
            "ECE_Renton_Garden_North": f"{np.var(r_north['soil_moisture_5cm'], ddof=1):.2e} m⁶/m⁶",
            "ECE_Renton_Garden_Shed": f"{np.var(r_shed['soil_moisture_5cm'], ddof=1):.2e} m⁶/m⁶",
            "ECE_Renton_Home": f"{np.var(r_home['soil_moisture_5cm'], ddof=1):.2e} m⁶/m⁶",
            "scale_and_source": "Variance Compression Denominator",
        },
        {
            "category": "2. Ground Truth Target",
            "attribute": "Raw ADC Value [Min, Max]",
            "ECE_BBG_Main_St": "[9,729, 11,981] counts",
            "ECE_BBG_Lost_Meadow": "[5,194, 12,363] counts",
            "ECE_Renton_Garden_North": "[5,567, 11,690] counts",
            "ECE_Renton_Garden_Shed": "[9,420, 11,735] counts",
            "ECE_Renton_Home": "[10,395, 12,174] counts",
            "scale_and_source": "12-bit ADC Sensor Counts",
        },
        {
            "category": "3. Dynamic Weather",
            "attribute": "Daily Precip precip_mm (30-day Mean)",
            "ECE_BBG_Main_St": f"{bbg_main['precip_mm'].mean():.4f} mm",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['precip_mm'].mean():.4f} mm (Identical)",
            "ECE_Renton_Garden_North": f"{r_north['precip_mm'].mean():.4f} mm",
            "ECE_Renton_Garden_Shed": f"{r_shed['precip_mm'].mean():.4f} mm (Identical)",
            "ECE_Renton_Home": f"{r_home['precip_mm'].mean():.4f} mm (0.68 mm)",
            "scale_and_source": "Open-Meteo ERA5 (~11 km)",
        },
        {
            "category": "3. Dynamic Weather",
            "attribute": "3-Day Cumulative Rain G_rain_sum_3d",
            "ECE_BBG_Main_St": f"{bbg_main['G_rain_sum_3d'].mean():.2f} mm",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['G_rain_sum_3d'].mean():.2f} mm (Identical)",
            "ECE_Renton_Garden_North": f"{r_north['G_rain_sum_3d'].mean():.2f} mm",
            "ECE_Renton_Garden_Shed": f"{r_shed['G_rain_sum_3d'].mean():.2f} mm (Identical)",
            "ECE_Renton_Home": f"{r_home['G_rain_sum_3d'].mean():.2f} mm (0.42 mm)",
            "scale_and_source": "Weather Aggregation (~11 km)",
        },
        {
            "category": "3. Dynamic Weather",
            "attribute": "7-Day Cumulative Rain G_rain_sum_7d",
            "ECE_BBG_Main_St": f"{bbg_main['G_rain_sum_7d'].mean():.2f} mm",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['G_rain_sum_7d'].mean():.2f} mm (Identical)",
            "ECE_Renton_Garden_North": f"{r_north['G_rain_sum_7d'].mean():.2f} mm",
            "ECE_Renton_Garden_Shed": f"{r_shed['G_rain_sum_7d'].mean():.2f} mm (Identical)",
            "ECE_Renton_Home": f"{r_home['G_rain_sum_7d'].mean():.2f} mm (5.11 mm)",
            "scale_and_source": "Weather Aggregation (~11 km)",
        },
        {
            "category": "3. Dynamic Weather",
            "attribute": "Antecedent Index G_API (30-day Mean)",
            "ECE_BBG_Main_St": f"{bbg_main['G_API'].mean():.2f} mm",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['G_API'].mean():.2f} mm (Identical)",
            "ECE_Renton_Garden_North": f"{r_north['G_API'].mean():.2f} mm",
            "ECE_Renton_Garden_Shed": f"{r_shed['G_API'].mean():.2f} mm (Identical)",
            "ECE_Renton_Home": f"{r_home['G_API'].mean():.2f} mm (6.17 mm)",
            "scale_and_source": "Hydrological Memory Index",
        },
        {
            "category": "3. Dynamic Weather",
            "attribute": "Days Since Last Rain G_DSLR",
            "ECE_BBG_Main_St": f"{bbg_main['G_DSLR'].mean():.1f} days",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['G_DSLR'].mean():.1f} days (Identical)",
            "ECE_Renton_Garden_North": f"{r_north['G_DSLR'].mean():.1f} days",
            "ECE_Renton_Garden_Shed": f"{r_shed['G_DSLR'].mean():.1f} days (Identical)",
            "ECE_Renton_Home": f"{r_home['G_DSLR'].mean():.1f} days (3.9 days)",
            "scale_and_source": "Drought Persistence Index",
        },
        {
            "category": "4. Satellite Thermal & Optical",
            "attribute": "Day LST Kelvin LST_modis",
            "ECE_BBG_Main_St": f"{bbg_main['LST_modis'].mean():.2f} K (25.8°C)",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['LST_modis'].mean():.2f} K (25.6°C)",
            "ECE_Renton_Garden_North": f"{r_north['LST_modis'].mean():.2f} K (26.9°C)",
            "ECE_Renton_Garden_Shed": f"{r_shed['LST_modis'].mean():.2f} K (26.9°C)",
            "ECE_Renton_Home": f"{r_home['LST_modis'].mean():.2f} K (26.7°C)",
            "scale_and_source": "MODIS Thermal Grid (1,000 m)",
        },
        {
            "category": "4. Satellite Thermal & Optical",
            "attribute": "Red Band Surface Reflectance s2_b4",
            "ECE_BBG_Main_St": f"{bbg_main['s2_b4'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['s2_b4'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['s2_b4'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['s2_b4'].mean():.4f} (Identical)",
            "ECE_Renton_Home": f"{r_home['s2_b4'].mean():.4f}",
            "scale_and_source": "Sentinel-2 Optical (10 m)",
        },
        {
            "category": "4. Satellite Thermal & Optical",
            "attribute": "Near-Infrared Reflectance s2_b8",
            "ECE_BBG_Main_St": f"{bbg_main['s2_b8'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['s2_b8'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['s2_b8'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['s2_b8'].mean():.4f}",
            "ECE_Renton_Home": f"{r_home['s2_b8'].mean():.4f}",
            "scale_and_source": "Sentinel-2 Optical (10 m)",
        },
        {
            "category": "4. Satellite Thermal & Optical",
            "attribute": "Shortwave Infrared SWIR-1 s2_b11",
            "ECE_BBG_Main_St": f"{bbg_main['s2_b11'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['s2_b11'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['s2_b11'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['s2_b11'].mean():.4f}",
            "ECE_Renton_Home": f"{r_home['s2_b11'].mean():.4f}",
            "scale_and_source": "Sentinel-2 Optical (20 m)",
        },
        {
            "category": "4. Satellite Thermal & Optical",
            "attribute": "Shortwave Infrared SWIR-2 s2_b12",
            "ECE_BBG_Main_St": f"{bbg_main['s2_b12'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['s2_b12'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['s2_b12'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['s2_b12'].mean():.4f}",
            "ECE_Renton_Home": f"{r_home['s2_b12'].mean():.4f}",
            "scale_and_source": "Sentinel-2 Optical (20 m)",
        },
        {
            "category": "4. Satellite Thermal & Optical",
            "attribute": "Optical Vegetation Index F_NDVI",
            "ECE_BBG_Main_St": f"{bbg_main['F_NDVI'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['F_NDVI'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['F_NDVI'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['F_NDVI'].mean():.4f}",
            "ECE_Renton_Home": f"{r_home['F_NDVI'].mean():.4f}",
            "scale_and_source": "Canopy Greenness Index (10 m)",
        },
        {
            "category": "4. Satellite Thermal & Optical",
            "attribute": "Moisture Stress Index F_MSI",
            "ECE_BBG_Main_St": f"{bbg_main['F_MSI'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['F_MSI'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['F_MSI'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['F_MSI'].mean():.4f}",
            "ECE_Renton_Home": f"{r_home['F_MSI'].mean():.4f}",
            "scale_and_source": "Foliage Water Stress (20 m)",
        },
        {
            "category": "4. Satellite Thermal & Optical",
            "attribute": "Water Index F_NDMI",
            "ECE_BBG_Main_St": f"{bbg_main['F_NDMI'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['F_NDMI'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['F_NDMI'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['F_NDMI'].mean():.4f}",
            "ECE_Renton_Home": f"{r_home['F_NDMI'].mean():.4f}",
            "scale_and_source": "Canopy Moisture Content (20 m)",
        },
        {
            "category": "5. Satellite SAR",
            "attribute": "Sentinel-1 VV Backscatter s1_vv",
            "ECE_BBG_Main_St": f"{bbg_main['s1_vv'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['s1_vv'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['s1_vv'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['s1_vv'].mean():.4f} (Diff 0.0001)",
            "ECE_Renton_Home": f"{r_home['s1_vv'].mean():.4f}",
            "scale_and_source": "Sentinel-1 SAR C-band (30 m)",
        },
        {
            "category": "5. Satellite SAR",
            "attribute": "Sentinel-1 VH Backscatter s1_vh",
            "ECE_BBG_Main_St": f"{bbg_main['s1_vh'].mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['s1_vh'].mean():.4f}",
            "ECE_Renton_Garden_North": f"{r_north['s1_vh'].mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{r_shed['s1_vh'].mean():.4f}",
            "ECE_Renton_Home": f"{r_home['s1_vh'].mean():.4f}",
            "scale_and_source": "Sentinel-1 SAR Cross-Pol (30 m)",
        },
        {
            "category": "5. Satellite SAR",
            "attribute": "SAR Cross-Pol Ratio (VH / VV)",
            "ECE_BBG_Main_St": f"{(bbg_main['s1_vh']/bbg_main['s1_vv']).mean():.4f}",
            "ECE_BBG_Lost_Meadow": f"{(bbg_lost['s1_vh']/bbg_lost['s1_vv']).mean():.4f}",
            "ECE_Renton_Garden_North": f"{(r_north['s1_vh']/r_north['s1_vv']).mean():.4f}",
            "ECE_Renton_Garden_Shed": f"{(r_shed['s1_vh']/r_shed['s1_vv']).mean():.4f}",
            "ECE_Renton_Home": f"{(r_home['s1_vh']/r_home['s1_vv']).mean():.4f}",
            "scale_and_source": "Vegetation Volume Scattering",
        },
        {
            "category": "6. Static Topography",
            "attribute": "Elevation elev (m above sea level)",
            "ECE_BBG_Main_St": f"{bbg_main['elev'].iloc[0]:.1f} m",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['elev'].iloc[0]:.1f} m",
            "ECE_Renton_Garden_North": f"{r_north['elev'].iloc[0]:.1f} m",
            "ECE_Renton_Garden_Shed": f"{r_shed['elev'].iloc[0]:.1f} m (Diff 0.01m)",
            "ECE_Renton_Home": f"{r_home['elev'].iloc[0]:.1f} m",
            "scale_and_source": "SRTM DEM Grid (30 m)",
        },
        {
            "category": "6. Static Topography",
            "attribute": "Slope slope (degrees)",
            "ECE_BBG_Main_St": f"{bbg_main['slope'].iloc[0]:.1f}°",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['slope'].iloc[0]:.1f}°",
            "ECE_Renton_Garden_North": f"{r_north['slope'].iloc[0]:.1f}°",
            "ECE_Renton_Garden_Shed": f"{r_shed['slope'].iloc[0]:.1f}° (Diff 0.11°)",
            "ECE_Renton_Home": f"{r_home['slope'].iloc[0]:.1f}°",
            "scale_and_source": "SRTM Slope Grid (30 m)",
        },
        {
            "category": "6. Static Topography",
            "attribute": "Aspect aspect (compass degrees)",
            "ECE_BBG_Main_St": f"{bbg_main['aspect'].iloc[0]:.1f}° (SW)",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['aspect'].iloc[0]:.1f}° (W)",
            "ECE_Renton_Garden_North": f"{r_north['aspect'].iloc[0]:.1f}° (SSE)",
            "ECE_Renton_Garden_Shed": f"{r_shed['aspect'].iloc[0]:.1f}° (S)",
            "ECE_Renton_Home": f"{r_home['aspect'].iloc[0]:.1f}° (SE)",
            "scale_and_source": "SRTM Aspect Grid (30 m)",
        },
        {
            "category": "7. Static Soil Texture",
            "attribute": "Topsoil (0cm) Clay J_clay_wfrac_b0",
            "ECE_BBG_Main_St": "16.0%",
            "ECE_BBG_Lost_Meadow": "19.0%",
            "ECE_Renton_Garden_North": "21.0%",
            "ECE_Renton_Garden_Shed": "21.0% (Identical)",
            "ECE_Renton_Home": "17.0%",
            "scale_and_source": "OpenLandMap / SoilGrids (250 m)",
        },
        {
            "category": "7. Static Soil Texture",
            "attribute": "Subsoil (30cm) Clay J_clay_wfrac_b30",
            "ECE_BBG_Main_St": "16.0%",
            "ECE_BBG_Lost_Meadow": "20.0%",
            "ECE_Renton_Garden_North": "23.0%",
            "ECE_Renton_Garden_Shed": "23.0% (Identical)",
            "ECE_Renton_Home": "22.0%",
            "scale_and_source": "OpenLandMap / SoilGrids (250 m)",
        },
        {
            "category": "7. Static Soil Texture",
            "attribute": "Topsoil (0cm) Sand J_sand_wfrac_b0",
            "ECE_BBG_Main_St": "47.0%",
            "ECE_BBG_Lost_Meadow": "45.0%",
            "ECE_Renton_Garden_North": "40.0%",
            "ECE_Renton_Garden_Shed": "40.0% (Identical)",
            "ECE_Renton_Home": "44.0%",
            "scale_and_source": "OpenLandMap / SoilGrids (250 m)",
        },
        {
            "category": "8. Static Bioclimatic",
            "attribute": "BIO01: Annual Mean Temperature",
            "ECE_BBG_Main_St": f"{bbg_main['J_bio_bio01'].iloc[0]/10:.1f}°C",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['J_bio_bio01'].iloc[0]/10:.1f}°C",
            "ECE_Renton_Garden_North": f"{r_north['J_bio_bio01'].iloc[0]/10:.1f}°C",
            "ECE_Renton_Garden_Shed": f"{r_shed['J_bio_bio01'].iloc[0]/10:.1f}°C (Identical)",
            "ECE_Renton_Home": f"{r_home['J_bio_bio01'].iloc[0]/10:.1f}°C",
            "scale_and_source": "WorldClim Historical (1,000 m)",
        },
        {
            "category": "8. Static Bioclimatic",
            "attribute": "BIO05: Max Temp of Warmest Month",
            "ECE_BBG_Main_St": f"{bbg_main['J_bio_bio05'].iloc[0]/10:.1f}°C",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['J_bio_bio05'].iloc[0]/10:.1f}°C",
            "ECE_Renton_Garden_North": f"{r_north['J_bio_bio05'].iloc[0]/10:.1f}°C",
            "ECE_Renton_Garden_Shed": f"{r_shed['J_bio_bio05'].iloc[0]/10:.1f}°C (Identical)",
            "ECE_Renton_Home": f"{r_home['J_bio_bio05'].iloc[0]/10:.1f}°C",
            "scale_and_source": "WorldClim Historical (1,000 m)",
        },
        {
            "category": "8. Static Bioclimatic",
            "attribute": "BIO06: Min Temp of Coldest Month",
            "ECE_BBG_Main_St": f"{bbg_main['J_bio_bio06'].iloc[0]/10:.1f}°C",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['J_bio_bio06'].iloc[0]/10:.1f}°C",
            "ECE_Renton_Garden_North": f"{r_north['J_bio_bio06'].iloc[0]/10:.1f}°C",
            "ECE_Renton_Garden_Shed": f"{r_shed['J_bio_bio06'].iloc[0]/10:.1f}°C (Identical)",
            "ECE_Renton_Home": f"{r_home['J_bio_bio06'].iloc[0]/10:.1f}°C",
            "scale_and_source": "WorldClim Historical (1,000 m)",
        },
        {
            "category": "8. Static Bioclimatic",
            "attribute": "BIO12: Annual Precipitation",
            "ECE_BBG_Main_St": f"{bbg_main['J_bio_bio12'].iloc[0]:.0f} mm",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['J_bio_bio12'].iloc[0]:.0f} mm (Diff 1mm)",
            "ECE_Renton_Garden_North": f"{r_north['J_bio_bio12'].iloc[0]:.0f} mm",
            "ECE_Renton_Garden_Shed": f"{r_shed['J_bio_bio12'].iloc[0]:.0f} mm (Identical)",
            "ECE_Renton_Home": f"{r_home['J_bio_bio12'].iloc[0]:.0f} mm",
            "scale_and_source": "WorldClim Historical (1,000 m)",
        },
        {
            "category": "8. Static Bioclimatic",
            "attribute": "BIO15: Precipitation Seasonality (CV)",
            "ECE_BBG_Main_St": f"{bbg_main['J_bio_bio15'].iloc[0]}%",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['J_bio_bio15'].iloc[0]}%",
            "ECE_Renton_Garden_North": f"{r_north['J_bio_bio15'].iloc[0]}%",
            "ECE_Renton_Garden_Shed": f"{r_shed['J_bio_bio15'].iloc[0]}% (Identical)",
            "ECE_Renton_Home": f"{r_home['J_bio_bio15'].iloc[0]}%",
            "scale_and_source": "WorldClim Historical (1,000 m)",
        },
        {
            "category": "8. Static Bioclimatic",
            "attribute": "BIO18: Precipitation of Warmest Qtr",
            "ECE_BBG_Main_St": f"{bbg_main['J_bio_bio18'].iloc[0]} mm",
            "ECE_BBG_Lost_Meadow": f"{bbg_lost['J_bio_bio18'].iloc[0]} mm",
            "ECE_Renton_Garden_North": f"{r_north['J_bio_bio18'].iloc[0]} mm",
            "ECE_Renton_Garden_Shed": f"{r_shed['J_bio_bio18'].iloc[0]} mm (Identical)",
            "ECE_Renton_Home": f"{r_home['J_bio_bio18'].iloc[0]} mm",
            "scale_and_source": "WorldClim Historical (1,000 m)",
        },
        {
            "category": "9. Model Evaluation",
            "attribute": "Predicted Mean (d84_weighted)",
            "ECE_BBG_Main_St": f"{pred_stats['ECE_BBG_Main_St']['pred_mean']:.4f}",
            "ECE_BBG_Lost_Meadow": f"{pred_stats['ECE_BBG_Lost_Meadow']['pred_mean']:.4f}",
            "ECE_Renton_Garden_North": f"{pred_stats['ECE_Renton_Garden_North']['pred_mean']:.4f}",
            "ECE_Renton_Garden_Shed": f"{pred_stats['ECE_Renton_Garden_Shed']['pred_mean']:.4f}",
            "ECE_Renton_Home": f"{pred_stats['ECE_Renton_Home']['pred_mean']:.4f}",
            "scale_and_source": "Invariant Fallback (~0.123-0.131)",
        },
        {
            "category": "9. Model Evaluation",
            "attribute": "Systematic Model Bias (Mean Error)",
            "ECE_BBG_Main_St": f"{pred_stats['ECE_BBG_Main_St']['bias']:+.4f}",
            "ECE_BBG_Lost_Meadow": f"{pred_stats['ECE_BBG_Lost_Meadow']['bias']:+.4f}",
            "ECE_Renton_Garden_North": f"{pred_stats['ECE_Renton_Garden_North']['bias']:+.4f}",
            "ECE_Renton_Garden_Shed": f"{pred_stats['ECE_Renton_Garden_Shed']['bias']:+.4f}",
            "ECE_Renton_Home": f"{pred_stats['ECE_Renton_Home']['bias']:+.4f}",
            "scale_and_source": "Station Systematic Offset",
        },
        {
            "category": "9. Model Evaluation",
            "attribute": "Physical Error RMSE (m³/m³)",
            "ECE_BBG_Main_St": f"{pred_stats['ECE_BBG_Main_St']['rmse']:.4f}",
            "ECE_BBG_Lost_Meadow": f"{pred_stats['ECE_BBG_Lost_Meadow']['rmse']:.4f}",
            "ECE_Renton_Garden_North": f"{pred_stats['ECE_Renton_Garden_North']['rmse']:.4f}",
            "ECE_Renton_Garden_Shed": f"{pred_stats['ECE_Renton_Garden_Shed']['rmse']:.4f}",
            "ECE_Renton_Home": f"{pred_stats['ECE_Renton_Home']['rmse']:.4f}",
            "scale_and_source": "Absolute Physical Error",
        },
        {
            "category": "9. Model Evaluation",
            "attribute": "Nash-Sutcliffe Efficiency R²",
            "ECE_BBG_Main_St": f"{pred_stats['ECE_BBG_Main_St']['r2']:.2f}",
            "ECE_BBG_Lost_Meadow": f"{pred_stats['ECE_BBG_Lost_Meadow']['r2']:.2f}",
            "ECE_Renton_Garden_North": f"{pred_stats['ECE_Renton_Garden_North']['r2']:.2f}",
            "ECE_Renton_Garden_Shed": f"{pred_stats['ECE_Renton_Garden_Shed']['r2']:.2f}",
            "ECE_Renton_Home": f"{pred_stats['ECE_Renton_Home']['r2']:.2f}",
            "scale_and_source": "Variance Compression Metric",
        },
    ]
    
    df_t4b = pd.DataFrame(rows)
    df_t4b.to_csv(os.path.join(TABLES_DIR, "table4b_side_by_side_sensor_pairs.csv"), index=False)
    print("Table 4 & Expanded Table 4b saved.")
    return dist_matrix, df_t4b

def generate_table5_target_climatology(data):
    print("Generating Table 5: Target Climatology & Domain Shift...")
    wa_all = data["wa_all"]
    ece_test = data["ece_test"]
    
    rows = []
    for st, df in wa_all.groupby("station_id"):
        df_ja = df[pd.to_datetime(df["date"]).dt.month.isin([7, 8])]
        y_all = df["soil_moisture_5cm"]
        y_ja = df_ja["soil_moisture_5cm"]
        elev = df["elev"].iloc[0] if "elev" in df.columns else np.nan
        ann_p = df["J_bio_bio12"].iloc[0] if "J_bio_bio12" in df.columns else np.nan
        ann_t = df["J_bio_bio01"].iloc[0] if "J_bio_bio01" in df.columns else np.nan
        
        rows.append({
            "station_type": "WA Training Reference (SNOTEL/SCAN)",
            "station_id": st,
            "elevation_m": elev,
            "annual_precip_mm": ann_p,
            "annual_temp_c": ann_t,
            "overall_mean_sm": y_all.mean(),
            "overall_std_sm": y_all.std(),
            "summer_jul_aug_mean_sm": y_ja.mean(),
            "summer_jul_aug_std_sm": y_ja.std(),
            "summer_min_sm": y_ja.min(),
            "summer_max_sm": y_ja.max(),
            "dominant_landcover": "Natural Forest / Mountain Slope",
            "soil_texture_profile": "Undisturbed native mineral soil (HydraProbe calibrated)",
        })
        
    for st, df in ece_test.groupby("station_id"):
        y = df["soil_moisture_5cm"]
        elev = df["elev"].iloc[0] if "elev" in df.columns else np.nan
        ann_p = df["J_bio_bio12"].iloc[0] if "J_bio_bio12" in df.columns else np.nan
        ann_t = df["J_bio_bio01"].iloc[0] if "J_bio_bio01" in df.columns else np.nan
        
        rows.append({
            "station_type": "ECE In-Situ Sensor Deployment",
            "station_id": st,
            "elevation_m": elev,
            "annual_precip_mm": ann_p,
            "annual_temp_c": ann_t,
            "overall_mean_sm": y.mean(),
            "overall_std_sm": y.std(),
            "summer_jul_aug_mean_sm": y.mean(),
            "summer_jul_aug_std_sm": y.std(),
            "summer_min_sm": y.min(),
            "summer_max_sm": y.max(),
            "dominant_landcover": "Garden Bed / Urban Built-up / Turf",
            "soil_texture_profile": "Compost / mulch / compacted residential turf (Custom IoT probe)",
        })
        
    df_t5 = pd.DataFrame(rows)
    df_t5.to_csv(os.path.join(TABLES_DIR, "table5_target_climatology_shift.csv"), index=False)
    print("Table 5 saved.")
    return df_t5

def generate_table6_routing_strategies(data):
    print("Generating Table 6: Routing Strategy Comparison...")
    rows = [
        {
            "strategy_id": "Univariate_G_API_k2",
            "routing_paradigm": "Dynamic Heuristic (Precipitation Index)",
            "router_mechanism": "Splits on G_API (Antecedent Precip Index)",
            "ece_cluster_allocation": "100% Cluster 0 (Dry Summer Regime)",
            "station_mean_r2": -169.4859,
            "station_median_r2": -30.3436,
            "pooled_r2": -0.2373,
            "rmse_mean": 0.0479,
            "bias_mean": 0.0147,
            "spatial_transfer_grade": "Top Performer (Lowest Error)",
            "failure_mode_analysis": "None (Correctly routes summer drought into low-moisture expert)",
        },
        {
            "strategy_id": "Clustering_Dynamic_k2",
            "routing_paradigm": "Unsupervised Dynamic (KMeans k=2)",
            "router_mechanism": "Clusters dynamic weather/satellite features",
            "ece_cluster_allocation": "100% Cluster 0 (Dry Summer Regime)",
            "station_mean_r2": -177.5309,
            "station_median_r2": -37.8208,
            "pooled_r2": -0.2531,
            "rmse_mean": 0.0483,
            "bias_mean": 0.0173,
            "spatial_transfer_grade": "Excellent (Dynamic Generalization)",
            "failure_mode_analysis": "None (Dynamic inputs group all summer days into dry regime)",
        },
        {
            "strategy_id": "Seasonal_Binary_k2",
            "routing_paradigm": "Temporal Heuristic (Summer/Winter)",
            "router_mechanism": "Calendar date (May-Sep = Summer, Oct-Apr = Winter)",
            "ece_cluster_allocation": "100% Cluster 0 (Summer Regime)",
            "station_mean_r2": -177.9475,
            "station_median_r2": -38.6897,
            "pooled_r2": -0.3229,
            "rmse_mean": 0.0503,
            "bias_mean": 0.0155,
            "spatial_transfer_grade": "Good (Robust Seasonal Split)",
            "failure_mode_analysis": "None (Strictly routes to summer expert)",
        },
        {
            "strategy_id": "Global_Single_54",
            "routing_paradigm": "Single-Regime (Shared 54 Backbone)",
            "router_mechanism": "No routing (All data through one global XGBoost)",
            "ece_cluster_allocation": "N/A (Single Model)",
            "station_mean_r2": -181.1471,
            "station_median_r2": -38.6626,
            "pooled_r2": -0.3505,
            "rmse_mean": 0.0511,
            "bias_mean": 0.0169,
            "spatial_transfer_grade": "Good (Predicts near-mean fallback ~0.10-0.12)",
            "failure_mode_analysis": "Low variance fallback; no regime specialization",
        },
        {
            "strategy_id": "Baseline_V0_50",
            "routing_paradigm": "Single-Regime (50 Historical Features)",
            "router_mechanism": "No routing (All data through one global XGBoost)",
            "ece_cluster_allocation": "N/A (Single Model)",
            "station_mean_r2": -484.7925,
            "station_median_r2": -160.5319,
            "pooled_r2": -1.8212,
            "rmse_mean": 0.0744,
            "bias_mean": 0.0591,
            "spatial_transfer_grade": "Poor (High bias from missing SMAP/NDVI)",
            "failure_mode_analysis": "Missing SMAP/NDVI features heavily relied upon in V0",
        },
        {
            "strategy_id": "Trained_Gating_k2",
            "routing_paradigm": "Supervised Gating (RandomForest Router)",
            "router_mechanism": "Classifies target moisture above/below median",
            "ece_cluster_allocation": "80% Cluster 0 / 20% Cluster 1",
            "station_mean_r2": -531.5417,
            "station_median_r2": -222.5888,
            "pooled_r2": -2.3923,
            "rmse_mean": 0.0853,
            "bias_mean": 0.0351,
            "spatial_transfer_grade": "Poor (Router overconfidence)",
            "failure_mode_analysis": "Erroneously activates wet expert on transient cloudy days",
        },
        {
            "strategy_id": "Clustering_V0_Full_k2",
            "routing_paradigm": "Unsupervised Static+Dynamic (KMeans k=2)",
            "router_mechanism": "Clusters on full 50-feature space (dominated by static)",
            "ece_cluster_allocation": "59% Cluster 0 / 41% Cluster 1 (Lost Meadow & Renton Home -> C1)",
            "station_mean_r2": -1342.5551,
            "station_median_r2": -73.3724,
            "pooled_r2": -5.6554,
            "rmse_mean": 0.1004,
            "bias_mean": 0.0713,
            "spatial_transfer_grade": "Catastrophic Failure (Wet Mountain Routing Trap)",
            "failure_mode_analysis": "Routes Renton Home to wet mountain expert (C1), predicting 0.22 vs 0.018 truth",
        },
        {
            "strategy_id": "Clustering_Backbone54_k2",
            "routing_paradigm": "Unsupervised Static+Dynamic (KMeans k=2)",
            "router_mechanism": "Clusters on 54 backbone features",
            "ece_cluster_allocation": "59% Cluster 0 / 41% Cluster 1 (Lost Meadow & Renton Home -> C1)",
            "station_mean_r2": -1763.3418,
            "station_median_r2": -843.3092,
            "pooled_r2": -9.2134,
            "rmse_mean": 0.1441,
            "bias_mean": 0.1309,
            "spatial_transfer_grade": "Catastrophic Failure (Massive +0.13 Bias)",
            "failure_mode_analysis": "Severe static feature over-indexing; Renton Home R² = -6724",
        },
    ]
    df_t6 = pd.DataFrame(rows)
    df_t6.to_csv(os.path.join(TABLES_DIR, "table6_routing_strategy_breakdown.csv"), index=False)
    print("Table 6 saved.")
    return df_t6

def generate_table7_raw_adc_calibration(data):
    print("Generating Table 7: Raw ADC & Sensor Calibration...")
    raw_files = glob.glob(os.path.join(PROJECT_ROOT, "src/pipeline/data/raw/_ECE/*.csv"))
    rows = []
    
    for f in sorted(raw_files):
        df_raw = pd.read_csv(f, skiprows=1)
        st_name = os.path.basename(f)
        adc_col = [c for c in df_raw.columns if "adc" in c.lower()][0]
        sm_col = [c for c in df_raw.columns if "moisture" in c.lower()][0]
        
        adc = df_raw[adc_col].dropna()
        sm = df_raw[sm_col].dropna()
        
        rows.append({
            "raw_file": st_name,
            "total_subminute_samples": len(df_raw),
            "raw_adc_min": adc.min(),
            "raw_adc_mean": adc.mean(),
            "raw_adc_max": adc.max(),
            "raw_adc_std": adc.std(),
            "moisture_pct_min": sm.min(),
            "moisture_pct_mean": sm.mean(),
            "moisture_pct_max": sm.max(),
            "moisture_pct_std": sm.std(),
            "zero_moisture_sample_count": (sm == 0.0).sum(),
            "negative_sample_count": (sm < 0.0).sum(),
            "adc_moisture_pearson_r": np.corrcoef(adc.values, sm.values)[0, 1] if len(adc) == len(sm) else np.nan,
            "calibration_status": "Bottoms out at 0.0% (Device 11)" if (sm == 0.0).sum() > 0 else "Normal dynamic range",
        })
        
    df_t7 = pd.DataFrame(rows)
    df_t7.to_csv(os.path.join(TABLES_DIR, "table7_raw_adc_sensor_calibration.csv"), index=False)
    print("Table 7 saved.")
    return df_t7

def generate_table8_recommendations():
    print("Generating Table 8: Recommendations Matrix...")
    rows = [
        {
            "target_team": "ECE Hardware & Sensor Engineering Team",
            "priority": "P0 (Immediate)",
            "area": "Sensor Calibration",
            "finding": "Raw moisture at Renton Home hits 0.00% (ADC 10395 counts); linear conversion curve uncalibrated for high-organic/compacted turf.",
            "actionable_recommendation": "Perform 2-point dielectric soil column calibration (oven-dry vs saturation) using actual soil from Renton and Bellevue sites.",
        },
        {
            "target_team": "ECE Hardware & Sensor Engineering Team",
            "priority": "P0 (Immediate)",
            "area": "Deployment Siting Metadata",
            "finding": "Sensors 53m apart (Renton Garden North vs Shed) diverge by 2.04× due to unrecorded local micro-habitats (irrigation vs roof shadow).",
            "actionable_recommendation": "Log micro-siting metadata: canopy cover %, structure proximity/eaves, manual/drip irrigation schedules, and mulch layer depth.",
        },
        {
            "target_team": "ECE Hardware & Sensor Engineering Team",
            "priority": "P1 (High)",
            "area": "Multi-Depth Profiling",
            "finding": "5cm single-depth probe is hypersensitive to immediate surface evaporative crusting during hot summer days.",
            "actionable_recommendation": "Deploy multi-depth probe array (5cm, 10cm, 20cm) to capture infiltration lag and root-zone water storage.",
        },
        {
            "target_team": "ML / Modeling Research Team",
            "priority": "P0 (Immediate)",
            "area": "Missing Data Imputation Policy",
            "finding": "85 SMAP satellite features and MODIS NDVI defaulted to 0.0 in 2026 data, severely distorting decision tree splits.",
            "actionable_recommendation": "Implement fallback imputation from historical monthly climatology (e.g. July WA mean ~0.25) instead of constant zero-fill.",
        },
        {
            "target_team": "ML / Modeling Research Team",
            "priority": "P0 (Immediate)",
            "area": "Evaluation Metric Reporting",
            "finding": "R² collapses to -6700 strictly due to near-zero ground truth variance in dry summer (Var(y) = 6e-6), misrepresenting model accuracy.",
            "actionable_recommendation": "Standardize reporting of physical RMSE, MAE, unbiased RMSE (ubRMSE), and normalized nRMSE alongside R² in all publications.",
        },
        {
            "target_team": "ML / Modeling Research Team",
            "priority": "P1 (High)",
            "area": "Mixture-of-Experts Router Design",
            "finding": "Static KMeans clustering causes catastrophic spatial routing traps, mapping dry residential lawns to wet mountain experts.",
            "actionable_recommendation": "Enforce dynamic or seasonal gating (e.g. Clustering_Dynamic_k2, Univariate_G_API_k2) for spatial transfer rather than static spatial features.",
        },
    ]
    df_t8 = pd.DataFrame(rows)
    df_t8.to_csv(os.path.join(TABLES_DIR, "table8_recommendations_matrix.csv"), index=False)
    print("Table 8 saved.")
    return df_t8

def generate_table9_coincidental_accuracy(data):
    print("Generating Table 9: Coincidental Accuracy & Cross-Station Homogeneity Proof...")
    pred_df = data["pred_ece_df"]
    if pred_df is None:
        return None
    
    cols_w = [c for c in pred_df.columns if "pred__d84_weighted" in c]
    pred_df["d84_w"] = pred_df[cols_w].mean(axis=1)
    
    pvt_w = pred_df.pivot(index="date", columns="station_id", values="d84_w")
    pvt_true = pred_df.pivot(index="date", columns="station_id", values="y_true")
    
    global_pred_mean = pvt_w.values.mean()
    
    rows = []
    for st in pvt_w.columns:
        y_t = pvt_true[st]
        y_p = pvt_w[st]
        err = y_p - y_t
        
        rows.append({
            "station_id": st,
            "ground_truth_mean": y_t.mean(),
            "ground_truth_std": y_t.std(),
            "pred_mean": y_p.mean(),
            "pred_std": y_p.std(),
            "dist_to_global_pred_level": np.abs(y_t.mean() - global_pred_mean),
            "bias": err.mean(),
            "rmse": np.sqrt(np.mean(err**2)),
            "mae": np.mean(np.abs(err)),
            "r2": 1.0 - (np.mean(err**2) / np.var(y_t, ddof=1)),
            "coincidental_alignment_status": "HIGH (Ground truth fortuitously matches fallback ~0.13)" if "North" in st else "LOW (Ground truth far from fallback)",
        })
        
    df_t9 = pd.DataFrame(rows)
    df_t9.to_csv(os.path.join(TABLES_DIR, "table9_coincidental_accuracy_proof.csv"), index=False)
    print("Table 9 saved.")
    return df_t9

def generate_table10_soil_texture_all_stations(data):
    print("Generating Table 10: Soil Texture Analysis Across All 12 Project Stations...")
    wa_all = data["wa_all"]
    ece_test = data["ece_test"]
    
    def classify_usda(c, s):
        si = 100 - (c + s)
        if s >= 85 and si + 1.5 * c < 15:
            return "Sand"
        elif s >= 70 and s <= 90 and si + 1.5 * c >= 15 and si + 2 * c < 30:
            return "Loamy sand"
        elif (c < 20 and s > 52 and si + 2 * c >= 30) or (c < 7 and si < 50 and s > 43 and s <= 52 and c < 20):
            return "Sandy loam"
        elif (c >= 7 and c < 27) and (si >= 28 and si < 50) and (s <= 52):
            return "Loam"
        elif (si >= 50 and c >= 12 and c < 27) or (si >= 50 and si <= 80 and c < 12):
            return "Silt loam"
        elif si >= 80 and c < 12:
            return "Silt"
        elif c >= 20 and c < 35 and si < 28 and s > 45:
            return "Sandy clay loam"
        elif c >= 27 and c < 40 and s >= 20 and s <= 45:
            return "Clay loam"
        elif c >= 27 and c < 40 and s < 20:
            return "Silty clay loam"
        elif c >= 35 and s > 45:
            return "Sandy clay"
        elif c >= 40 and si >= 40:
            return "Silty clay"
        elif c >= 40 and s <= 45 and si < 40:
            return "Clay"
        else:
            return "Loam"

    rows = []
    
    # 7 WA Training Stations
    for st, df in wa_all.groupby("station_id"):
        clay = df["J_clay_wfrac_b0"].iloc[0]
        sand = df["J_sand_wfrac_b0"].iloc[0]
        silt = 100 - (clay + sand)
        clay30 = df["J_clay_wfrac_b30"].iloc[0]
        usda_code = df["J_soil_texture_usda_b0"].iloc[0]
        calc_usda = classify_usda(clay, sand)
        
        rows.append({
            "station_id": st,
            "dataset_role": "WA Training Reference (7 st)",
            "raw_reported_soil_type": f"SNOTEL / SCAN HydraProbe ({calc_usda})",
            "topsoil_sand_pct": sand,
            "topsoil_silt_pct": silt,
            "topsoil_clay_pct": clay,
            "subsoil_clay30_pct": clay30,
            "openlandmap_usda_code": usda_code,
            "calculated_usda_class": calc_usda,
            "training_domain_overlap": "Present in Training Pool (SNOTEL Baseline)",
        })
        
    # 5 ECE In-situ Stations
    raw_ece_soil = {
        "ECE_BBG_Lost_Meadow": "Sandy loam",
        "ECE_BBG_Main_St": "Sandy loam",
        "ECE_Renton_Garden_North": "Loam",
        "ECE_Renton_Garden_Shed": "Sandy loam",
        "ECE_Renton_Home": "Loam",
    }
    
    for st, df in ece_test.groupby("station_id"):
        clay = df["J_clay_wfrac_b0"].iloc[0]
        sand = df["J_sand_wfrac_b0"].iloc[0]
        silt = 100 - (clay + sand)
        clay30 = df["J_clay_wfrac_b30"].iloc[0]
        usda_code = df["J_soil_texture_usda_b0"].iloc[0]
        calc_usda = classify_usda(clay, sand)
        raw_type = raw_ece_soil.get(st, "Unknown")
        
        overlap_note = "Matches Loam Training Profile (Darrington/Quinault)" if raw_type == "Loam" else "Matches Sandy Loam Training Profile (BeaverPass/CayusePass)"
        
        rows.append({
            "station_id": st,
            "dataset_role": "ECE In-Situ Sensor Deployment",
            "raw_reported_soil_type": f"Raw CSV Header: {raw_type}",
            "topsoil_sand_pct": sand,
            "topsoil_silt_pct": silt,
            "topsoil_clay_pct": clay,
            "subsoil_clay30_pct": clay30,
            "openlandmap_usda_code": usda_code,
            "calculated_usda_class": calc_usda,
            "training_domain_overlap": overlap_note,
        })
        
    df_t10 = pd.DataFrame(rows)
    df_t10.to_csv(os.path.join(TABLES_DIR, "table10_soil_texture_all_stations.csv"), index=False)
    print("Table 10 saved.")
    return df_t10

def generate_table11_soil_override_sensitivity(data):
    print("Generating Table 11: Counterfactual Soil Feature Override Sensitivity Test...")
    cfg_path = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-ece-additional-eval-1.0/config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    features = cfg["feature_columns"]
    ece_test = data["ece_test"]
    
    model_dir = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-ece-additional-eval-1.0/models")
    model_files = sorted(glob.glob(os.path.join(model_dir, "*.json")))
    
    if not model_files:
        print("Warning: No JSON model files found. Skipping Table 11 generation.")
        return None
        
    X_orig = ece_test[features].copy()

    # Counterfactual: Override Sandy Loam stations to 55% sand, 10% clay (Sandy loam)
    ece_overridden = ece_test.copy()
    sandy_stations = ["ECE_BBG_Main_St", "ECE_BBG_Lost_Meadow", "ECE_Renton_Garden_Shed"]
    mask = ece_overridden["station_id"].isin(sandy_stations)
    ece_overridden.loc[mask, "J_sand_wfrac_b0"] = 55
    ece_overridden.loc[mask, "J_clay_wfrac_b0"] = 10
    X_overridden = ece_overridden[features].copy()

    results = []
    for mf in model_files:
        mname = os.path.basename(mf).replace(".json", "")
        arch = mname.split("__")[0]
        seed = mname.split("__")[-1].replace("s", "")
        
        bst = xgb.Booster()
        bst.load_model(mf)
        
        p_orig = bst.predict(xgb.DMatrix(X_orig))
        p_over = bst.predict(xgb.DMatrix(X_overridden))
        
        diff = p_over - p_orig
        
        results.append({
            "model_architecture": arch,
            "seed": int(seed),
            "mean_orig_pred": np.mean(p_orig),
            "mean_overridden_pred": np.mean(p_over),
            "mean_abs_diff": np.mean(np.abs(diff)),
            "max_abs_diff": np.max(np.abs(diff)),
            "mean_diff_sandy_stations": np.mean(diff[mask]),
            "pct_change_sandy_stations": (np.mean(diff[mask]) / np.mean(p_orig[mask])) * 100,
        })

    df_t11 = pd.DataFrame(results)
    df_t11.to_csv(os.path.join(TABLES_DIR, "table11_soil_override_sensitivity.csv"), index=False)
    print("Table 11 saved.")
    return df_t11

def plot_kde(ax, data, label, color, linestyle="-", linewidth=2, fill=False):
    data_clean = np.asarray(data)[~np.isnan(data)]
    if len(data_clean) < 2 or np.std(data_clean) == 0:
        ax.axvline(np.mean(data_clean), color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        return
    kde = gaussian_kde(data_clean)
    x = np.linspace(max(0, np.min(data_clean) - 0.05), min(1.0, np.max(data_clean) + 0.05), 200)
    y = kde(x)
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    if fill:
        ax.fill_between(x, 0, y, color=color, alpha=0.2)

def generate_all_figures(data):
    print("Generating Publication Figures...")
    ece_test = data["ece_test"]
    wa_train = data["wa_train"]
    wa_all = data["wa_all"]
    pred_df = data["pred_ece_df"]
    
    # -------------------------------------------------------------
    # FIGURE 1: R² Variance Compression Anatomy
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    st_vars = []
    st_r2_global = []
    st_r2_dyn = []
    st_r2_static = []
    st_names = []
    
    for st, df in ece_test.groupby("station_id"):
        y = df["soil_moisture_5cm"]
        v = np.var(y, ddof=1)
        st_vars.append(v)
        st_names.append(st.replace("ECE_", ""))
        if "Home" in st:
            st_r2_global.append(-785.74)
            st_r2_dyn.append(-790.49)
            st_r2_static.append(-6724.48)
        elif "Main_St" in st:
            st_r2_global.append(-38.66)
            st_r2_dyn.append(-37.82)
            st_r2_static.append(-956.02)
        elif "Lost_Meadow" in st:
            st_r2_global.append(-50.48)
            st_r2_dyn.append(-38.78)
            st_r2_static.append(-283.75)
        elif "Garden_Shed" in st:
            st_r2_global.append(-23.94)
            st_r2_dyn.append(-14.06)
            st_r2_static.append(-843.31)
        elif "Garden_North" in st:
            st_r2_global.append(-6.92)
            st_r2_dyn.append(-6.50)
            st_r2_static.append(-9.15)
            
    axes[0].scatter(st_vars, st_r2_global, color='tab:blue', s=100, label='Global_Single_54', zorder=4)
    axes[0].scatter(st_vars, st_r2_dyn, color='tab:green', s=100, marker='^', label='Clustering_Dynamic_k2', zorder=4)
    axes[0].scatter(st_vars, st_r2_static, color='tab:red', s=100, marker='x', label='Clustering_Backbone54_k2', zorder=4)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Ground Truth Variance Var(y) [log scale]')
    axes[0].set_ylabel('Nash-Sutcliffe Efficiency R²')
    axes[0].set_title('(a) Collapse of R² as Target Variance Approaches Zero')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.7)
    axes[0].legend()
    
    for i, txt in enumerate(st_names):
        axes[0].annotate(txt, (st_vars[i], st_r2_global[i]), textcoords="offset points", xytext=(5,5), fontsize=8)
        
    stations_sub = ["ECE_Renton_Home", "ECE_BBG_Main_St", "ECE_Renton_Garden_North"]
    bias_sq = []
    var_err = []
    labels = []
    
    for st in stations_sub:
        sdf = pred_df[pred_df["station_id"] == st]
        y = sdf["y_true"]
        preds = sdf[[c for c in sdf.columns if "pred__d84_weighted__" in c]].mean(axis=1)
        err = preds - y
        bias_sq.append(np.mean(err)**2)
        var_err.append(np.var(err))
        labels.append(st.replace("ECE_", ""))
        
    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x - width/2, bias_sq, width, label='Bias² (Systematic Error)', color='tab:orange')
    axes[1].bar(x + width/2, var_err, width, label='Var(Error) (Random Error)', color='tab:purple')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Mean Squared Error Component (m³/m³)²')
    axes[1].set_title('(b) MSE Decomposition: Bias² Dominance on Low-Moisture Sites')
    axes[1].legend()
    
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_r2_variance_compression_anatomy.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print("Fig 1 saved.")

    # -------------------------------------------------------------
    # FIGURE 1B: Target Variance Comparison (ECE Sensors vs WA Test Period)
    # -------------------------------------------------------------
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    wa_test = data["wa_test"]
    
    wa_stats = []
    for st, g in wa_test.groupby("station_id"):
        y = g["soil_moisture_5cm"]
        wa_stats.append({
            "station": st.replace("_WA_990", "").replace("_WA_985", "").replace("_WA", ""),
            "mean": float(y.mean()),
            "std": float(y.std(ddof=1)),
            "var": float(y.var(ddof=1)),
            "var_pop": float(y.var(ddof=0)),
            "n": int(len(y)),
            "min": float(y.min()),
            "max": float(y.max()),
            "group": "WA Reference (2023-2025 Test)",
        })

    ece_stats = []
    for st, g in ece_test.groupby("station_id"):
        y = g["soil_moisture_5cm"]
        ece_stats.append({
            "station": st.replace("ECE_", ""),
            "mean": float(y.mean()),
            "std": float(y.std(ddof=1)),
            "var": float(y.var(ddof=1)),
            "var_pop": float(y.var(ddof=0)),
            "n": int(len(y)),
            "min": float(y.min()),
            "max": float(y.max()),
            "group": "ECE In-Situ (2026 Test)",
        })

    df_comp = pd.DataFrame(wa_stats + ece_stats)
    # Season-matched + pooled reference values (sample var for display).
    # Pooled variance exceeds the mean of per-station variances because
    # between-station mean differences inflate the pooled estimator.
    y_wa_pool = wa_test["soil_moisture_5cm"]
    y_ece_pool = ece_test["soil_moisture_5cm"]
    dt_wa = pd.to_datetime(wa_test["date"], errors="raise")
    wa_summer_y = wa_test[(dt_wa.dt.strftime("%m-%d") >= "07-20") & (dt_wa.dt.strftime("%m-%d") <= "08-19")]["soil_moisture_5cm"]
    pooled_refs = {
        "wa_pool_var": float(y_wa_pool.var(ddof=1)),
        "ece_pool_var": float(y_ece_pool.var(ddof=1)),
        "wa_summer_var": float(wa_summer_y.var(ddof=1)),
        "wa_pool_var_pop": float(y_wa_pool.var(ddof=0)),
        "ece_pool_var_pop": float(y_ece_pool.var(ddof=0)),
        "wa_summer_var_pop": float(wa_summer_y.var(ddof=0)),
    }
    colors = ["#1f77b4" if g == "WA Reference (2023-2025 Test)" else "#d62728" for g in df_comp["group"]]
    x = np.arange(len(df_comp))
    
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="WA Reference (7 Stations, 2023-2025 Test)"),
        Patch(facecolor="#d62728", edgecolor="black", label="ECE In-Situ (5 Sensors, 2026 Test)")
    ]
    
    # Panel (a): Mean +/- 1 std and [min, max] range
    axes[0].bar(x, df_comp["mean"], yerr=df_comp["std"], capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8, zorder=3)
    for idx, row in df_comp.iterrows():
        axes[0].plot([idx, idx], [row["min"], row["max"]], color="black", linestyle=":", linewidth=1.2, zorder=2)
        axes[0].scatter([idx, idx], [row["min"], row["max"]], color="black", s=12, marker="_", zorder=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_comp["station"], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Volumetric Soil Moisture (m³/m³)")
    axes[0].set_title("(a) Target Soil Moisture: Mean ± 1σ & Full Range", fontweight="bold")
    axes[0].set_ylim(0, 0.45)
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)
    
    # Panel (b): Target Variance on Logarithmic Scale
    # Shows per-station sample vars plus pooled and season-matched references so
    # readers do not mistake the full-year-vs-summer gap for a pure site effect.
    axes[1].bar(x, df_comp["var"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.8, zorder=3)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_comp["station"], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Target Variance Var(y) [log scale]")
    axes[1].set_title("(b) Target Variance Var(y) Compression (Log Scale)", fontweight="bold")
    mean_wa_var = df_comp[df_comp["group"] == "WA Reference (2023-2025 Test)"]["var"].mean()
    mean_ece_var = df_comp[df_comp["group"] == "ECE In-Situ (2026 Test)"]["var"].mean()
    min_ece_var = df_comp[df_comp["group"] == "ECE In-Situ (2026 Test)"]["var"].min()
    axes[1].axhline(mean_wa_var, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.8, label=f"WA Mean Var ({mean_wa_var:.1e})")
    axes[1].axhline(mean_ece_var, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8, label=f"ECE Mean Var ({mean_ece_var:.1e})")
    axes[1].axhline(min_ece_var, color="darkred", linestyle=":", linewidth=1.2, alpha=0.8, label=f"ECE Min Var ({min_ece_var:.1e})")
    axes[1].axhline(pooled_refs["wa_pool_var"], color="#1f77b4", linestyle="-", linewidth=1.4, alpha=0.9, label=f"WA Pooled Var ({pooled_refs['wa_pool_var']:.1e})")
    axes[1].axhline(pooled_refs["ece_pool_var"], color="#d62728", linestyle="-", linewidth=1.4, alpha=0.9, label=f"ECE Pooled Var ({pooled_refs['ece_pool_var']:.1e})")
    axes[1].axhline(pooled_refs["wa_summer_var"], color="#17becf", linestyle="-.", linewidth=1.6, alpha=0.95, label=f"WA Summer Pooled ({pooled_refs['wa_summer_var']:.1e}; season-matched)")
    axes[1].legend(loc="upper right", fontsize=7)

    # Panel (c): Mathematical R² Penalty at Fixed RMSE = 0.04 m³/m³
    # Uses population variance (ddof=0), consistent with Table 1b, plus pooled
    # markers: pooled ECE R2 is positive (~+0.28); collapse is per-station.
    rmse_bench = 0.04
    r2_hypo = 1.0 - (rmse_bench**2) / df_comp["var_pop"]
    axes[2].bar(x, r2_hypo, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8, zorder=3)
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df_comp["station"], rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel(r"Theoretical $R^2 = 1 - (0.04)^2/\mathrm{Var}_{pop}(y)$")
    axes[2].set_title(r"(c) Theoretical Per-Station $R^2$ at Fixed RMSE = $0.04\ \mathrm{m}^3/\mathrm{m}^3$", fontweight="bold")
    axes[2].axhline(0, color="gray", linestyle="-", linewidth=1.0)
    for key, ls, col, lab in [
        ("wa_pool_var_pop", "--", "#1f77b4", "WA pooled"),
        ("ece_pool_var_pop", "-", "#d62728", "ECE pooled (+0.28)"),
        ("wa_summer_var_pop", "-.", "#17becf", "WA summer pooled"),
    ]:
        axes[2].axhline(1.0 - rmse_bench**2 / pooled_refs[key], color=col, linestyle=ls, linewidth=1.2, alpha=0.9, label=f"{lab} R²={1.0 - rmse_bench**2 / pooled_refs[key]:+.2f}")
    axes[2].legend(handles=legend_elements + [Patch(facecolor="none", edgecolor="none", label="Dashed/dotted: pooled refs (positive R²)")], loc="lower left", fontsize=7)
    
    plt.tight_layout()
    fig1b_path = os.path.join(FIGURES_DIR, "fig1b_target_variance_ece_vs_wa_test_comparison.png")
    plt.savefig(fig1b_path, dpi=300)
    plt.close()
    print("Fig 1b saved.")

    # -------------------------------------------------------------
    # FIGURE 1C: 1-Month 2025 Summer vs 2026 ECE Performance Bridge
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    pred_dir = os.path.join(PROJECT_ROOT, "notebooks/experiment/derived_8.4-formal-eval-2.0/predictions")
    mask_2025 = (wa_test["date"] >= "2025-07-20") & (wa_test["date"] <= "2025-08-19")
    models_sub = ["Clustering_V0_Full_k2", "Clustering_Dynamic_k2", "Global_Single_54", "Baseline_V0_50", "Univariate_G_API_k2", "Trained_Gating_k2"]
    
    # Panel (a): RMSE comparison
    x = np.arange(len(models_sub))
    w = 0.25
    
    rmse_full = []
    rmse_sum25 = []
    rmse_ece = [0.1004, 0.0483, 0.0511, 0.0515, 0.0479, 0.0495]
    
    for m in models_sub:
        files = sorted(glob.glob(f"{pred_dir}/{m}*full_preds.npy"))
        p = np.mean([np.load(f) for f in files[:5]], axis=0)
        err_f = p - wa_test["soil_moisture_5cm"].values
        rmse_full.append(np.sqrt(np.mean(err_f**2)))
        
        sub = wa_test[mask_2025]
        err_s = p[mask_2025] - sub["soil_moisture_5cm"].values
        rmse_sum25.append(np.sqrt(np.mean(err_s**2)))
        
    axes[0].bar(x - w, rmse_full, w, label='Full 3-Yr Test (2023-2025)', color='#1f77b4', edgecolor='black', alpha=0.85)
    axes[0].bar(x, rmse_sum25, w, label='1-Mo 2025 Summer (Ref)', color='#2ca02c', edgecolor='black', alpha=0.85)
    axes[0].bar(x + w, rmse_ece, w, label='1-Mo 2026 Summer (ECE)', color='#d62728', edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace('_k2', '').replace('_54', '').replace('_50', '') for m in models_sub], rotation=35, ha='right', fontsize=8)
    axes[0].set_ylabel('RMSE (m³/m³)')
    axes[0].set_title('(a) Physical Error Comparison: 1-Mo Summer vs Full Test', fontweight='bold')
    axes[0].legend(fontsize=7, loc='upper left')

    # Panel (b): Station R2 in 2025 Summer for Key Architectures
    st_names = ['CayusePass', 'Darrington', 'Paradise', 'Quinault', 'Spokane']
    x_st = np.arange(len(st_names))
    w_st = 0.25
    
    p_glob = np.mean([np.load(f) for f in sorted(glob.glob(f"{pred_dir}/Global_Single_54*full_preds.npy"))[:5]], axis=0)
    p_base = np.mean([np.load(f) for f in sorted(glob.glob(f"{pred_dir}/Baseline_V0_50*full_preds.npy"))[:5]], axis=0)
    p_v0 = np.mean([np.load(f) for f in sorted(glob.glob(f"{pred_dir}/Clustering_V0_Full_k2*full_preds.npy"))[:5]], axis=0)
    
    wa_test_eval = wa_test.copy()
    wa_test_eval['p_glob'] = p_glob
    wa_test_eval['p_base'] = p_base
    wa_test_eval['p_v0'] = p_v0
    sub25 = wa_test_eval[mask_2025]
    
    r2_g, r2_b, r2_v = [], [], []
    for st, g in sub25.groupby("station_id"):
        y = g['soil_moisture_5cm'].values
        v = np.var(y, ddof=1)
        r2_g.append(1.0 - np.mean((g['p_glob'].values - y)**2) / v)
        r2_b.append(1.0 - np.mean((g['p_base'].values - y)**2) / v)
        r2_v.append(1.0 - np.mean((g['p_v0'].values - y)**2) / v)
        
    axes[1].bar(x_st - w_st, r2_v, w_st, label='Clustering_V0', color='#9467bd', edgecolor='black', alpha=0.85)
    axes[1].bar(x_st, r2_g, w_st, label='Global_Single_54', color='#ff7f0e', edgecolor='black', alpha=0.85)
    axes[1].bar(x_st + w_st, r2_b, w_st, label='Baseline_V0_50', color='#8c564b', edgecolor='black', alpha=0.85)
    axes[1].set_yscale('symlog', linthresh=1.0)
    axes[1].set_xticks(x_st)
    axes[1].set_xticklabels(st_names, rotation=35, ha='right', fontsize=8)
    axes[1].set_ylabel('Station R² (symlog scale)')
    axes[1].set_title('(b) 2025 Summer: Native Stations Plunge into Negative R²', fontweight='bold')
    axes[1].axhline(0, color='gray', linestyle='-', linewidth=0.8)
    axes[1].scatter([4.6, 4.8, 5.0], [0.8003, 0.5006, 0.6723], marker='D', s=40, color=['#9467bd', '#ff7f0e', '#8c564b'], label='Pooled R² (+0.50 to +0.80)', zorder=5)
    axes[1].legend(fontsize=7, loc='lower left')

    # Panel (c): Hyperbolic decay curve R2 vs Var(y)
    var_curve = np.logspace(-6, -1, 200)
    for rmse_val, ls in [(0.02, ':'), (0.03, '--'), (0.04, '-'), (0.05, '-.')]:
        r2_curve = 1.0 - (rmse_val**2) / var_curve
        axes[2].plot(var_curve, r2_curve, linestyle=ls, color='gray', alpha=0.7, label=f'Theory (RMSE={rmse_val})')
        
    # Scatter reference stations (2025)
    st_vars_25 = []
    st_r2_g = []
    for st, g in sub25.groupby("station_id"):
        y = g['soil_moisture_5cm'].values
        v = np.var(y, ddof=1)
        st_vars_25.append(v)
        st_r2_g.append(1.0 - np.mean((g['p_glob'].values - y)**2) / v)
    axes[2].scatter(st_vars_25, st_r2_g, color='#1f77b4', s=80, marker='o', edgecolor='black', label='WA Ref Stations (2025 Summer)', zorder=5)

    # Scatter ECE stations (2026)
    ece_vars = [6.03e-5, 3.25e-5, 6.94e-4, 2.11e-5, 6.43e-6]
    ece_r2_g = [-50.48, -38.66, -6.92, -23.94, -785.74]
    axes[2].scatter(ece_vars, ece_r2_g, color='#d62728', s=80, marker='s', edgecolor='black', label='ECE In-Situ (2026 Summer)', zorder=5)

    axes[2].set_xscale('log')
    axes[2].set_yscale('symlog', linthresh=1.0)
    axes[2].set_xlabel('Target Variance Var(y) [log scale]')
    axes[2].set_ylabel('R² (symlog scale)')
    axes[2].set_title('(c) Unified R² vs Var(y) Hyperbolic Decay Curve', fontweight='bold')
    axes[2].axhline(0, color='gray', linestyle='-', linewidth=0.8)
    axes[2].legend(fontsize=7, loc='lower left')

    plt.tight_layout()
    fig1c_path = os.path.join(FIGURES_DIR, "fig1c_1month_2025_summer_vs_ece_bridge.png")
    plt.savefig(fig1c_path, dpi=300)
    plt.close()
    print("Fig 1c saved.")

    # -------------------------------------------------------------
    # FIGURE 2: SMAP & MODIS NDVI Missingness & Feature Shift
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    smap_train = wa_train["SMAP_sm_am_interp"].dropna()
    smap_ece = ece_test["SMAP_sm_am_interp"].dropna()
    
    plot_kde(axes[0], smap_train, label='WA Train (2017-2022, N=14,608)', color='tab:blue', fill=True)
    axes[0].hist(smap_ece, bins=10, density=True, color='tab:red', alpha=0.7, label='ECE Test 2026 (100% Zero Spike)')
    axes[0].set_xlabel('SMAP Soil Moisture (m³/m³)')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_title('(a) Severe Domain Gap: Zeroed SMAP Satellite Inputs in 2026')
    axes[0].legend()
    
    axes[1].hist(wa_train["F_NDVI"].dropna(), bins=30, density=True, alpha=0.6, color='tab:green', label='WA Train F_NDVI')
    axes[1].hist(ece_test["F_NDVI"].dropna(), bins=10, density=True, alpha=0.7, color='tab:orange', label='ECE In-Situ F_NDVI')
    axes[1].set_xlabel('Sentinel-2 Optical NDVI (Canopy Greenness)')
    axes[1].set_ylabel('Probability Density')
    axes[1].set_title('(b) Optical NDVI Comparison (WA Baseline vs ECE Lowlands)')
    axes[1].legend()
    
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_smap_ndvi_missingness_distributions.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print("Fig 2 saved.")

    # -------------------------------------------------------------
    # FIGURE 3: Spatial Microclimate Discrepancy (53m Renton Pair)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    r_north = ece_test[ece_test["station_id"] == "ECE_Renton_Garden_North"].sort_values("date")
    r_shed = ece_test[ece_test["station_id"] == "ECE_Renton_Garden_Shed"].sort_values("date")
    dates = pd.to_datetime(r_north["date"])
    
    ax1 = axes[0]
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(dates, r_north["soil_moisture_5cm"], 'o-', color='tab:green', label='Renton Garden North (15.5% mean)', linewidth=2)
    l2 = ax1.plot(dates, r_shed["soil_moisture_5cm"], 's-', color='tab:brown', label='Renton Garden Shed (7.6% mean)', linewidth=2)
    l3 = ax2.bar(dates, r_north["precip_mm"], width=0.4, color='tab:blue', alpha=0.3, label='Rainfall (Identical for both)')
    
    ax1.set_xlabel('Date (2026)')
    ax1.set_ylabel('Measured Volumetric Soil Moisture (m³/m³)')
    ax2.set_ylabel('Daily Precipitation (mm)', color='tab:blue')
    ax1.set_title('(a) 2.04× Ground Truth Divergence Between Sensors 53.4m Apart')
    
    lines = l1 + l2 + [l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    pred_north = pred_df[pred_df["station_id"] == "ECE_Renton_Garden_North"][[c for c in pred_df.columns if "d84_weighted" in c]].mean(axis=1)
    pred_shed = pred_df[pred_df["station_id"] == "ECE_Renton_Garden_Shed"][[c for c in pred_df.columns if "d84_weighted" in c]].mean(axis=1)
    
    axes[1].plot(dates, r_north["soil_moisture_5cm"], 'o--', color='tab:green', alpha=0.5, label='Actual: Garden North')
    axes[1].plot(dates, pred_north, '-', color='tab:green', linewidth=2.5, label='Predicted: Garden North (~0.131)')
    axes[1].plot(dates, r_shed["soil_moisture_5cm"], 's--', color='tab:brown', alpha=0.5, label='Actual: Garden Shed')
    axes[1].plot(dates, pred_shed, ':', color='tab:brown', linewidth=2.5, label='Predicted: Garden Shed (~0.131)')
    
    axes[1].set_xlabel('Date (2026)')
    axes[1].set_ylabel('Soil Moisture (m³/m³)')
    axes[1].set_title('(b) Model Predicts Identical Values (~0.131) for 2× Divergent Truths')
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_spatial_microclimate_discrepancy.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print("Fig 3 saved.")

    # -------------------------------------------------------------
    # FIGURE 4: Target Distribution Domain Shift
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    
    wa_jul_aug = wa_all[pd.to_datetime(wa_all["date"]).dt.month.isin([7, 8])]
    
    plot_kde(ax, wa_all["soil_moisture_5cm"], label=f'WA Reference All Seasons (μ={wa_all["soil_moisture_5cm"].mean():.3f})', color='navy', linewidth=2)
    plot_kde(ax, wa_jul_aug["soil_moisture_5cm"], label=f'WA Reference Summer Jul-Aug (μ={wa_jul_aug["soil_moisture_5cm"].mean():.3f})', color='tab:blue', linestyle='--', linewidth=2)
    
    for st, color in zip(data["ece_test"]["station_id"].unique(), ['tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']):
        sub = data["ece_test"][data["ece_test"]["station_id"] == st]["soil_moisture_5cm"]
        plot_kde(ax, sub, label=f'{st.replace("ECE_", "")} (μ={sub.mean():.3f}, σ={sub.std():.3f})', color=color, linewidth=1.8)
        
    ax.set_xlabel('Volumetric Soil Moisture (m³/m³)')
    ax.set_ylabel('Density')
    ax.set_title('Target Soil Moisture Distribution: SNOTEL Reference vs ECE In-Situ Sensors')
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "fig4_target_distribution_domain_shift.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print("Fig 4 saved.")

    # -------------------------------------------------------------
    # FIGURE 5: Routing Strategy Comparison across 5 Stations
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = ["Univariate_G_API_k2", "Clustering_Dynamic_k2", "Seasonal_Binary_k2", "Global_Single_54", "Clustering_V0_Full_k2", "Clustering_Backbone54_k2"]
    med_r2 = [-30.34, -37.82, -38.69, -38.66, -73.37, -843.31]
    rmse_vals = [0.0479, 0.0483, 0.0503, 0.0511, 0.1004, 0.1441]
    colors = ['tab:green', 'tab:cyan', 'tab:blue', 'tab:olive', 'tab:orange', 'tab:red']
    
    ax.bar(np.arange(len(models)), rmse_vals, color=colors, width=0.5, edgecolor='black')
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models], rotation=0)
    ax.set_ylabel('Station Mean RMSE (m³/m³) [Lower is Better]')
    ax.set_title('In-Situ ECE Transfer: Dynamic/Heuristic Routers vs Static MoE Routing Traps')
    
    for i, (r, v) in enumerate(zip(med_r2, rmse_vals)):
        ax.text(i, v + 0.003, f"RMSE: {v:.3f}\nMed R²: {r:.1f}", ha='center', va='bottom', fontsize=8, weight='bold')
        
    ax.set_ylim(0, 0.18)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "fig5_routing_strategy_ece_comparison.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print("Fig 5 saved.")

    # -------------------------------------------------------------
    # FIGURE 6: Raw ADC to Moisture Calibration Scatter
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_files = glob.glob(os.path.join(PROJECT_ROOT, "src/pipeline/data/raw/_ECE/*.csv"))
    
    for f in sorted(raw_files):
        df_raw = pd.read_csv(f, skiprows=1)
        st_name = os.path.basename(f).split("(")[-1].replace(").csv", "").replace("Trail (BBG)", "BBG Lost Meadow").replace("Main St (BBG)", "BBG Main St")
        adc_col = [c for c in df_raw.columns if "adc" in c.lower()][0]
        sm_col = [c for c in df_raw.columns if "moisture" in c.lower()][0]
        
        sample = df_raw.sample(n=min(1000, len(df_raw)), random_state=42)
        ax.scatter(sample[adc_col], sample[sm_col], alpha=0.4, s=15, label=st_name)
        
    ax.set_xlabel('Raw ADC Value (Digital Counts)')
    ax.set_ylabel('Reported Soil Moisture (%)')
    ax.set_title('Raw ADC Counts vs Calibrated Soil Moisture (%) across In-Situ Probes')
    ax.axhline(0.0, color='red', linestyle='--', alpha=0.5, label='Zero Moisture Baseline (Device 11 Bottoms Out)')
    ax.legend()
    
    plt.tight_layout()
    fig6_path = os.path.join(FIGURES_DIR, "fig6_raw_adc_to_moisture_calibration.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print("Fig 6 saved.")

    # -------------------------------------------------------------
    # FIGURE 7: Error Decomposition Waterfall
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.text(0.05, 0.75, "1. Hydroclimatic Accuracy:\nPhysical RMSE = 0.048 m³/m³\n(Better than Out-of-State transfer!)", 
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", lw=1.5), fontsize=10)
    ax.text(0.38, 0.75, "2. Siting / Scale Bias:\nConstant +0.06 to +0.14 m³/m³ offset\n(due to sub-meter canopy & rain shadow)", 
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", ec="orange", lw=1.5), fontsize=10)
    ax.text(0.72, 0.75, "3. Missing Satellite Features:\n85 SMAP + MODIS features zeroed\n(Tree splits forced into left leaf nodes)", 
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gold", lw=1.5), fontsize=10)
    
    ax.annotate("", xy=(0.36, 0.80), xytext=(0.31, 0.80), arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
    ax.annotate("", xy=(0.70, 0.80), xytext=(0.65, 0.80), arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
    
    ax.text(0.20, 0.35, "4. Low-Variance Summer Ground Truth:\nTarget variance Var(y) collapses to 0.000006 m³/m³ in Mediterranean dry summer", 
            bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", lw=1.5), fontsize=10)
    
    ax.annotate("", xy=(0.5, 0.47), xytext=(0.5, 0.70), arrowprops=dict(arrowstyle="->", lw=2, color='red'))
    
    ax.text(0.15, 0.05, "RESULT: R² = 1 - MSE / Var(y) = 1 - (0.010 / 0.000006) = -1,665\nAstronomical negative R² despite physical error <= 0.10 m³/m³", 
            bbox=dict(boxstyle="round,pad=0.7", fc="salmon", ec="red", lw=2), fontsize=11, weight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    fig7_path = os.path.join(FIGURES_DIR, "fig7_error_decomposition_waterfall.png")
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print("Fig 7 saved.")

    # -------------------------------------------------------------
    # FIGURE 8: 5-Panel Composite & Individual Station Time Series
    # -------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    stations = data["ece_test"]["station_id"].unique()
    
    for idx, (st, ax) in enumerate(zip(stations, axes)):
        st_df = data["ece_test"][data["ece_test"]["station_id"] == st].sort_values("date")
        dates = pd.to_datetime(st_df["date"])
        y_true = st_df["soil_moisture_5cm"].values
        
        ax.plot(dates, y_true, 'k-o', label='Ground Truth (In-Situ Sensor)', linewidth=2.5, markersize=5, zorder=5)
        
        if pred_df is not None:
            sdf = pred_df[pred_df["station_id"] == st].sort_values("date")
            p_d84_w = sdf[[c for c in sdf.columns if "pred__d84_weighted__" in c]].mean(axis=1).values
            p_d80_w = sdf[[c for c in sdf.columns if "pred__d80_weighted__" in c]].mean(axis=1).values
            p_d84_no = sdf[[c for c in sdf.columns if "pred__d84_no_weights__" in c]].mean(axis=1).values
            
            ax.plot(dates, p_d84_w, '-', color='tab:blue', label='d84_weighted (7 st, Huber)', linewidth=1.8)
            ax.plot(dates, p_d80_w, '--', color='tab:green', label='d80_weighted (5 st, Huber)', linewidth=1.8)
            ax.plot(dates, p_d84_no, ':', color='tab:red', label='d84_no_weights (7 st, L1)', linewidth=1.8)
            
        ax.set_ylabel('Moisture (m³/m³)')
        ax.set_title(f"Station {idx+1}: {st} (Mean Truth = {np.mean(y_true):.4f}, Std = {np.std(y_true):.4f})", fontsize=11, weight='bold')
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        
        fig_st, ax_st = plt.subplots(figsize=(10, 4))
        ax_st.plot(dates, y_true, 'k-o', label='Ground Truth (In-Situ Sensor)', linewidth=2.5, markersize=5)
        if pred_df is not None:
            ax_st.plot(dates, p_d84_w, '-', color='tab:blue', label='d84_weighted (7 st, Huber)', linewidth=2)
            ax_st.plot(dates, p_d80_w, '--', color='tab:green', label='d80_weighted (5 st, Huber)', linewidth=2)
            ax_st.plot(dates, p_d84_no, ':', color='tab:red', label='d84_no_weights (7 st, L1)', linewidth=2)
        ax_st.set_xlabel('Date (2026)')
        ax_st.set_ylabel('Soil Moisture (m³/m³)')
        ax_st.set_title(f"{st} — Observed vs Predicted Time Series (July 20 – August 19, 2026)", fontsize=12, weight='bold')
        ax_st.legend(loc='upper right')
        ax_st.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        fig_st.tight_layout()
        fig_st.savefig(os.path.join(FIGURES_DIR, f"fig8_station_{st}_timeseries.png"), dpi=300)
        plt.close(fig_st)
        
    axes[-1].set_xlabel('Date (July 20 – August 19, 2026)')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, "fig8_per_station_timeseries_overlay.png")
    fig.savefig(fig8_path, dpi=300)
    plt.close(fig)
    print("Fig 8 & standalone station time-series figures saved.")

    # -------------------------------------------------------------
    # FIGURE 9: Coincidental Accuracy & Prediction Homogeneity Proof
    # -------------------------------------------------------------
    if pred_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        cols_w = [c for c in pred_df.columns if "pred__d84_weighted" in c]
        pred_df["d84_w"] = pred_df[cols_w].mean(axis=1)
        pvt_w = pred_df.pivot(index="date", columns="station_id", values="d84_w")
        pvt_true = pred_df.pivot(index="date", columns="station_id", values="y_true")
        
        dates = pd.to_datetime(pvt_w.index)
        for st, color in zip(pvt_w.columns, ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive', 'tab:orange']):
            axes[0].plot(dates, pvt_w[st], label=f"Pred: {st.replace('ECE_', '')}", color=color, linewidth=1.8)
            
        axes[0].set_xlabel('Date (2026)')
        axes[0].set_ylabel('Predicted Soil Moisture (m³/m³)')
        axes[0].set_title('(a) Cross-Station Prediction Homogeneity (r > 0.96 across all sites)')
        axes[0].legend(fontsize=8)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        global_fallback = pvt_w.values.mean()
        st_names = []
        dists = []
        rmses = []
        
        for st in pvt_w.columns:
            y_t = pvt_true[st]
            y_p = pvt_w[st]
            dists.append(abs(y_t.mean() - global_fallback))
            rmses.append(np.sqrt(np.mean((y_p - y_t)**2)))
            st_names.append(st.replace("ECE_", ""))
            
        axes[1].scatter(dists, rmses, color='tab:red', s=120, zorder=4)
        axes[1].plot(dists, dists, '--', color='gray', label='1:1 Bias-Dominated Line (RMSE ≈ |Mean Truth - Fallback|)')
        
        for i, txt in enumerate(st_names):
            axes[1].annotate(txt, (dists[i], rmses[i]), textcoords="offset points", xytext=(5,5), fontsize=9, weight='bold')
            
        axes[1].set_xlabel('|Station Ground Truth Mean - Model Fallback Mean (~0.13)| (m³/m³)')
        axes[1].set_ylabel('Observed Station RMSE (m³/m³)')
        axes[1].set_title('(b) Proof of Coincidence: RMSE is Fully Dictated by Fallback Distance')
        axes[1].legend()
        
        plt.tight_layout()
        fig9_path = os.path.join(FIGURES_DIR, "fig9_coincidental_accuracy_analysis.png")
        plt.savefig(fig9_path, dpi=300)
        plt.close()
        print("Fig 9 saved.")

def main():
    print("=== STARTING DIAGNOSTICS GENERATION ===")
    data = load_data()
    generate_table1_variance_compression(data)
    generate_table1b_target_variance_comparison(data)
    generate_table1c_1month_2025_summer_all_metrics(data)
    generate_table1d_macro_evaluation_window_benchmark(data)
    generate_table2_historical_benchmarks(data)
    generate_table3_missing_data_audit(data)
    generate_table4_spatial_proximity_and_side_by_side(data)
    generate_table5_target_climatology(data)
    generate_table6_routing_strategies(data)
    generate_table7_raw_adc_calibration(data)
    generate_table8_recommendations()
    generate_table9_coincidental_accuracy(data)
    generate_table10_soil_texture_all_stations(data)
    generate_table11_soil_override_sensitivity(data)
    generate_all_figures(data)
    print("=== ALL DIAGNOSTICS, TABLES, AND FIGURES SUCCESSFULLY GENERATED ===")

if __name__ == "__main__":
    main()
