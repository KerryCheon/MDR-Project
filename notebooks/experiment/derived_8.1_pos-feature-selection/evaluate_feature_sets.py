import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Set up project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from Modeling.Utils.config import load_config
from Modeling.Src.soilmoist_fl.Data.load import load_splits
from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split
from Modeling.Src.soilmoist_fl.Selectors.mi import select_mi
from Modeling.Src.soilmoist_fl.Selectors.elasticnet import select_elasticnet

# Import Feature Set B from dataset_metadata
sys.path.append(str(project_root / "data" / "splits" / "derived_8.1_pos"))
import dataset_metadata
FEATURE_SET_B = dataset_metadata.OVERALL_SELECTED_FEATURES

FEATURE_SET_A = [
    'SMAP_sm_pm_interp_ema02',
    'V_rollmin_LST_modis_kobs30',
    'D_sin_DOY', 'G_rain_sum_3d',
    'V_ema_G_API_kobs7',
    'V_rollmin_G_API_kobs30',
    'G_rain_sum_7d',
    'C_lag_LST_modis_kobs30',
    'C_lag_G_API_kobs1',
    'V_ema_G_API_kobs14',
    'V_rollmean_G_API_kobs14',
    'G_API', 'G_DSLR',
    'SMAP_ampm_diff_interp',
    'V_rollmax_G_API_kobs30',
    'V_ema_G_API_kobs30',
    'V_rollmean_s2_b11_kobs7',
    'V_ema_LST_modis_kobs7',
    'V_rollmean_G_API_kobs7',
    'C_lag_s2_b11_kobs30',
    'A_d_E_SAR_diff_kobs14',
    'C_lag_LST_modis_kobs6',
    'A_d_LST_modis_kobs14',
    'A_d_SMAP_sm_interp_kobs14',
    'V_rollstd_SMAP_sm_interp_kobs30',
    'SMAP_sm_interp_grad7',
    'year_frac', 'sin_year', 'cos_year',
    'API_x_year', 'SMAP_x_year',
    'slope', 'elev', 'K_slope_sin',
    'K_slope_cos', 'K_aspect_cos',
    'J_clay_wfrac_b0', 'J_sand_wfrac_b0'
]

XGB_PARAMS_W = dict(
    objective="reg:pseudohubererror",
    random_state=42,
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
    device="cuda",
)

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    ubrmse = np.sqrt(np.mean(((y_true - np.mean(y_true)) - (y_pred - np.mean(y_pred))) ** 2))
    bias = np.mean(err)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = float("nan")
    else:
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
    return {"R2": r2, "RMSE": rmse, "ubRMSE": ubrmse, "Bias": bias, "MAE": mae, "Pearson": pearson}

def run_hybrid_selection(X, y):
    print("Running Hybrid/Bypass selection...")
    # Identify bypass features
    bypass_cols = [c for c in X.columns if c.startswith('J_') or c.startswith('K_') or c.startswith('D_') or c.startswith('G_') or 'year' in c or c in ['longitude', 'latitude', 'elev', 'slope', 'aspect', 'DOY', 'precip_mm']]
    ts_cols = [c for c in X.columns if c not in bypass_cols]
    
    # MI on time-series features only
    mi_out = select_mi(X[ts_cols], y, k=100)
    selected_ts = mi_out["selected"]
    
    # Combine selected time-series features with bypass features
    candidate_cols = selected_ts + bypass_cols
    print(f"Candidate features count: {len(candidate_cols)} ({len(selected_ts)} TS + {len(bypass_cols)} Bypass)")
    
    # ElasticNet selection on Candidate cols directly (k=40)
    X_candidates = X[candidate_cols]
    enet_out = select_elasticnet(X_candidates, y, k=40)
    return enet_out["selected"]

def main():
    config_path = "notebooks/experiment/derived_8.1_pos-feature-selection/config.yaml"
    cfg = load_config(config_path)
    
    loaded = load_splits(cfg)
    fold = loaded.folds[0]
    
    # Preprocess splits for feature selection (on Train)
    target = cfg["data"]["target"]
    id_cols = list(cfg["data"].get("id_cols", []) or [])
    time_col = cfg["data"].get("time_col")
    drop_cols = list(id_cols)
    if time_col:
        drop_cols.append(time_col)
        
    X_tr_fs, y_tr_fs, _, _ = preprocess_split(fold.train, target, drop_cols=drop_cols)
    
    # ------------------ Run Selections ------------------
    # Set C: No MI
    print("\n--- Running Selection C: No MI Filter ---")
    enet_c = select_elasticnet(X_tr_fs, y_tr_fs, k=40)
    feature_set_c = enet_c["selected"]
    print(f"Set C selection completed: {len(feature_set_c)} features selected")
    
    # Set D: High MI
    print("\n--- Running Selection D: High MI Filter (k=300) ---")
    mi_d = select_mi(X_tr_fs, y_tr_fs, k=300)
    X_mi_d = X_tr_fs[mi_d["selected"]]
    enet_d = select_elasticnet(X_mi_d, y_tr_fs, k=40)
    feature_set_d = enet_d["selected"]
    print(f"Set D selection completed: {len(feature_set_d)} features selected")
    
    # Set E: Hybrid
    print("\n--- Running Selection E: Hybrid/Bypass ---")
    feature_set_e = run_hybrid_selection(X_tr_fs, y_tr_fs)
    print(f"Set E selection completed: {len(feature_set_e)} features selected")
    
    # ------------------ Prepare Data for Modeling ------------------
    train_df = fold.train.copy()
    val_df = fold.val.copy()
    test_df = fold.test.copy()
    
    for df in [train_df, val_df, test_df]:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month.astype(int)
        df["year"] = df["date"].dt.year.astype(float)
        
    trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    y_trainval = np.asarray(trainval_df[target]).ravel()
    y_test = np.asarray(test_df[target]).ravel()
    
    # recency weights
    years_tv = trainval_df["year"]
    max_year = years_tv.max()
    beta = 0.2
    w_trainval = np.exp(beta * (years_tv - max_year))
    w_trainval = w_trainval / w_trainval.mean()
    
    feature_sets = {
        "A (v25 Baseline)": FEATURE_SET_A,
        "B (Default: MI=120)": FEATURE_SET_B,
        "C (No MI Filter)": feature_set_c,
        "D (High MI: k=300)": feature_set_d,
        "E (Hybrid: Bypass MI)": feature_set_e
    }
    
    results = []
    
    for name, f_set in feature_sets.items():
        print(f"\nEvaluating Feature Set: {name} (Size: {len(f_set)})")
        
        # Build training matrix
        X_trainval = trainval_df[f_set].copy()
        X_test = test_df[f_set].copy()
        
        for col in f_set:
            X_trainval[col] = pd.to_numeric(X_trainval[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            
        model = XGBRegressor(**XGB_PARAMS_W)
        model.fit(X_trainval, y_trainval, sample_weight=w_trainval, verbose=0)
        
        preds = np.asarray(model.predict(X_test)).ravel()
        metrics = compute_metrics(y_test, preds)
        metrics["Feature Set"] = name
        metrics["Size"] = len(f_set)
        results.append(metrics)
        
        print(f"  R2: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} | Pearson: {metrics['Pearson']:.4f}")
        
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("="*80)
    results_df = pd.DataFrame(results)
    results_df = results_df[["Feature Set", "Size", "R2", "RMSE", "ubRMSE", "Bias", "MAE", "Pearson"]]
    print(results_df.to_string(index=False, formatters={
        'R2': '{:,.4f}'.format,
        'RMSE': '{:,.4f}'.format,
        'ubRMSE': '{:,.4f}'.format,
        'Bias': '{:+,.4f}'.format,
        'MAE': '{:,.4f}'.format,
        'Pearson': '{:,.4f}'.format
    }))
    print("="*80)
    
    # Save the selected feature lists to a JSON for reference
    import json
    feature_dump = {
        "Set_C": list(feature_set_c),
        "Set_D": list(feature_set_d),
        "Set_E": list(feature_set_e)
    }
    with open("notebooks/experiment/derived_8.1_pos-feature-selection/selected_features_comparison.json", "w") as f:
        json.dump(feature_dump, f, indent=2)

if __name__ == "__main__":
    main()
