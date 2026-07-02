import os
import sys
import random
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Configure stdout to use UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Resolve project root relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
print(f"Project root resolved to: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/train.csv")
VAL_PATH = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/val.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/test.csv")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)
trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

import importlib.util
metadata_path = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/dataset_metadata.py")
spec = importlib.util.spec_from_file_location("dataset_metadata", metadata_path)
dataset_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_metadata)

T1 = dataset_metadata.T1
T2 = dataset_metadata.T2
OVERALL_SELECTED_FEATURES = dataset_metadata.OVERALL_SELECTED_FEATURES
DRY_SELECTED_FEATURES = dataset_metadata.DRY_SELECTED_FEATURES
TRANSITION_SELECTED_FEATURES = dataset_metadata.TRANSITION_SELECTED_FEATURES
WET_SELECTED_FEATURES = dataset_metadata.WET_SELECTED_FEATURES

T_2REGIME = dataset_metadata.T_2REGIME
DRY_2REGIME_SELECTED_FEATURES = dataset_metadata.DRY_2REGIME_SELECTED_FEATURES
WET_2REGIME_SELECTED_FEATURES = dataset_metadata.WET_2REGIME_SELECTED_FEATURES

TARGET_COL = "soil_moisture_5cm"

y_trainval = np.asarray(trainval_df[TARGET_COL]).ravel()
y_test = np.asarray(test_df[TARGET_COL]).ravel()

mask_dry_tv = y_trainval < T1
mask_transition_tv = (y_trainval >= T1) & (y_trainval < T2)
mask_wet_tv = y_trainval >= T2

mask_dry_test = y_test < T1
mask_transition_test = (y_test >= T1) & (y_test < T2)
mask_wet_test = y_test >= T2

mask_dry_tv_2r = y_trainval < T_2REGIME
mask_wet_tv_2r = y_trainval >= T_2REGIME
mask_dry_test_2r = y_test < T_2REGIME
mask_wet_test_2r = y_test >= T_2REGIME

# Verify CUDA/GPU availability for XGBoost
try:
    dummy = XGBRegressor(n_estimators=1, device="cuda")
    dummy.fit(np.array([[1.0]]), np.array([1.0]))
    XGB_DEVICE = "cuda"
except Exception as e:
    XGB_DEVICE = "cpu"

def _ubrmse(y_true, y_pred):
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def _safe_corr(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def run_experiment_with_beta(beta):
    # Compute temporal weights
    if beta == 0.0:
        w_trainval = np.ones(len(trainval_df))
    else:
        years_tv = trainval_df["year"].astype(float)
        max_year = years_tv.max()
        w_trainval = np.exp(beta * (years_tv - max_year))
        w_trainval = w_trainval / w_trainval.mean()
        
    # Model 1
    xgb_global = XGBRegressor(
        objective="reg:absoluteerror", random_state=SEED, n_jobs=-1,
        subsample=0.9, colsample_bytree=0.8, max_depth=8, min_child_weight=2,
        n_estimators=5500, learning_rate=0.04, reg_lambda=1.5, reg_alpha=0.03,
        gamma=0.0, device=XGB_DEVICE
    )
    xgb_global.fit(trainval_df[OVERALL_SELECTED_FEATURES], trainval_df[TARGET_COL], sample_weight=w_trainval, verbose=0)
    pred_global = np.asarray(xgb_global.predict(test_df[OVERALL_SELECTED_FEATURES])).ravel()
    
    # Model 2 & 3
    xgb_dry = XGBRegressor(
        objective="reg:absoluteerror", random_state=SEED, n_jobs=-1,
        subsample=0.9, colsample_bytree=0.8, max_depth=8, min_child_weight=2,
        n_estimators=5500, learning_rate=0.04, reg_lambda=1.5, reg_alpha=0.03,
        gamma=0.0, device=XGB_DEVICE
    )
    xgb_dry.fit(trainval_df.loc[mask_dry_tv, DRY_SELECTED_FEATURES], trainval_df.loc[mask_dry_tv, TARGET_COL], sample_weight=w_trainval[mask_dry_tv], verbose=0)
    pred_dry_test = np.asarray(xgb_dry.predict(test_df[DRY_SELECTED_FEATURES])).ravel()

    xgb_transition = XGBRegressor(
        objective="reg:absoluteerror", random_state=SEED, n_jobs=-1,
        max_depth=7, min_child_weight=5, subsample=0.9, colsample_bytree=0.85,
        n_estimators=8000, learning_rate=0.03, reg_lambda=3.0, reg_alpha=0.05,
        device=XGB_DEVICE
    )
    xgb_transition.fit(trainval_df.loc[mask_transition_tv, TRANSITION_SELECTED_FEATURES], trainval_df.loc[mask_transition_tv, TARGET_COL], sample_weight=w_trainval[mask_transition_tv], verbose=0)
    pred_transition_test = np.asarray(xgb_transition.predict(test_df[TRANSITION_SELECTED_FEATURES])).ravel()

    xgb_wet = XGBRegressor(
        objective="reg:squarederror", random_state=SEED, n_jobs=-1,
        max_depth=10, min_child_weight=1, subsample=1.0, colsample_bytree=0.9,
        n_estimators=6000, learning_rate=0.03, reg_lambda=0.3, reg_alpha=0.0,
        device=XGB_DEVICE
    )
    xgb_wet.fit(trainval_df.loc[mask_wet_tv, WET_SELECTED_FEATURES], trainval_df.loc[mask_wet_tv, TARGET_COL], sample_weight=w_trainval[mask_wet_tv], verbose=0)
    pred_wet_test = np.asarray(xgb_wet.predict(test_df[WET_SELECTED_FEATURES])).ravel()

    pred_combined = np.zeros_like(y_test, dtype=float)
    pred_combined[mask_dry_test] = pred_dry_test[mask_dry_test]
    pred_combined[mask_transition_test] = pred_transition_test[mask_transition_test]
    pred_combined[mask_wet_test] = pred_wet_test[mask_wet_test]

    # Model 4 & 5
    xgb_dry_overall = XGBRegressor(
        objective="reg:absoluteerror", random_state=SEED, n_jobs=-1,
        subsample=0.9, colsample_bytree=0.8, max_depth=8, min_child_weight=2,
        n_estimators=5500, learning_rate=0.04, reg_lambda=1.5, reg_alpha=0.03,
        gamma=0.0, device=XGB_DEVICE
    )
    xgb_dry_overall.fit(trainval_df.loc[mask_dry_tv, OVERALL_SELECTED_FEATURES], trainval_df.loc[mask_dry_tv, TARGET_COL], sample_weight=w_trainval[mask_dry_tv], verbose=0)
    pred_dry_overall_test = np.asarray(xgb_dry_overall.predict(test_df[OVERALL_SELECTED_FEATURES])).ravel()

    xgb_transition_overall = XGBRegressor(
        objective="reg:absoluteerror", random_state=SEED, n_jobs=-1,
        max_depth=7, min_child_weight=5, subsample=0.9, colsample_bytree=0.85,
        n_estimators=8000, learning_rate=0.03, reg_lambda=3.0, reg_alpha=0.05,
        device=XGB_DEVICE
    )
    xgb_transition_overall.fit(trainval_df.loc[mask_transition_tv, OVERALL_SELECTED_FEATURES], trainval_df.loc[mask_transition_tv, TARGET_COL], sample_weight=w_trainval[mask_transition_tv], verbose=0)
    pred_transition_overall_test = np.asarray(xgb_transition_overall.predict(test_df[OVERALL_SELECTED_FEATURES])).ravel()

    xgb_wet_overall = XGBRegressor(
        objective="reg:squarederror", random_state=SEED, n_jobs=-1,
        max_depth=10, min_child_weight=1, subsample=1.0, colsample_bytree=0.9,
        n_estimators=6000, learning_rate=0.03, reg_lambda=0.3, reg_alpha=0.0,
        device=XGB_DEVICE
    )
    xgb_wet_overall.fit(trainval_df.loc[mask_wet_tv, OVERALL_SELECTED_FEATURES], trainval_df.loc[mask_wet_tv, TARGET_COL], sample_weight=w_trainval[mask_wet_tv], verbose=0)
    pred_wet_overall_test = np.asarray(xgb_wet_overall.predict(test_df[OVERALL_SELECTED_FEATURES])).ravel()

    pred_combined_overall = np.zeros_like(y_test, dtype=float)
    pred_combined_overall[mask_dry_test] = pred_dry_overall_test[mask_dry_test]
    pred_combined_overall[mask_transition_test] = pred_transition_overall_test[mask_transition_test]
    pred_combined_overall[mask_wet_test] = pred_wet_overall_test[mask_wet_test]

    # Model 6
    xgb_dry_2r = XGBRegressor(
        objective="reg:absoluteerror", random_state=SEED, n_jobs=-1,
        subsample=0.9, colsample_bytree=0.8, max_depth=8, min_child_weight=2,
        n_estimators=5500, learning_rate=0.04, reg_lambda=1.5, reg_alpha=0.03,
        gamma=0.0, device=XGB_DEVICE
    )
    xgb_dry_2r.fit(trainval_df.loc[mask_dry_tv_2r, DRY_2REGIME_SELECTED_FEATURES], trainval_df.loc[mask_dry_tv_2r, TARGET_COL], sample_weight=w_trainval[mask_dry_tv_2r], verbose=0)
    pred_dry_test_2r = np.asarray(xgb_dry_2r.predict(test_df[DRY_2REGIME_SELECTED_FEATURES])).ravel()

    xgb_wet_2r = XGBRegressor(
        objective="reg:squarederror", random_state=SEED, n_jobs=-1,
        max_depth=10, min_child_weight=1, subsample=1.0, colsample_bytree=0.9,
        n_estimators=6000, learning_rate=0.03, reg_lambda=0.3, reg_alpha=0.0,
        device=XGB_DEVICE
    )
    xgb_wet_2r.fit(trainval_df.loc[mask_wet_tv_2r, WET_2REGIME_SELECTED_FEATURES], trainval_df.loc[mask_wet_tv_2r, TARGET_COL], sample_weight=w_trainval[mask_wet_tv_2r], verbose=0)
    pred_wet_test_2r = np.asarray(xgb_wet_2r.predict(test_df[WET_2REGIME_SELECTED_FEATURES])).ravel()

    pred_combined_2r = np.zeros_like(y_test, dtype=float)
    pred_combined_2r[mask_dry_test_2r] = pred_dry_test_2r[mask_dry_test_2r]
    pred_combined_2r[mask_wet_test_2r] = pred_wet_test_2r[mask_wet_test_2r]

    res = {
        "beta": beta,
        "global_r2": r2_score(y_test, pred_global),
        "model3_r2": r2_score(y_test, pred_combined),
        "model5_r2": r2_score(y_test, pred_combined_overall),
        "model6_r2": r2_score(y_test, pred_combined_2r),
        "dry_sp_r2": r2_score(y_test[mask_dry_test], pred_dry_overall_test[mask_dry_test]),
        "trans_sp_r2": r2_score(y_test[mask_transition_test], pred_transition_overall_test[mask_transition_test]),
        "wet_sp_r2": r2_score(y_test[mask_wet_test], pred_wet_overall_test[mask_wet_test]),
    }
    return res

if __name__ == "__main__":
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    for b in betas:
        print(f"Running experiment with beta = {b}...")
        start = time.time()
        res = run_experiment_with_beta(b)
        results.append(res)
        print(f"Beta = {b} completed in {time.time() - start:.2f} seconds.")

    print("\n--- BETA SWEEP RESULTS TABLE ---")
    print("| Beta | Global R² | Model 3 R² (Regime Specific) | Model 5 R² (Overall) | Model 6 R² (2-Regime) | Dry Specialist R² | Transition Specialist R² | Wet Specialist R² |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['beta']:.1f} | {r['global_r2']:.4f} | {r['model3_r2']:.4f} | {r['model5_r2']:.4f} | {r['model6_r2']:.4f} | {r['dry_sp_r2']:.4f} | {r['trans_sp_r2']:.4f} | {r['wet_sp_r2']:.4f} |")
