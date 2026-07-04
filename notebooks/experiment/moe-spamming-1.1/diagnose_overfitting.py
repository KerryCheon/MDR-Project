import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import importlib.util

# Resolve project root relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
print(f"Project root resolved to: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/train.csv")
VAL_PATH = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/val.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/test.csv")

# Load data
print("Loading dataset splits...")
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)
trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
print(f"Loaded trainval: {trainval_df.shape}, test: {test_df.shape}")

TARGET_COL = "soil_moisture_5cm"
y_trainval = np.asarray(trainval_df[TARGET_COL], dtype=float).ravel()
y_test = np.asarray(test_df[TARGET_COL], dtype=float).ravel()

# Load features
metadata_path = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/dataset_metadata.py")
spec = importlib.util.spec_from_file_location("dataset_metadata", metadata_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load metadata from {metadata_path}")
dataset_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_metadata)
OVERALL_SELECTED_FEATURES = dataset_metadata.OVERALL_SELECTED_FEATURES

# Weights
years_tv = trainval_df["year"].astype(float)
max_year = years_tv.max()
beta = 0.4
w_trainval = np.exp(beta * (years_tv - max_year))
w_trainval = w_trainval / w_trainval.mean()

# Verify CUDA/GPU availability for XGBoost
try:
    dummy = xgb.XGBRegressor(n_estimators=1, device="cuda")
    dummy.fit(np.array([[1.0]]), np.array([1.0]))
    XGB_DEVICE = "cuda"
    print("XGBoost CUDA support enabled.")
except Exception as e:
    XGB_DEVICE = "cpu"
    print(f"XGBoost CUDA dummy test failed ({e}). Using CPU.")

XGB_PARAMS = dict(
    objective="reg:absoluteerror",
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
    device=XGB_DEVICE
)

results = []

for k in range(1, 5):
    print(f"\n==================== ANALYZING K = {k} ====================")
    
    if k == 1:
        percentiles = [0.0, 100.0]
        thresholds = [0.0, 1.0]
        
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(
            trainval_df[OVERALL_SELECTED_FEATURES],
            y_trainval,
            sample_weight=w_trainval,
            verbose=0
        )
        
        pred_train = model.predict(trainval_df[OVERALL_SELECTED_FEATURES])
        pred_test = model.predict(test_df[OVERALL_SELECTED_FEATURES])
        
        results.append({
            "k": k,
            "bin": 1,
            "train_target_std": np.std(y_trainval),
            "train_pred_std": np.std(pred_train),
            "test_target_std": np.std(y_test),
            "test_pred_std": np.std(pred_test),
            "train_mae": np.mean(np.abs(y_trainval - pred_train)),
            "test_mae": np.mean(np.abs(y_test - pred_test))
        })
        
        print("Global Model (k=1):")
        print(f"  Train Target Std: {np.std(y_trainval):.4f} | Train Pred Std: {np.std(pred_train):.4f} | Train MAE: {np.mean(np.abs(y_trainval - pred_train)):.4f}")
        print(f"  Test Target Std:  {np.std(y_test):.4f} | Test Pred Std:  {np.std(pred_test):.4f}  | Test MAE:  {np.mean(np.abs(y_test - pred_test)):.4f}")
    else:
        percentiles = np.linspace(0, 100, k + 1)
        thresholds = np.percentile(y_trainval, percentiles)
        
        for i in range(k):
            if i == k - 1:
                mask_tv = (y_trainval >= thresholds[i]) & (y_trainval <= thresholds[i+1])
                mask_test = (y_test >= thresholds[i]) & (y_test <= thresholds[i+1])
            else:
                mask_tv = (y_trainval >= thresholds[i]) & (y_trainval < thresholds[i+1])
                mask_test = (y_test >= thresholds[i]) & (y_test < thresholds[i+1])
                
            if mask_tv.sum() == 0 or mask_test.sum() == 0:
                continue
                
            expert = XGBRegressor(**XGB_PARAMS)
            expert.fit(
                trainval_df.loc[mask_tv, OVERALL_SELECTED_FEATURES],
                y_trainval[mask_tv],
                sample_weight=w_trainval[mask_tv],
                verbose=0
            )
            
            pred_train = expert.predict(trainval_df.loc[mask_tv, OVERALL_SELECTED_FEATURES])
            pred_test = expert.predict(test_df.loc[mask_test, OVERALL_SELECTED_FEATURES])
            
            train_t_std = np.std(y_trainval[mask_tv])
            train_p_std = np.std(pred_train)
            test_t_std = np.std(y_test[mask_test])
            test_p_std = np.std(pred_test)
            train_m = np.mean(np.abs(y_trainval[mask_tv] - pred_train))
            test_m = np.mean(np.abs(y_test[mask_test] - pred_test))
            
            results.append({
                "k": k,
                "bin": i + 1,
                "train_target_std": train_t_std,
                "train_pred_std": train_p_std,
                "test_target_std": test_t_std,
                "test_pred_std": test_p_std,
                "train_mae": train_m,
                "test_mae": test_m
            })
            
            print(f"Expert {i+1}/{k} (Range: [{thresholds[i]:.3f}, {thresholds[i+1]:.3f}]):")
            print(f"  Train Target Std: {train_t_std:.4f} | Train Pred Std: {train_p_std:.4f} | Train MAE: {train_m:.4f}")
            print(f"  Test Target Std:  {test_t_std:.4f} | Test Pred Std:  {test_p_std:.4f}  | Test MAE:  {test_m:.4f}")

print("\n--- Summary CSV-like output ---")
df = pd.DataFrame(results)
print(df.to_string(index=False))
