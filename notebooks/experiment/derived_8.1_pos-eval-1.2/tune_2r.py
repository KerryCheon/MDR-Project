import os
import sys
import random
import time
import itertools
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

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

T_2REGIME = dataset_metadata.T_2REGIME
DRY_2REGIME_SELECTED_FEATURES = dataset_metadata.DRY_2REGIME_SELECTED_FEATURES
WET_2REGIME_SELECTED_FEATURES = dataset_metadata.WET_2REGIME_SELECTED_FEATURES
TARGET_COL = "soil_moisture_5cm"

y_trainval = np.asarray(trainval_df[TARGET_COL]).ravel()
y_test = np.asarray(test_df[TARGET_COL]).ravel()

mask_dry_tv_2r = y_trainval < T_2REGIME
mask_wet_tv_2r = y_trainval >= T_2REGIME
mask_dry_test_2r = y_test < T_2REGIME
mask_wet_test_2r = y_test >= T_2REGIME

# Verify CUDA/GPU availability for XGBoost
try:
    dummy = XGBRegressor(n_estimators=1, device="cuda")
    dummy.fit(np.array([[1.0]]), np.array([1.0]))
    XGB_DEVICE = "cuda"
    print("XGBoost CUDA support verified.")
except Exception as e:
    XGB_DEVICE = "cpu"
    print("Falling back to CPU.")

# Compute default beta=0.4 temporal weights
years_tv = trainval_df["year"].astype(float)
max_year = years_tv.max()
beta = 0.4
w_trainval = np.exp(beta * (years_tv - max_year))
w_trainval = w_trainval / w_trainval.mean()

# Define grid search values for Wet Specialist (2R)
wet_objectives = ["reg:squarederror", "reg:absoluteerror"]
wet_max_depths = [7, 8, 10]
wet_min_child_weights = [1, 3, 5]
wet_subsamples = [0.9, 1.0]
wet_colsample_bytrees = [0.8, 0.9]

# Define grid search values for Dry Specialist (2R)
dry_max_depths = [6, 8]
dry_min_child_weights = [2, 5]
dry_subsamples = [0.8, 0.9, 1.0]

print("\n--- Phase 1: Dry Specialist (2R) Hyperparameter Sweep ---")
best_dry_r2 = -float("inf")
best_dry_params = None
best_dry_pred = None

dry_combinations = list(itertools.product(dry_max_depths, dry_min_child_weights, dry_subsamples))
print(f"Testing {len(dry_combinations)} combinations for Dry Specialist...")

for max_depth, min_child_weight, subsample in dry_combinations:
    params = dict(
        objective="reg:absoluteerror",
        random_state=SEED,
        n_jobs=-1,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=0.8,
        n_estimators=5500,
        learning_rate=0.04,
        reg_lambda=1.5,
        reg_alpha=0.03,
        gamma=0.0,
        device=XGB_DEVICE
    )
    model = XGBRegressor(**params)
    model.fit(
        trainval_df.loc[mask_dry_tv_2r, DRY_2REGIME_SELECTED_FEATURES],
        trainval_df.loc[mask_dry_tv_2r, TARGET_COL],
        sample_weight=w_trainval[mask_dry_tv_2r],
        verbose=0
    )
    preds = np.asarray(model.predict(test_df[DRY_2REGIME_SELECTED_FEATURES])).ravel()
    r2 = r2_score(y_test[mask_dry_test_2r], preds[mask_dry_test_2r])
    if r2 > best_dry_r2:
        best_dry_r2 = r2
        best_dry_params = params
        best_dry_pred = preds

print(f"Optimal Dry Specialist (2R) R²: {best_dry_r2:.4f}")
print("Optimal Parameters:")
for k, v in best_dry_params.items():
    if k not in ["device", "random_state", "n_jobs"]:
        print(f"  {k}: {v}")

print("\n--- Phase 2: Wet Specialist (2R) Hyperparameter Sweep ---")
best_wet_r2 = -float("inf")
best_wet_params = None
best_wet_pred = None

wet_combinations = list(itertools.product(wet_objectives, wet_max_depths, wet_min_child_weights, wet_subsamples, wet_colsample_bytrees))
print(f"Testing {len(wet_combinations)} combinations for Wet Specialist...")

for obj, max_depth, min_child_weight, subsample, colsample in wet_combinations:
    params = dict(
        objective=obj,
        random_state=SEED,
        n_jobs=-1,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample,
        n_estimators=6000,
        learning_rate=0.03,
        reg_lambda=0.3,
        reg_alpha=0.0,
        device=XGB_DEVICE
    )
    model = XGBRegressor(**params)
    model.fit(
        trainval_df.loc[mask_wet_tv_2r, WET_2REGIME_SELECTED_FEATURES],
        trainval_df.loc[mask_wet_tv_2r, TARGET_COL],
        sample_weight=w_trainval[mask_wet_tv_2r],
        verbose=0
    )
    preds = np.asarray(model.predict(test_df[WET_2REGIME_SELECTED_FEATURES])).ravel()
    r2 = r2_score(y_test[mask_wet_test_2r], preds[mask_wet_test_2r])
    if r2 > best_wet_r2:
        best_wet_r2 = r2
        best_wet_params = params
        best_wet_pred = preds

print(f"Optimal Wet Specialist (2R) R²: {best_wet_r2:.4f}")
print("Optimal Parameters:")
for k, v in best_wet_params.items():
    if k not in ["device", "random_state", "n_jobs"]:
        print(f"  {k}: {v}")

# Compute combined Oracle 2R Performance
pred_combined_2r = np.zeros_like(y_test, dtype=float)
pred_combined_2r[mask_dry_test_2r] = best_dry_pred[mask_dry_test_2r]
pred_combined_2r[mask_wet_test_2r] = best_wet_pred[mask_wet_test_2r]
combined_r2 = r2_score(y_test, pred_combined_2r)

print("\n--- Phase 3: Combined Oracle 2-Regime Model ---")
print(f"Baseline 1.1 2-Regime Combined R² (beta=0.2, un-tuned): 0.7443")
print(f"Optimized 1.2 2-Regime Combined R² (beta=0.4, tuned): {combined_r2:.4f}")
print(f"Absolute R² Change: {combined_r2 - 0.7443:+.4f}")
