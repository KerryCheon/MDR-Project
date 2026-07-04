import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from scipy.stats import binned_statistic

# Resolve project root relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
print(f"Project root resolved to: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import metrics dashboard from project utilities
try:
    from u_dashboard.dashboard import _ubrmse, _safe_corr
    print("Imported _ubrmse and _safe_corr from u_dashboard.dashboard")
except ImportError:
    def _ubrmse(y_true, y_pred):
        yt = y_true - np.mean(y_true)
        yp = y_pred - np.mean(y_pred)
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
        
    def _safe_corr(y_true, y_pred):
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return float("nan")
        return float(np.corrcoef(y_true, y_pred)[0, 1])

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

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

# Load metadata features
import importlib.util
metadata_path = os.path.join(PROJECT_ROOT, "data/splits/derived_8.1_pos/dataset_metadata.py")
spec = importlib.util.spec_from_file_location("dataset_metadata", metadata_path)
dataset_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_metadata)

OVERALL_SELECTED_FEATURES = dataset_metadata.OVERALL_SELECTED_FEATURES
TARGET_COL = "soil_moisture_5cm"

y_trainval = np.asarray(trainval_df[TARGET_COL]).ravel()
y_test = np.asarray(test_df[TARGET_COL]).ravel()

# Temporal recency weighting (Drift, beta=0.4)
years_tv = trainval_df["year"].astype(float)
max_year = years_tv.max()
beta = 0.4
w_trainval = np.exp(beta * (years_tv - max_year))
w_trainval = w_trainval / w_trainval.mean()
print(f"Temporal Drift Weighting initialized (beta={beta})")

# Verify CUDA/GPU availability for XGBoost
try:
    dummy = xgb.XGBRegressor(n_estimators=1, device="cuda")
    dummy.fit(np.array([[1.0]]), np.array([1.0]))
    XGB_DEVICE = "cuda"
    print("XGBoost CUDA support enabled.")
except Exception as e:
    XGB_DEVICE = "cpu"
    print(f"XGBoost CUDA dummy test failed ({e}). Using CPU.")

# XGBoost parameter dictionary
XGB_PARAMS = dict(
    objective="reg:absoluteerror",
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
    device=XGB_DEVICE
)

# Output directory for plots
plots_dir = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Helper function to plot residuals with running mean
def save_residual_plot(y_true, y_pred, title, filename):
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 5.5))
    # Scatter plot of residuals
    plt.scatter(y_true, residuals, s=8, alpha=0.4, color='#7209b7', label="Test Samples")
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Compute binned mean of residuals
    bin_means, bin_edges, _ = binned_statistic(y_true, residuals, statistic='mean', bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot binned mean line
    plt.plot(bin_centers, bin_means, color='#f72585', linewidth=2.5, marker='o', label="Binned Mean Residual")
    
    plt.xlabel("True Soil Moisture")
    plt.ylabel("Residual (True - Predicted)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")
    plt.xlim(0.0, 0.45)
    plt.ylim(-0.2, 0.2)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved residual plot to {plot_path}")

# Evaluation dictionary helper
from sklearn.metrics import mean_absolute_error, r2_score
def evaluate_predictions(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ubrmse = _ubrmse(y_true, y_pred)
    bias = np.mean(y_true - y_pred)
    pearson = _safe_corr(y_true, y_pred)
    med_ae = np.median(np.abs(y_true - y_pred))
    return {
        "r2": r2,
        "rmse": rmse,
        "ubrmse": ubrmse,
        "bias": bias,
        "mae": mae,
        "med_ae": med_ae,
        "pearson": pearson
    }

results = []

# Range of experts: 1 to 10
for k in range(1, 11):
    print(f"\n==================== TRAINING K = {k} EXPERTS ====================")
    start_time = time.time()
    
    pred_combined = np.zeros_like(y_test, dtype=float)
    
    if k == 1:
        # Standard Global Model
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(
            trainval_df[OVERALL_SELECTED_FEATURES],
            y_trainval,
            sample_weight=w_trainval,
            verbose=0
        )
        pred_combined = np.asarray(model.predict(test_df[OVERALL_SELECTED_FEATURES])).ravel()
    else:
        # Quantile-based thresholds
        percentiles = np.linspace(0, 100, k + 1)
        thresholds = np.percentile(y_trainval, percentiles)
        print(f"Bin thresholds for k={k}: {['.3f' for _ in thresholds]}")
        print("Threshold values: " + ", ".join([f"{t:.4f}" for t in thresholds]))
        
        experts = []
        for i in range(k):
            # Subset trainval data
            if i == k - 1:
                mask_tv = (y_trainval >= thresholds[i]) & (y_trainval <= thresholds[i+1])
            else:
                mask_tv = (y_trainval >= thresholds[i]) & (y_trainval < thresholds[i+1])
                
            print(f"  Expert {i+1}/{k} training sample count: {mask_tv.sum():,}")
            
            expert = XGBRegressor(**XGB_PARAMS)
            expert.fit(
                trainval_df.loc[mask_tv, OVERALL_SELECTED_FEATURES],
                y_trainval[mask_tv],
                sample_weight=w_trainval[mask_tv],
                verbose=0
            )
            experts.append(expert)
            
            # Predict on corresponding test subset (oracle gating)
            if i == k - 1:
                mask_test = (y_test >= thresholds[i]) & (y_test <= thresholds[i+1])
            else:
                mask_test = (y_test >= thresholds[i]) & (y_test < thresholds[i+1])
                
            if mask_test.sum() > 0:
                pred_subset = np.asarray(expert.predict(test_df.loc[mask_test, OVERALL_SELECTED_FEATURES])).ravel()
                pred_combined[mask_test] = pred_subset
                
    elapsed = time.time() - start_time
    print(f"Completed k={k} in {elapsed:.2f} seconds.")
    
    # Evaluate
    metrics = evaluate_predictions(y_test, pred_combined)
    metrics["k"] = f"{k} Experts (Quantile)"
    results.append(metrics)
    
    # Plot residuals
    save_residual_plot(
        y_test,
        pred_combined,
        title=f"Residuals vs True SM | Oracle Hard Gating ({k} Experts - Quantile Bins)",
        filename=f"residuals_k{k}.png"
    )

# --- Calibrated 3-Regime Baseline ---
print("\n==================== TRAINING CALIBRATED 3-REGIME BASELINE ====================")
start_time = time.time()

cal_thresholds = [0.0, 0.159, 0.248, 1.0] # Bounds for Dry, Transition, Wet
pred_cal = np.zeros_like(y_test, dtype=float)
cal_names = ["Dry", "Transition", "Wet"]

for i in range(3):
    mask_tv = (y_trainval >= cal_thresholds[i]) & (y_trainval < cal_thresholds[i+1])
    print(f"  Calibrated Expert {cal_names[i]} sample count: {mask_tv.sum():,}")
    
    expert = XGBRegressor(**XGB_PARAMS)
    expert.fit(
        trainval_df.loc[mask_tv, OVERALL_SELECTED_FEATURES],
        y_trainval[mask_tv],
        sample_weight=w_trainval[mask_tv],
        verbose=0
    )
    
    mask_test = (y_test >= cal_thresholds[i]) & (y_test < cal_thresholds[i+1])
    if mask_test.sum() > 0:
        pred_subset = np.asarray(expert.predict(test_df.loc[mask_test, OVERALL_SELECTED_FEATURES])).ravel()
        pred_cal[mask_test] = pred_subset

elapsed = time.time() - start_time
print(f"Completed calibrated 3-regime baseline in {elapsed:.2f} seconds.")

metrics_cal = evaluate_predictions(y_test, pred_cal)
metrics_cal["k"] = "3 Experts (Calibrated T1/T2)"
results.append(metrics_cal)

save_residual_plot(
    y_test,
    pred_cal,
    title="Residuals vs True SM | Oracle Hard Gating (3 Experts - Calibrated T1/T2)",
    filename="residuals_k3_calibrated.png"
)

# Print Summary Table
print("\n==================== METRICS SUMMARY ====================")
df_res = pd.DataFrame(results)
cols = ["k", "r2", "rmse", "ubrmse", "bias", "mae", "med_ae", "pearson"]
df_res = df_res[cols]

# Dependency-free markdown table formatting
headers = list(df_res.columns)
lines = []
lines.append("| " + " | ".join(headers) + " |")
lines.append("| " + " | ".join(["---" for _ in headers]) + " |")
for _, row in df_res.iterrows():
    row_str = []
    for h in headers:
        val = row[h]
        if isinstance(val, float):
            row_str.append(f"{val:.5f}")
        else:
            row_str.append(str(val))
    lines.append("| " + " | ".join(row_str) + " |")
table_str = "\n".join(lines)
print(table_str)

# Write out to a CSV for ease of use
csv_path = os.path.join(SCRIPT_DIR, "moe_results.csv")
df_res.to_csv(csv_path, index=False)
print(f"\nSaved metrics CSV to {csv_path}")
