import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from scipy.stats import binned_statistic, gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score

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

# Helper function to compute local R2
def get_local_r2(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - (num / denom)

# Helper to plot distributions of True vs Pred within bins
def save_bin_distribution_plot(y_true, y_pred, bin_indices, thresholds, k, filename):
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4.5), squeeze=False)
    fig.suptitle(f"Output Value Frequency Analysis (k={k} Experts)", fontsize=14, fontweight='bold')
    
    colors_true = '#1d3557'
    colors_pred = '#e63946'
    
    for i in range(k):
        ax = axes[0, i]
        mask = bin_indices == i
        yt = y_true[mask]
        yp = y_pred[mask]
        
        low, high = thresholds[i], thresholds[i+1]
        ax.set_title(f"Expert {i+1} (Range: [{low:.3f}, {high:.3f}])\nSamples: {len(yt)}")
        
        if len(yt) < 2:
            ax.text(0.5, 0.5, "Insufficient samples", ha='center', va='center')
            continue
            
        # Draw histograms
        bins = np.linspace(low, high, 20)
        ax.hist(yt, bins=bins, alpha=0.5, color=colors_true, label='True Target', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(yp, bins=bins, alpha=0.6, color=colors_pred, label='Prediction', density=True, edgecolor='black', linewidth=0.5)
        
        # Overlay KDE curves if possible
        try:
            # True KDE
            if np.std(yt) > 1e-5:
                kde_t = gaussian_kde(yt)
                xs = np.linspace(low, high, 200)
                ax.plot(xs, kde_t(xs), color=colors_true, linewidth=2, linestyle='-')
            # Pred KDE
            if np.std(yp) > 1e-5:
                kde_p = gaussian_kde(yp)
                xs = np.linspace(low, high, 200)
                ax.plot(xs, kde_p(xs), color=colors_pred, linewidth=2, linestyle='-')
        except Exception as e:
            pass
            
        ax.set_xlabel("Soil Moisture")
        if i == 0:
            ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize='small')
        
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {plot_path}")

# Helper to plot True vs Pred scatter plot colored by bin
def save_scatter_by_bin_plot(y_true, y_pred, bin_indices, thresholds, k, filename):
    plt.figure(figsize=(7, 6))
    
    # Generate distinct colors for each bin
    cmap = plt.get_cmap('tab10')
    
    for i in range(k):
        mask = bin_indices == i
        yt = y_true[mask]
        yp = y_pred[mask]
        low, high = thresholds[i], thresholds[i+1]
        
        plt.scatter(yt, yp, s=10, alpha=0.6, color=cmap(i), label=f"Expert {i+1} ([{low:.3f}, {high:.3f}])")
        
    # Draw reference line y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1.5, label="y = x")
    
    # Draw bin boundaries
    for t in thresholds[1:-1]:
        plt.axvline(t, color='grey', linestyle=':', alpha=0.5)
        plt.axhline(t, color='grey', linestyle=':', alpha=0.5)
        
    plt.xlabel("True Soil Moisture")
    plt.ylabel("Predicted Soil Moisture")
    plt.title(f"True vs Predicted | k={k} Experts (Oracle Gating)", fontsize=12, fontweight='bold')
    plt.legend(loc="upper left", fontsize='small')
    plt.grid(alpha=0.2)
    plt.xlim(0.0, 0.45)
    plt.ylim(0.0, 0.45)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot to {plot_path}")

# List to accumulate granular statistics
expert_stats = []

# Range of experts k = 1 to 4
for k in range(1, 5):
    print(f"\n==================== TRAINING K = {k} EXPERTS ====================")
    start_time = time.time()
    
    pred_combined = np.zeros_like(y_test, dtype=float)
    bin_indices = np.zeros_like(y_test, dtype=int)
    
    if k == 1:
        # Standard Global Model
        percentiles = [0.0, 100.0]
        thresholds = [0.0, 1.0] # standard bounds
        
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(
            trainval_df[OVERALL_SELECTED_FEATURES],
            y_trainval,
            sample_weight=w_trainval,
            verbose=0
        )
        pred_combined = np.asarray(model.predict(test_df[OVERALL_SELECTED_FEATURES])).ravel()
        bin_indices[:] = 0
        
        # Calculate stats for the single bin
        yt = y_test
        yp = pred_combined
        
        # Dummy prediction (mean of trainval)
        y_dummy = np.full_like(yt, np.mean(y_trainval))
        
        # Within-bin metrics
        train_mean = np.mean(y_trainval)
        train_std = np.std(y_trainval)
        test_mean = np.mean(yt)
        test_std = np.std(yt)
        pred_mean = np.mean(yp)
        pred_std = np.std(yp)
        pred_min = np.min(yp)
        pred_max = np.max(yp)
        
        pred_spread_ratio = pred_std / test_std if test_std > 0 else 0.0
        pearson_r = _safe_corr(yt, yp)
        mae_xgb = mean_absolute_error(yt, yp)
        mae_dummy = mean_absolute_error(yt, y_dummy)
        mae_imp = (mae_dummy - mae_xgb) / mae_dummy if mae_dummy > 0 else 0.0
        
        r2_xgb = get_local_r2(yt, yp)
        r2_dummy = get_local_r2(yt, y_dummy)
        
        expert_stats.append({
            "k": k,
            "bin_index": 1,
            "threshold_low": 0.0,
            "threshold_high": 1.0,
            "num_train": len(y_trainval),
            "num_test": len(yt),
            "train_mean": train_mean,
            "train_std": train_std,
            "test_target_mean": test_mean,
            "test_target_std": test_std,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "pred_spread_ratio": pred_spread_ratio,
            "pearson_r": pearson_r,
            "mae_xgb": mae_xgb,
            "mae_dummy": mae_dummy,
            "mae_improvement_pct": mae_imp * 100,
            "r2_xgb_local": r2_xgb,
            "r2_dummy_local": r2_dummy
        })
        
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
                
            bin_indices[mask_test] = i
            
            if mask_test.sum() > 0:
                pred_subset = np.asarray(expert.predict(test_df.loc[mask_test, OVERALL_SELECTED_FEATURES])).ravel()
                pred_combined[mask_test] = pred_subset
                
                # Calculate stats for this specific bin
                yt_bin = y_test[mask_test]
                yp_bin = pred_subset
                y_dummy_bin = np.full_like(yt_bin, np.mean(y_trainval[mask_tv]))
                
                train_mean = np.mean(y_trainval[mask_tv])
                train_std = np.std(y_trainval[mask_tv])
                test_mean = np.mean(yt_bin)
                test_std = np.std(yt_bin)
                pred_mean = np.mean(yp_bin)
                pred_std = np.std(yp_bin)
                pred_min = np.min(yp_bin)
                pred_max = np.max(yp_bin)
                
                pred_spread_ratio = pred_std / test_std if test_std > 0 else 0.0
                pearson_r = _safe_corr(yt_bin, yp_bin)
                mae_xgb = mean_absolute_error(yt_bin, yp_bin)
                mae_dummy = mean_absolute_error(yt_bin, y_dummy_bin)
                mae_imp = (mae_dummy - mae_xgb) / mae_dummy if mae_dummy > 0 else 0.0
                
                r2_xgb = get_local_r2(yt_bin, yp_bin)
                r2_dummy = get_local_r2(yt_bin, y_dummy_bin)
                
                expert_stats.append({
                    "k": k,
                    "bin_index": i + 1,
                    "threshold_low": thresholds[i],
                    "threshold_high": thresholds[i+1],
                    "num_train": mask_tv.sum(),
                    "num_test": mask_test.sum(),
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "test_target_mean": test_mean,
                    "test_target_std": test_std,
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "pred_min": pred_min,
                    "pred_max": pred_max,
                    "pred_spread_ratio": pred_spread_ratio,
                    "pearson_r": pearson_r,
                    "mae_xgb": mae_xgb,
                    "mae_dummy": mae_dummy,
                    "mae_improvement_pct": mae_imp * 100,
                    "r2_xgb_local": r2_xgb,
                    "r2_dummy_local": r2_dummy
                })
                
    elapsed = time.time() - start_time
    print(f"Completed k={k} in {elapsed:.2f} seconds.")
    
    # Save the distribution plots
    save_bin_distribution_plot(
        y_test,
        pred_combined,
        bin_indices,
        thresholds if k > 1 else [0.0, 1.0],
        k,
        f"distribution_k{k}.png"
    )
    
    # Save the scatter plot colored by bin
    save_scatter_by_bin_plot(
        y_test,
        pred_combined,
        bin_indices,
        thresholds if k > 1 else [0.0, 1.0],
        k,
        f"scatter_k{k}.png"
    )

# Write out to a CSV for ease of use
df_stats = pd.DataFrame(expert_stats)
csv_path = os.path.join(SCRIPT_DIR, "moe_expert_stats.csv")
df_stats.to_csv(csv_path, index=False)
print(f"\nSaved metrics CSV to {csv_path}")

# Print summary table formatted nicely
print("\n==================== GRANULAR EXPERT METRICS ====================")
headers = ["k", "bin_index", "num_test", "train_mean", "test_target_std", "pred_std", "pred_spread_ratio", "pearson_r", "mae_improvement_pct", "r2_xgb_local"]
print("| " + " | ".join(headers) + " |")
print("| " + " | ".join(["---" for _ in headers]) + " |")
for _, row in df_stats.iterrows():
    row_str = []
    for h in headers:
        val = row[h]
        if isinstance(val, (int, np.integer)):
            row_str.append(f"{val:,}")
        elif isinstance(val, (float, np.floating)):
            row_str.append(f"{val:.4f}")
        else:
            row_str.append(str(val))
    print("| " + " | ".join(row_str) + " |")
