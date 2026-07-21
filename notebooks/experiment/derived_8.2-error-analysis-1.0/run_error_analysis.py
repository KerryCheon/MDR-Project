import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def main():
    exp_dir = Path(__file__).resolve().parent
    notebooks_dir = exp_dir.parent.parent
    project_root = notebooks_dir.parent

    sys.path.insert(0, str(project_root))

    eval33_dir = notebooks_dir / "experiment" / "derived_8.2-eval-3.3"
    eval33_models_dir = eval33_dir / "models"
    split_dir = project_root / "data" / "splits" / "derived_8.2"

    print(f"Project root: {project_root}")
    print(f"Eval-3.3 dir: {eval33_dir}")

    # Load test split
    test_df = pd.read_csv(split_dir / "test.csv")
    y_test = test_df["soil_moisture_5cm"].values
    print(f"Loaded test set with {len(test_df)} samples. Target mean={y_test.mean():.4f}, std={y_test.std():.4f}, var={y_test.var():.6f}")

    # Load metrics from eval-3.3
    summary_df = pd.read_csv(eval33_dir / "metrics_summary.csv")
    per_regime_summary = pd.read_csv(eval33_dir / "per_regime_metrics_summary.csv")
    
    # -------------------------------------------------------------
    # STEP 1: Target Variance Decomposition per Regime
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: Target Variance Decomposition per Regime")
    print("="*80)

    # Load previous and selected features json to get regime definitions
    with open(eval33_dir / "previous_features.json", "r") as f:
        prev_features = json.load(f)
    with open(eval33_dir / "selected_features.json", "r") as f:
        sel_features = json.load(f)

    # Parse date to extract month
    test_df["date_parsed"] = pd.to_datetime(test_df["date"])
    test_df["month"] = test_df["date_parsed"].dt.month

    # Detailed per-model metric & target variance table
    detailed_metrics = []
    
    # Map model metadata to model files
    for idx, row in per_regime_summary.iterrows():
        mid = int(row["Model ID"])
        mname = row["Model Name"]
        strat = row["Strategy"]
        arm = row["Arm"]

        # Search for preds file in eval33_models_dir
        preds_files = list(eval33_models_dir.glob(f"model_{mid}_*_preds.npy"))
        labels_files = list(eval33_models_dir.glob(f"model_{mid}_*_labels_te.npy"))
        
        if not preds_files or not labels_files:
            print(f"Warning: files for Model {mid} not found.")
            continue
        
        preds = np.load(preds_files[0])
        labels_te = np.load(labels_files[0])

        mask0 = (labels_te == 0)
        mask1 = (labels_te == 1)

        y0 = y_test[mask0]
        p0 = preds[mask0]
        y1 = y_test[mask1]
        p1 = preds[mask1]

        var_y0 = np.var(y0) if len(y0) > 0 else np.nan
        var_y1 = np.var(y1) if len(y1) > 0 else np.nan

        mse0 = np.mean((y0 - p0)**2) if len(y0) > 0 else np.nan
        mse1 = np.mean((y1 - p1)**2) if len(y1) > 0 else np.nan

        rmse0 = np.sqrt(mse0) if len(y0) > 0 else np.nan
        rmse1 = np.sqrt(mse1) if len(y1) > 0 else np.nan

        r2_0 = 1.0 - (mse0 / var_y0) if var_y0 > 0 else np.nan
        r2_1 = 1.0 - (mse1 / var_y1) if var_y1 > 0 else np.nan

        nrmse_0 = rmse0 / np.std(y0) if np.std(y0) > 0 else np.nan
        nrmse_1 = rmse1 / np.std(y1) if np.std(y1) > 0 else np.nan

        print(f"Model {mid:2d} ({strat:22s} | {arm:9s}): R0 N={len(y0):4d}, Var(y)={var_y0:.6f}, MSE={mse0:.6f}, R2={r2_0:+.4f}, nRMSE={nrmse_0:.4f} | R1 N={len(y1):4d}, Var(y)={var_y1:.6f}, MSE={mse1:.6f}, R2={r2_1:+.4f}, nRMSE={nrmse_1:.4f}")

        detailed_metrics.append({
            "Model ID": mid,
            "Model Name": mname,
            "Strategy": strat,
            "Arm": arm,
            "N_R0": len(y0),
            "Var_y_R0": var_y0,
            "MSE_R0": mse0,
            "RMSE_R0": rmse0,
            "nRMSE_R0": nrmse_0,
            "R2_R0": r2_0,
            "N_R1": len(y1),
            "Var_y_R1": var_y1,
            "MSE_R1": mse1,
            "RMSE_R1": rmse1,
            "nRMSE_R1": nrmse_1,
            "R2_R1": r2_1,
        })

    det_df = pd.DataFrame(detailed_metrics)
    det_df.to_csv(exp_dir / "detailed_per_regime_metrics.csv", index=False)
    print("\nSaved detailed per-regime metrics to detailed_per_regime_metrics.csv")

    # -------------------------------------------------------------
    # STEP 2: Deep Dive into Model 7 & Model 8 Discrete Output Values
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: Deep Dive into Model 7 & Model 8 Discrete Output Values")
    print("="*80)

    g_api_mask1 = (test_df["G_API"] >= 0.16).values

    m7_preds_file = list(eval33_models_dir.glob("model_7_*_preds.npy"))[0]
    m8_preds_file = list(eval33_models_dir.glob("model_8_*_preds.npy"))[0]
    m10_preds_file = list(eval33_models_dir.glob("model_10_*_preds.npy"))[0]

    m7_preds = np.load(m7_preds_file)
    m8_preds = np.load(m8_preds_file)
    m10_preds = np.load(m10_preds_file)

    # Check regime 1 predictions for Model 7 and 8
    m7_r1_preds = m7_preds[g_api_mask1]
    m8_r1_preds = m8_preds[g_api_mask1]
    m10_r1_preds = m10_preds[g_api_mask1]

    u7_r1 = np.unique(np.round(m7_r1_preds, 6))
    u8_r1 = np.unique(np.round(m8_r1_preds, 6))
    u10_r1 = np.unique(np.round(m10_r1_preds, 6))

    print(f"Model 7 (Spec-old) Regime 1 unique prediction values count: {len(u7_r1)}")
    print(f"Model 7 Regime 1 Unique Values: {np.round(u7_r1, 5)}")
    print(f"Model 8 (Spec-new) Regime 1 unique prediction values count: {len(u8_r1)}")
    print(f"Model 8 Regime 1 Unique Values: {np.round(u8_r1, 5)}")
    print(f"Model 10 (Global-c1) Regime 1 unique prediction values count: {len(u10_r1)} (continuous!)")

    # Inspect feature count for cluster 1 in previous and selected features
    m7_features_c1 = prev_features["clusters"]["Univariate_G_API_k2"]["1"]["features"]
    m8_features_c1 = sel_features["clusters"]["Univariate_G_API_k2"]["1"]["features"]

    print(f"\nFeature Selection Check:")
    print(f"  Model 7 (Spec-old) Cluster 1 Features (count={len(m7_features_c1)}): {m7_features_c1}")
    print(f"  Model 8 (Spec-new) Cluster 1 Features (count={len(m8_features_c1)}): {m8_features_c1}")

    # Inspect relation between J_aspect_deg and Model 7 / Model 8 predictions
    aspect_vals = test_df.loc[g_api_mask1, "J_aspect_deg"].values
    
    discrete_df = pd.DataFrame({
        "J_aspect_deg": aspect_vals,
        "Model_7_Pred": m7_r1_preds,
        "Model_8_Pred": m8_r1_preds,
        "Model_10_Pred": m10_r1_preds,
        "Ground_Truth": y_test[g_api_mask1]
    })
    discrete_df.to_csv(exp_dir / "model7_8_discrete_predictions_analysis.csv", index=False)

    # Group by J_aspect_deg to check mapping
    grouped = discrete_df.groupby("J_aspect_deg").agg(
        Count=("Model_7_Pred", "count"),
        M7_Pred_Mean=("Model_7_Pred", "mean"),
        M7_Pred_Std=("Model_7_Pred", "std"),
        M8_Pred_Mean=("Model_8_Pred", "mean"),
        M8_Pred_Std=("Model_8_Pred", "std"),
        GroundTruth_Mean=("Ground_Truth", "mean"),
        GroundTruth_Std=("Ground_Truth", "std")
    ).reset_index()

    print("\nGrouped predictions by J_aspect_deg in Cluster 1:")
    print(grouped.to_string(index=False))

    # -------------------------------------------------------------
    # STEP 3: Residual vs Precipitation & Seasonal Month Analysis
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: Residual vs Precipitation & Month Analysis")
    print("="*80)

    # Compute month-by-month target statistics and model performance
    test_df["residual_M14"] = y_test - np.load(list(eval33_models_dir.glob("model_14_*_preds.npy"))[0])
    test_df["residual_M2"] = y_test - np.load(list(eval33_models_dir.glob("model_2_*_preds.npy"))[0])

    monthly_stats = []
    for m in range(1, 13):
        m_mask = (test_df["month"] == m).values
        y_m = y_test[m_mask]
        res14_m = test_df.loc[m_mask, "residual_M14"].values
        res2_m = test_df.loc[m_mask, "residual_M2"].values
        precip_m = test_df.loc[m_mask, "precip_mm"].values if "precip_mm" in test_df.columns else np.zeros_like(y_m)
        rain3d_m = test_df.loc[m_mask, "G_rain_sum_3d"].values if "G_rain_sum_3d" in test_df.columns else np.zeros_like(y_m)

        if len(y_m) > 0:
            monthly_stats.append({
                "Month": m,
                "N": len(y_m),
                "Target_Mean": np.mean(y_m),
                "Target_Std": np.std(y_m),
                "Target_Var": np.var(y_m),
                "Precip_Mean": np.mean(precip_m),
                "Rain3d_Mean": np.mean(rain3d_m),
                "RMSE_M14": np.sqrt(np.mean(res14_m**2)),
                "Bias_M14": np.mean(res14_m),
                "R2_M14": 1.0 - np.mean(res14_m**2)/np.var(y_m) if np.var(y_m)>0 else np.nan,
                "RMSE_M2": np.sqrt(np.mean(res2_m**2)),
                "R2_M2": 1.0 - np.mean(res2_m**2)/np.var(y_m) if np.var(y_m)>0 else np.nan,
            })

    month_df = pd.DataFrame(monthly_stats)
    month_df.to_csv(exp_dir / "monthly_target_and_error_breakdown.csv", index=False)
    print("\nMonthly Target & Error Breakdown:")
    print(month_df[["Month", "N", "Target_Mean", "Target_Var", "Precip_Mean", "RMSE_M14", "R2_M14"]].round(4).to_string(index=False))

    # -------------------------------------------------------------
    # STEP 4: Visual Figure Generation
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: Generating Diagnostic Figures")
    print("="*80)

    # Figure 1: Target Variance & R2 Comparison across Regimes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1A: Target Variance vs Residual Variance by Regime for Model 14 & Model 10
    ax = axes[0, 0]
    m14_row = det_df[det_df["Model ID"] == 14].iloc[0]
    m10_row = det_df[det_df["Model ID"] == 10].iloc[0]

    categories = ["M14 (Cluster K=2)\nRegime 0", "M14 (Cluster K=2)\nRegime 1", "M10 (G_API K=2)\nRegime 0", "M10 (G_API K=2)\nRegime 1"]
    var_y_vals = [m14_row["Var_y_R0"], m14_row["Var_y_R1"], m10_row["Var_y_R0"], m10_row["Var_y_R1"]]
    mse_vals = [m14_row["MSE_R0"], m14_row["MSE_R1"], m10_row["MSE_R0"], m10_row["MSE_R1"]]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, var_y_vals, width, label="Target Variance Var(y)", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, mse_vals, width, label="Residual Variance (MSE)", color="#d62728", alpha=0.85)
    ax.set_ylabel("Variance ($m^6/m^6$)", fontweight="bold")
    ax.set_title("A. Target Variance Var(y) vs Model MSE by Regime", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 1B: R2 vs nRMSE Comparison by Regime
    ax = axes[0, 1]
    r2_vals = [m14_row["R2_R0"], m14_row["R2_R1"], m10_row["R2_R0"], m10_row["R2_R1"]]
    nrmse_vals = [m14_row["nRMSE_R0"], m14_row["nRMSE_R1"], m10_row["nRMSE_R0"], m10_row["nRMSE_R1"]]

    ax.bar(x - width/2, r2_vals, width, label="$R^2$ Score", color="#2ca02c", alpha=0.85)
    ax.bar(x + width/2, nrmse_vals, width, label="nRMSE (RMSE / Std(y))", color="#ff7f0e", alpha=0.85)
    ax.set_ylabel("Score / Ratio", fontweight="bold")
    ax.set_title("B. $R^2$ Score vs Normalized RMSE (nRMSE)", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 1C: All MoE Models R2 in Regime 0 vs Regime 1
    ax = axes[1, 0]
    moe_df = det_df.sort_values("Model ID")
    x_m = np.arange(len(moe_df))
    ax.bar(x_m - width/2, moe_df["R2_R0"], width, label="Regime 0 (Dry)", color="#1f77b4", alpha=0.85)
    ax.bar(x_m + width/2, moe_df["R2_R1"], width, label="Regime 1 (Wet)", color="#ff7f0e", alpha=0.85)
    ax.set_ylabel("$R^2$ Score", fontweight="bold")
    ax.set_title("C. $R^2$ Score Breakdown Across All 2-Regime Models", fontweight="bold", fontsize=11)
    ax.set_xticks(x_m)
    ax.set_xticklabels([f"M{mid}" for mid in moe_df["Model ID"]], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", linestyle="--", alpha=0.7)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 1D: All MoE Models nRMSE in Regime 0 vs Regime 1
    ax = axes[1, 1]
    ax.bar(x_m - width/2, moe_df["nRMSE_R0"], width, label="Regime 0 (Dry)", color="#1f77b4", alpha=0.85)
    ax.bar(x_m + width/2, moe_df["nRMSE_R1"], width, label="Regime 1 (Wet)", color="#ff7f0e", alpha=0.85)
    ax.set_ylabel("nRMSE (Lower is Better)", fontweight="bold")
    ax.set_title("D. nRMSE (Normalized RMSE) Across All 2-Regime Models", fontweight="bold", fontsize=11)
    ax.set_xticks(x_m)
    ax.set_xticklabels([f"M{mid}" for mid in moe_df["Model ID"]], rotation=45, ha="right", fontsize=8)
    ax.axhline(1.0, color="red", linestyle=":", label="nRMSE = 1.0 (Baseline Mean)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Target Variance & Per-Regime Performance Gap Analysis (derived_8.2-eval-3.3)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(exp_dir / "target_variance_and_r2_by_regime.png", dpi=150)
    plt.close()
    print("Saved target_variance_and_r2_by_regime.png")

    # Figure 2: Model 7 & Model 8 Discrete Step Output Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2A: Model 7 & Model 8 Predictions vs J_aspect_deg
    ax = axes[0]
    ax.scatter(discrete_df["J_aspect_deg"], discrete_df["Model_7_Pred"], color="#d62728", alpha=0.6, label="Model 7 (Spec-old: 1 feature)", s=25)
    ax.scatter(discrete_df["J_aspect_deg"], discrete_df["Model_10_Pred"], color="#2ca02c", alpha=0.3, label="Model 10 (Global-c1: 50 features)", s=10)
    ax.set_xlabel("J_aspect_deg (Discrete Aspect Angle Feature)", fontweight="bold")
    ax.set_ylabel("Model Prediction (Soil Moisture)", fontweight="bold")
    ax.set_title("A. Model 7 Discrete Step Predictions vs Model 10 Continuous Predictions", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2B: Histogram of Prediction Distributions in Regime 1
    ax = axes[1]
    ax.hist(m7_r1_preds, bins=50, color="#d62728", alpha=0.7, label="Model 7 (Spec-old: 10 discrete spikes)", density=True)
    ax.hist(m10_r1_preds, bins=50, color="#2ca02c", alpha=0.5, label="Model 10 (Global-c1: Continuous)", density=True)
    ax.set_xlabel("Predicted Soil Moisture ($m^3/m^3$)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("B. Prediction Distribution Density in High G_API Regime (Regime 1)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Model 7 & Model 8 Discrete Output Anomaly Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(exp_dir / "model7_8_discrete_step_analysis.png", dpi=150)
    plt.close()
    print("Saved model7_8_discrete_step_analysis.png")

    # Figure 3: Monthly Target Variance, Precipitation, and Residual Analysis
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    months_str = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # 3A: Monthly Target Mean, Std, and Precipitation
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    p1 = ax1.plot(month_df["Month"], month_df["Target_Mean"], "b-o", label="Target Mean", linewidth=2)
    p2 = ax1.fill_between(month_df["Month"], month_df["Target_Mean"] - month_df["Target_Std"], month_df["Target_Mean"] + month_df["Target_Std"], color="b", alpha=0.15, label="Target ± 1 Std")
    p3 = ax1_twin.bar(month_df["Month"], month_df["Precip_Mean"], color="#1f77b4", alpha=0.3, width=0.4, label="Mean Precip (mm)")

    ax1.set_ylabel("Soil Moisture ($m^3/m^3$)", fontweight="bold", color="b")
    ax1_twin.set_ylabel("Precipitation (mm)", fontweight="bold", color="#1f77b4")
    ax1.set_title("A. Seasonal Cycle of Soil Moisture vs Precipitation in WA Test Stations", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # 3B: Monthly R2 Performance (Model 14 vs Model 2)
    ax2 = axes[1]
    ax2.plot(month_df["Month"], month_df["R2_M14"], "g-s", label="Model 14 (Cluster K=2 Global-c1)", linewidth=2)
    ax2.plot(month_df["Month"], month_df["R2_M2"], "r--o", label="Model 2 (Baseline c1)", linewidth=1.8)
    ax2.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Month", fontweight="bold")
    ax2.set_ylabel("$R^2$ Score", fontweight="bold")
    ax2.set_title("B. Monthly $R^2$ Breakdown: Cluster MoE (Model 14) vs Global Baseline (Model 2)", fontweight="bold")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(months_str)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Seasonal Hydrological Error Drivers & Performance Collapse in Wet Months", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(exp_dir / "monthly_residual_and_variance.png", dpi=150)
    plt.close()
    print("Saved monthly_residual_and_variance.png")

    print("\nAll error analysis computations and diagnostic figures completed successfully!")

if __name__ == "__main__":
    main()
