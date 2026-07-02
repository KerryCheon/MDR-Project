import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Set premium style for matplotlib
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 15,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def main():
    # Resolve paths (runnable from project root or notebooks/ directory)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    split_dir = os.path.join(project_root, "data", "splits", "derived_8.1_pos")
    output_dir = os.path.join(project_root, "notebooks", "experiment", "derived_8.1_pos-data-exploration")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"Data split dir: {split_dir}")
    print(f"Output dir: {output_dir}")
    
    # 1. Load data
    print("\n--- Loading derived_8.1_pos splits ---")
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")
    
    cols_to_load = [
        "station_id", "date", "soil_moisture_5cm", 
        "G_API", "LST_modis", "SMAP_sm_pm_interp", "precip_mm"
    ]
    
    train_df = pd.read_csv(train_path, usecols=cols_to_load)
    val_df = pd.read_csv(val_path, usecols=cols_to_load)
    test_df = pd.read_csv(test_path, usecols=cols_to_load)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df["month"] = pd.DatetimeIndex(all_df["date"]).month.astype(int)
    
    print(f"Loaded {len(all_df)} total observations (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    
    # 2. Define regimes based on thresholds
    # Recalibrated valley thresholds for derived_8.1_pos
    t1_cal, t2_cal = 0.159, 0.248
    # Original thresholds (for comparison)
    t1_orig, t2_orig = 0.20, 0.313
    # 2-Regime threshold
    t_2regime = 0.159
    
    def assign_regime(df, t1, t2):
        sm = df["soil_moisture_5cm"]
        conds = [sm < t1, (sm >= t1) & (sm < t2), sm >= t2]
        return np.select(conds, [0, 1, 2], default=2) # 0: Dry, 1: Transition, 2: Wet
        
    all_df["regime_cal"] = assign_regime(all_df, t1_cal, t2_cal)
    all_df["regime_orig"] = assign_regime(all_df, t1_orig, t2_orig)
    all_df["regime_cal_2r"] = np.where(all_df["soil_moisture_5cm"] < t_2regime, 0, 1) # 0: Dry, 1: Wet
    
    # 3. Monthly regime distribution
    print("\n--- Calculating Monthly Regime Distribution (Recalibrated) ---")
    monthly_cal = all_df.groupby(["month", "regime_cal"]).size().unstack(fill_value=0)
    monthly_cal_pct = monthly_cal.div(monthly_cal.sum(axis=1), axis=0) * 100
    
    print("\nMonthly counts (Recalibrated):")
    print(monthly_cal)
    print("\nMonthly percentages (Recalibrated):")
    print(monthly_cal_pct.round(1))
    
    # Figure 1: Monthly Regime Proportions (Stacked Bar)
    months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colors = ["#FFBB78", "#AEC7E8", "#98DF8A"] # Dry (Warm orange), Transition (Soft blue), Wet (Green)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Recalibrated Stacked Bar
    monthly_cal_pct.plot(kind="bar", stacked=True, color=colors, ax=axes[0], width=0.7, edgecolor="none")
    axes[0].set_title(f"Recalibrated Thresholds (t1={t1_cal}, t2={t2_cal})")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xticklabels(months_names, rotation=0)
    axes[0].legend(["Dry", "Transition", "Wet"], title="Regime")
    
    # Original Stacked Bar
    monthly_orig = all_df.groupby(["month", "regime_orig"]).size().unstack(fill_value=0)
    monthly_orig_pct = monthly_orig.div(monthly_orig.sum(axis=1), axis=0) * 100
    monthly_orig_pct.plot(kind="bar", stacked=True, color=colors, ax=axes[1], width=0.7, edgecolor="none")
    axes[1].set_title(f"Original Thresholds (t1={t1_orig}, t2={t2_orig})")
    axes[1].set_xlabel("Month")
    axes[1].set_xticklabels(months_names, rotation=0)
    axes[1].legend(["Dry", "Transition", "Wet"], title="Regime")
    
    fig.suptitle("Monthly Soil Moisture Regime Distributions (derived_8.1_pos)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_regime_distribution.png"))
    plt.close()
    
    # Figure 1b: Monthly 2-Regime Proportions (Stacked Bar)
    print("\n--- Calculating Monthly 2-Regime Distribution (T=0.159) ---")
    monthly_cal_2r = all_df.groupby(["month", "regime_cal_2r"]).size().unstack(fill_value=0)
    monthly_cal_2r_pct = monthly_cal_2r.div(monthly_cal_2r.sum(axis=1), axis=0) * 100
    
    fig_2r, ax_2r = plt.subplots(figsize=(8, 6))
    colors_2r = ["#FFBB78", "#98DF8A"] # Warm Dry, Rich Wet
    monthly_cal_2r_pct.plot(kind="bar", stacked=True, color=colors_2r, ax=ax_2r, width=0.7, edgecolor="none")
    ax_2r.set_title(f"Monthly 2-Regime Soil Moisture Distributions (T={t_2regime:.3f})", fontweight="bold")
    ax_2r.set_xlabel("Month")
    ax_2r.set_ylabel("Percentage (%)")
    ax_2r.set_xticklabels(months_names, rotation=0)
    ax_2r.legend(["Dry", "Wet"], title="Regime")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_regime_distribution_2r.png"))
    plt.close()
    
    # Figure 2: Monthly Target Densities
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    
    for i, month in enumerate(range(1, 13)):
        ax = axes[i]
        month_data = all_df[all_df["month"] == month]["soil_moisture_5cm"]
        ax.hist(month_data, bins=30, density=True, color="#1F77B4", alpha=0.6, edgecolor="none")
        ax.axvline(t1_cal, color="#D62728", linestyle="--", linewidth=1.2, label=f"t1={t1_cal}")
        ax.axvline(t2_cal, color="#2CA02C", linestyle="-.", linewidth=1.2, label=f"t2={t2_cal}")
        ax.set_title(f"{months_names[i]} (n={len(month_data)})")
        ax.set_xlim(0, 0.55)
        ax.set_ylim(0, 15)
        if i == 0:
            ax.legend(loc="upper right")
            
    fig.suptitle("Target Soil Moisture Density by Month (derived_8.1_pos)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_sm_density.png"))
    plt.close()
    
    # 4. Correlation with other key metrics
    print("\n--- Analyzing Correlation with Key Metrics ---")
    corr_cols = ["soil_moisture_5cm", "G_API", "LST_modis", "SMAP_sm_pm_interp", "precip_mm", "month"]
    corr_df = all_df[corr_cols].dropna()
    print(f"Correlation matrix (based on {len(corr_df)} complete rows):")
    print(corr_df.corr().round(3))
    
    # Correlation by month
    monthly_corrs = []
    for month in range(1, 13):
        m_df = all_df[(all_df["month"] == month) & all_df[corr_cols].notna().all(axis=1)]
        if len(m_df) > 10:
            api_corr = m_df["soil_moisture_5cm"].corr(m_df["G_API"])
            lst_corr = m_df["soil_moisture_5cm"].corr(m_df["LST_modis"])
            smap_corr = m_df["soil_moisture_5cm"].corr(m_df["SMAP_sm_pm_interp"])
            monthly_corrs.append({
                "Month": month,
                "API_Corr": api_corr,
                "LST_Corr": lst_corr,
                "SMAP_Corr": smap_corr,
                "N": len(m_df)
            })
    monthly_corrs_df = pd.DataFrame(monthly_corrs)
    print("\nMonthly correlation coefficients with soil moisture:")
    print(monthly_corrs_df.round(3).to_string(index=False))
    
    # Figure 3: Correlation trends across the year
    plt.figure(figsize=(10, 5))
    plt.plot(list(monthly_corrs_df["Month"]), list(monthly_corrs_df["API_Corr"]), "o-", color="#D62728", label="API Correlation")
    plt.plot(list(monthly_corrs_df["Month"]), list(monthly_corrs_df["LST_Corr"]), "s-", color="#FF7F0E", label="LST Correlation")
    plt.plot(list(monthly_corrs_df["Month"]), list(monthly_corrs_df["SMAP_Corr"]), "d-", color="#1F77B4", label="SMAP Correlation")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(range(1, 13), months_names)
    plt.xlabel("Month")
    plt.ylabel("Pearson Correlation (r)")
    plt.title("Correlation of Soil Moisture vs. Key Metrics across the Year", pad=15)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_correlations.png"))
    plt.close()
    
    # Figure 4: 2D Regime Separability (Scatter plots)
    # Subsample to avoid overcrowded scatter plots
    sample_df = all_df.sample(n=min(5000, len(all_df)), random_state=42).dropna(subset=["G_API", "LST_modis", "SMAP_sm_pm_interp"])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # API vs SM
    for r in [0, 1, 2]:
        mask = sample_df["regime_cal"] == r
        axes[0].scatter(sample_df.loc[mask, "G_API"], sample_df.loc[mask, "soil_moisture_5cm"],
                        color=colors[r], label=["Dry", "Transition", "Wet"][r], s=10, alpha=0.6)
    axes[0].set_xlabel("Antecedent Precipitation Index (G_API)")
    axes[0].set_ylabel("Soil Moisture (5cm)")
    axes[0].set_title("G_API vs. Soil Moisture")
    axes[0].legend()
    
    # LST vs SM
    for r in [0, 1, 2]:
        mask = sample_df["regime_cal"] == r
        axes[1].scatter(sample_df.loc[mask, "LST_modis"], sample_df.loc[mask, "soil_moisture_5cm"],
                        color=colors[r], label=["Dry", "Transition", "Wet"][r], s=10, alpha=0.6)
    axes[1].set_xlabel("Land Surface Temperature (LST_modis)")
    axes[1].set_ylabel("Soil Moisture (5cm)")
    axes[1].set_title("LST vs. Soil Moisture")
    axes[1].legend()
    
    # SMAP vs SM
    for r in [0, 1, 2]:
        mask = sample_df["regime_cal"] == r
        axes[2].scatter(sample_df.loc[mask, "SMAP_sm_pm_interp"], sample_df.loc[mask, "soil_moisture_5cm"],
                        color=colors[r], label=["Dry", "Transition", "Wet"][r], s=10, alpha=0.6)
    axes[2].set_xlabel("Satellite Soil Moisture (SMAP)")
    axes[2].set_ylabel("Soil Moisture (5cm)")
    axes[2].set_title("SMAP vs. Soil Moisture")
    axes[2].legend()
    
    fig.suptitle("Feature Separability vs. Soil Moisture Regimes", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "separability_scatter_plots.png"))
    plt.close()

    # Figure 4b: 2D 2-Regime Separability (Scatter plots)
    fig_2r, axes_2r = plt.subplots(1, 3, figsize=(18, 5))
    
    # API vs SM
    for r in [0, 1]:
        mask = sample_df["regime_cal_2r"] == r
        axes_2r[0].scatter(sample_df.loc[mask, "G_API"], sample_df.loc[mask, "soil_moisture_5cm"],
                           color=colors_2r[r], label=["Dry", "Wet"][r], s=10, alpha=0.6)
    axes_2r[0].axhline(t_2regime, color="#D62728", linestyle="--", linewidth=1.5, label=f"Boundary (T={t_2regime})")
    axes_2r[0].set_xlabel("Antecedent Precipitation Index (G_API)")
    axes_2r[0].set_ylabel("Soil Moisture (5cm)")
    axes_2r[0].set_title("G_API vs. Soil Moisture")
    axes_2r[0].legend()
    
    # LST vs SM
    for r in [0, 1]:
        mask = sample_df["regime_cal_2r"] == r
        axes_2r[1].scatter(sample_df.loc[mask, "LST_modis"], sample_df.loc[mask, "soil_moisture_5cm"],
                           color=colors_2r[r], label=["Dry", "Wet"][r], s=10, alpha=0.6)
    axes_2r[1].axhline(t_2regime, color="#D62728", linestyle="--", linewidth=1.5, label=f"Boundary (T={t_2regime})")
    axes_2r[1].set_xlabel("Land Surface Temperature (LST_modis)")
    axes_2r[1].set_ylabel("Soil Moisture (5cm)")
    axes_2r[1].set_title("LST vs. Soil Moisture")
    axes_2r[1].legend()
    
    # SMAP vs SM
    for r in [0, 1]:
        mask = sample_df["regime_cal_2r"] == r
        axes_2r[2].scatter(sample_df.loc[mask, "SMAP_sm_pm_interp"], sample_df.loc[mask, "soil_moisture_5cm"],
                           color=colors_2r[r], label=["Dry", "Wet"][r], s=10, alpha=0.6)
    axes_2r[2].axhline(t_2regime, color="#D62728", linestyle="--", linewidth=1.5, label=f"Boundary (T={t_2regime})")
    axes_2r[2].set_xlabel("Satellite Soil Moisture (SMAP)")
    axes_2r[2].set_ylabel("Soil Moisture (5cm)")
    axes_2r[2].set_title("SMAP vs. Soil Moisture")
    axes_2r[2].legend()
    
    fig_2r.suptitle("Feature Separability vs. 2-Regime Soil Moisture Boundary", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "separability_scatter_plots_2r.png"))
    plt.close()
    
    # 5. Gating Classification Evaluation
    print("\n=== EVALUATING GATING STRATEGIES (validation + test sets) ===")
    
    # We construct the dataset for evaluating gating routers:
    eval_df = all_df[all_df["split"].isin(["val", "test"])].copy()
    
    # We define Heuristic 1: Season-only Gating
    # Wet Season: Nov (11) to Mar (3)
    # Dry Season: Jul (7) to Sep (9)
    # Transition Season: Apr (4), May (5), Jun (6), Oct (10)
    def month_gating(month):
        if month in [11, 12, 1, 2, 3]:
            return 2 # Wet
        elif month in [7, 8, 9]:
            return 0 # Dry
        else:
            return 1 # Transition
            
    eval_df["pred_month_gating"] = eval_df["month"].apply(month_gating)
    
    # Let's check performance on Calibrated regimes
    y_true = eval_df["regime_cal"]
    y_pred_m = eval_df["pred_month_gating"]
    
    print("\n--- 1. Heuristic Month-Only Gating ---")
    print(classification_report(y_true, y_pred_m, target_names=["Dry", "Transition", "Wet"]))
    cm_m = confusion_matrix(y_true, y_pred_m)
    
    # Now let's train a simple decision tree on Month + G_API
    # To do this cleanly, we train on the TRAIN split, evaluate on VAL + TEST split
    train_clean = all_df[all_df["split"] == "train"].dropna(subset=["G_API", "LST_modis", "SMAP_sm_pm_interp"]).copy()
    eval_clean = all_df[all_df["split"].isin(["val", "test"])].dropna(subset=["G_API", "LST_modis", "SMAP_sm_pm_interp"]).copy()
    
    print(f"\nTraining ML models on train set ({len(train_clean)} rows)...")
    
    # Simple DT: Month + G_API
    dt_api = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_api.fit(train_clean[["month", "G_API"]], train_clean["regime_cal"])
    y_pred_dt_api = dt_api.predict(eval_clean[["month", "G_API"]])
    
    print("\n--- 2. Decision Tree Gating (Month + G_API) ---")
    print(classification_report(eval_clean["regime_cal"], y_pred_dt_api, target_names=["Dry", "Transition", "Wet"]))
    cm_dt_api = confusion_matrix(eval_clean["regime_cal"], y_pred_dt_api)
    
    # Decision Tree: Month + G_API + LST_modis + SMAP
    dt_all = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_all.fit(train_clean[["month", "G_API", "LST_modis", "SMAP_sm_pm_interp"]], train_clean["regime_cal"])
    y_pred_dt_all = dt_all.predict(eval_clean[["month", "G_API", "LST_modis", "SMAP_sm_pm_interp"]])
    
    print("\n--- 3. Decision Tree Gating (Month + G_API + LST + SMAP) ---")
    print(classification_report(eval_clean["regime_cal"], y_pred_dt_all, target_names=["Dry", "Transition", "Wet"]))
    cm_dt_all = confusion_matrix(eval_clean["regime_cal"], y_pred_dt_all)
    
    # Random Forest: Month + G_API + LST_modis + SMAP
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(train_clean[["month", "G_API", "LST_modis", "SMAP_sm_pm_interp"]], train_clean["regime_cal"])
    y_pred_rf = rf.predict(eval_clean[["month", "G_API", "LST_modis", "SMAP_sm_pm_interp"]])
    
    print("\n--- 4. Random Forest Gating (Month + G_API + LST + SMAP) ---")
    print(classification_report(eval_clean["regime_cal"], y_pred_rf, target_names=["Dry", "Transition", "Wet"]))
    cm_rf = confusion_matrix(eval_clean["regime_cal"], y_pred_rf)
    
    # Print DT rules to log
    print("\n=== DECISION TREE RULES (Month + G_API) ===")
    from sklearn.tree import export_text
    tree_rules = export_text(dt_api, feature_names=["month", "G_API"])
    print(tree_rules)
    
    # Figure 5: Confusion Matrices Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    
    # Month Gating
    ConfusionMatrixDisplay(cm_m, display_labels=["Dry", "Transition", "Wet"]).plot(
        ax=axes[0, 0], cmap="Blues", values_format="d", colorbar=False
    )
    axes[0, 0].set_title("Heuristic Month-Only Gating")
    
    # DT (Month + G_API)
    ConfusionMatrixDisplay(cm_dt_api, display_labels=["Dry", "Transition", "Wet"]).plot(
        ax=axes[0, 1], cmap="Blues", values_format="d", colorbar=False
    )
    axes[0, 1].set_title("DT Gating (Month + G_API)")
    
    # DT (Month + G_API + LST + SMAP)
    ConfusionMatrixDisplay(cm_dt_all, display_labels=["Dry", "Transition", "Wet"]).plot(
        ax=axes[1, 0], cmap="Blues", values_format="d", colorbar=False
    )
    axes[1, 0].set_title("DT Gating (Month + G_API + LST + SMAP)")
    
    # RF (Month + G_API + LST + SMAP)
    ConfusionMatrixDisplay(cm_rf, display_labels=["Dry", "Transition", "Wet"]).plot(
        ax=axes[1, 1], cmap="Blues", values_format="d", colorbar=False
    )
    axes[1, 1].set_title("RF Gating (Month + G_API + LST + SMAP)")
    
    fig.suptitle("Gating Performance Comparison on Val+Test Sets", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gating_confusion_matrices.png"))
    plt.close()
    
    # Figure 6: Decision Tree Structure Visualization
    plt.figure(figsize=(12, 6))
    plot_tree(dt_api, feature_names=["month", "G_API"], class_names=["Dry", "Transition", "Wet"], filled=True, rounded=True, fontsize=9)
    plt.title("Interpretable Decision Tree (Month + G_API) for Regime Routing", pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_tree_gating_structure.png"))
    plt.close()
    
    # 2-Regime Gating Evaluation
    print("\n=== EVALUATING 2-REGIME GATING STRATEGIES (validation + test sets) ===")
    
    # Heuristic Month-only Gating for 2-regime
    def month_gating_2r(month):
        if month in [7, 8, 9]:
            return 0 # Dry
        else:
            return 1 # Wet
            
    eval_df["pred_month_gating_2r"] = eval_df["month"].apply(month_gating_2r)
    y_true_2r = eval_df["regime_cal_2r"]
    y_pred_m_2r = eval_df["pred_month_gating_2r"]
    
    print("\n--- 1. Heuristic Month-Only Binary Gating ---")
    print(classification_report(y_true_2r, y_pred_m_2r, target_names=["Dry", "Wet"]))
    cm_m_2r = confusion_matrix(y_true_2r, y_pred_m_2r)
    
    # Train binary models
    print(f"\nTraining ML models for 2-regime gating on train set ({len(train_clean)} rows)...")
    
    # Simple DT: Month + G_API
    dt_api_2r = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_api_2r.fit(train_clean[["month", "G_API"]], train_clean["regime_cal_2r"])
    y_pred_dt_api_2r = dt_api_2r.predict(eval_clean[["month", "G_API"]])
    
    print("\n--- 2. Decision Tree Gating (Month + G_API) [2-Regime] ---")
    print(classification_report(eval_clean["regime_cal_2r"], y_pred_dt_api_2r, target_names=["Dry", "Wet"]))
    cm_dt_api_2r = confusion_matrix(eval_clean["regime_cal_2r"], y_pred_dt_api_2r)
    
    # Decision Tree: Month + G_API + LST + SMAP
    dt_all_2r = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_all_2r.fit(train_clean[["month", "G_API", "LST_modis", "SMAP_sm_pm_interp"]], train_clean["regime_cal_2r"])
    y_pred_dt_all_2r = dt_all_2r.predict(eval_clean[["month", "G_API", "LST_modis", "SMAP_sm_pm_interp"]])
    
    print("\n--- 3. Decision Tree Gating (Month + G_API + LST + SMAP) [2-Regime] ---")
    print(classification_report(eval_clean["regime_cal_2r"], y_pred_dt_all_2r, target_names=["Dry", "Wet"]))
    cm_dt_all_2r = confusion_matrix(eval_clean["regime_cal_2r"], y_pred_dt_all_2r)
    
    # Random Forest: Month + G_API + LST + SMAP
    rf_2r = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf_2r.fit(train_clean[["month", "G_API", "LST_modis", "SMAP_sm_pm_interp"]], train_clean["regime_cal_2r"])
    y_pred_rf_2r = rf_2r.predict(eval_clean[["month", "G_API", "LST_modis", "SMAP_sm_pm_interp"]])
    
    print("\n--- 4. Random Forest Gating (Month + G_API + LST + SMAP) [2-Regime] ---")
    print(classification_report(eval_clean["regime_cal_2r"], y_pred_rf_2r, target_names=["Dry", "Wet"]))
    cm_rf_2r = confusion_matrix(eval_clean["regime_cal_2r"], y_pred_rf_2r)
    
    # Print DT rules to log
    print("\n=== DECISION TREE RULES (Month + G_API) [2-Regime] ===")
    tree_rules_2r = export_text(dt_api_2r, feature_names=["month", "G_API"])
    print(tree_rules_2r)
    
    # Figure 5b: Confusion Matrices Comparison (2-Regime)
    fig_cm_2r, axes_cm_2r = plt.subplots(2, 2, figsize=(12, 11))
    
    ConfusionMatrixDisplay(cm_m_2r, display_labels=["Dry", "Wet"]).plot(
        ax=axes_cm_2r[0, 0], cmap="Blues", values_format="d", colorbar=False
    )
    axes_cm_2r[0, 0].set_title("Heuristic Month-Only Gating (2-Regime)")
    
    ConfusionMatrixDisplay(cm_dt_api_2r, display_labels=["Dry", "Wet"]).plot(
        ax=axes_cm_2r[0, 1], cmap="Blues", values_format="d", colorbar=False
    )
    axes_cm_2r[0, 1].set_title("DT Gating (Month + G_API) (2-Regime)")
    
    ConfusionMatrixDisplay(cm_dt_all_2r, display_labels=["Dry", "Wet"]).plot(
        ax=axes_cm_2r[1, 0], cmap="Blues", values_format="d", colorbar=False
    )
    axes_cm_2r[1, 0].set_title("DT Gating (Month + G_API + LST + SMAP) (2-Regime)")
    
    ConfusionMatrixDisplay(cm_rf_2r, display_labels=["Dry", "Wet"]).plot(
        ax=axes_cm_2r[1, 1], cmap="Blues", values_format="d", colorbar=False
    )
    axes_cm_2r[1, 1].set_title("RF Gating (Month + G_API + LST + SMAP) (2-Regime)")
    
    fig_cm_2r.suptitle("2-Regime Gating Performance Comparison on Val+Test Sets", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gating_confusion_matrices_2r.png"))
    plt.close()
    
    # Figure 6b: Decision Tree Structure Visualization (2-Regime)
    plt.figure(figsize=(12, 6))
    plot_tree(dt_api_2r, feature_names=["month", "G_API"], class_names=["Dry", "Wet"], filled=True, rounded=True, fontsize=9)
    plt.title("Interpretable Decision Tree (Month + G_API) for 2-Regime Routing", pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_tree_gating_structure_2r.png"))
    plt.close()
    
    print("\nDone. All figures saved in the experiment directory.")

if __name__ == "__main__":
    main()
