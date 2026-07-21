import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add splits/derived_8.2 directory to sys.path to import the metadata module
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(os.path.join(project_root, 'data', 'splits', 'derived_8.2'))
from dataset_metadata import TERNARY_REGIME_DRY_THRESHOLD as t1_cal, TERNARY_REGIME_WET_THRESHOLD as t2_cal, BINARY_REGIME_THRESHOLD as t_2regime

# Set style for premium aesthetics using standard matplotlib
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def main():
    split_dir = os.path.join(project_root, "data", "splits", "derived_8.2")
    output_dir = os.path.join(project_root, "notebooks", "experiment", "derived_8.2-data-exploration")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading derived_8.2 data splits...")
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print("Loading baseline derived_8.0 data split for density comparison...")
    d80_dir = os.path.join(project_root, "data", "splits", "derived_8.0")
    d80_train = pd.read_csv(os.path.join(d80_dir, "train.csv"), usecols=["soil_moisture_5cm"])
    
    # 1. Dataset stats summary
    print("\n=== Dataset Row and Station Counts ===")
    print(f"derived_8.2 Total rows: {len(all_df)} (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    stations = sorted(all_df['station_id'].unique())
    print(f"derived_8.2 Stations ({len(stations)}): {stations}")
    
    # 2. Target Quantiles
    quantiles = [0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 1.0]
    q_80 = d80_train["soil_moisture_5cm"].quantile(quantiles)
    q_82 = train_df["soil_moisture_5cm"].quantile(quantiles)
    
    q_df = pd.DataFrame({
        "Percentile": [f"{int(q*100)}%" for q in quantiles],
        "derived_8.0 (Train)": q_80.values,
        "derived_8.2 (Train)": q_82.values
    })
    print("\n=== Target (soil_moisture_5cm) Quantiles in Train Set ===")
    print(q_df.to_string(index=False))
    
    t1_orig, t2_orig = 0.20, 0.313
    print(f"\n3-Regime Thresholds: Original (t1={t1_orig}, t2={t2_orig}), Valleys (t1={t1_cal}, t2={t2_cal})")
    print(f"2-Regime Threshold: T={t_2regime}")
    
    # 3. Density Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    d80_train["soil_moisture_5cm"].plot.kde(ax=ax, label="derived_8.0 Train (5 stations)", color="#1f77b4", linewidth=2)
    train_df["soil_moisture_5cm"].plot.kde(ax=ax, label="derived_8.2 Train (12 stations)", color="#ff7f0e", linewidth=2.5)
    
    ax.axvline(t1_cal, color="#d62728", linestyle="--", alpha=0.8, label=f"Calibrated t1 = {t1_cal:.2f}")
    ax.axvline(t2_cal, color="#2ca02c", linestyle="--", alpha=0.8, label=f"Calibrated t2 = {t2_cal:.2f}")
    ax.axvline(t1_orig, color="gray", linestyle=":", alpha=0.6, label=f"Original t1 = {t1_orig:.2f}")
    ax.axvline(t2_orig, color="gray", linestyle=":", alpha=0.6, label=f"Original t2 = {t2_orig:.3f}")
    
    ax.set_title("Soil Moisture Target Density Distribution: derived_8.0 vs derived_8.2", pad=15)
    ax.set_xlabel("Soil Moisture 5cm ($m^3/m^3$)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 0.5)
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "soil_moisture_density_comparison.png"))
    plt.close()
    
    # Helper to categorize 3 regimes
    def get_regime_3r(sm, t1, t2):
        conds = [sm < t1, (sm >= t1) & (sm < t2), sm >= t2]
        return np.select(conds, ["Dry", "Transition", "Wet"], default="Wet")

    def get_regime_2r(sm, t2r):
        return np.where(sm < t2r, "Dry", "Wet")

    # 4. Aggregated 3-Regime Comparison Plot
    d80_train_reg_orig = get_regime_3r(d80_train["soil_moisture_5cm"], t1_orig, t2_orig)
    d82_train_reg_orig = get_regime_3r(train_df["soil_moisture_5cm"], t1_orig, t2_orig)
    d82_train_reg_cal = get_regime_3r(train_df["soil_moisture_5cm"], t1_cal, t2_cal)
    
    reg_comparison = pd.DataFrame({
        "derived_8.0 Train (Orig)": pd.Series(d80_train_reg_orig).value_counts(normalize=True)[["Dry", "Transition", "Wet"]] * 100,
        "derived_8.2 Train (Orig)": pd.Series(d82_train_reg_orig).value_counts(normalize=True)[["Dry", "Transition", "Wet"]] * 100,
        "derived_8.2 Train (Valleys)": pd.Series(d82_train_reg_cal).value_counts(normalize=True)[["Dry", "Transition", "Wet"]] * 100,
    }).T

    colors_3r = ["#FFBB78", "#AEC7E8", "#98DF8A"]
    fig, ax = plt.subplots(figsize=(10, 6))
    reg_comparison.plot(kind="bar", stacked=True, color=colors_3r, ax=ax, width=0.55, edgecolor="none")
    ax.set_title("Aggregated 3-Regime Split Proportions Comparison", pad=15)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(reg_comparison.index, rotation=0)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    for i, (idx, row) in enumerate(reg_comparison.iterrows()):
        cum = 0
        for cat in ["Dry", "Transition", "Wet"]:
            val = row[cat]
            if val > 3:
                ax.text(i, cum + val / 2, f"{val:.1f}%", ha="center", va="center", color="#222222", fontweight="bold", fontsize=9)
            cum += val

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_regime_comparison.png"))
    plt.close()

    # 5. Station-by-Station 3-Regime Plot
    all_df["regime_cal"] = get_regime_3r(all_df["soil_moisture_5cm"], t1_cal, t2_cal)
    st_reg_3r = all_df.groupby("station_id")["regime_cal"].value_counts(normalize=True).unstack(fill_value=0)[["Dry", "Transition", "Wet"]] * 100
    st_reg_3r = st_reg_3r.sort_values(by="Wet", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    st_reg_3r.plot(kind="barh", stacked=True, color=colors_3r, ax=ax, width=0.65, edgecolor="none")
    ax.set_title(f"3-Regime Distribution by Station (Valleys t1={t1_cal:.2f}, t2={t2_cal:.2f})", pad=15)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 100)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    for i, (st, row) in enumerate(st_reg_3r.iterrows()):
        cum = 0
        for cat in ["Dry", "Transition", "Wet"]:
            val = row[cat]
            if val > 5:
                ax.text(cum + val / 2, i, f"{val:.1f}%", ha="center", va="center", color="#222222", fontweight="bold", fontsize=8)
            cum += val
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regime_distribution_by_station.png"))
    plt.close()

    # 6. Aggregated 2-Regime Comparison Plot
    d82_train_reg_2r = get_regime_2r(train_df["soil_moisture_5cm"], t_2regime)
    d82_val_reg_2r = get_regime_2r(val_df["soil_moisture_5cm"], t_2regime)
    d82_test_reg_2r = get_regime_2r(test_df["soil_moisture_5cm"], t_2regime)

    reg_2r_df = pd.DataFrame({
        "Train": pd.Series(d82_train_reg_2r).value_counts(normalize=True)[["Dry", "Wet"]] * 100,
        "Val": pd.Series(d82_val_reg_2r).value_counts(normalize=True)[["Dry", "Wet"]] * 100,
        "Test": pd.Series(d82_test_reg_2r).value_counts(normalize=True)[["Dry", "Wet"]] * 100,
    }).T

    colors_2r = ["#FFBB78", "#98DF8A"]
    fig, ax = plt.subplots(figsize=(8, 6))
    reg_2r_df.plot(kind="bar", stacked=True, color=colors_2r, ax=ax, width=0.5, edgecolor="none")
    ax.set_title(f"Aggregated 2-Regime Split Proportions (Threshold T={t_2regime:.2f})", pad=15)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(reg_2r_df.index, rotation=0)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    for i, (idx, row) in enumerate(reg_2r_df.iterrows()):
        cum = 0
        for cat in ["Dry", "Wet"]:
            val = row[cat]
            if val > 3:
                ax.text(i, cum + val / 2, f"{val:.1f}%", ha="center", va="center", color="#222222", fontweight="bold", fontsize=10)
            cum += val
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_regime_comparison_2r.png"))
    plt.close()

    # 7. Station-by-Station 2-Regime Plot
    all_df["regime_2r"] = get_regime_2r(all_df["soil_moisture_5cm"], t_2regime)
    st_reg_2r = all_df.groupby("station_id")["regime_2r"].value_counts(normalize=True).unstack(fill_value=0)[["Dry", "Wet"]] * 100
    st_reg_2r = st_reg_2r.sort_values(by="Wet", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    st_reg_2r.plot(kind="barh", stacked=True, color=colors_2r, ax=ax, width=0.65, edgecolor="none")
    ax.set_title(f"2-Regime Distribution by Station (Boundary T={t_2regime:.2f})", pad=15)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 100)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    for i, (st, row) in enumerate(st_reg_2r.iterrows()):
        cum = 0
        for cat in ["Dry", "Wet"]:
            val = row[cat]
            if val > 5:
                ax.text(cum + val / 2, i, f"{val:.1f}%", ha="center", va="center", color="#222222", fontweight="bold", fontsize=8)
            cum += val
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regime_distribution_by_station_2r.png"))
    plt.close()

    # 8. Target Histograms Grid by Station
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, st in enumerate(stations):
        ax = axes[i]
        sub = all_df[all_df["station_id"] == st]["soil_moisture_5cm"]
        ax.hist(sub, bins=30, color="#2b5c8f", alpha=0.75, edgecolor="white", density=True)
        ax.axvline(t1_cal, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(t2_cal, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_title(st, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

    for j in range(len(stations), len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle("Soil Moisture Density Histograms by Station (with Valleys t1=0.16, t2=0.25)", fontsize=16, y=0.98)
    fig.text(0.5, 0.02, "Soil Moisture 5cm ($m^3/m^3$)", ha="center", fontsize=12)
    fig.text(0.02, 0.5, "Density", va="center", rotation="vertical", fontsize=12)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "soil_moisture_by_station_grid.png"))
    plt.close()

    # 9. Target Histograms Grid by Month
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df["month"] = all_df["date"].dt.month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for m in range(1, 13):
        ax = axes[m-1]
        sub = all_df[all_df["month"] == m]["soil_moisture_5cm"]
        ax.hist(sub, bins=30, color="#386cb0", alpha=0.75, edgecolor="white", density=True)
        ax.axvline(t1_cal, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(t2_cal, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_title(f"{month_names[m-1]} (m={m})", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        
    fig.suptitle("Soil Moisture Density Histograms by Month Across All Stations", fontsize=16, y=0.98)
    fig.text(0.5, 0.02, "Soil Moisture 5cm ($m^3/m^3$)", ha="center", fontsize=12)
    fig.text(0.02, 0.5, "Density", va="center", rotation="vertical", fontsize=12)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "soil_moisture_by_month_grid.png"))
    plt.close()

    print("Successfully completed analyze_regimes.py for derived_8.2!")

if __name__ == "__main__":
    main()
