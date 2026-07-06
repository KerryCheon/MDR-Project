import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add splits/derived_8.1_pos directory to sys.path to import the metadata module
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'data', 'splits', 'derived_8.1_pos')))
from dataset_metadata import T1 as t1_cal, T2 as t2_cal, T_2REGIME as t_2regime

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
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    split_dir = os.path.join(project_root, "data", "splits", "derived_8.1_pos")
    output_dir = os.path.join(project_root, "notebooks", "experiment", "derived_8.1_pos-data-exploration")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading derived_8.1_pos splits...")
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df["year"] = pd.DatetimeIndex(all_df["date"]).year.astype(int)
    
    # 3-Regime Thresholds
    t1_orig, t2_orig = 0.20, 0.313
    
    print(f"Loaded {len(all_df)} rows spanning years: {sorted(all_df['year'].unique())}")
    
    # Categorize functions
    def get_regime_3r(sm, t1, t2):
        conds = [sm < t1, (sm >= t1) & (sm < t2), sm >= t2]
        return np.select(conds, ["Dry", "Transition", "Wet"], default="Wet")
        
    def get_regime_2r(sm, t_2r):
        return np.where(sm < t_2r, "Dry", "Wet")
        
    all_df["regime_cal"] = get_regime_3r(all_df["soil_moisture_5cm"], t1_cal, t2_cal)
    all_df["regime_orig"] = get_regime_3r(all_df["soil_moisture_5cm"], t1_orig, t2_orig)
    all_df["regime_2r"] = get_regime_2r(all_df["soil_moisture_5cm"], t_2regime)
    
    years = sorted(all_df["year"].unique())
    
    # Helper to compute and print annual tables
    def analyze_distribution(col_name, label, classes):
        print(f"\n=== Annual Regime Distribution: {label} ===")
        annual_counts = all_df.groupby(["year", col_name]).size().unstack(fill_value=0).reindex(columns=classes)
        annual_pcts = annual_counts.div(annual_counts.sum(axis=1), axis=0) * 100
        
        # Display combined table
        table_df = pd.DataFrame(index=years)
        for cls in classes:
            table_df[f"{cls} (Count)"] = annual_counts[cls]
            table_df[f"{cls} (%)"] = annual_pcts[cls].round(2)
        table_df["Total"] = annual_counts.sum(axis=1)
        print(table_df.to_string())
        return annual_pcts
        
    pcts_orig = analyze_distribution("regime_orig", f"Original 3-Regime (t1={t1_orig:.2f}, t2={t2_orig:.3f})", ["Dry", "Transition", "Wet"])
    pcts_cal = analyze_distribution("regime_cal", f"Valley-Calibrated 3-Regime (t1={t1_cal:.3f}, t2={t2_cal:.3f})", ["Dry", "Transition", "Wet"])
    pcts_2r = analyze_distribution("regime_2r", f"2-Regime (T={t_2regime:.3f})", ["Dry", "Wet"])
    
    # Let's generate stacked bar charts for each threshold type
    colors_3r = ["#FFBB78", "#AEC7E8", "#98DF8A"]  # Warm Dry, Soft Transition, Rich Wet
    colors_2r = ["#FFBB78", "#98DF8A"]  # Warm Dry, Rich Wet
    
    # Figure 1: Original 3-Regime
    fig, ax = plt.subplots(figsize=(10, 6))
    pcts_orig.plot(kind="bar", stacked=True, color=colors_3r, ax=ax, width=0.6, edgecolor="none")
    ax.set_title(f"Annual 3-Regime Distributions: Original Thresholds\n(t1 = {t1_orig:.2f}, t2 = {t2_orig:.3f})", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(years, rotation=0)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    # Annotate percentages
    for i, year in enumerate(years):
        accum = 0
        for col in ["Dry", "Transition", "Wet"]:
            val = pcts_orig.loc[year, col]
            if val > 3.0:
                ax.text(i, accum + val/2, f"{val:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=9)
            accum += val
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "annual_regime_distribution_original.png"))
    plt.close()
    
    # Figure 2: Valley-Calibrated 3-Regime
    fig, ax = plt.subplots(figsize=(10, 6))
    pcts_cal.plot(kind="bar", stacked=True, color=colors_3r, ax=ax, width=0.6, edgecolor="none")
    ax.set_title(f"Annual 3-Regime Distributions: Valley-Calibrated Thresholds\n(t1 = {t1_cal:.3f}, t2 = {t2_cal:.3f})", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(years, rotation=0)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    # Annotate percentages
    for i, year in enumerate(years):
        accum = 0
        for col in ["Dry", "Transition", "Wet"]:
            val = pcts_cal.loc[year, col]
            if val > 3.0:
                ax.text(i, accum + val/2, f"{val:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=9)
            accum += val
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "annual_regime_distribution_calibrated.png"))
    plt.close()
    
    # Figure 3: 2-Regime
    fig, ax = plt.subplots(figsize=(10, 6))
    pcts_2r.plot(kind="bar", stacked=True, color=colors_2r, ax=ax, width=0.6, edgecolor="none")
    ax.set_title(f"Annual 2-Regime Distributions: Threshold T = {t_2regime:.3f}", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(years, rotation=0)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    # Annotate percentages
    for i, year in enumerate(years):
        accum = 0
        for col in ["Dry", "Wet"]:
            val = pcts_2r.loc[year, col]
            if val > 3.0:
                ax.text(i, accum + val/2, f"{val:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=9)
            accum += val
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "annual_regime_distribution_2r.png"))
    plt.close()

    # Figure 4a: 3-Regime Valley-Calibrated trends
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(years, pcts_cal["Dry"], "o-", color="#FF7F0E", linewidth=2.5, label="Dry Regime (<0.159)")
    ax.plot(years, pcts_cal["Transition"], "s-", color="#1F77B4", linewidth=2.5, label="Transition Regime (0.159-0.248)")
    ax.plot(years, pcts_cal["Wet"], "d-", color="#2CA02C", linewidth=2.5, label="Wet Regime (>=0.248)")
    
    # Calculate and plot average reference lines
    dry_mean_3r = pcts_cal["Dry"].mean()
    trans_mean_3r = pcts_cal["Transition"].mean()
    wet_mean_3r = pcts_cal["Wet"].mean()
    
    ax.axhline(y=dry_mean_3r, color="#FF7F0E", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Dry Avg ({dry_mean_3r:.1f}%)")
    ax.axhline(y=trans_mean_3r, color="#1F77B4", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Trans Avg ({trans_mean_3r:.1f}%)")
    ax.axhline(y=wet_mean_3r, color="#2CA02C", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Wet Avg ({wet_mean_3r:.1f}%)")
    
    ax.set_title("3-Regime Valley-Calibrated Trends (2017 - 2025)", fontweight="bold", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 70)
    ax.set_xticks(years)
    ax.legend(frameon=True, facecolor="white", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "annual_regime_trends_3r.png"))
    plt.close()
    
    # Figure 4b: 2-Regime trends
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(years, pcts_2r["Dry"], "o-", color="#FF7F0E", linewidth=2.5, label="Dry Regime (<0.159)")
    ax.plot(years, pcts_2r["Wet"], "d-", color="#2CA02C", linewidth=2.5, label="Wet Regime (>=0.159)")
    
    # Calculate and plot average reference lines
    dry_mean_2r = pcts_2r["Dry"].mean()
    wet_mean_2r = pcts_2r["Wet"].mean()
    
    ax.axhline(y=dry_mean_2r, color="#FF7F0E", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Dry Avg ({dry_mean_2r:.1f}%)")
    ax.axhline(y=wet_mean_2r, color="#2CA02C", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Wet Avg ({wet_mean_2r:.1f}%)")
    
    ax.set_title("2-Regime Trends (2017 - 2025)", fontweight="bold", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 95)
    ax.set_xticks(years)
    ax.legend(frameon=True, facecolor="white", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "annual_regime_trends_2r.png"))
    plt.close()

    print("Done. Generated figures and tables for annual regime distribution analysis.")

if __name__ == "__main__":
    main()
