import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(os.path.join(project_root, 'data', 'splits', 'derived_8.2'))
from dataset_metadata import TERNARY_REGIME_DRY_THRESHOLD as t1_cal, TERNARY_REGIME_WET_THRESHOLD as t2_cal, BINARY_REGIME_THRESHOLD as t_2regime

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
    
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df["month"] = all_df["date"].dt.month
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    def get_regime_3r(sm, t1, t2):
        conds = [sm < t1, (sm >= t1) & (sm < t2), sm >= t2]
        return np.select(conds, ["Dry", "Transition", "Wet"], default="Wet")

    def get_regime_2r(sm, t2r):
        return np.where(sm < t2r, "Dry", "Wet")

    all_df["regime_3r"] = get_regime_3r(all_df["soil_moisture_5cm"], t1_cal, t2_cal)
    all_df["regime_2r"] = get_regime_2r(all_df["soil_moisture_5cm"], t_2regime)
    
    monthly_3r = all_df.groupby("month")["regime_3r"].value_counts(normalize=True).unstack(fill_value=0)[["Dry", "Transition", "Wet"]] * 100
    monthly_2r = all_df.groupby("month")["regime_2r"].value_counts(normalize=True).unstack(fill_value=0)[["Dry", "Wet"]] * 100
    
    # Figure 1: 3-Regime Monthly
    colors_3r = ["#FFBB78", "#AEC7E8", "#98DF8A"]
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_3r.plot(kind="bar", stacked=True, color=colors_3r, ax=ax, width=0.6, edgecolor="none")
    ax.set_title(f"Monthly 3-Regime Proportions Across All Stations (t1={t1_cal:.2f}, t2={t2_cal:.2f})", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(month_names, rotation=0)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    for i in range(12):
        cum = 0
        for cat in ["Dry", "Transition", "Wet"]:
            val = monthly_3r.iloc[i][cat]
            if val > 5:
                ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center", va="center", color="#222222", fontweight="bold", fontsize=8)
            cum += val
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_regime_distribution.png"))
    plt.close()

    # Figure 2: 2-Regime Monthly
    colors_2r = ["#FFBB78", "#98DF8A"]
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_2r.plot(kind="bar", stacked=True, color=colors_2r, ax=ax, width=0.6, edgecolor="none")
    ax.set_title(f"Monthly 2-Regime Proportions Across All Stations (Boundary T={t_2regime:.2f})", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(month_names, rotation=0)
    ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
    
    for i in range(12):
        cum = 0
        for cat in ["Dry", "Wet"]:
            val = monthly_2r.iloc[i][cat]
            if val > 5:
                ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center", va="center", color="#222222", fontweight="bold", fontsize=9)
            cum += val
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_regime_distribution_2r.png"))
    plt.close()
    
    print("Saved monthly regime distribution plots for derived_8.2!")

if __name__ == "__main__":
    main()
