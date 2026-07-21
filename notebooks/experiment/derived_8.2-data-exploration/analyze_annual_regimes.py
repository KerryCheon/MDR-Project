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
    all_df["year"] = all_df["date"].dt.year
    years = sorted(all_df["year"].unique())
    
    t1_orig, t2_orig = 0.20, 0.313
    
    def get_regime_3r(sm, t1, t2):
        conds = [sm < t1, (sm >= t1) & (sm < t2), sm >= t2]
        return np.select(conds, ["Dry", "Transition", "Wet"], default="Wet")

    def get_regime_2r(sm, t2r):
        return np.where(sm < t2r, "Dry", "Wet")

    all_df["regime_orig"] = get_regime_3r(all_df["soil_moisture_5cm"], t1_orig, t2_orig)
    all_df["regime_cal"] = get_regime_3r(all_df["soil_moisture_5cm"], t1_cal, t2_cal)
    all_df["regime_2r"] = get_regime_2r(all_df["soil_moisture_5cm"], t_2regime)
    
    def plot_annual(col_name, title, colors, classes, filename):
        counts = all_df.groupby(["year", col_name]).size().unstack(fill_value=0).reindex(columns=classes)
        pcts = counts.div(counts.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pcts.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.6, edgecolor="none")
        ax.set_title(title, pad=15)
        ax.set_xlabel("Year")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)
        ax.set_xticklabels(years, rotation=0)
        ax.legend(title="Regime", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor="white")
        
        for i, year in enumerate(years):
            cum = 0
            for cls in classes:
                val = pcts.loc[year, cls]
                if val > 4:
                    ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center", va="center", color="#222222", fontweight="bold", fontsize=8)
                cum += val
                
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    colors_3r = ["#FFBB78", "#AEC7E8", "#98DF8A"]
    colors_2r = ["#FFBB78", "#98DF8A"]
    
    plot_annual("regime_orig", f"Annual 3-Regime Proportions: Original Thresholds (t1={t1_orig:.2f}, t2={t2_orig:.3f})", colors_3r, ["Dry", "Transition", "Wet"], "annual_regime_distribution_original.png")
    plot_annual("regime_cal", f"Annual 3-Regime Proportions: Valleys Thresholds (t1={t1_cal:.2f}, t2={t2_cal:.2f})", colors_3r, ["Dry", "Transition", "Wet"], "annual_regime_distribution_calibrated.png")
    plot_annual("regime_2r", f"Annual 2-Regime Proportions (Boundary T={t_2regime:.2f})", colors_2r, ["Dry", "Wet"], "annual_regime_distribution_2r.png")
    
    print("Saved annual regime distribution plots for derived_8.2!")

if __name__ == "__main__":
    main()
