import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    # Paths (assuming script is run from notebooks/ directory)
    project_root = ".."
    d80_dir = os.path.join(project_root, "data", "splits", "derived_8.0")
    d81_dir = os.path.join(project_root, "data", "splits", "derived_8.1")
    output_dir = os.path.join("experiment", "derived_8.1-data-exploration")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading derived_8.0 data splits...")
    d80_train = pd.read_csv(os.path.join(d80_dir, "train.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    d80_val = pd.read_csv(os.path.join(d80_dir, "val.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    d80_test = pd.read_csv(os.path.join(d80_dir, "test.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    
    d80_train["split"] = "train"
    d80_val["split"] = "val"
    d80_test["split"] = "test"
    d80_all = pd.concat([d80_train, d80_val, d80_test], ignore_index=True)
    
    print("Loading derived_8.1 data splits...")
    d81_train = pd.read_csv(os.path.join(d81_dir, "train.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    d81_val = pd.read_csv(os.path.join(d81_dir, "val.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    d81_test = pd.read_csv(os.path.join(d81_dir, "test.csv"), usecols=["station_id", "date", "soil_moisture_5cm"])
    
    d81_train["split"] = "train"
    d81_val["split"] = "val"
    d81_test["split"] = "test"
    d81_all = pd.concat([d81_train, d81_val, d81_test], ignore_index=True)
    
    # 1. Dataset stats summary
    print("\n=== Dataset Row and Station Counts ===")
    print(f"derived_8.0 Total rows: {len(d80_all)} (Train: {len(d80_train)}, Val: {len(d80_val)}, Test: {len(d80_test)})")
    print(f"derived_8.0 Stations: {d80_all['station_id'].nunique()} -> {d80_all['station_id'].unique()}")
    print(f"derived_8.1 Total rows: {len(d81_all)} (Train: {len(d81_train)}, Val: {len(d81_val)}, Test: {len(d81_test)})")
    print(f"derived_8.1 Stations: {d81_all['station_id'].nunique()} -> {sorted(d81_all['station_id'].unique())}")
    
    # 2. Compute Target Quantiles
    print("\n=== Target (soil_moisture_5cm) Quantiles in Train Set ===")
    quantiles = [0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 1.0]
    q_80 = d80_train["soil_moisture_5cm"].quantile(quantiles)
    q_81 = d81_train["soil_moisture_5cm"].quantile(quantiles)
    
    q_df = pd.DataFrame({
        "Percentile": [f"{int(q*100)}%" for q in quantiles],
        "derived_8.0 (Train)": q_80.values,
        "derived_8.1 (Train)": q_81.values
    })
    print(q_df.to_string(index=False))
    
    t1_80_orig, t2_80_orig = 0.20, 0.313
    t1_90, t2_90 = 0.0993, 0.2115
    t1_81_cal, t2_81_cal = 0.16, 0.25
    
    print("\n=== Threshold Calibration Options ===")
    print(f"1. Original 8.0 thresholds:       t1 = {t1_80_orig:.4f}, t2 = {t2_80_orig:.4f}")
    print(f"2. Current 9.0 thresholds:        t1 = {t1_90:.4f}, t2 = {t2_90:.4f}")
    print(f"3. Recalibrated 8.1 thresholds (valleys-based on 8.1 Train): t1 = {t1_81_cal:.4f}, t2 = {t2_81_cal:.4f}")
    
    # Analyze regimes helper
    def analyze_regime_distribution(df, t1, t2, label):
        sm = df["soil_moisture_5cm"]
        conditions = [
            sm < t1,
            (sm >= t1) & (sm < t2),
            sm >= t2
        ]
        choices = ["Dry", "Transition", "Wet"]
        regimes = np.select(conditions, choices, default="Wet")
        df_copy = df.copy()
        df_copy["regime"] = regimes
        counts = df_copy["regime"].value_counts().reindex(choices)
        pcts = df_copy["regime"].value_counts(normalize=True).reindex(choices) * 100
        
        summary = pd.DataFrame({
            "Count": counts,
            "Percentage": pcts
        })
        print(f"\nRegime distribution ({label}) [t1={t1:.4f}, t2={t2:.4f}]:")
        print(summary.to_string())
        return df_copy
    
    # Analyze global splits under different thresholds
    print("\n--- GLOBAL REGIME DISTRIBUTIONS ---")
    analyze_regime_distribution(d80_train, t1_80_orig, t2_80_orig, "derived_8.0 Train with Original thresholds")
    analyze_regime_distribution(d81_train, t1_80_orig, t2_80_orig, "derived_8.1 Train with Original thresholds")
    
    print("\n--- derived_8.1 splits with recalibrated thresholds (t1={:.4f}, t2={:.4f}) ---".format(t1_81_cal, t2_81_cal))
    df81_train_reg = analyze_regime_distribution(d81_train, t1_81_cal, t2_81_cal, "derived_8.1 Train")
    df81_val_reg   = analyze_regime_distribution(d81_val, t1_81_cal, t2_81_cal, "derived_8.1 Val")
    df81_test_reg  = analyze_regime_distribution(d81_test, t1_81_cal, t2_81_cal, "derived_8.1 Test")
    df81_all_reg   = pd.concat([df81_train_reg, df81_val_reg, df81_test_reg], ignore_index=True)
    
    # Station-by-station analysis on derived_8.1 with recalibrated thresholds
    print("\n=== Station-by-Station Details (derived_8.1 with Recalibrated Thresholds) ===")
    station_regimes = []
    for station, group in df81_all_reg.groupby("station_id"):
        total = len(group)
        counts = group["regime"].value_counts().reindex(["Dry", "Transition", "Wet"], fill_value=0)
        pcts = (counts / total * 100)
        min_sm = group["soil_moisture_5cm"].min()
        max_sm = group["soil_moisture_5cm"].max()
        mean_sm = group["soil_moisture_5cm"].mean()
        
        station_regimes.append({
            "Station": station,
            "Total Obs": total,
            "Mean SM": f"{mean_sm:.4f}",
            "Min SM": f"{min_sm:.4f}",
            "Max SM": f"{max_sm:.4f}",
            "Dry Count": counts["Dry"],
            "Dry %": f"{pcts['Dry']:.1f}%",
            "Trans Count": counts["Transition"],
            "Trans %": f"{pcts['Transition']:.1f}%",
            "Wet Count": counts["Wet"],
            "Wet %": f"{pcts['Wet']:.1f}%",
        })
    station_reg_df = pd.DataFrame(station_regimes)
    print(station_reg_df.to_string(index=False))
    
    # Compare original 5 stations vs 8 new stations in derived_8.1
    orig_stations = ["Spokane", "Darrington", "Quinault", "Touchet_WA_824", "SourdoughGulch_WA_985"]
    df81_all_reg["station_type"] = np.where(df81_all_reg["station_id"].isin(orig_stations), "Original 5 Stations", "New 8 SNOTEL Stations")
    
    print("\n=== Original vs New Stations Regime Distribution (derived_8.1, Recalibrated) ===")
    for s_type, gp in df81_all_reg.groupby("station_type"):
        total = len(gp)
        counts = gp["regime"].value_counts().reindex(["Dry", "Transition", "Wet"])
        pcts = counts / total * 100
        print(f"\n{s_type} (Total observations: {total}):")
        for r in ["Dry", "Transition", "Wet"]:
            print(f"  {r}: {counts[r]} ({pcts[r]:.2f}%)")

    # Generate Figures
    print("\nGenerating figures...")
    
    # Figure 1: Target distributions (Histogram / Density) using matplotlib hist/kde approximation
    plt.figure(figsize=(10, 6))
    
    # Histogram plots for approximation of density
    plt.hist(d80_train["soil_moisture_5cm"], bins=50, density=True, alpha=0.4, label="derived_8.0 Train (5 stations)", color="#1F77B4", edgecolor="none")
    plt.hist(d81_train["soil_moisture_5cm"], bins=50, density=True, alpha=0.4, label="derived_8.1 Train (13 stations)", color="#2CA02C", edgecolor="none")
    
    plt.axvline(t1_80_orig, color="#D62728", linestyle="--", linewidth=1.5, label=f"Original t1 = {t1_80_orig:.2f}")
    plt.axvline(t2_80_orig, color="#D62728", linestyle="-.", linewidth=1.5, label=f"Original t2 = {t2_80_orig:.3f}")
    plt.axvline(t1_81_cal, color="#9467BD", linestyle="--", linewidth=1.5, label=f"Recalibrated 8.1 t1 = {t1_81_cal:.3f}")
    plt.axvline(t2_81_cal, color="#9467BD", linestyle="-.", linewidth=1.5, label=f"Recalibrated 8.1 t2 = {t2_81_cal:.3f}")
    
    plt.title("Soil Moisture Distribution: derived_8.0 vs. derived_8.1 Train Sets", pad=15)
    plt.xlabel("Soil Moisture (5cm) [cm³/cm³]")
    plt.ylabel("Density")
    plt.xlim(0, 0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "soil_moisture_density_comparison.png"))
    plt.close()
    
    # Figure 2: Regime Distribution by Station (Stacked Bar Chart)
    # Prepare data
    pivot_df = df81_all_reg.groupby(["station_id", "regime"]).size().unstack(fill_value=0).reindex(columns=["Dry", "Transition", "Wet"])
    # Sort stations by total count
    pivot_df["Total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(by="Total", ascending=True)
    pivot_plot = pivot_df.drop(columns="Total")
    
    # Plot counts
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#FFBB78", "#AEC7E8", "#98DF8A"] # Warm Dry, Soft Transition, Rich Wet
    pivot_plot.plot(kind="barh", stacked=True, color=colors, ax=ax, width=0.7)
    
    # Annotate bar percentages
    for i, (idx, row) in enumerate(pivot_df.iterrows()):
        total = row["Total"]
        accum = 0
        for col_idx, col in enumerate(["Dry", "Transition", "Wet"]):
            val = row[col]
            if val > 0:
                pct = val / total * 100
                if pct > 4.0: # Only label if text fits
                    ax.text(accum + val/2, i, f"{pct:.0f}%", 
                            ha='center', va='center', color='black', 
                            fontweight='bold', fontsize=9)
            accum += val
            
    ax.set_title("derived_8.1: Soil Moisture Regime Counts & Percentages by Station", pad=15)
    ax.set_xlabel("Observations (Days)")
    ax.set_ylabel("Station ID")
    ax.legend(title="Regime", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regime_distribution_by_station.png"))
    plt.close()
    
    # Figure 3: Histogram of values per station (Small multiples)
    stations_sorted = sorted(df81_all_reg["station_id"].unique())
    n_stations = len(stations_sorted)
    
    # 4 columns, compute rows needed
    n_cols = 4
    n_rows = (n_stations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.flatten()
    
    for idx, station in enumerate(stations_sorted):
        ax = axes[idx]
        data = df81_all_reg[df81_all_reg["station_id"] == station]["soil_moisture_5cm"]
        ax.hist(data, bins=25, color="#1F77B4", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(t1_81_cal, color="#9467BD", linestyle="--", linewidth=1.2, label="t1")
        ax.axvline(t2_81_cal, color="#9467BD", linestyle="-.", linewidth=1.2, label="t2")
        ax.set_title(station, fontsize=11, pad=5)
        ax.set_xlim(0, 0.6)
        
    # Hide any unused subplots
    for idx in range(n_stations, len(axes)):
        fig.delaxes(axes[idx])
        
    fig.suptitle("derived_8.1: Target Distributions across Stations with Recalibrated Thresholds", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "soil_moisture_by_station_grid.png"))
    plt.close()

    # Figure 4: Aggregated Regime Proportions Comparison (Dry + Transition + Wet)
    categories = [
        "derived_8.0 Train\n(Orig thresholds: t1=0.20, t2=0.313)",
        "derived_8.1 Train\n(Orig thresholds: t1=0.20, t2=0.313)",
        "derived_8.1 Train\n(Recalibrated: t1=0.160, t2=0.250)"
    ]
    dry_pcts = [47.7, 47.8, 36.8]
    trans_pcts = [39.5, 31.1, 24.4]
    wet_pcts = [12.8, 21.1, 38.8]

    plot_df = pd.DataFrame({
        'Dry': dry_pcts,
        'Transition': trans_pcts,
        'Wet': wet_pcts
    }, index=categories)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#FFBB78", "#AEC7E8", "#98DF8A"]
    plot_df.plot(kind="barh", stacked=True, color=colors, ax=ax, width=0.55)

    # Add percentages annotations
    for i in range(len(categories)):
        accum = 0
        for col in ['Dry', 'Transition', 'Wet']:
            val = plot_df.loc[categories[i], col]
            ax.text(accum + val/2, i, f"{val:.1f}%", 
                    ha='center', va='center', color='black', 
                    fontweight='bold', fontsize=10)
            accum += val

    ax.set_title("Aggregated Soil Moisture Regime Proportions Comparison", pad=15)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 100)
    ax.legend(title="Regime", loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_regime_comparison.png"))
    plt.close()

    print("Done generating figures.")

if __name__ == "__main__":
    main()
