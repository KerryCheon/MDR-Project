import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

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

    print("Loading derived_8.2 data for Soil Moisture vs Precipitation analysis...")
    cols = ["station_id", "date", "soil_moisture_5cm", "precip_mm", "G_API", "G_rain_sum_3d", "G_rain_sum_7d"]
    
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"), usecols=cols)
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"), usecols=cols)
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"), usecols=cols)

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    stations = sorted(all_df["station_id"].unique())

    # Model R2 scores from derived_8.2-eval-3.3 for context
    model_r2 = {
        "BurntMountain_WA": -0.783,
        "Touchet_WA_824": -1.621,
        "MartenRidge_WA_999": -0.186,
        "RainyPass_WA_711": 0.501,
        "HartsPass_WA_515": 0.540,
        "BeaverPass_WA_990": 0.681,
        "CayusePass_WA": 0.690,
        "SourdoughGulch_WA_985": 0.695,
        "Paradise_WA": 0.730,
        "Quinault": 0.742,
        "Darrington": 0.785,
        "Spokane": 0.912
    }

    # 1. Compute Station-level Correlations and Anomalies
    stats_list = []
    for st in stations:
        sub = all_df[all_df["station_id"] == st].dropna(subset=["soil_moisture_5cm", "precip_mm", "G_API"])
        sm = sub["soil_moisture_5cm"]
        prcp = sub["precip_mm"]
        api = sub["G_API"]
        
        corr_prcp = sm.corr(prcp)
        corr_api = sm.corr(api)
        
        near_zero_pct = (sm < 0.01).mean() * 100
        high_p_low_sm_count = ((prcp > 10) & (sm < 0.05)).sum()
        
        stats_list.append({
            "station_id": st,
            "count": len(sub),
            "sm_mean": sm.mean(),
            "sm_median": sm.median(),
            "sm_std": sm.std(),
            "precip_mean": prcp.mean(),
            "api_mean": api.mean(),
            "corr_prcp": corr_prcp,
            "corr_api": corr_api,
            "near_zero_pct": near_zero_pct,
            "high_p_low_sm_count": high_p_low_sm_count,
            "model_r2": model_r2.get(st, np.nan)
        })

    stats_df = pd.DataFrame(stats_list)
    print("\n=== Station Soil Moisture vs Precipitation Diagnostics ===")
    print(stats_df.to_string(index=False))

    # Figure 1: Soil Moisture vs Daily Precip Scatter Grid per Station
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, st in enumerate(stations):
        ax = axes[i]
        sub = all_df[all_df["station_id"] == st]
        
        # Hexbin density plot for smooth visualization of dense points
        hb = ax.hexbin(sub["precip_mm"], sub["soil_moisture_5cm"], gridsize=25, cmap="YlGnBu", mincnt=1, alpha=0.85)
        
        # Linear regression trendline
        sns.regplot(
            data=sub, x="precip_mm", y="soil_moisture_5cm", ax=ax,
            scatter=False, color="red", line_kws={"linewidth": 1.5, "linestyle": "--"}
        )
        
        r2 = model_r2.get(st, 0.0)
        corr = sub["soil_moisture_5cm"].corr(sub["precip_mm"])
        
        # Title color: red if model R2 < 0, green if R2 > 0.7, blue otherwise
        title_color = "#d62728" if r2 < 0 else ("#2ca02c" if r2 > 0.7 else "#1f77b4")
        
        ax.set_title(f"{st}\n(Corr={corr:.2f}, Model R²={r2:.2f})", fontsize=10, fontweight="bold", color=title_color)
        ax.tick_params(labelsize=8)

    fig.suptitle("Daily Soil Moisture (5cm) vs Daily Precipitation (precip_mm) per Station", fontsize=16, y=0.98)
    fig.text(0.5, 0.02, "Daily Precipitation (mm)", ha="center", fontsize=12)
    fig.text(0.02, 0.5, "Soil Moisture 5cm ($m^3/m^3$)", va="center", rotation="vertical", fontsize=12)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "sm_vs_precip_by_station.png"))
    plt.close()

    # Figure 2: Soil Moisture vs Antecedent Precipitation Index (G_API) Scatter Grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, st in enumerate(stations):
        ax = axes[i]
        sub = all_df[all_df["station_id"] == st]
        
        hb = ax.hexbin(sub["G_API"], sub["soil_moisture_5cm"], gridsize=25, cmap="viridis", mincnt=1, alpha=0.85)
        
        sns.regplot(
            data=sub, x="G_API", y="soil_moisture_5cm", ax=ax,
            scatter=False, color="red", line_kws={"linewidth": 1.5, "linestyle": "--"}
        )
        
        r2 = model_r2.get(st, 0.0)
        corr_api = sub["soil_moisture_5cm"].corr(sub["G_API"])
        title_color = "#d62728" if r2 < 0 else ("#2ca02c" if r2 > 0.7 else "#1f77b4")
        
        ax.set_title(f"{st}\n(API Corr={corr_api:.2f}, Model R²={r2:.2f})", fontsize=10, fontweight="bold", color=title_color)
        ax.tick_params(labelsize=8)

    fig.suptitle("Soil Moisture (5cm) vs Antecedent Precipitation Index (G_API) per Station", fontsize=16, y=0.98)
    fig.text(0.5, 0.02, "Antecedent Precipitation Index (G_API)", ha="center", fontsize=12)
    fig.text(0.02, 0.5, "Soil Moisture 5cm ($m^3/m^3$)", va="center", rotation="vertical", fontsize=12)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "sm_vs_g_api_by_station.png"))
    plt.close()

    # Figure 3: Correlation Comparison vs Model Performance R2
    stats_df_sorted = stats_df.sort_values(by="model_r2", ascending=True)
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    y_pos = np.arange(len(stats_df_sorted))
    bar_width = 0.35
    
    rects1 = ax1.barh(y_pos - bar_width/2, stats_df_sorted["corr_prcp"], bar_width, label="Corr(SM, Daily Precip)", color="#6baed6", alpha=0.85)
    rects2 = ax1.barh(y_pos + bar_width/2, stats_df_sorted["corr_api"], bar_width, label="Corr(SM, G_API Index)", color="#3182bd", alpha=0.85)
    
    ax1.set_xlabel("Pearson Correlation Coefficient ($r$)", fontsize=12, color="#3182bd")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stats_df_sorted["station_id"], fontsize=10)
    ax1.axvline(0, color="gray", linestyle="-", alpha=0.5)
    ax1.set_xlim(-0.2, 0.85)
    
    # Overlay Model R2 on secondary axis
    ax2 = ax1.twiny()
    ax2.plot(stats_df_sorted["model_r2"], y_pos, "ro-", linewidth=2, markersize=8, label="Global Model Baseline $R^2$")
    ax2.set_xlabel("Model Baseline $R^2$ Score", fontsize=12, color="#d62728")
    ax2.axvline(0, color="red", linestyle=":", alpha=0.7)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True, facecolor="white")

    plt.title("Soil Moisture - Precipitation Coupling Strength vs. Model Performance R²", pad=20, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sm_precip_correlation_by_station.png"))
    plt.close()

    # Figure 4: Dual-Axis Time Series Diagnostics comparing Problem vs Well-Behaved Stations
    sample_year = 2020
    diag_stations = ["BurntMountain_WA", "Touchet_WA_824", "Darrington", "Spokane"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for i, st in enumerate(diag_stations):
        ax1 = axes[i]
        sub = all_df[(all_df["station_id"] == st) & (all_df["date"].dt.year == sample_year)].sort_values(by="date")
        
        if len(sub) == 0: # If year missing, pick available year
            avail_yr = all_df[all_df["station_id"] == st]["date"].dt.year.min()
            sub = all_df[(all_df["station_id"] == st) & (all_df["date"].dt.year == avail_yr)].sort_values(by="date")
            yr_str = str(avail_yr)
        else:
            yr_str = str(sample_year)
            
        color_sm = "#1f77b4"
        ax1.plot(sub["date"], sub["soil_moisture_5cm"], color=color_sm, linewidth=1.8, label="Soil Moisture 5cm")
        ax1.set_ylabel("Soil Moisture ($m^3/m^3$)", color=color_sm, fontsize=10)
        ax1.tick_params(axis="y", labelcolor=color_sm)
        ax1.set_ylim(0, 0.45)
        
        ax2 = ax1.twinx()
        color_prcp = "#e377c2"
        ax2.bar(sub["date"], sub["precip_mm"], color=color_prcp, alpha=0.4, width=1.0, label="Daily Precip (mm)")
        ax2.set_ylabel("Precip (mm)", color=color_prcp, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=color_prcp)
        ax2.set_ylim(0, 100)

        r2 = model_r2.get(st, 0.0)
        ax1.set_title(f"{st} ({yr_str}) — Model R²: {r2:.2f}", fontsize=11, fontweight="bold")
        ax1.tick_params(axis="x", rotation=30)

    fig.suptitle("Diagnostic Time-Series (2020): Soil Moisture Response to Precipitation", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "sm_vs_precip_diagnostics_time.png"))
    plt.close()

    print("Saved Soil Moisture vs Precipitation diagnostic figures successfully!")

if __name__ == "__main__":
    main()
