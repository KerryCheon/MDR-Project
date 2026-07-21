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
    'axes.grid': False
})

def main():
    split_dir = os.path.join(project_root, "data", "splits", "derived_8.2")
    output_dir = os.path.join(project_root, "notebooks", "experiment", "derived_8.2-data-exploration")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading derived_8.2 data splits for station data entry count analysis...")
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"), usecols=["station_id", "date"])
    val_df = pd.read_csv(os.path.join(split_dir, "val.csv"), usecols=["station_id", "date"])
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"), usecols=["station_id", "date"])

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df["year"] = all_df["date"].dt.year

    years = sorted(all_df["year"].unique())
    stations = sorted(all_df["station_id"].unique())

    # Build count matrix (Station x Year)
    counts_matrix = all_df.groupby(["station_id", "year"]).size().unstack(fill_value=0).reindex(index=stations, columns=years)

    print("\n=== Data Entry Counts by Station and Year ===")
    print(counts_matrix.to_string())

    total_by_station = counts_matrix.sum(axis=1)
    print("\n=== Total Data Entry Counts per Station ===")
    print(total_by_station.to_string())

    # Figure 1: Annotated Heatmap of Data Entry Counts by Year per Station
    plt.figure(figsize=(12, 8))
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    cmap.set_under("#f0f0f0") # Color for 0 entries

    ax = sns.heatmap(
        counts_matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=1,
        linecolor="white",
        cbar_kws={'label': 'Data Entry Count (Days/Year)'},
        vmin=1,
        vmax=366
    )

    # Highlight missing cells (count == 0) with distinct color/text
    for i, st in enumerate(stations):
        for j, yr in enumerate(years):
            val = counts_matrix.loc[st, yr]
            if val == 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='#ffe6e6', edgecolor='white', lw=1))
                ax.text(j + 0.5, i + 0.5, '0 (MISSING)', ha='center', va='center', color='#d62728', fontweight='bold', fontsize=8)
            elif val < 100:
                ax.text(j + 0.5, i + 0.5, f'{val}\n(SPARSE)', ha='center', va='center', color='#d95f02', fontweight='bold', fontsize=8)

    plt.title("Station Data Entry Counts by Year (derived_8.2 Dataset)", pad=15, fontsize=15, fontweight="bold")
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Station ID", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "station_data_entries_heatmap.png"))
    plt.close()

    # Figure 2: Grouped Bar Chart of Total Data Entry Counts per Station with Split Breakdown
    split_counts = all_df.groupby(["station_id", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    split_counts["total"] = split_counts.sum(axis=1)
    split_counts = split_counts.sort_values(by="total", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    split_counts[["train", "val", "test"]].plot(
        kind="barh",
        stacked=True,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        ax=ax,
        width=0.65,
        edgecolor="white"
    )

    ax.set_title("Total Data Entries per Station by Dataset Split (derived_8.2)", pad=15, fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of Data Entries", fontsize=12)
    ax.set_ylabel("Station ID", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    ax.legend(title="Split", loc="lower right", frameon=True, facecolor="white")

    # Annotate total counts on bars
    for i, (st, row) in enumerate(split_counts.iterrows()):
        tot = row["total"]
        ax.text(tot + 30, i, f"{tot:,}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "station_data_entries_by_year.png"))
    plt.close()

    print("Saved station data entry count visualizations successfully!")

if __name__ == "__main__":
    main()
