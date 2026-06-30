import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def main():
    parser = argparse.ArgumentParser(description="Programmatically identify density peaks and valleys to calibrate regime thresholds.")
    parser.add_argument("--split-dir", type=str, default=None, 
                        help="Path to the split directory containing train.csv. Defaults to data/splits/derived_8.1/")
    parser.add_argument("--bw", type=float, default=0.15, help="Bandwidth factor for Kernel Density Estimation (default: 0.15)")
    parser.add_argument("--plot", action="store_true", default=True, help="Save a visualization of the peaks and valleys")
    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    if args.split_dir:
        split_dir = os.path.abspath(args.split_dir)
    else:
        split_dir = os.path.join(project_root, "data", "splits", "derived_8.1")

    train_path = os.path.join(split_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Could not find training split at: {train_path}")

    print(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path, usecols=["soil_moisture_5cm"])
    sm = df["soil_moisture_5cm"].dropna().values

    # Perform Kernel Density Estimation (KDE)
    print(f"Computing Kernel Density Estimation (bandwidth={args.bw})...")
    kde = gaussian_kde(sm, bw_method=args.bw)
    x_grid = np.linspace(0.0, 0.5, 1000)
    density = kde(x_grid)

    # Find peaks (modes)
    peaks, _ = find_peaks(density, distance=20)
    peak_x = x_grid[peaks]
    peak_y = density[peaks]

    # Find valleys (local minima between peaks)
    valleys, _ = find_peaks(-density, distance=20)
    valley_x = x_grid[valleys]
    valley_y = density[valleys]

    print("\n=== Programmatically Identified Modes (Peaks) ===")
    for px, py in zip(peak_x, peak_y):
        print(f"Peak at soil moisture = {px:.3f} (density = {py:.3f})")

    print("\n=== Programmatically Identified Valleys (Minima) ===")
    for vx, vy in zip(valley_x, valley_y):
        print(f"Valley at soil moisture = {vx:.3f} (density = {vy:.3f})")

    # Save a visualization if requested
    if args.plot:
        out_plot_path = os.path.join(script_dir, "programmatic_valleys_calibration.png")
        print(f"\nSaving density visualization to {out_plot_path}...")
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, density, label="KDE Density", color="#2CA02C", linewidth=2)
        plt.fill_between(x_grid, density, alpha=0.1, color="#2CA02C")
        
        # Plot peaks and valleys
        plt.scatter(peak_x, peak_y, color="red", s=80, zorder=5, label="Modes (Peaks)")
        for px, py in zip(peak_x, peak_y):
            plt.annotate(f"Peak: {px:.3f}", (px, py), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontweight='bold')
            
        plt.scatter(valley_x, valley_y, color="purple", s=80, marker="v", zorder=5, label="Valleys (Minima)")
        for vx, vy in zip(valley_x, valley_y):
            plt.annotate(f"Valley: {vx:.3f}", (vx, vy), textcoords="offset points", 
                         xytext=(0,-15), ha='center', color="purple", fontweight='bold')
            plt.axvline(vx, color="purple", linestyle="--", alpha=0.5)

        plt.title("Programmatic Valley-Based Threshold Calibration (derived_8.1 Train)", pad=15)
        plt.xlabel("Soil Moisture (5cm) [cm³/cm³]")
        plt.ylabel("Density")
        plt.xlim(0, 0.5)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        plt.savefig(out_plot_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
