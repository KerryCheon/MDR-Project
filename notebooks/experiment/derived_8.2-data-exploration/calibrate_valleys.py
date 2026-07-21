import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    split_dir = os.path.join(project_root, "data", "splits", "derived_8.2")
    output_dir = os.path.join(project_root, "notebooks", "experiment", "derived_8.2-data-exploration")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading derived_8.2 training split for valley calibration...")
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"), usecols=["soil_moisture_5cm"])
    sm_train = train_df["soil_moisture_5cm"].dropna().values

    # Fit KDE
    kde = gaussian_kde(sm_train, bw_method="scott")
    x_eval = np.linspace(0, 0.45, 1000)
    density = kde(x_eval)

    # Find peaks (modes)
    peaks, _ = find_peaks(density, distance=30, prominence=0.1)
    
    # Find valleys (density minima between peaks)
    inv_density = -density
    valleys, _ = find_peaks(inv_density, distance=30, prominence=0.05)

    peak_x = x_eval[peaks]
    valley_x = x_eval[valleys]

    print("\n=== KDE Peak and Valley Detection ===")
    print(f"Modes (Peaks) at SM values: {np.round(peak_x, 4)}")
    print(f"Valleys (Minima) at SM values: {np.round(valley_x, 4)}")

    # Plot programmatic calibration
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_eval, density, label="KDE Target Density", color="#1f77b4", linewidth=2)
    ax.plot(peak_x, density[peaks], "ro", markersize=8, label="Detected Modes")
    ax.plot(valley_x, density[valleys], "gs", markersize=8, label="Detected Valleys")

    for v in valley_x:
        ax.axvline(v, color="green", linestyle="--", alpha=0.7, label=f"Valley T={v:.3f}")

    ax.set_title("Programmatic Valley Threshold Calibration (derived_8.2 Train)", fontsize=14, pad=15)
    ax.set_xlabel("Soil Moisture 5cm ($m^3/m^3$)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(0, 0.45)
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "programmatic_valleys_calibration.png"), dpi=300)
    plt.close()
    print("Saved programmatic_valleys_calibration.png!")

if __name__ == "__main__":
    main()
