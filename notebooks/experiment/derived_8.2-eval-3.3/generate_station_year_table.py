import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error

# Find project root
def find_project_root():
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for cand in candidates:
        if (cand / "data").exists() and (cand / "d_models").exists():
            return cand
    raise FileNotFoundError("Could not locate repo root")

PROJECT_ROOT = find_project_root()
EXP_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.2-eval-3.3"
MODELS_DIR = EXP_DIR / "models"
TEST_PATH = PROJECT_ROOT / "data/splits/derived_8.2/test.csv"

# Load test data
test_df = pd.read_csv(TEST_PATH)
y_te = test_df["soil_moisture_5cm"].values

# Load metrics summary to find best model
summary_path = EXP_DIR / "metrics_summary.csv"
if not summary_path.exists():
    print(f"Error: {summary_path} does not exist. Did you run the notebook?")
    sys.exit(1)

summary = pd.read_csv(summary_path)
best_row = summary.sort_values("R2", ascending=False).iloc[0]
best_id = int(best_row["Model ID"])
best_name = best_row["Model Name"]
print(f"Best model: {best_name} (ID: {best_id}) with overall R2 = {best_row['R2']:.4f}")

# Find predictions file
pred_files = list(MODELS_DIR.glob(f"model_{best_id}_*_preds.npy"))
if not pred_files:
    print(f"Error: Could not find predictions file for model {best_id} in {MODELS_DIR}")
    sys.exit(1)

pred_file = pred_files[0]
print(f"Loading predictions from: {pred_file.name}")
preds = np.load(pred_file)

# Align station_id and year
test_df["pred"] = preds
stations = sorted(test_df["station_id"].unique())
years = sorted(test_df["year"].unique())

# Initialize data structures for table
cell_data = {}

for s in stations:
    cell_data[s] = {}
    for y in years:
        mask = (test_df["station_id"] == s) & (test_df["year"] == y)
        sub = test_df[mask]
        n = len(sub)
        if n > 1:
            r2 = r2_score(sub["soil_moisture_5cm"], sub["pred"])
            rmse = root_mean_squared_error(sub["soil_moisture_5cm"], sub["pred"])
        else:
            r2, rmse = float("nan"), float("nan")
        cell_data[s][y] = {"r2": r2, "rmse": rmse, "n": n}

    # Station overall
    mask_s = test_df["station_id"] == s
    sub_s = test_df[mask_s]
    n_s = len(sub_s)
    if n_s > 1:
        r2_s = r2_score(sub_s["soil_moisture_5cm"], sub_s["pred"])
        rmse_s = root_mean_squared_error(sub_s["soil_moisture_5cm"], sub_s["pred"])
    else:
        r2_s, rmse_s = float("nan"), float("nan")
    cell_data[s]["Overall"] = {"r2": r2_s, "rmse": rmse_s, "n": n_s}

# Year overall (row at the bottom)
cell_data["Overall"] = {}
for y in years:
    mask_y = test_df["year"] == y
    sub_y = test_df[mask_y]
    n_y = len(sub_y)
    if n_y > 1:
        r2_y = r2_score(sub_y["soil_moisture_5cm"], sub_y["pred"])
        rmse_y = root_mean_squared_error(sub_y["soil_moisture_5cm"], sub_y["pred"])
    else:
        r2_y, rmse_y = float("nan"), float("nan")
    cell_data["Overall"][y] = {"r2": r2_y, "rmse": rmse_y, "n": n_y}

# Global overall (bottom right cell)
n_tot = len(test_df)
r2_tot = r2_score(test_df["soil_moisture_5cm"], test_df["pred"])
rmse_tot = root_mean_squared_error(test_df["soil_moisture_5cm"], test_df["pred"])
cell_data["Overall"]["Overall"] = {"r2": r2_tot, "rmse": rmse_tot, "n": n_tot}

# Now, we plot this beautiful table
col_labels = ["Station"] + [str(int(y)) for y in years] + ["Overall"]
row_labels = stations + ["Overall"]

nrows = len(row_labels)
ncols = len(col_labels)

# We draw the table using custom boxes in Matplotlib for complete aesthetic control
fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0f172a")
ax.set_facecolor("#0f172a")

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Cell dimensions
dx = 1.0 / ncols
dy = 1.0 / (nrows + 1) # +1 for header

# Color palette
BG_HEADER = "#1e293b"
BG_STATION_COL = "#1e293b"
BG_CELL_EVEN = "#0f172a"
BG_CELL_ODD = "#1e293b"
BG_OVERALL = "#334155" # highlighted overall row/col
BORDER_COLOR = "#475569"

TEXT_WHITE = "#ffffff"
TEXT_MUTED = "#94a3b8"
COLOR_R2_POS = "#22d3ee" # Vibrant cyan
COLOR_R2_NEG = "#f87171" # Coral red
COLOR_RMSE = "#fb923c"   # Amber/Orange

# Draw headers
for c_idx, label in enumerate(col_labels):
    rect = plt.Rectangle(
        (c_idx * dx, 1 - dy), dx, dy,
        facecolor=BG_HEADER, edgecolor=BORDER_COLOR, linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(
        c_idx * dx + dx/2, 1 - dy/2, label,
        color=TEXT_WHITE, weight="bold", fontsize=12,
        ha="center", va="center"
    )

# Draw rows
for r_idx, row_name in enumerate(row_labels):
    y_pos = 1 - (r_idx + 2) * dy
    is_overall_row = (row_name == "Overall")

    for c_idx in range(ncols):
        rect_x = c_idx * dx
        is_overall_col = (col_labels[c_idx] == "Overall")
        
        # Decide cell background
        if is_overall_row or is_overall_col:
            cell_bg = BG_OVERALL
        elif c_idx == 0:
            cell_bg = BG_STATION_COL
        else:
            cell_bg = BG_CELL_EVEN if r_idx % 2 == 0 else BG_CELL_ODD

        rect = plt.Rectangle(
            (rect_x, y_pos), dx, dy,
            facecolor=cell_bg, edgecolor=BORDER_COLOR, linewidth=1
        )
        ax.add_patch(rect)

        # Draw cell contents
        if c_idx == 0:
            # Station label
            label_text = row_name
            font_w = "bold" if is_overall_row else "normal"
            ax.text(
                rect_x + 0.02, y_pos + dy/2, label_text,
                color=TEXT_WHITE, weight=font_w, fontsize=11,
                ha="left", va="center"
            )
        else:
            col_name = col_labels[c_idx]
            # Retrieve metric data
            y_key = float(col_name) if col_name != "Overall" else "Overall"
            cell_metrics = cell_data[row_name][y_key]
            
            r2 = cell_metrics["r2"]
            rmse = cell_metrics["rmse"]
            n = cell_metrics["n"]

            if n == 0 or np.isnan(r2):
                ax.text(
                    rect_x + dx/2, y_pos + dy/2, "No Data",
                    color=TEXT_MUTED, fontsize=10, ha="center", va="center"
                )
            else:
                # Top text: R2
                r2_color = COLOR_R2_POS if r2 >= 0 else COLOR_R2_NEG
                r2_text = f"R²: {r2:.4f}" if not np.isnan(r2) else "R²: N/A"
                ax.text(
                    rect_x + dx/2, y_pos + 2*dy/3, r2_text,
                    color=r2_color, weight="bold", fontsize=10,
                    ha="center", va="center"
                )
                
                # Bottom text: RMSE
                rmse_text = f"RMSE: {rmse:.4f}"
                ax.text(
                    rect_x + dx/2, y_pos + dy/3, rmse_text,
                    color=COLOR_RMSE, fontsize=10,
                    ha="center", va="center"
                )
                
                # Subscript text: sample count (n)
                n_text = f"n={n}"
                ax.text(
                    rect_x + dx/2, y_pos + dy/10, n_text,
                    color=TEXT_MUTED, fontsize=8,
                    ha="center", va="center"
                )

# Title
ax.text(
    0.5, 1.05, f"Station × Year Metrics Breakdown\nBest Model: {best_name}",
    color=TEXT_WHITE, fontsize=16, weight="bold", ha="center", va="bottom", transform=ax.transAxes
)

# Tighten layout and save
plt.tight_layout()
output_path = EXP_DIR / "station_year_metrics.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print(f"Successfully generated and saved styled breakdown table image to: {output_path.name}")
