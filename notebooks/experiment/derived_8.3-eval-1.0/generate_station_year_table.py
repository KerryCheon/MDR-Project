import os
import sys
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error

def find_project_root():
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for cand in candidates:
        if (cand / "data").exists() and (cand / "d_models").exists():
            return cand
    raise FileNotFoundError("Could not locate repo root")

PROJECT_ROOT = find_project_root()
EXP_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.3-eval-1.0"
MODELS_DIR = EXP_DIR / "models"
TEST_PATH = PROJECT_ROOT / "data/splits/derived_8.3/test.csv"

def generate_table(model_id: int, output_filename: str, title_suffix: str = ""):
    test_df = pd.read_csv(TEST_PATH)

    summary_path = EXP_DIR / "metrics_summary.csv"
    if not summary_path.exists():
        print(f"Error: {summary_path} does not exist. Did you run the notebook?")
        sys.exit(1)

    summary = pd.read_csv(summary_path)
    model_row = summary[summary["Model ID"] == model_id]
    if model_row.empty:
        print(f"Error: Model ID {model_id} not found in {summary_path}")
        return

    model_row = model_row.iloc[0]
    model_name = model_row["Model Name"]
    overall_r2 = model_row["R2"]
    print(f"Generating table for: {model_name} (ID: {model_id}) with R2 = {overall_r2:.4f}")

    pred_files = list(MODELS_DIR.glob(f"model_{model_id}_*_preds.npy"))
    if not pred_files:
        print(f"Error: Could not find predictions file for model {model_id} in {MODELS_DIR}")
        return

    pred_file = pred_files[0]
    preds = np.load(pred_file)

    df = test_df.copy()
    df["pred"] = preds
    stations = sorted(df["station_id"].unique())
    years = sorted(df["year"].unique())

    cell_data = {}

    for s in stations:
        cell_data[s] = {}
        for y in years:
            mask = (df["station_id"] == s) & (df["year"] == y)
            sub = df[mask]
            n = len(sub)
            if n > 1:
                r2 = r2_score(sub["soil_moisture_5cm"], sub["pred"])
                rmse = root_mean_squared_error(sub["soil_moisture_5cm"], sub["pred"])
            else:
                r2, rmse = float("nan"), float("nan")
            cell_data[s][y] = {"r2": r2, "rmse": rmse, "n": n}

        mask_s = df["station_id"] == s
        sub_s = df[mask_s]
        n_s = len(sub_s)
        if n_s > 1:
            r2_s = r2_score(sub_s["soil_moisture_5cm"], sub_s["pred"])
            rmse_s = root_mean_squared_error(sub_s["soil_moisture_5cm"], sub_s["pred"])
        else:
            r2_s, rmse_s = float("nan"), float("nan")
        cell_data[s]["Overall"] = {"r2": r2_s, "rmse": rmse_s, "n": n_s}

    cell_data["Overall"] = {}
    for y in years:
        mask_y = df["year"] == y
        sub_y = df[mask_y]
        n_y = len(sub_y)
        if n_y > 1:
            r2_y = r2_score(sub_y["soil_moisture_5cm"], sub_y["pred"])
            rmse_y = root_mean_squared_error(sub_y["soil_moisture_5cm"], sub_y["pred"])
        else:
            r2_y, rmse_y = float("nan"), float("nan")
        cell_data["Overall"][y] = {"r2": r2_y, "rmse": rmse_y, "n": n_y}

    n_tot = len(df)
    r2_tot = r2_score(df["soil_moisture_5cm"], df["pred"])
    rmse_tot = root_mean_squared_error(df["soil_moisture_5cm"], df["pred"])
    cell_data["Overall"]["Overall"] = {"r2": r2_tot, "rmse": rmse_tot, "n": n_tot}

    col_labels = ["Station"] + [str(int(y)) for y in years] + ["Overall"]
    row_labels = stations + ["Overall"]

    nrows = len(row_labels)
    ncols = len(col_labels)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    dx = 1.0 / ncols
    dy = 1.0 / (nrows + 1)

    BG_HEADER = "#1e293b"
    BG_STATION_COL = "#1e293b"
    BG_CELL_EVEN = "#0f172a"
    BG_CELL_ODD = "#1e293b"
    BG_OVERALL = "#334155"
    BORDER_COLOR = "#475569"

    TEXT_WHITE = "#ffffff"
    TEXT_MUTED = "#94a3b8"
    COLOR_R2_POS = "#22d3ee"
    COLOR_R2_NEG = "#f87171"
    COLOR_RMSE = "#fb923c"

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

    for r_idx, row_name in enumerate(row_labels):
        y_pos = 1 - (r_idx + 2) * dy
        is_overall_row = (row_name == "Overall")

        for c_idx in range(ncols):
            rect_x = c_idx * dx
            is_overall_col = (col_labels[c_idx] == "Overall")

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

            if c_idx == 0:
                label_text = row_name
                font_w = "bold" if is_overall_row else "normal"
                ax.text(
                    rect_x + 0.02, y_pos + dy/2, label_text,
                    color=TEXT_WHITE, weight=font_w, fontsize=11,
                    ha="left", va="center"
                )
            else:
                col_name = col_labels[c_idx]
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
                    r2_color = COLOR_R2_POS if r2 >= 0 else COLOR_R2_NEG
                    r2_text = f"R²: {r2:.4f}" if not np.isnan(r2) else "R²: N/A"
                    ax.text(
                        rect_x + dx/2, y_pos + 2*dy/3, r2_text,
                        color=r2_color, weight="bold", fontsize=10,
                        ha="center", va="center"
                    )

                    rmse_text = f"RMSE: {rmse:.4f}"
                    ax.text(
                        rect_x + dx/2, y_pos + dy/3, rmse_text,
                        color=COLOR_RMSE, fontsize=10,
                        ha="center", va="center"
                    )

                    n_text = f"n={n}"
                    ax.text(
                        rect_x + dx/2, y_pos + dy/10, n_text,
                        color=TEXT_MUTED, fontsize=8,
                        ha="center", va="center"
                    )

    title_text = f"Station × Year Metrics Breakdown{title_suffix}\nModel: {model_name}"
    ax.text(
        0.5, 1.05, title_text,
        color=TEXT_WHITE, fontsize=16, weight="bold", ha="center", va="bottom", transform=ax.transAxes
    )

    plt.tight_layout()
    output_path = EXP_DIR / output_filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"Saved: {output_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Station x Year breakdown table images.")
    parser.add_argument("--model-id", type=int, default=None, help="Model ID to generate table for")
    args = parser.parse_args()

    if args.model_id is not None:
        generate_table(args.model_id, f"station_year_metrics_model_{args.model_id}.png")
    else:
        generate_table(1, "station_year_metrics_global_v0.png", title_suffix=" (Global V0 Baseline)")
        generate_table(10, "station_year_metrics_clustering_dynamic_k2.png", title_suffix=" (Clustering Dynamic K=2)")
        generate_table(16, "station_year_metrics_clustering_v0_full_k2.png", title_suffix=" (Clustering V0 Full K=2)")
