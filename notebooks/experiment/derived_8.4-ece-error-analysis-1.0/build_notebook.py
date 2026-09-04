"""Builds the reproducible derived_8.4-ece-error-analysis-1.0.ipynb notebook."""

import json
from pathlib import Path
import uuid

EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = EXP_DIR / "derived_8.4-ece-error-analysis-1.0.ipynb"


def make_cell(cell_type: str, source: str) -> dict:
    """Create a standard nbformat v4.5 notebook cell."""
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": cell_type,
        "id": str(uuid.uuid4()),
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main():
    cells = []

    # Cell 0: Title & Executive Context
    cells.append(make_cell("markdown", r"""# Diagnostic Report: `derived_8.4-ece-error-analysis-1.0`
## Comprehensive Investigation into In-Situ ECE Sensor Evaluation Performance

### Executive Context
This diagnostic report investigates and explains the causes of severe negative $R^2$ performance observed across models evaluated on the **5 in-situ ECE soil moisture sensor stations** (`derived_8.4-ece`, 150 daily observations recorded between July 20 and August 19, 2026 across Bellevue and Renton, Washington).

While models trained on the 7 Washington state reference stations achieve state-of-the-art in-distribution accuracy ($R^2 = 0.8126$, $\text{RMSE} = 0.0441\text{ m}^3/\text{m}^3$), evaluation on the in-situ sensors yielded extreme negative $R^2$ scores ($-0.24$ to $-6,724$). This investigation provides mathematical, physical, and hardware-level explanations showing that the models maintain respectable physical error ($\text{RMSE} \approx 0.048\text{ m}^3/\text{m}^3$, better than out-of-state transfer), but $R^2$ collapses due to a severe variance compression paradox combined with missing satellite data and hyper-local micro-habitat scale mismatch."""))

    # Cell 1: Setup & Imports
    cells.append(make_cell("markdown", r"""## 1. Setup, Environment & Imports
Loads required scientific libraries, verifies directory structure, and initializes diagnostic report engine."""))

    cells.append(make_cell("code", r"""import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from IPython.display import Image, display

# Find project root robustly
cur = Path.cwd().resolve()
while cur != cur.parent:
    if (cur / "data" / "splits").exists() and (cur / "notebooks").exists():
        PROJECT_ROOT = cur
        break
    cur = cur.parent

EXP_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.4-ece-error-analysis-1.0"
TABLES_DIR = EXP_DIR / "tables"
FIGURES_DIR = EXP_DIR / "figures"

with open(EXP_DIR / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Loaded Diagnostic Report Configuration: {config['experiment_name']}")
print(f"Tables Directory: {TABLES_DIR}")
print(f"Figures Directory: {FIGURES_DIR}")
"""))

    # Cell 2: Section 2 - Variance Compression
    cells.append(make_cell("markdown", r"""## 2. Mathematical Anatomy of Negative $R^2$ (The Variance Compression Paradox)

The Nash-Sutcliffe Efficiency ($R^2$) is defined as:
$$R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)} = 1 - \frac{\text{Bias}^2 + \text{Var}(\hat{y}) - 2\text{Cov}(y, \hat{y})}{\text{Var}(y)}$$

During the 30-day ECE deployment window in late summer (July 20 – August 19, 2026), Western Washington experienced dry Mediterranean summer conditions with near-zero precipitation. Consequently, ground-truth soil moisture was statically low and flat:
- Standard deviation $\sigma_y \in [0.0025, 0.0078]\text{ m}^3/\text{m}^3$ across 4 of the 5 stations.
- Ground truth variance $\text{Var}(y) \in [6\times 10^{-6}, 6\times 10^{-5}]\text{ m}^3/\text{m}^3$.

Even a modest physical prediction error ($\text{RMSE} \approx 0.048 - 0.08\text{ m}^3/\text{m}^3$) yields $\text{MSE} \approx 0.0023 - 0.0064$. Dividing by $6\times 10^{-6}$ produces $\frac{\text{MSE}}{\text{Var}(y)} \approx 400 - 1,100$, driving $R^2$ to $-400$ to $-1,100$ by purely mathematical necessity."""))

    cells.append(make_cell("code", r"""t1 = pd.read_csv(TABLES_DIR / "table1_variance_compression_r2.csv")
print("=== TABLE 1: VARIANCE COMPRESSION & ERROR METRICS PER STATION ===")
display(t1)
display(Image(filename=str(FIGURES_DIR / "fig1_r2_variance_compression_anatomy.png")))
"""))

    # Cell 2b: Section 2.2 - Direct Variance Comparison: 5 ECE Sensors vs 7 WA Training Stations (Test Period)
    cells.append(make_cell("markdown", r"""### 2.2 Direct Variance & Standard Deviation Comparison: ECE Sensors vs. WA Reference Test Period

To empirically prove that the negative $R^2$ collapse on ECE is driven by target variance compression, Table 1b and Figure 1b contrast the target soil moisture distribution ($\text{soil\_moisture\_5cm}$) of the 5 in-situ ECE sensor stations (150 obs total, 30 per station with 2026-08-01 missing, July 20 – August 19, 2026) against the 7 Washington training reference stations during their official test period (6,620 obs, 2023–2025). Table 1b also includes a season-matched WA summer subset (547 obs, Jul 20 – Aug 19 across 2023–2025) as the like-for-like comparator:

- **Standard Deviation Collapse**: WA reference stations maintain a mean per-station standard deviation of $\sigma = 0.0953\text{ m}^3/\text{m}^3$ ($0.0694$ to $0.1195$), while ECE sensors average $\sigma = 0.0094\text{ m}^3/\text{m}^3$ (down to $0.0025\text{ m}^3/\text{m}^3$ at `ECE_Renton_Home`) — **~10× on average, up to 47× for the most extreme station pair** (mean-to-min $37.6×$, max-to-min $47.1×$).
- **Target Variance Ratios (do not mix estimators)**: the mean of per-station sample variances is $9.36\times 10^{-3}\text{ (m}^3/\text{m}^3)^2$ (WA, $N=6{,}620$ total) vs $1.63\times 10^{-4}\text{ (m}^3/\text{m}^3)^2$ (ECE, $N=150$ total), i.e. **57× mean-of-variances and 1,456× vs `ECE_Renton_Home`** ($6.43\times 10^{-6}$). These compare full-year WA (wet-winter variance included) against summer-drought ECE. The pooled variances — the estimator matching the pooled $N$ — are $0.0103786$ (WA) vs $0.0022285$ (ECE), only **4.66×**; season-matched pooled WA summer ($0.0025261$) vs pooled ECE ($0.0022285$) is only **1.13×**. Between-station mean differences inflate pooled variance, so never pair pooled $N$ with a mean-of-variances.
- **The Mathematical Per-Station $R^2$ Penalty**: if a model achieves an identical, highly respectable physical prediction error ($\text{RMSE} = 0.040\text{ m}^3/\text{m}^3$), the resulting Nash-Sutcliffe Efficiency on the reference test stations is $R^2 \in [+0.668, +0.888]$ (strong positive skill). On the 5 ECE stations individually, that exact same physical error produces per-station $R^2 \in [-256.53, -1.38]$ (population-variance form; sample-variance form gives $[-247.94, -1.30]$) purely because each station's denominator $\text{Var}(y)$ is negligible. The pooled ECE set, by contrast, has theoretical $R^2 \approx +0.28$ at RMSE $0.04$ (WA summer pooled $\approx +0.37$) — the collapse is a per-station variance artifact, not a pooled-set failure. Theoretical $R^2$ uses population variance ($\text{ddof}=0$) since $R^2 = 1 - \text{SSE}/\text{SST}$ with $\text{SST} = N\cdot\text{Var}_{pop}$."""))

    cells.append(make_cell("code", r"""t1b = pd.read_csv(TABLES_DIR / "table1b_target_variance_ece_vs_wa_test.csv")
print("=== TABLE 1B: TARGET VARIANCE COMPARISON: 5 ECE SENSORS vs 7 WA REFERENCE TEST PERIOD ===")
display(t1b)
display(Image(filename=str(FIGURES_DIR / "fig1b_target_variance_ece_vs_wa_test_comparison.png")))
"""))

    # Cell 3: Section 3 - Historical Benchmarks
    cells.append(make_cell("markdown", r"""## 3. Historical Cross-Experiment Reference Benchmarks

To contextualize the ECE evaluation, we benchmark performance across three distinct operational domains:
1. **In-Distribution Temporal Evaluation** (`derived_8.4` test, 7 WA stations, 2023–2025).
2. **Out-of-State Spatial Transfer** (`derived_8.4-oos` test, 5 stations in OR/ID/CA, 2017–2025).
3. **In-Situ ECE Spatial Generalization** (`derived_8.4-ece` test, 5 micro-stations in WA, 2026).

Notice that for models like `Global_Single_54` and `Clustering_Dynamic_k2`, the **physical RMSE on ECE ($0.048 - 0.051\text{ m}^3/\text{m}^3$) is lower than Out-of-State spatial error ($0.062\text{ m}^3/\text{m}^3$)**, confirming that regional hydroclimatic predictions remain physically sound."""))

    cells.append(make_cell("code", r"""t2 = pd.read_csv(TABLES_DIR / "table2_historical_benchmark_ref.csv")
print("=== TABLE 2: CROSS-EXPERIMENT BENCHMARK (TEMPORAL vs OOS vs ECE) ===")
display(t2)
"""))

    # Cell 4: Section 4 - Missing Data Audit
    cells.append(make_cell("markdown", r"""## 4. Latent Data Quality Gap in 2026: Missing Satellite Products

An audit of the pipeline features reveals a significant data latency gap for recent 2026 observations:
- **SMAP L3/L4 Surface Soil Moisture**: 85 derived features are **100% missing (NaN imputed to 0.0)** because SMAP products had not yet been ingested into Google Earth Engine for July/August 2026.
- **MODIS 250m NDVI**: 12 derived features are **100% missing (imputed to 0.0)**.
- **Impact**: In tree-based models, defaulting 97 top-tier satellite features to `0.0` forces tree traversals down unvisited decision splits, creating an artificial dry-bias offset."""))

    cells.append(make_cell("code", r"""t3 = pd.read_csv(TABLES_DIR / "table3_missing_data_audit.csv")
print("=== TABLE 3: 2026 MISSING SATELLITE DATA PRODUCT AUDIT ===")
display(t3)
display(Image(filename=str(FIGURES_DIR / "fig2_smap_ndvi_missingness_distributions.png")))
"""))

    # Cell 5: Section 5 - Spatial Proximity & Table 4b
    cells.append(make_cell("markdown", r"""## 5. Spatial Scale Mismatch & Empirical 5-Station Side-by-Side Comparisons

Sensors deployed in close proximity receive nearly identical gridded satellite and weather inputs, but exhibit dramatically different in-situ soil moisture due to sub-meter siting micro-climates:
- `ECE_Renton_Garden_North` (15.5% mean) vs `ECE_Renton_Garden_Shed` (7.6% mean) are separated by **only 53.4 meters**.
- Gridded inputs (ERA5 rain, MODIS LST, SRTM elevation) are identical or near-identical, yet ground truth differs by **2.04×** due to localized shading and eaves rain shadowing.
- Table 4b details all 30 empirical features across all 5 deployment stations."""))

    cells.append(make_cell("code", r"""t4 = pd.read_csv(TABLES_DIR / "table4_spatial_proximity_inputs.csv", index_col=0)
t4b = pd.read_csv(TABLES_DIR / "table4b_side_by_side_sensor_pairs.csv")
print("=== TABLE 4: PAIRWISE GEODESIC DISTANCE MATRIX (KM) ===")
display(t4)
print("\n=== TABLE 4B: EMPIRICAL SIDE-BY-SIDE FEATURE COMPARISONS (ALL 5 STATIONS) ===")
display(t4b)
display(Image(filename=str(FIGURES_DIR / "fig3_spatial_microclimate_discrepancy.png")))
"""))

    # Cell 6: Section 6 - Time Series Overlays & Day 30 Drop
    cells.append(make_cell("markdown", r"""## 6. Per-Station 30-Day Observed vs Predicted Time Series & Anomaly Analysis

### 6.1 Time Series Overlays Across All 5 Stations
Line charts comparing actual ground-truth moisture against multi-seed predictions for `d84_weighted`, `d80_weighted`, and `d84_no_weights`.

### 6.2 Explanation for Final-Day (August 19) Prediction Drop
On the final day (`2026-08-19`), predicted moisture drops sharply across all stations from $\sim 0.11 - 0.12\text{ m}^3/\text{m}^3$ down to $\sim 0.034 - 0.068\text{ m}^3/\text{m}^3$.
- **Root Cause**: The ECE dataset starts on July 20 without historical warmup buffer. For Days 1–29, 30-day rolling window features (`V_rollmin_G_API_kobs30`, `V_rollmean_G_API_kobs30`) evaluate to `NaN` and XGBoost follows its default missing branch.
- **Day-30 Activation**: On Day 30, exactly 30 days accumulated, transitioning `V_rollmin_G_API_kobs30` from `NaN` to `0.000`. Because this single feature represents **23.9% of total feature importance** in `d84_weighted`, the activation of the numeric split `V_rollmin_G_API_kobs30 <= threshold` immediately routes predictions to the extreme dry terminal leaf node."""))

    cells.append(make_cell("code", r"""display(Image(filename=str(FIGURES_DIR / "fig8_per_station_timeseries_overlay.png")))
"""))

    # Cell 7: Section 7 - Coincidental Accuracy Proof
    cells.append(make_cell("markdown", r"""## 7. Cross-Station Prediction Homogeneity & "Coincidental Accuracy" Proof

### Hypothesis:
The models output a single station-agnostic regional response curve. Stations with lower prediction error (e.g. `ECE_Renton_Garden_North`) perform well purely because their actual moisture happens to coincide with the model's global fallback level ($\sim 0.13\text{ m}^3/\text{m}^3$).

### Proof:
1. **Prediction Correlation**: Cross-station prediction correlation is **$r \ge 0.960$** across all 5 stations (and $r = 0.999998$ between Renton Garden North and Shed).
2. **Error Linearity**: Observed station RMSE is strictly proportional to $|\bar{y}_{\text{true}} - \bar{\hat{y}}_{\text{fallback}}|$ ($R^2 > 0.99$), confirming 100% coincidental alignment at Renton Garden North."""))

    cells.append(make_cell("code", r"""t9 = pd.read_csv(TABLES_DIR / "table9_coincidental_accuracy_proof.csv")
print("=== TABLE 9: COINCIDENTAL ACCURACY PROOF ACROSS ALL 5 STATIONS ===")
display(t9)
display(Image(filename=str(FIGURES_DIR / "fig9_coincidental_accuracy_analysis.png")))
"""))

    # Cell 8: Section 8 - Target Climatology & Domain Shift
    cells.append(make_cell("markdown", r"""## 8. Target Climatology & Macro-Ecological Domain Shift

Comparison between the 7 high-elevation mountain SNOTEL stations in training and the 5 low-elevation urban/garden ECE sensor sites:
- **Elevation**: Training mean $= 890\text{ m}$ vs ECE mean $= 83\text{ m}$.
- **Precipitation**: Training mean $= 1,850\text{ mm/yr}$ vs ECE mean $= 1,130\text{ mm/yr}$.
- **Soil & Siting**: Undisturbed forest mineral soils vs residential turf and garden mulch."""))

    cells.append(make_cell("code", r"""t5 = pd.read_csv(TABLES_DIR / "table5_target_climatology_shift.csv")
print("=== TABLE 5: HYDROCLIMATIC PROFILE & DOMAIN SHIFT ===")
display(t5)
display(Image(filename=str(FIGURES_DIR / "fig4_target_distribution_domain_shift.png")))
"""))

    # Cell 9: Section 9 - Mixture of Experts Breakdown
    cells.append(make_cell("markdown", r"""## 9. Mixture-of-Experts (MoE) Routing Strategy Comparison

Evaluating 8 different routing paradigms on the ECE dataset:
- **Dynamic Routers** (`Univariate_G_API_k2`, `Clustering_Dynamic_k2`, `Seasonal_Binary_k2`): Successfully route 100% of samples into the dry summer regime, achieving optimal RMSE ($0.0479 - 0.0503\text{ m}^3/\text{m}^3$).
- **Static MoE Routing Trap** (`Clustering_V0_Full_k2`, `Clustering_Backbone54_k2`): Static geographic features dominate the KMeans clustering space, erroneously routing low-elevation dry residential lawns (`ECE_Renton_Home`) to the wet mountain expert ($R^2 = -6,724$, $\text{Bias} = +0.13\text{ m}^3/\text{m}^3$)."""))

    cells.append(make_cell("code", r"""t6 = pd.read_csv(TABLES_DIR / "table6_routing_strategy_breakdown.csv")
print("=== TABLE 6: MOE ROUTING STRATEGY COMPARISON ON ECE DATA ===")
display(t6)
display(Image(filename=str(FIGURES_DIR / "fig5_routing_strategy_ece_comparison.png")))
"""))

    # Cell 10: Section 10 - Soil Texture Benchmark & Counterfactual Sensitivity Test
    cells.append(make_cell("markdown", r"""## 10. Soil Texture Benchmark Across 12 Stations & Feature Override Sensitivity Analysis

### 10.1 Soil Texture Comparison Across All 12 Stations (Table 10)
We benchmark the soil types and percentage-based physical properties across all 7 Washington training reference stations and all 5 in-situ ECE stations:
- **Training Set Composition**: 5 stations are **Loam** (`Darrington`, `Paradise`, `Quinault`, `SourdoughGulch`, `Spokane` — 71.4% of training rows), and 2 stations are **Sandy Loam** (`BeaverPass`, `CayusePass` — 28.6% of training rows).
- **Encountered Soil Types**: The models have **definitely encountered both Loam and Sandy Loam** during training.

### 10.2 Counterfactual Feature Override Sensitivity Test (Table 11)
To determine whether manual feature override is worthwhile, we execute a counterfactual simulation across all 20 trained XGBoost models (5 seeds $\times$ 4 architectures), overriding the soil features for all 3 Sandy Loam stations (`ECE_BBG_Main_St`, `ECE_BBG_Lost_Meadow`, `ECE_Renton_Garden_Shed`) to $55\%$ Sand and $10\%$ Clay."""))

    cells.append(make_cell("code", r"""# Executable Counterfactual Sensitivity Test across all 20 models and seeds
cfg_path = PROJECT_ROOT / "notebooks/experiment/derived_8.4-ece-additional-eval-1.0/config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

features = cfg["feature_columns"]
ece_test = pd.read_csv(PROJECT_ROOT / "data/splits/derived_8.4-ece/test.csv")
model_dir = PROJECT_ROOT / "notebooks/experiment/derived_8.4-ece-additional-eval-1.0/models"
model_files = sorted(glob.glob(str(model_dir / "*.json")))

X_orig = ece_test[features].copy()
ece_over = ece_test.copy()
sandy_stations = ["ECE_BBG_Main_St", "ECE_BBG_Lost_Meadow", "ECE_Renton_Garden_Shed"]
mask = ece_over["station_id"].isin(sandy_stations)
ece_over.loc[mask, "J_sand_wfrac_b0"] = 55
ece_over.loc[mask, "J_clay_wfrac_b0"] = 10
X_over = ece_over[features].copy()

sim_results = []
for mf in model_files:
    mname = Path(mf).stem
    arch, seed = mname.split("__")[0], mname.split("__")[-1].replace("s", "")
    bst = xgb.Booster()
    bst.load_model(mf)
    p_orig = bst.predict(xgb.DMatrix(X_orig))
    p_over = bst.predict(xgb.DMatrix(X_over))
    diff = p_over - p_orig
    sim_results.append({
        "model_architecture": arch,
        "seed": int(seed),
        "mean_orig_pred": np.mean(p_orig),
        "mean_over_pred": np.mean(p_over),
        "mean_abs_diff": np.mean(np.abs(diff)),
        "max_abs_diff": np.max(np.abs(diff)),
        "diff_sandy_stations": np.mean(diff[mask]),
    })

sim_df = pd.DataFrame(sim_results)
print("=== COUNTERFACTUAL OVERRIDE SENSITIVITY TEST SUMMARY ===")
print(f"Total Models Evaluated: {len(sim_df)}")
print(f"Mean Prediction Shift Across Ensemble: {sim_df['mean_abs_diff'].mean():.6f} m3/m3 ({sim_df['mean_abs_diff'].mean()*100:.4f}%)")
print(f"Max Prediction Shift on Any Sample:     {sim_df['max_abs_diff'].max():.6f} m3/m3 ({sim_df['max_abs_diff'].max()*100:.4f}%)")
print(f"Mean Shift on Sandy Loam Stations:     {sim_df['diff_sandy_stations'].mean():+.6f} m3/m3 ({sim_df['diff_sandy_stations'].mean()*100:+.4f}%)")

t10 = pd.read_csv(TABLES_DIR / "table10_soil_texture_all_stations.csv")
t11 = pd.read_csv(TABLES_DIR / "table11_soil_override_sensitivity.csv")
print("\n=== TABLE 10: SOIL TEXTURE BENCHMARK ACROSS ALL 12 STATIONS ===")
display(t10)
print("\n=== TABLE 11: COUNTERFACTUAL OVERRIDE SENSITIVITY BREAKDOWN ===")
display(t11)
"""))

    # Cell 11: Section 11 - Sensor Hardware & Calibration
    cells.append(make_cell("markdown", r"""## 11. Sensor Hardware, Calibration & Negative Value Clarification

Analysis of raw sensor recordings and ADC counts:
1. **Negative Value Clarification**: Raw soil moisture percentages are strictly non-negative ($\ge 0.0\%$). Negative values in results represent: (1) negative $R^2$ scores, (2) negative Pearson correlation ($r = -0.33$ to $-0.68$), and (3) negative bias at specific models.
2. **ADC Zero-Drift**: Device 11 (`ECE_Renton_Home`) bottoms out at $0.00\%$ ($1.78\%$ mean), indicating potential probe air gaps or uncalibrated zero-point offset for compacted residential turf."""))

    cells.append(make_cell("code", r"""t7 = pd.read_csv(TABLES_DIR / "table7_raw_adc_sensor_calibration.csv")
print("=== TABLE 7: RAW ADC COUNTS & SENSOR CALIBRATION SUMMARY ===")
display(t7)
display(Image(filename=str(FIGURES_DIR / "fig6_raw_adc_to_moisture_calibration.png")))
"""))

    # Cell 12: Section 12 - Error Decomposition Waterfall
    cells.append(make_cell("markdown", r"""## 12. Error Decomposition Synthesis

Synthesis of error contributions illustrating how a baseline physical error ($\text{ubRMSE} \approx 0.048\text{ m}^3/\text{m}^3$) combined with siting bias ($+0.06\text{ m}^3/\text{m}^3$) and missing satellite data produces large negative $R^2$ when divided by near-zero ground truth variance."""))

    cells.append(make_cell("code", r"""display(Image(filename=str(FIGURES_DIR / "fig7_error_decomposition_waterfall.png")))
"""))

    # Cell 13: Section 13 - Recommendations Matrix
    cells.append(make_cell("markdown", r"""## 13. Actionable Recommendations & Future Roadmap

Specific protocols and recommendations tailored for the ECE Hardware Engineering Team and the ML / Modeling Research Team."""))

    cells.append(make_cell("code", r"""t8 = pd.read_csv(TABLES_DIR / "table8_recommendations_matrix.csv")
print("=== TABLE 8: ACTIONABLE RECOMMENDATIONS MATRIX ===")
display(t8)
"""))

    # Assemble notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Successfully generated notebook at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
