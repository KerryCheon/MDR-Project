"""Builds the reproducible derived_8.4-ece-error-analysis.ipynb notebook."""

import json
from pathlib import Path
import uuid

EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = EXP_DIR / "derived_8.4-ece-error-analysis.ipynb"


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
    cells.append(make_cell("markdown", r"""# Diagnostic Report: `derived_8.4-ece-error-analysis`
## Comprehensive Investigation into In-Situ ECE Sensor Evaluation Performance

### Executive Context
This diagnostic report investigates and explains the causes of severe negative $R^2$ performance observed across models evaluated on the **5 in-situ ECE soil moisture sensor stations** (`derived_8.4-ece`, 150 daily observations recorded between July 20 and August 19, 2026 across Bellevue and Renton, Washington).

While models trained on the 7 Washington state reference stations achieve state-of-the-art in-distribution accuracy ($R^2 = 0.8126$, $\text{RMSE} = 0.0441\text{ m}^3/\text{m}^3$), evaluation on the in-situ sensors yielded extreme negative $R^2$ scores ($-0.24$ to $-6,724$). This investigation provides mathematical, physical, and hardware-level explanations showing that the models maintain respectable physical error ($\text{RMSE} \approx 0.048\text{ m}^3/\text{m}^3$, better than out-of-state transfer), but $R^2$ collapses due to a severe variance compression paradox combined with missing satellite data and hyper-local micro-habitat scale mismatch."""))

    # Cell 1: Setup & Imports
    cells.append(make_cell("markdown", r"""## 1. Setup, Environment & Imports
Loads required scientific libraries, verifies directory structure, and initializes diagnostic report engine."""))

    cells.append(make_cell("code", r"""import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from IPython.display import Image, display

# Experiment root
EXP_DIR = Path.cwd()
if not (EXP_DIR / "config.yaml").exists():
    # If run from notebooks/ root
    EXP_DIR = Path.cwd() / "experiment" / "derived_8.4-ece-error-analysis"

TABLES_DIR = EXP_DIR / "tables"
FIGURES_DIR = EXP_DIR / "figures"

with open(EXP_DIR / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(f"Loaded Diagnostic Report Configuration: {config['experiment_name']}")
print(f"Tables Directory: {TABLES_DIR}")
print(f"Figures Directory: {FIGURES_DIR}")
"""))

    # Cell 2: Section 1 Narrative - Variance Compression
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

    # Cell 3: Section 2 - Historical Benchmarks
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

    # Cell 4: Section 3 - Missing Satellite Data Audit
    cells.append(make_cell("markdown", r"""## 4. Data Quality & Missingness Audit for 2026 Recency Gap

Because July–August 2026 is the most recent dataset ever processed in the project, latency in satellite data products created two severe data gaps in Google Earth Engine:
1. **SMAP L3/L4 Surface Soil Moisture**: Completely missing ($30/30$ NaNs per station) in GEE image collections, resulting in all 85 derived SMAP features defaulting to `0.0`. In training, SMAP has mean $\approx 0.343\text{ m}^3/\text{m}^3$ (range $0.07 - 0.68$).
2. **MODIS 250m NDVI (`NDVI_modis`)**: Completely missing ($30/30$ NaNs per station) due to 16-day compositing delay, defaulting to `0.0`.

These features represent over $20\%$ of model split decisions in baseline architectures, forcing decision tree traversals down unvisited out-of-range leaf paths."""))

    cells.append(make_cell("code", r"""t3 = pd.read_csv(TABLES_DIR / "table3_missing_data_audit.csv")
print("=== TABLE 3: SATELLITE & WEATHER PRODUCT AUDIT (2026 vs TRAINING) ===")
display(t3)
display(Image(filename=str(FIGURES_DIR / "fig2_smap_ndvi_missingness_distributions.png")))
"""))

    # Cell 5: Section 4 - Spatial Scale Mismatch
    cells.append(make_cell("markdown", r"""## 5. Spatial Scale Mismatch & Empirical 5-Station Side-by-Side Comparisons

Macro-scale remote sensing and weather grids operate at resolutions of $1\text{ km}$ (MODIS), $11\text{ km}$ (Open-Meteo ERA5), and $9–36\text{ km}$ (SMAP).
In contrast, the in-situ ECE sensors are deployed within sub-meter micro-habitats:
- `ECE_Renton_Garden_North` vs `ECE_Renton_Garden_Shed` are only **53.4 meters apart**.
- Their macro-weather and satellite reflectance inputs are virtually $100\%$ identical.
- Yet ground truth moisture is **$15.49\%$ (North)** vs **$7.58\%$ (Shed)** — a **$2.04\times$ divergence** caused by unmodelled tree shade, rich garden compost, and roof eaves rain shadowing."""))

    cells.append(make_cell("code", r"""t4 = pd.read_csv(TABLES_DIR / "table4_spatial_proximity_inputs.csv", index_col=0)
t4b = pd.read_csv(TABLES_DIR / "table4b_side_by_side_sensor_pairs.csv")
print("=== TABLE 4: PAIRWISE GEOGRAPHIC DISTANCE MATRIX (KM) ===")
display(t4)
print("\n=== TABLE 4B: EMPIRICAL SIDE-BY-SIDE FEATURE COMPARISONS ACROSS ALL 5 STATIONS ===")
display(t4b)
display(Image(filename=str(FIGURES_DIR / "fig3_spatial_microclimate_discrepancy.png")))
"""))

    # Cell 6: Section 5 - Per-Station 30-Day Time Series
    cells.append(make_cell("markdown", r"""## 6. Per-Station 30-Day Ground Truth vs Model Prediction Time Series

Daily time series overlays comparing ground truth measurements against multi-seed average predictions from major model architectures across July 20 to August 19, 2026.
Because dynamic inputs are low/missing, models predict a near-constant baseline ($\sim 0.10 - 0.16\text{ m}^3/\text{m}^3$), which aligns well with Renton Garden North ($15.5\%$) but creates substantial systematic positive bias at dry sites ($1.8\% - 5.8\%$)."""))

    cells.append(make_cell("code", r"""display(Image(filename=str(FIGURES_DIR / "fig8_per_station_timeseries_overlay.png")))
"""))

    # Cell 7: Section 5b - Day-30 Warmup Drop
    cells.append(make_cell("markdown", r"""## 7. Analysis of the Final-Day (August 19) Prediction Drop

In the 30-day time-series plots, all models exhibit a sharp drop in predicted soil moisture on the final day (`2026-08-19`) from $\sim 0.11 - 0.12\text{ m}^3/\text{m}^3$ down to $\sim 0.034 - 0.068\text{ m}^3/\text{m}^3$.

### Root Cause: 30-Day Rolling Window Warmup Boundary Transition
1. **The Rolling Window Lag**: The ECE dataset begins on July 20, 2026 without prior historical buffer days. Consequently, for Days 1 through 29 (July 20 to August 18), all 30-day rolling window features (`V_rollmin_G_API_kobs30`, `V_rollmean_G_API_kobs30`, `V_rollmin_LST_modis_kobs30`, `SMAP_sm_am_interp_rollmean30`) evaluate to `NaN`.
2. **XGBoost Missing-Value Routing**: During Days 1–29, tree traversals hit splits on these 30-day rolling features and take the default `missing_value` branch, outputting the general summer median value ($\sim 0.11 - 0.13\text{ m}^3/\text{m}^3$).
3. **Day-30 Transition (NaN -> 0.000)**: On Day 30 (`2026-08-19`), exactly 30 daily observations have accumulated. Features like `V_rollmin_G_API_kobs30` transition from `NaN` to `0.000000` for the first time.
4. **Massive Feature Importance Activation**: Because `V_rollmin_G_API_kobs30` accounts for **$23.9\%$ of total split gain** in `d84_weighted` (and `V_rollmin_LST_modis_kobs30` accounts for **$27.8\%$** in `d80_weighted`), the sudden presence of a valid numeric value ($0.0$) satisfies the condition `V_rollmin_G_API_kobs30 <= threshold`, actively routing tree traversals to the extreme dry terminal leaf node.

This confirms that the drop is an artifact of the 30-day feature pipeline warmup boundary rather than a real meteorological event."""))

    # Cell 8: Section 5c - Cross-Station Homogeneity & Coincidental Accuracy
    cells.append(make_cell("markdown", r"""## 8. Cross-Station Prediction Homogeneity & The "Coincidental Accuracy" Proof

### Hypothesis:
The models are essentially predicting a single station-agnostic regional response curve across all 5 deployment sites. Stations with lower error (e.g. `ECE_Renton_Garden_North`) perform better not because the model possesses genuine site-specific sensitivity, but purely because that station's actual moisture level happens to coincide with the model's regional fallback level ($\sim 0.13\text{ m}^3/\text{m}^3$).

### Quantitative Proof:
1. **Cross-Station Prediction Correlation $r \ge 0.960$**:
   - Between `ECE_Renton_Garden_North` and `ECE_Renton_Garden_Shed` (53m apart), prediction correlation is **$r = 0.999998$** with a mean absolute difference of **$0.000016\text{ m}^3/\text{m}^3$** ($0.0016\%$).
   - Between Renton and Bellevue (13km apart), prediction correlation is **$r = 0.960 - 0.975$** with a pairwise difference $< 0.008\text{ m}^3/\text{m}^3$.
2. **Error Dictated Strictly by Fallback Distance**:
   - At `ECE_Renton_Garden_North`, where ground truth happens to be $\bar{y} = 0.155\text{ m}^3/\text{m}^3$, distance from fallback is only $|0.155 - 0.131| = 0.024\text{ m}^3/\text{m}^3 \implies \text{RMSE} = 0.037\text{ m}^3/\text{m}^3$, $R^2 \approx -0.98$.
   - At `ECE_Renton_Home`, ground truth is $\bar{y} = 0.018\text{ m}^3/\text{m}^3 \implies |0.018 - 0.129| = 0.111\text{ m}^3/\text{m}^3 \implies \text{RMSE} = 0.113\text{ m}^3/\text{m}^3$, $R^2 \approx -1985$.
   - Observed station RMSE correlates $1:1$ with distance to the global fallback level ($R^2_{\text{dist}} > 0.99$)."""))

    cells.append(make_cell("code", r"""t9 = pd.read_csv(TABLES_DIR / "table9_coincidental_accuracy_proof.csv")
print("=== TABLE 9: COINCIDENTAL ACCURACY PROOF ACROSS 5 STATIONS ===")
display(t9)
display(Image(filename=str(FIGURES_DIR / "fig9_coincidental_accuracy_analysis.png")))
"""))

    # Cell 9: Section 6 - Target Distribution & Domain Shift
    cells.append(make_cell("markdown", r"""## 9. Target Distribution & Climatological Domain Shift

The 7 Washington reference stations (SNOTEL/SCAN) are located in high-elevation montane/forest environments (e.g. CayusePass, Paradise, Darrington) with high annual rainfall ($1,000 - 3,500\text{ mm}$) and deep organic soil layers that retain moisture ($\mu = 0.217\text{ m}^3/\text{m}^3$).
The ECE deployment sites represent Puget Sound lowland residential turf, manicured garden beds, and built-up areas that dry down rapidly to $1.8\% - 5.8\%$ in Mediterranean summer."""))

    cells.append(make_cell("code", r"""t5 = pd.read_csv(TABLES_DIR / "table5_target_climatology_shift.csv")
print("=== TABLE 5: TARGET CLIMATOLOGY & HYDROCLIMATIC PROFILES ===")
display(t5)
display(Image(filename=str(FIGURES_DIR / "fig4_target_distribution_domain_shift.png")))
"""))

    # Cell 10: Section 7 - Routing Strategies & MoE Traps
    cells.append(make_cell("markdown", r"""## 10. Routing Strategy Comparison & MoE Failure Modes

Evaluating the 8 routing architectures reveals why static KMeans clustering fails catastrophically under out-of-distribution spatial transfer:
- `Clustering_V0_Full_k2` and `Clustering_Backbone54_k2` route `ECE_Renton_Home` and `ECE_BBG_Lost_Meadow` to Cluster 1 (the wet mountain regime trained on Paradise/CayusePass). The wet expert predicts $0.22 - 0.25\text{ m}^3/\text{m}^3$ for a site at $0.018\text{ m}^3/\text{m}^3$, driving $R^2$ to $-6,724$.
- Dynamic routers (`Clustering_Dynamic_k2`, `Univariate_G_API_k2`, `Seasonal_Binary_k2`) route $100\%$ of summer samples to the dry regime (Cluster 0), avoiding static misrouting and achieving $\text{RMSE} = 0.0479\text{ m}^3/\text{m}^3$."""))

    cells.append(make_cell("code", r"""t6 = pd.read_csv(TABLES_DIR / "table6_routing_strategy_breakdown.csv")
print("=== TABLE 6: ROUTING STRATEGY COMPARISON ACROSS 5 ECE SENSORS ===")
display(t6)
display(Image(filename=str(FIGURES_DIR / "fig5_routing_strategy_ece_comparison.png")))
"""))

    # Cell 11: Section 8 - Sensor Hardware & Calibration
    cells.append(make_cell("markdown", r"""## 11. Sensor Hardware, Calibration & Negative Value Clarification

Analysis of raw sensor recordings and ADC counts:
1. **Negative Value Clarification**: Raw soil moisture percentages are strictly non-negative ($\ge 0.0\%$). Negative values in results represent: (1) negative $R^2$ scores, (2) negative Pearson correlation ($r = -0.33$ to $-0.68$), and (3) negative bias at specific models.
2. **ADC Zero-Drift**: Device 11 (`ECE_Renton_Home`) bottoms out at $0.00\%$ ($1.78\%$ mean), indicating potential probe air gaps or uncalibrated zero-point offset for compacted residential turf."""))

    cells.append(make_cell("code", r"""t7 = pd.read_csv(TABLES_DIR / "table7_raw_adc_sensor_calibration.csv")
print("=== TABLE 7: RAW ADC COUNTS & SENSOR CALIBRATION SUMMARY ===")
display(t7)
display(Image(filename=str(FIGURES_DIR / "fig6_raw_adc_to_moisture_calibration.png")))
"""))

    # Cell 12: Section 9 - Error Decomposition Waterfall
    cells.append(make_cell("markdown", r"""## 12. Error Decomposition Synthesis

Synthesis of error contributions illustrating how a baseline physical error ($\text{ubRMSE} \approx 0.048\text{ m}^3/\text{m}^3$) combined with siting bias ($+0.06\text{ m}^3/\text{m}^3$) and missing satellite data produces large negative $R^2$ when divided by near-zero ground truth variance."""))

    cells.append(make_cell("code", r"""display(Image(filename=str(FIGURES_DIR / "fig7_error_decomposition_waterfall.png")))
"""))

    # Cell 13: Section 10 - Actionable Recommendations
    cells.append(make_cell("markdown", r"""## 13. Actionable Recommendations & Future Roadmap

Specific protocols and recommendations tailored for the ECE Hardware Engineering Team and the ML / Modeling Research Team."""))

    cells.append(make_cell("code", r"""t8 = pd.read_csv(TABLES_DIR / "table8_recommendations_matrix.csv")
print("=== TABLE 8: ACTIONABLE RECOMMENDATIONS MATRIX ===")
display(t8)
"""))

    # Write notebook
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)

    print(f"Successfully generated notebook at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
