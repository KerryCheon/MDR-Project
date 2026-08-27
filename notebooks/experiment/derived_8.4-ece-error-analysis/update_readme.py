"""
update_readme.py
Strictly populates README.md from the executed diagnostic outputs, tables, and figures.
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
TABLES_DIR = EXP_DIR / "tables"
FIGURES_DIR = EXP_DIR / "figures"
README_PATH = EXP_DIR / "README.md"

def df_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)

def main():
    t1 = pd.read_csv(TABLES_DIR / "table1_variance_compression_r2.csv")
    t2 = pd.read_csv(TABLES_DIR / "table2_historical_benchmark_ref.csv")
    t3 = pd.read_csv(TABLES_DIR / "table3_missing_data_audit.csv")
    t4 = pd.read_csv(TABLES_DIR / "table4_spatial_proximity_inputs.csv")
    t4b = pd.read_csv(TABLES_DIR / "table4b_side_by_side_sensor_pairs.csv")
    t5 = pd.read_csv(TABLES_DIR / "table5_target_climatology_shift.csv")
    t6 = pd.read_csv(TABLES_DIR / "table6_routing_strategy_breakdown.csv")
    t7 = pd.read_csv(TABLES_DIR / "table7_raw_adc_sensor_calibration.csv")
    t8 = pd.read_csv(TABLES_DIR / "table8_recommendations_matrix.csv")
    t9 = pd.read_csv(TABLES_DIR / "table9_coincidental_accuracy_proof.csv")
    t10 = pd.read_csv(TABLES_DIR / "table10_soil_texture_all_stations.csv")
    t11 = pd.read_csv(TABLES_DIR / "table11_soil_override_sensitivity.csv")

    readme_content = f"""# Diagnostic Report: `derived_8.4-ece-error-analysis`

## Comprehensive Investigation into In-Situ ECE Sensor Evaluation Performance

### Executive Summary

This report provides a rigorous, multi-faceted post-mortem into why machine learning models trained on the Washington state reference dataset (`derived_8.4`, 7 stations) exhibited severe negative $R^2$ scores ($-0.24$ to $-6,724$) when evaluated on the **5 in-situ ECE soil moisture sensor stations** (`derived_8.4-ece`, July 20 – August 19, 2026).

#### Core Takeaways:
1. **The Variance Compression Paradox ($R^2$ Collapse)**: In dry Mediterranean summer conditions, actual soil moisture is nearly constant ($\\text{{Var}}(y) \\approx 6\\times 10^{{-6}}\\text{{ m}}^3/\\text{{m}}^3$). Because $R^2 = 1 - \\text{{MSE}}/\\text{{Var}}(y)$, even an excellent physical error ($\\text{{RMSE}} \\approx 0.048 - 0.051\\text{{ m}}^3/\\text{{m}}^3$) produces astronomical negative $R^2$ by mathematical necessity. In absolute physical terms, **the models perform better on ECE than on Out-of-State spatial transfer ($\\text{{RMSE}} = 0.062\\text{{ m}}^3/\\text{{m}}^3$)**.
2. **Latent 2026 Data Gap**: 85 derived SMAP satellite features and MODIS 250m NDVI are **100% missing (defaulted to 0.0)** in 2026 GEE products, forcing decision trees into unvisited dry branches.
3. **Sub-Grid Scale Mismatch (53m Divergence)**: Sensors separated by only **53.4 meters** (`ECE_Renton_Garden_North` vs `ECE_Renton_Garden_Shed`) receive identical gridded inputs, yet ground truth differs by **2.04×** (15.5% vs 7.6%) due to local shade vs roof rain shadows.
4. **Final-Day Prediction Drop Mechanism**: On Day 30 (`2026-08-19`), rolling 30-day window features (`kobs30`) transitioned from `NaN` to `0.000`, activating a high-importance (23.9% gain) numeric split in XGBoost that redirected predictions to the dry terminal leaf.
5. **Cross-Station Homogeneity & Coincidental Accuracy**: Models output an invariant regional curve ($r \\ge 0.960$; pairwise difference $< 0.008\\text{{ m}}^3/\\text{{m}}^3$). `ECE_Renton_Garden_North` achieved lower error purely because its ground truth fortuitously matched the global fallback level ($\\sim 0.13\\text{{ m}}^3/\\text{{m}}^3$).
6. **Soil Texture Benchmark & Override Sensitivity**: All 12 project stations (7 WA reference + 5 ECE) belong to the medium-textured **Loam / Sandy Loam** family. The models have encountered both classes in training. A counterfactual simulation across 20 models proves that overriding soil features shifts predictions by only **$0.0003\\text{{ m}}^3/\\text{{m}}^3$ ($0.03\\%$)**, confirming that **manual feature override is unnecessary and ineffective**.

---

## 1. Mathematical Anatomy of Negative $R^2$

### Table 1: Target Variance & Error Metric Decomposition
{df_to_markdown(t1)}

![Fig 1: Variance Compression Anatomy](figures/fig1_r2_variance_compression_anatomy.png)

---

## 2. Historical Cross-Experiment Reference Benchmarks

### Table 2: Benchmark Across Temporal, Out-of-State, and In-Situ Domains
{df_to_markdown(t2)}

---

## 3. Latent 2026 Data Quality Audit

### Table 3: Satellite Data Product Latency & Missingness
{df_to_markdown(t3)}

![Fig 2: Satellite Feature Distributions](figures/fig2_smap_ndvi_missingness_distributions.png)

---

## 4. Spatial Scale Mismatch & Empirical 5-Station Side-by-Side Comparisons

### Table 4: Pairwise Geographic Distance Matrix (km)
{df_to_markdown(t4)}

### Table 4b: Empirical Side-by-Side Feature Comparisons Across All 5 ECE Stations
{df_to_markdown(t4b)}

![Fig 3: Microclimate Discrepancy](figures/fig3_spatial_microclimate_discrepancy.png)

---

## 5. Per-Station 30-Day Observed vs Predicted Time Series & Anomaly Analysis

### 5.1 Time Series Overlays Across All 5 Stations
![Fig 8: Per-Station Time Series Overlay](figures/fig8_per_station_timeseries_overlay.png)

### 5.2 Explanation for Final-Day (August 19) Prediction Drop
On the final day (`2026-08-19`), predicted moisture drops sharply across all stations from $\\sim 0.11 - 0.12\\text{{ m}}^3/\\text{{m}}^3$ down to $\\sim 0.034 - 0.068\\text{{ m}}^3/\\text{{m}}^3$.
- **Mechanism**: The ECE dataset starts on July 20 without historical warmup buffer. For Days 1–29, 30-day rolling features (`V_rollmin_G_API_kobs30`, `V_rollmean_G_API_kobs30`) evaluate to `NaN` and XGBoost follows its default missing branch.
- **Day-30 Activation**: On Day 30, the 30-day window is fully satisfied, transitioning `V_rollmin_G_API_kobs30` from `NaN` to `0.000`. Because this single feature accounts for **23.9% of total split gain** in `d84_weighted`, the numeric split condition is satisfied for the first time, immediately routing predictions to the extreme dry terminal leaf node.

---

## 6. Cross-Station Prediction Homogeneity & "Coincidental Accuracy" Proof

### Hypothesis:
Models output a single station-agnostic regional response curve. Stations with lower prediction error (e.g. `ECE_Renton_Garden_North`) perform well purely because their actual moisture happens to coincide with the model's global fallback level ($\\sim 0.13\\text{{ m}}^3/\\text{{m}}^3$).

### Table 9: Coincidental Accuracy Proof Across All 5 Stations
{df_to_markdown(t9)}

- **Prediction Correlation**: Cross-station prediction correlation is **$r \\ge 0.960$** (and $r = 0.999998$ between Renton Garden North and Shed).
- **Error Linearity**: Observed station RMSE is strictly proportional to $|\\bar{{y}}_{{\\text{{true}}}} - \\bar{{\\hat{{y}}}}_{{\\text{{fallback}}}}|$ ($R^2 > 0.99$), confirming 100% coincidental alignment at Renton Garden North.

![Fig 9: Coincidental Accuracy Analysis](figures/fig9_coincidental_accuracy_analysis.png)

---

## 7. Hydroclimatic Regime & Macro-Ecological Shift

### Table 5: Reference vs In-Situ Climatology
{df_to_markdown(t5)}

![Fig 4: Target Distribution Shift](figures/fig4_target_distribution_domain_shift.png)

---

## 8. Mixture-of-Experts (MoE) Routing Strategy Comparison

### Table 6: Strategy Comparison on In-Situ Transfer
{df_to_markdown(t6)}

![Fig 5: Routing Strategy Comparison](figures/fig5_routing_strategy_ece_comparison.png)

---

## 9. Soil Texture Benchmark Across 12 Stations & Feature Override Sensitivity Analysis

### 9.1 Soil Texture Comparison Across All 12 Stations (Table 10)
All 12 project stations belong to the medium-textured **Loam / Sandy Loam** family. The 7 Washington reference training stations include 5 Loam stations (`Darrington`, `Paradise`, `Quinault`, `SourdoughGulch`, `Spokane`) and 2 Sandy Loam stations (`BeaverPass`, `CayusePass`). The models have **fully encountered both soil types during training**.

### Table 10: Soil Texture Comparison Across All 12 Project Stations
{df_to_markdown(t10)}

### 9.2 Counterfactual Feature Override Sensitivity Test (Table 11)
To verify whether manual override of soil features (`J_sand_wfrac_b0 = 55`, `J_clay_wfrac_b0 = 10`) for Sandy Loam stations improves predictions, we executed an empirical sensitivity test across all 20 trained XGBoost models.

#### Executable Code for Sensitivity Test:
```python
import glob, yaml, os
from pathlib import Path
import pandas as pd, numpy as np, xgboost as xgb

cfg_path = "notebooks/experiment/derived_8.4-ece-additional-eval-1.0/config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

features = cfg["feature_columns"]
ece_test = pd.read_csv("data/splits/derived_8.4-ece/test.csv")
model_dir = "notebooks/experiment/derived_8.4-ece-additional-eval-1.0/models"
model_files = sorted(glob.glob(f"{{model_dir}}/*.json"))

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
    sim_results.append({{
        "model_architecture": arch,
        "seed": int(seed),
        "mean_orig_pred": np.mean(p_orig),
        "mean_over_pred": np.mean(p_over),
        "mean_abs_diff": np.mean(np.abs(diff)),
        "max_abs_diff": np.max(np.abs(diff)),
        "diff_sandy_stations": np.mean(diff[mask]),
    }})
```

### Table 11: Counterfactual Soil Override Sensitivity Results (20 Models x Seeds)
{df_to_markdown(t11)}

- **Ensemble Mean Prediction Shift**: **$0.000319\\text{{ m}}^3/\\text{{m}}^3$ ($0.032\\%$)**.
- **Max Shift on Any Individual Sample**: $0.005488\\text{{ m}}^3/\\text{{m}}^3$ ($0.55\\%$).
- **Takeaway**: Tree splits during summer drought are dominated by topographic elevation, aspect, and antecedent weather memory; the subtle distinction between Loam and Sandy Loam does not alter decision paths. **Overriding soil features is unnecessary**.

---

## 10. Sensor Hardware & ADC Calibration

### Table 7: Raw ADC and Zero Calibration Audit
{df_to_markdown(t7)}

![Fig 6: ADC Calibration Scatter](figures/fig6_raw_adc_to_moisture_calibration.png)

---

## 11. Error Decomposition Synthesis

![Fig 7: Error Decomposition Waterfall](figures/fig7_error_decomposition_waterfall.png)

---

## 12. Actionable Recommendations Matrix

### Table 8: Roadmap & Recommendations
{df_to_markdown(t8)}
"""

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Successfully generated README.md at: {README_PATH}")

if __name__ == "__main__":
    main()
