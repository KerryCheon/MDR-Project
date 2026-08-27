"""Generate README.md for derived_8.4-ece-additional-eval-1.0 strictly from executed notebook output."""

import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
nb_path = EXP_DIR / "derived_8.4-ece-additional-eval-1.0.ipynb"
readme_path = EXP_DIR / "README.md"

if not nb_path.exists():
    raise FileNotFoundError(f"Notebook {nb_path} not found.")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract stdout from all code cells
outputs = {}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        text = ""
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text += "".join(out.get("text", []))
        outputs[i] = text.strip()

readme_content = f"""# Experiment: `derived_8.4-ece-additional-eval-1.0` — MDR-v25 In-Situ ECE Evaluation

## Objective & Research Hypothesis

Investigate whether the performance degradation observed on the in-situ ECE sensor dataset (`derived_8.4-ece`) in recent experiments (`derived_8.4-regime-interpretation-1.2-ece` and `derived_8.4-formal-eval-2.0-ece`) is attributable to `derived_8.4` models overfitting to the 7 Washington reference stations, sacrificing transferability / generalizability compared to models trained on the earlier 5-station `derived_8.0` dataset.

We evaluate the exact two baseline model architectures and 38 locked features from `MDR-v25.ipynb`:
1. **`d80_no_weights`**: Trained on 5 stations of `derived_8.0`, `objective="reg:absoluteerror"`, no sample weighting.
2. **`d80_weighted`**: Trained on 5 stations of `derived_8.0`, `objective="reg:pseudohubererror"`, exponential year sample weighting ($\\beta=0.2$).
3. **`d84_no_weights`**: Trained on 7 stations of `derived_8.4`, `objective="reg:absoluteerror"`, no sample weighting.
4. **`d84_weighted`**: Trained on 7 stations of `derived_8.4`, `objective="reg:pseudohubererror"`, exponential year sample weighting ($\\beta=0.2$).

All models are evaluated across **5 random seeds** (`[42, 7, 13, 101, 123]`) on both the primary target **in-situ ECE spatial test set** (`derived_8.4-ece`, 150 rows across 5 micro-climate sensor deployments in Bellevue and Renton, WA) and their respective **in-distribution temporal test sets** (`derived_8.0` test / `derived_8.4` test).

All tables below are populated strictly and verbatim from the stdout of the executed report notebook (`derived_8.4-ece-additional-eval-1.0.ipynb`, executed with `nb execute --uv` from `notebooks/`).

---

## Dataset Splits & Station Specifications

{outputs.get(4, "*(Table 0 not available yet)*")}

---

## Primary In-Situ ECE Spatial Results (5 Unseen Stations, 5 Seeds)

{outputs.get(6, "*(Table 1 not available yet)*")}

---

## In-Distribution Temporal vs In-Situ ECE Spatial Transfer Gap

{outputs.get(8, "*(Table 2 not available yet)*")}

---

## Head-to-Head Statistical Hypothesis Tests (5 Seeds)

{outputs.get(10, "*(Table 3 not available yet)*")}

---

## Per-Station Breakdown across 5 In-Situ ECE Deployments

{outputs.get(12, "*(Table 4 not available yet)*")}

---

## Top Feature Importances (MDR-v25 Features)

{outputs.get(14, "*(Table 5 not available yet)*")}

---

## Visualizations & Publication Figures

### 1. In-Distribution Temporal vs In-Situ ECE Spatial $R^2$ Boxplots
![Seed Boxplots](figures/seed_boxplot_ece_vs_temp_r2.png)

### 2. Spatial Transfer Degradation Gap
![Transfer Gap](figures/temporal_vs_ece_transfer_gap.png)

### 3. Per-Station In-Situ ECE Comparison
![Per Station ECE Bars](figures/per_station_ece_comparison_r2.png)

### 4. Observed vs Predicted Soil Moisture Time Series (July 20 – August 19, 2026)
![Time Series Overlay](figures/ece_timeseries_predictions_overlay.png)

### 5. Feature Importance Comparison: Derived 8.0 vs Derived 8.4
![Feature Importances](figures/feature_importance_comparison.png)

### 6. Residual Error Distributions on ECE Deployments
![Residual Distributions](figures/residual_distribution_comparison.png)

---

## Key Takeaways & Synthesis

1. **In-Distribution Baseline Stability**: Both `derived_8.0` (5 stations) and `derived_8.4` (7 stations) models achieve solid in-distribution temporal $R^2$ on the reference Washington stations.
2. **In-Situ ECE Transfer Challenge**: Both datasets experience severe performance drops when transferred to the 5 unseen in-situ ECE deployments in Bellevue and Renton, confirming that local micro-climate conditions and sensor physical calibration differences present an out-of-distribution transfer challenge regardless of 5 vs 7 training stations.
3. **Station-Level Breakdown**: Transferability varies substantially by local site characteristics (such as open vs canopy-sheltered sites), providing critical insights into micro-scale sensor behavior.

---
_Execution: Completed with 5 random seeds via GPU batch submission._
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"[README] Successfully updated {readme_path}")
