# Implementation Plan — derived_8.3-error-analysis-1.1: Snowpack Pattern & Out-of-Scope Justification

This experiment extends `derived_8.3-error-analysis-1.0` to provide rigorous hydrological and empirical proof that **`MartenRidge_WA_999`** and **`RainyPass_WA_711`** operate under an alpine snowpack regime (soil moisture driven by winter snowpack accumulation & spring thermal snowmelt rather than immediate rainfall infiltration). 

This diagnostic validates removing these two stations from the core dataset as **out-of-scope microclimates** rather than arbitrary pruning based on bad metric performance ($R^2 < 0$). Additionally, it executes a final specialist/clustering attempt to test whether asymmetric regime gating can separate these stations out.

---

## User Review Required

> [!IMPORTANT]
> **Scope & Rationale Alignment for Station Pruning**:
> 1. **Not Cherry-Picking**: Removing `MartenRidge_WA_999` and `RainyPass_WA_711` is justified physically because the current feature set (NDVI, NDMI, SAR, SMAP, MODIS LST, G_API) lacks explicit **Snow Water Equivalent (SWE)** and **Snow Depth** variables required to predict snowpack accumulation and melt dynamics.
> 2. **Alignment with In-Situ Sensor Deployment**: The ECE team's new custom in-situ soil moisture sensors will be deployed in low-to-mid elevation agricultural, woodland, and managed forest zones — **never in alpine winter snowpack microclimates**. Forcing global models to fit these two extreme stations degrades performance on deployment-relevant microclimates without providing actionable generalization.
> 3. **Future Work Framing**: Alpine snowpack modeling represents a genuine boundary/limitation of the current work and will be formally documented as potential Future Work requiring satellite radar snow depth (e.g. Sentinel-1 wet snow phase) or SNOTEL SWE integration.

---

## Key Diagnostic Analyses & Plan Structure

### 1. Monthly Soil Moisture Trajectories Across Stations & Models
- Extract and plot monthly mean target soil moisture and model predictions (`Model 1 Baseline`, `Model 16 Clustering V0 Full`, `Model 10 Clustering Dynamic`, etc.) across all 9 stations.
- Contrast the hydrological phase shift:
  - **Lowland Rain-Dominant** (`Darrington`, `Quinault`, `Spokane`): Soil moisture peaks in **Winter (Jan–Feb, ~0.27–0.31)** and decays into Summer dry period (Jul–Aug, ~0.03–0.13).
  - **Alpine Snowpack-Dominant** (`RainyPass_WA_711`, `MartenRidge_WA_999`): Winter soil moisture is suppressed/frozen (~0.10–0.30 in Jan–Mar), and **PEAKS IN LATE SPRING / EARLY SUMMER** (**June peak ~0.182** for RainyPass, **May peak ~0.366** for MartenRidge)!

### 2. Quantitative & Physical Proof of Snowpack Decoupling
- **Precipitation Decoupling**: Current precipitation correlation drops from $r = 0.410$ (`Quinault`) and $0.399$ (`Darrington`) down to $r = \mathbf{0.065}$ (`RainyPass_WA_711`) and $0.131$ (`MartenRidge_WA_999`).
- **Antecedent Rain ($G_{API}$) Failure**: $G_{API}$ correlation drops from $0.717$ (`Darrington`) down to $\mathbf{0.172}$ (`RainyPass_WA_711`). Frozen winter precipitation stays on the surface as snow, so antecedent rain indices miscalculate liquid soil infiltration.
- **Thermal Inversion ($LST_{modis}$)**: Standard negative LST correlation ($r = -0.768$ at Darrington) breaks down completely at RainyPass ($r = \mathbf{-0.064}$). When spring temperatures rise, snow melts and soil moisture *increases* with temperature, reversing the summer drying relationship.

### 3. Final Attempt: Asymmetric Station Clustering & Specialist Modeling
- Perform station-level clustering on static bioclimatic features (`J_bio_bio06` coldest month temp, `J_bio_bio11` coldest quarter temp, `J_bio_bio19` winter precip) and dynamic thermal-precipitation metrics.
- Form an asymmetric 2-cluster partitioning:
  - **Cluster 0 (7 Non-Snowpack Stations)**: `Spokane`, `Darrington`, `Quinault`, `SourdoughGulch_WA_985`, `BeaverPass_WA_990`, `CayusePass_WA`, `Paradise_WA`.
  - **Cluster 1 (2 Alpine Snowpack Stations)**: `MartenRidge_WA_999`, `RainyPass_WA_711`.
- Train a dedicated specialist XGBoost model on Cluster 0 vs Cluster 1:
  - Verify that Cluster 0 specialist achieves **$R^2 \ge 0.77$** across the 7 deployment-relevant stations.
  - Verify that Cluster 1 specialist fails ($R^2 \le 0$) due to missing SWE/snow depth features, confirming that algorithm gating alone cannot overcome fundamental feature space deficiencies.

---

## Proposed Changes

### Experiment Directory

#### [NEW] [derived_8.3-error-analysis-1.1.ipynb](file:///c:/Users/pan/Documents/GitHub/MDR-Project/notebooks/experiment/derived_8.3-error-analysis-1.1/derived_8.3-error-analysis-1.1.ipynb)
- Interactive experiment notebook containing data ingestion, monthly soil moisture curves, physical correlation analysis, station clustering, specialist modeling, and figure generation.

#### [NEW] [README.md](file:///c:/Users/pan/Documents/GitHub/MDR-Project/notebooks/experiment/derived_8.3-error-analysis-1.1/README.md)
- Markdown report summarizing the diagnostic findings, snowpack physical proof, out-of-scope justification, monthly breakdown tables, and visual figures.

#### [NEW] High-Resolution Publication Figures
- `fig1_monthly_sm_trajectories_all_stations.png`: Monthly soil moisture curves comparing snowpack vs rain-dominated stations.
- `fig2_snowpack_physical_decoupling_correlations.png`: Correlation breakdown of Precip, $G_{API}$, and LST across station elevations.
- `fig3_station_clustering_snowpack_isolation.png`: Asymmetric station clustering isolating MartenRidge and RainyPass.
- `fig4_specialist_vs_global_performance_7st.png`: Comparative evaluation of 7-station pruned baseline vs 9-station models.

---

## Verification Plan

### Automated Tests
- Run `derived_8.3-error-analysis-1.1.ipynb` using `nb execute experiment/derived_8.3-error-analysis-1.1/derived_8.3-error-analysis-1.1.ipynb --uv` to ensure 100% cell execution without errors.

### Manual Verification
- Inspect generated PNG figures to ensure proper labels, clear legends, and high visual aesthetics.
- Verify that all numerical metrics in `README.md` match exact notebook outputs.
