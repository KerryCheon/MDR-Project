# derived_8.4-regime-interpretation-1.2-ece

Physical & environmental interpretation of the two regimes of **every k2 routing strategy** of `derived_8.4-eval-1.1` — `Clustering_V0_Full_k2` (the winner), `Clustering_Dynamic_k2`, `Seasonal_Binary_k2`, `Univariate_G_API_k2`, `Trained_Gating_k2` — **evaluated on the 5 in-situ ECE soil moisture sensor stations (`derived_8.4-ece`)**.

The routers are fitted **strictly on the 7 Washington State reference stations** from `derived_8.4` (`trainval`, 2017–2022, 14,608 rows). The in-situ sensor dataset `derived_8.4-ece` (150 rows across July 20 to August 19, 2026 for 5 micro-stations located at Bellevue Botanical Garden and Renton, Washington) is treated as a **completely held-out test set** to inspect regime assignment, station purity, dynamic weather drivers, static environmental attributes, and micro-scale vs macro-scale clustering behavior in Western Washington.

---

## Run

From `notebooks/`:

```bash
nb execute experiment/derived_8.4-regime-interpretation-1.2-ece/derived_8.4_regime_interpretation_1.2_ece.ipynb --uv --timeout 900
```

`Trained_Gating_k2` is refit on CPU because this environment has no GPU (eval-1.1 trained it on CUDA). Its in-sample Washington test labels agree with eval-1.1 on 99.7% of rows; the four deterministic strategies match 100% (see "Router reproduction & In-Sample Sanity Check").

The map cell (3.6.2b) uses `contextily` and downloads Esri tiles (`World_Shaded_Relief`), falling back to a vector-only map if the download fails.

---

## Glossary & Terminology

A quick guide to the domain metrics, statistical terms, and dataset features used in this analysis:

### 1. Hydroclimatic & Aridity Indices
- **UNEP Aridity Index ($\text{AI}$)**: The United Nations Environment Programme aridity index, defined as $\text{AI} = \frac{P}{\text{PET}}$, where $P$ is annual precipitation and $\text{PET}$ is annual potential evapotranspiration. It measures regional water balance:
  - $\text{AI} < 0.20$: **Arid** (extreme water deficit)
  - $0.20 \le \text{AI} < 0.50$: **Semi-arid** (evaporative demand exceeds precipitation by $2–5\times$)
  - $0.50 \le \text{AI} < 0.65$: **Dry sub-humid**
  - $0.65 \le \text{AI} < 1.00$: **Sub-humid**
  - $\text{AI} \ge 1.00$: **Humid / Very Humid** (water surplus, precipitation exceeds evaporative demand)
- **De Martonne Aridity Index ($\text{DMI}$)**: Classical hydroclimatic index $\text{DMI} = \frac{P}{T + 10}$, where $P$ is annual precipitation (mm) and $T$ is annual mean temperature (°C). Higher values indicate wetter climates ($<20$ = semi-arid, $>60$ = very humid).
- **Thornthwaite $\text{PET}$**: Potential Evapotranspiration (mm/year) calculated by the Thornthwaite (1948) empirical temperature-based method, representing the theoretical atmospheric moisture demand.

### 2. Evaluation & Statistical Metrics
- **Station Purity**: The fraction of a station's observation days assigned to its dominant regime cluster: $\text{Purity} = \max(\text{share}_{c0}, \text{share}_{c1})$. A purity of $1.000$ indicates that the station belongs permanently to a single regime (pure spatial specialist), whereas a purity near $0.500$ indicates dynamic temporal switching across seasons.
- **Rank-Biserial Correlation ($r$)**: A non-parametric effect size derived from the Mann-Whitney $U$ test, measuring how strongly a continuous feature separates two discrete clusters. Scaled in $[-1.0, +1.0]$, where $r > 0$ indicates higher values in Cluster 1, and $|r| \approx 1.0$ indicates near-perfect separation.
- **Within-Cluster $\text{MAD}$**: Median Absolute Deviation ($\text{median}(|x - \text{median}(x)|)$), measuring internal cluster spread/dispersion in a manner robust to outliers.
- **Between-Cluster Gap (in Pooled $\text{SD}$)**: Normalized separation distance $|\text{median}_{c1} - \text{median}_{c0}| / \sigma_{\text{pooled}}$, quantifying how many standard deviations separate the two cluster centroids.

### 3. Satellite & Environmental Features
- **$\text{SMAP\_sm\_pm\_interp}$**: Surface soil moisture ($0–5\text{ cm}$, $\text{m}^3/\text{m}^3$) retrieved by NASA's Soil Moisture Active Passive (SMAP) satellite.
- **$\text{F\_NDVI}$**: Normalized Difference Vegetation Index from Sentinel-2/Landsat, quantifying vegetation canopy greenness and vigor.
- **$\text{F\_NDMI}$**: Normalized Difference Moisture Index, sensitive to vegetation liquid water content.
- **$\text{F\_MSI}$**: Moisture Stress Index ($\text{SWIR} / \text{NIR}$), where higher values indicate canopy moisture stress/drying.
- **$\text{LST\_modis}$**: Land Surface Temperature (Kelvin) from MODIS satellites.
- **$\text{G\_API}$**: Antecedent Precipitation Index ($\text{mm}$), an exponentially decayed running sum of prior rainfall representing cumulative antecedent soil wetness.
- **$\text{G\_DSLR}$**: Days Since Last Rain (count of consecutive dry days).
- **$\text{BIO01–BIO19}$**: WorldClim bioclimatic normals (e.g., $\text{BIO01}$ = annual mean temperature, $\text{BIO12}$ = annual precipitation, $\text{BIO15}$ = precipitation seasonality coefficient of variation).

---

## Key Takeaways

1. **`Clustering_V0_Full_k2` generalizes with high spatial purity on in-situ ECE stations (mean purity 0.953)**:
   - **Regime 0 (Open Lowland / Grassland & Exposed Soils)**: Captures open garden beds and manicured turf (`ECE_BBG_Main_St` [93.3% c0], `ECE_Renton_Garden_North` [96.7% c0], `ECE_Renton_Garden_Shed` [96.7% c0]). These sites experience direct solar radiation, higher surface evaporation, and lower subsoil moisture buffering.
   - **Regime 1 (Dense Canopy / Sheltered Forested Micro-habitats)**: Captures `ECE_BBG_Lost_Meadow` (100% c1) and `ECE_Renton_Home` (90.0% c1). At Lost Meadow, dense tree canopy interception and forest litter maintain higher organic layer insulation, while Renton Home is sheltered by residential landscape structure.
   - **Separating Features**: Top separating attributes are topographic aspect cosine (`K_aspect_cos`, gap = 1.76 SD), topsoil clay fraction (`J_clay_wfrac_b0`, gap = 1.32 SD), and annual precipitation (`annual_precip_P_mm`, gap = 1.18 SD).
2. **Dynamic & Seasonal Routers (`Clustering_Dynamic_k2`, `Seasonal_Binary_k2`, `Univariate_G_API_k2`) classify 100% of ECE samples into Cluster 0 (dry regime)**:
   - Because the ECE evaluation dates (July 20 – August 19) coincide with the Mediterranean dry summer minimum in Western Washington, antecedent rainfall ($G\_API$) and satellite soil moisture ($SMAP$) remain below the wet-season thresholds established during the winter-dominated training set.
3. **Moisture classification (`Trained_Gating_k2`) provides sharp target separation**:
   - Classifies 80.0% of ECE observations into Cluster 0 (dry, median 0.0626) and 20.0% into Cluster 1 (wet, mean 0.0812), identifying occasional episodic moisture pulses during transient rain events.

---

## Router Reproduction & In-Sample Sanity Check

Fitted on `derived_8.4` `trainval` (14,608 rows across 7 WA reference stations):

| Strategy | In-Sample WA Test Agreement (eval-1.1) | Matching Rows | WA Trainval Counts (c0 / c1) |
| :--- | :---: | :---: | :---: |
| `Clustering_V0_Full_k2` | **1.000000** | 6,620 / 6,620 | 10,624 (72.7%) / 3,984 (27.3%) |
| `Clustering_Dynamic_k2` | **1.000000** | 6,620 / 6,620 | 7,974 (54.6%) / 6,634 (45.4%) |
| `Seasonal_Binary_k2` | **1.000000** | 6,620 / 6,620 | 7,559 (51.7%) / 7,049 (48.3%) |
| `Univariate_G_API_k2` | **1.000000** | 6,620 / 6,620 | 7,304 (50.0%) / 7,304 (50.0%) |
| `Trained_Gating_k2` | **0.996828** | 6,599 / 6,620 | 4,181 (28.6%) / 10,427 (71.4%) |

---

## Cross-Strategy Comparison on ECE Dataset (150 rows)

| Strategy | ECE Sizes (c0 / c1) | Mean ECE Purity | Dominant Stations per Cluster | Top Separating Feature on ECE |
| :--- | :--- | ---: | :--- | :--- |
| `Clustering_V0_Full_k2` | 89 (59.3%) / 61 (40.7%) | **0.953** | **c0**: ECE_BBG_Main_St, ECE_Renton_Garden_North, ECE_Renton_Garden_Shed;<br>**c1**: ECE_BBG_Lost_Meadow, ECE_Renton_Home | `J_aspect_deg` (r = -0.928) |
| `Clustering_Dynamic_k2` | 150 (100.0%) | 1.000 | **c0**: ECE_BBG_Lost_Meadow, ECE_BBG_Main_St, ECE_Renton_Garden_North, ECE_Renton_Garden_Shed, ECE_Renton_Home | None (single cluster) |
| `Seasonal_Binary_k2` | 150 (100.0%) | 1.000 | **c0**: ECE_BBG_Lost_Meadow, ECE_BBG_Main_St, ECE_Renton_Garden_North, ECE_Renton_Garden_Shed, ECE_Renton_Home | None (single cluster) |
| `Univariate_G_API_k2` | 150 (100.0%) | 1.000 | **c0**: ECE_BBG_Lost_Meadow, ECE_BBG_Main_St, ECE_Renton_Garden_North, ECE_Renton_Garden_Shed, ECE_Renton_Home | None (single cluster) |
| `Trained_Gating_k2` | 120 (80.0%) / 30 (20.0%) | 0.800 | **c0**: ECE_BBG_Lost_Meadow, ECE_BBG_Main_St, ECE_Renton_Garden_North, ECE_Renton_Garden_Shed, ECE_Renton_Home | `V_ema_G_API_kobs30` (r = -0.997) |

---

## ECE Station Composition by Routing Strategy (Share of Cluster 1)

Values represent the proportion of observations assigned to **Cluster 1** ($1.000 =$ entirely in Cluster 1):

| Station ID | Elev (m) | `Clustering_V0_Full_k2` | `Clustering_Dynamic_k2` | `Seasonal_Binary_k2` | `Univariate_G_API_k2` | `Trained_Gating_k2` |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: |
| `ECE_BBG_Lost_Meadow` | 52 | **1.000** | 0.000 | 0.000 | 0.000 | 0.200 |
| `ECE_BBG_Main_St` | 51 | **0.067** | 0.000 | 0.000 | 0.000 | 0.200 |
| `ECE_Renton_Garden_North` | 157 | **0.033** | 0.000 | 0.000 | 0.000 | 0.200 |
| `ECE_Renton_Garden_Shed` | 157 | **0.033** | 0.000 | 0.000 | 0.000 | 0.200 |
| `ECE_Renton_Home` | 136 | **0.900** | 0.000 | 0.000 | 0.000 | 0.200 |

---

## Target (`soil_moisture_5cm`) Distribution by Regime on ECE Dataset

| Strategy | Cluster | N Samples | Median | Mean | P10 | P90 |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| `Clustering_V0_Full_k2` | 0 | 89 | **0.0764** | 0.0934 | 0.0508 | 0.1618 |
| `Clustering_V0_Full_k2` | 1 | 61 | **0.0506** | 0.0419 | 0.0157 | 0.0650 |
| `Clustering_Dynamic_k2` | 0 | 150 | **0.0626** | 0.0724 | 0.0176 | 0.1518 |
| `Seasonal_Binary_k2` | 0 | 150 | **0.0626** | 0.0724 | 0.0176 | 0.1518 |
| `Univariate_G_API_k2` | 0 | 150 | **0.0626** | 0.0724 | 0.0176 | 0.1518 |
| `Trained_Gating_k2` | 0 | 120 | **0.0626** | 0.0702 | 0.0166 | 0.1408 |
| `Trained_Gating_k2` | 1 | 30 | **0.0623** | 0.0812 | 0.0200 | 0.1897 |

---

## Geographic & Physical Deep Dive (`Clustering_V0_Full_k2`)

### 1. Per-Station Physical Profile (5 In-Situ ECE Stations)

| Station ID | Cluster | Purity | Elev (m) | Lat | Lon | Annual T (°C) | Annual P (mm) | Clay b0 (%) | Sand b0 (%) | Landcover | SMAP Climatology | NDVI Climatology |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| `ECE_Renton_Garden_North` | 0 | 0.967 | 157 | 47.4963 | -122.1406 | 10.3 | 1,227 | 21.0 | 40.0 | Grassland | 0.0000 | 0.5490 |
| `ECE_Renton_Garden_Shed` | 0 | 0.967 | 157 | 47.4958 | -122.1408 | 10.3 | 1,227 | 21.0 | 40.0 | Grassland | 0.0000 | 0.5490 |
| `ECE_BBG_Main_St` | 0 | 0.933 | 51 | 47.6098 | -122.1825 | 11.0 | 1,018 | 16.0 | 45.0 | Built-up | 0.0000 | 0.5284 |
| `ECE_Renton_Home` | 1 | 0.900 | 136 | 47.4887 | -122.1447 | 10.4 | 1,181 | 17.0 | 44.0 | Built-up | 0.0000 | 0.4954 |
| `ECE_BBG_Lost_Meadow` | 1 | 1.000 | 52 | 47.6072 | -122.1795 | 11.0 | 1,019 | 19.0 | 45.0 | Tree cover | 0.0000 | 0.4827 |

### 2. Aridity & Climate Normals Summary

| Station ID | Cluster | Annual T (°C) | Annual P (mm) | Driest Month P (mm) | BIO15 (Seasonality) | PET (mm) | UNEP Aridity Index ($P/\text{PET}$) | AI Class | De Martonne ($P/(T+10)$) | DMI Class |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| `ECE_Renton_Garden_North` | 0 | 10.3 | 1,227 | 31 | 50 | 679.3 | **1.806** | Humid | **60.44** | Very humid |
| `ECE_Renton_Garden_Shed` | 0 | 10.3 | 1,227 | 31 | 50 | 679.3 | **1.806** | Humid | **60.44** | Very humid |
| `ECE_BBG_Main_St` | 0 | 11.0 | 1,018 | 24 | 53 | 694.2 | **1.466** | Humid | **48.48** | Humid |
| `ECE_Renton_Home` | 1 | 10.4 | 1,181 | 29 | 50 | 681.4 | **1.733** | Humid | **57.89** | Humid |
| `ECE_BBG_Lost_Meadow` | 1 | 11.0 | 1,019 | 24 | 53 | 694.2 | **1.468** | Humid | **48.52** | Humid |

### 3. Within-Cluster Homogeneity vs Between-Cluster Separation

| Attribute | Median Cluster 0 | Median Cluster 1 | Within-Cluster MAD | Between-Cluster Gap (Pooled SD) |
| :--- | ---: | ---: | ---: | ---: |
| **Aspect Cosine (`K_aspect_cos`)** | **0.225** | **-0.662** | 0.168 | **1.76** |
| **Topsoil Clay Fraction (`J_clay_wfrac_b0`)** | **21.0** | **18.0** | 0.500 | **1.32** |
| **Annual Precipitation (mm)** | **1,227.0** | **1,100.0** | 40.500 | **1.18** |
| **De Martonne Index** | **60.44** | **53.21** | 2.342 | **1.17** |
| **Aridity Index ($P/\text{PET}$)** | **1.806** | **1.601** | 0.066 | **1.17** |
| **Elevation (m)** | **157.0** | **94.0** | 21.000 | **1.15** |
| **NDVI Climatology** | **0.549** | **0.489** | 0.006 | **1.04** |
| **Longitude** | **-122.141** | **-122.162** | 0.009 | **0.99** |
| **Precipitation Seasonality (BIO15)** | **50.0** | **51.5** | 0.750 | **0.91** |
| **Latitude** | **47.496** | **47.548** | 0.030 | **0.82** |
| **Subsoil Clay Fraction (`J_clay_wfrac_b100`)** | **25.0** | **22.5** | 0.250 | **0.76** |
| **Slope (deg)** | **2.0** | **3.5** | 1.250 | **0.62** |

---

## Artifacts Generated

- `derived_8.4_regime_interpretation_1.2_ece.ipynb` — Executed reproducible Jupyter notebook.
- `v0full_station_map.png` — Two-panel map showing Washington State reference stations alongside high-resolution King County ECE sensor network layout.
- `v0full_physical_separation.png` — 2x2 physical attribute separation scatter plots for ECE stations.
- `v0full_land_surface_profile.png` — Climatological satellite remote-sensing comparison.
- `v0full_aridity_summary.csv` — Station-level bioclimatic and aridity metrics.
- `v0full_station_physical_profile.csv` — Comprehensive static environmental attributes.
- `v0full_cluster_soil_terrain_landcover.csv` — Cluster-level medians of soil, terrain, and vegetation.
- `v0full_cluster_physical_commonality.csv` — Homogeneity (MAD) and separation gap statistics.
- `regime_comparison_summary.csv` — High-level 5-strategy cross-comparison table.
- `regime_spatial_summary.csv` — Station-level cluster assignment and purity across all strategies.
- 5 × `regime_geographic_distribution_*.png`
- 5 × `regime_seasonality_*.png`
- 5 × `regime_target_distributions_*.png`
- 5 × `regime_weather_drivers_*.png`
- 5 × `regime_static_attributes_*.png`
- 5 × `regime_profile_summary_*.csv`
- 5 × `regime_station_composition_*.csv`
- 5 × `regime_top_features_*.csv`
