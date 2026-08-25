# derived_8.4-regime-interpretation-1.2-oos

Physical & environmental interpretation of the two regimes of **every k2 routing strategy** of `derived_8.4-eval-1.1` — `Clustering_V0_Full_k2` (the winner), `Clustering_Dynamic_k2`, `Seasonal_Binary_k2`, `Univariate_G_API_k2`, `Trained_Gating_k2` — **evaluated on the 10 out-of-state held-out stations (`derived_8.4-oos`)**.

The routers are fitted **strictly on the 7 Washington State stations** from `derived_8.4` (`trainval`, 2017–2022, 14,608 rows). The out-of-state dataset `derived_8.4-oos` (25,176 rows across 2017–2025 for 10 stations across Oregon, Idaho, California, Colorado, Wyoming, and Montana) is treated as a **completely held-out test set** to inspect regime assignment, station purity, dynamic weather drivers, static environmental attributes, and macro-climatic clustering behavior across the Western US.

---

## Run

From `notebooks/`:

```bash
nb execute experiment/derived_8.4-regime-interpretation-1.2-oos/derived_8.4_regime_interpretation_1.2_oos.ipynb --uv --timeout 900
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
- **Station Purity**: The fraction of a station's observation days assigned to its dominant regime cluster: $\text{Purity} = \max(\text{share}_{c0}, \text{share}_{c1})$. A purity of $1.000$ indicates that the station belongs permanently to a single regime (pure spatial/regional specialist), whereas a purity near $0.500$ indicates dynamic temporal switching across seasons.
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

1. **`Clustering_V0_Full_k2` generalizes as a macro-regional climate specialist across the Western US** (mean OOS station purity **0.871**):
   - **Regime 0 (Pacific Maritime & Humid Cascades)**: Captures the wet, low-aridity maritime stations west of the Cascade crest (`Clackamas_Lake_398` [100% c0] and `Corvallis_10_SSW` [66.8% c0]).
   - **Regime 1 (Semi-Arid Continental & Rain-Shadow Basins)**: Captures continental stations in the Northern Great Basin, Snake River Plain, High Plains, and Rocky Mountains (`Riley_10_WSW` [100% c1], `Murphy_10_W` [100% c1], `Wolf_Point_29_ENE` [100% c1], `John_Day_35_WNW` [99.3% c1], `Lander_11_SSE` [98.0% c1], `Boulder_14_W` [79.2% c1], `Rock_Springs_721` [77.4% c1]).
   - `Redding_12_WNW` splits 50.0% c0 / 50.0% c1, reflecting a Mediterranean climate transition between wet winter (c0) and hot, dry summer (c1).
2. **Dynamic / Weather-driven routers (`Clustering_Dynamic_k2` and `Univariate_G_API_k2`) classify ~80% of out-of-state samples into Cluster 0 (dry regime)**:
   - Because the out-of-state stations in the Great Basin, Wyoming Basin, and Great Plains receive substantially lower precipitation than Western Washington, dynamic moisture and antecedent rainfall indicators consistently place them into the dry specialist.
3. **Calendar routing (`Seasonal_Binary_k2`) is geographically blind**:
   - Fixed May–Oct vs Nov–Apr split produces ~50/50 splits on Mediterranean/PNW stations, but fails for continental/mountain stations (`Boulder_14_W`, `Wolf_Point_29_ENE`) where peak precipitation occurs during late spring and summer convective events.
4. **Moisture classification (`Trained_Gating_k2`) provides sharp target separation**:
   - Classifies samples with $\text{soil\_moisture\_5cm} < 0.16$ into Cluster 0 (dry, median 0.106) and $\ge 0.16$ into Cluster 1 (wet, median 0.222).

---

## Router Reproduction & In-Sample Sanity Check

Fitted on `derived_8.4` `trainval` (14,608 rows across 7 WA stations):

| Strategy | In-Sample WA Test Agreement (eval-1.1) | Matching Rows | WA Trainval Counts (c0 / c1) |
| :--- | :---: | :---: | :---: |
| `Clustering_V0_Full_k2` | **1.000000** | 6,620 / 6,620 | 10,624 (72.7%) / 3,984 (27.3%) |
| `Clustering_Dynamic_k2` | **1.000000** | 6,620 / 6,620 | 7,974 (54.6%) / 6,634 (45.4%) |
| `Seasonal_Binary_k2` | **1.000000** | 6,620 / 6,620 | 7,559 (51.7%) / 7,049 (48.3%) |
| `Univariate_G_API_k2` | **1.000000** | 6,620 / 6,620 | 7,304 (50.0%) / 7,304 (50.0%) |
| `Trained_Gating_k2` | **0.996828** | 6,599 / 6,620 | 4,181 (28.6%) / 10,427 (71.4%) |

---

## Cross-Strategy Comparison on Out-of-State Dataset (25,176 rows)

| Strategy | OOS Sizes (c0 / c1) | Mean OOS Purity | Dominant Stations per Cluster | Top Separating Feature on OOS |
| :--- | :--- | ---: | :--- | :--- |
| `Clustering_V0_Full_k2` | 6,691 (26.6%) / 18,485 (73.4%) | **0.871** | **c0**: Clackamas_Lake_398, Corvallis_10_SSW;<br>**c1**: Boulder_14_W, John_Day_35_WNW, Lander_11_SSE, Murphy_10_W, Redding_12_WNW, Riley_10_WSW, Rock_Springs_721, Wolf_Point_29_ENE | `K_aspect_cos` (r = 0.999) |
| `Clustering_Dynamic_k2` | 19,995 (79.4%) / 5,181 (20.6%) | 0.817 | **c0**: Boulder_14_W, Clackamas_Lake_398, Corvallis_10_SSW, John_Day_35_WNW, Lander_11_SSE, Murphy_10_W, Redding_12_WNW, Riley_10_WSW, Rock_Springs_721, Wolf_Point_29_ENE | `V_ema_LST_modis_kobs30` (r = -0.923) |
| `Seasonal_Binary_k2` | 14,579 (57.9%) / 10,597 (42.1%) | 0.602 | **c0**: Boulder_14_W, John_Day_35_WNW, Lander_11_SSE, Murphy_10_W, Riley_10_WSW, Rock_Springs_721, Wolf_Point_29_ENE;<br>**c1**: Clackamas_Lake_398, Corvallis_10_SSW, Redding_12_WNW | `V_ema_LST_modis_kobs30` (r = -0.916) |
| `Univariate_G_API_k2` | 20,066 (79.7%) / 5,110 (20.3%) | 0.771 | **c0**: Corvallis_10_SSW, John_Day_35_WNW, Lander_11_SSE, Murphy_10_W, Redding_12_WNW, Riley_10_WSW, Rock_Springs_721, Wolf_Point_29_ENE;<br>**c1**: Boulder_14_W, Clackamas_Lake_398 | `V_rollmin_G_API_kobs14` (r = 0.996) |
| `Trained_Gating_k2` | 11,834 (47.0%) / 13,342 (53.0%) | 0.633 | **c0**: Lander_11_SSE, Murphy_10_W, Redding_12_WNW, Riley_10_WSW, Wolf_Point_29_ENE;<br>**c1**: Boulder_14_W, Clackamas_Lake_398, Corvallis_10_SSW, John_Day_35_WNW, Rock_Springs_721 | `V_ema_LST_modis_kobs30` (r = -0.872) |

---

## Out-of-State Station Composition by Routing Strategy (Share of Cluster 1)

Values represent the proportion of observations assigned to **Cluster 1** ($1.000 =$ entirely in Cluster 1):

| Station ID | State | Elev (m) | `Clustering_V0_Full_k2` | `Clustering_Dynamic_k2` | `Seasonal_Binary_k2` | `Univariate_G_API_k2` | `Trained_Gating_k2` |
| :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: |
| `Boulder_14_W` | CO | 2,965 | **0.792** | 0.201 | 0.151 | 0.520 | 0.960 |
| `Clackamas_Lake_398` | OR | 1,051 | **0.000** | 0.500 | 0.525 | 0.529 | 0.705 |
| `Corvallis_10_SSW` | OR | 95 | **0.332** | 0.288 | 0.516 | 0.407 | 0.627 |
| `John_Day_35_WNW` | OR | 684 | **0.993** | 0.167 | 0.475 | 0.064 | 0.526 |
| `Lander_11_SSE` | WY | 1,748 | **0.980** | 0.046 | 0.290 | 0.075 | 0.461 |
| `Murphy_10_W` | ID | 1,204 | **1.000** | 0.113 | 0.433 | 0.033 | 0.459 |
| `Redding_12_WNW` | CA | 432 | **0.500** | 0.210 | 0.509 | 0.392 | 0.490 |
| `Riley_10_WSW` | OR | 1,397 | **1.000** | 0.091 | 0.373 | 0.017 | 0.332 |
| `Rock_Springs_721` | OR | 1,599 | **0.774** | 0.327 | 0.498 | 0.059 | 0.521 |
| `Wolf_Point_29_ENE` | MT | 632 | **1.000** | 0.019 | 0.211 | 0.040 | 0.266 |

---

## Target (`soil_moisture_5cm`) Distribution by Regime on Out-of-State Dataset

| Strategy | Cluster | N Samples | Median | Mean | P10 | P90 |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| `Clustering_V0_Full_k2` | 0 | 6,691 | **0.2520** | 0.2458 | 0.1090 | 0.3660 |
| `Clustering_V0_Full_k2` | 1 | 18,485 | **0.1340** | 0.1557 | 0.0470 | 0.2930 |
| `Clustering_Dynamic_k2` | 0 | 19,995 | **0.1450** | 0.1578 | 0.0490 | 0.2910 |
| `Clustering_Dynamic_k2` | 1 | 5,181 | **0.2620** | 0.2642 | 0.1350 | 0.3790 |
| `Seasonal_Binary_k2` | 0 | 14,579 | **0.1400** | 0.1504 | 0.0480 | 0.2810 |
| `Seasonal_Binary_k2` | 1 | 10,597 | **0.2220** | 0.2201 | 0.0820 | 0.3540 |
| `Univariate_G_API_k2` | 0 | 20,066 | **0.1470** | 0.1593 | 0.0500 | 0.2910 |
| `Univariate_G_API_k2` | 1 | 5,110 | **0.2570** | 0.2598 | 0.1330 | 0.3780 |
| `Trained_Gating_k2` | 0 | 11,834 | **0.1060** | 0.1118 | 0.0450 | 0.1870 |
| `Trained_Gating_k2` | 1 | 13,342 | **0.2220** | 0.2400 | 0.1140 | 0.3600 |

---

## Geographic & Physical Deep Dive (`Clustering_V0_Full_k2`)

### 1. Per-Station Physical Profile (10 Out-of-State Stations)

| Station ID | Cluster | Purity | Elev (m) | Lat | Lon | Annual T (°C) | Annual P (mm) | Clay b0 (%) | Sand b0 (%) | Landcover | SMAP Climatology | NDVI Climatology |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| `Clackamas_Lake_398` | 0 | 1.000 | 1,051 | 45.097 | -121.754 | 6.5 | 1,782 | 11.0 | 50.0 | Tree cover | 0.3553 | 0.6385 |
| `Corvallis_10_SSW` | 0 | 0.668 | 90 | 44.419 | -123.326 | 11.1 | 1,129 | 26.0 | 30.0 | Cropland | 0.3341 | 0.4719 |
| `Boulder_14_W` | 1 | 0.792 | 2,965 | 40.035 | -105.541 | 1.4 | 620 | 10.0 | 57.0 | Tree cover | 0.1417 | 0.4190 |
| `Redding_12_WNW` | 1 | 0.500 | 489 | 40.651 | -122.607 | 15.0 | 1,370 | 20.0 | 44.0 | Grassland | 0.2078 | 0.3951 |
| `Wolf_Point_29_ENE` | 1 | 1.000 | 632 | 48.308 | -105.102 | 5.4 | 327 | 30.0 | 27.0 | Grassland | 0.1824 | 0.2798 |
| `Rock_Springs_721` | 1 | 0.774 | 1,599 | 44.009 | -118.838 | 4.0 | 390 | 13.0 | 41.0 | Grassland | 0.2195 | 0.2872 |
| `John_Day_35_WNW` | 1 | 0.993 | 678 | 44.556 | -119.646 | 10.3 | 299 | 18.0 | 50.0 | Grassland | 0.1704 | 0.2582 |
| `Murphy_10_W` | 1 | 1.000 | 1,202 | 43.204 | -116.751 | 8.5 | 293 | 20.0 | 44.0 | Grassland | 0.1418 | 0.1709 |
| `Riley_10_WSW` | 1 | 1.000 | 1,396 | 43.471 | -119.692 | 7.2 | 269 | 15.0 | 42.0 | Grassland | 0.1542 | 0.1741 |
| `Lander_11_SSE` | 1 | 0.980 | 1,748 | 42.675 | -108.669 | 6.6 | 262 | 21.0 | 40.0 | Grassland | 0.1264 | 0.1802 |

### 2. Aridity & Climate Normals Summary

| Station ID | Cluster | Annual T (°C) | Annual P (mm) | Driest Month P (mm) | BIO15 (Seasonality) | PET (mm) | UNEP Aridity Index ($P/\text{PET}$) | AI Class | De Martonne ($P/(T+10)$) | DMI Class |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| `Clackamas_Lake_398` | 0 | 6.5 | 1,782 | 23 | 61 | 477.3 | **3.733** | Humid | **108.00** | Very humid |
| `Corvallis_10_SSW` | 0 | 11.1 | 1,129 | 13 | 68 | 610.0 | **1.851** | Humid | **53.51** | Humid |
| `Redding_12_WNW` | 1 | 15.0 | 1,370 | 7 | 80 | 860.0 | 1.593 | Humid | 54.80 | Humid |
| `Boulder_14_W` | 1 | 1.4 | 620 | 38 | 17 | 406.3 | 1.526 | Humid | 54.39 | Humid |
| `Rock_Springs_721` | 1 | 4.0 | 390 | 16 | 31 | 487.9 | 0.799 | Sub-humid | 27.86 | Sub-humid |
| `Wolf_Point_29_ENE` | 1 | 5.4 | 327 | 7 | 70 | 615.6 | 0.531 | Dry sub-humid | 21.23 | Sub-humid |
| `John_Day_35_WNW` | 1 | 10.3 | 299 | 12 | 30 | 657.6 | 0.455 | Semi-arid | 14.73 | Semi-arid |
| `Murphy_10_W` | 1 | 8.5 | 293 | 9 | 28 | 608.5 | 0.482 | Semi-arid | 15.84 | Semi-arid |
| `Riley_10_WSW` | 1 | 7.2 | 269 | 12 | 26 | 562.7 | 0.478 | Semi-arid | 15.64 | Semi-arid |
| `Lander_11_SSE` | 1 | 6.6 | 262 | 10 | 50 | 610.0 | 0.429 | Semi-arid | 15.78 | Semi-arid |

### 3. Within-Cluster Homogeneity vs Between-Cluster Separation

| Attribute | Median Cluster 0 | Median Cluster 1 | Within-Cluster MAD | Between-Cluster Gap (Pooled SD) |
| :--- | ---: | ---: | ---: | ---: |
| **Aridity Index ($P/\text{PET}$)** | **2.792** | **0.506** | 0.633 | **2.30** |
| **De Martonne Index** | **80.755** | **18.535** | 19.641 | **2.19** |
| **Annual Precipitation (mm)** | **1,455.5** | **313.0** | 262.25 | **2.18** |
| **NDVI Climatology** | **0.555** | **0.269** | 0.063 | **2.13** |
| **SMAP Soil Moisture Climatology** | **0.345** | **0.162** | 0.022 | **2.09** |
| **Precipitation Seasonality (BIO15)** | **64.5** | **30.5** | 9.875 | **1.60** |
| **NDMI Moisture Index Climatology** | **0.262** | **-0.023** | 0.127 | **1.58** |
| **MSI Moisture Stress Index** | **0.619** | **1.077** | 0.199 | **1.46** |
| **Elevation (m)** | **570.5** | **1,299.0** | 534.438 | **0.94** |
| **Aspect Cosine (`K_aspect_cos`)** | **-0.118** | **0.350** | 0.431 | **0.86** |
| **Driest Month Precipitation (mm)** | **18.0** | **11.0** | 5.312 | **0.78** |
| **Longitude** | **-122.540** | **-117.794** | 3.188 | **0.71** |

---

## Artifacts Generated

- `derived_8.4_regime_interpretation_1.2_oos.ipynb` — Executed reproducible Jupyter notebook.
- `v0full_station_map.png` — Western US topographic map of all 10 out-of-state stations + 7 Washington reference stations colored by regime.
- `v0full_physical_separation.png` — 2x2 physical attribute separation scatter plots.
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
