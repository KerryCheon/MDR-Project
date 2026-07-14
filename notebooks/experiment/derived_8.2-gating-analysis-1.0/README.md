# Unsupervised Gating Proxy and Quality Analysis Report (derived_8.2)

This report details the findings of our exploratory study on alternative, unsupervised grouping methods for the Washington-only `derived_8.2` soil moisture dataset. 

Our goal was to find a grouping strategy that **does not rely on the target variable** (to avoid training a complex gating router and prevent target leakage) and **groups the observations in a physically meaningful way**, so that different specialist models can use the feature sets that are most helpful to them.

---

## 1. Summary of Gating Quality Metrics

We evaluated 11 different grouping strategies for both $K=2$ (binary) and $K=3$ (3-class) configurations on the training set (N=15,704 samples after filtering). The strategies were evaluated on three dimensions:
1. **Target Separability**: Measured via target variance reduction (VR) and Kolmogorov-Smirnov (KS) statistics.
2. **Feature Relationship Divergence (Core Metric)**: Measured via the standard deviation of feature-target correlations across clusters for the V3 feature set ($D_{V3}$) and the top 20 globally correlated V3 features ($D_{Top20}$). We also track the maximum correlation drift on a single feature.
3. **Group Balance**: Measured via normalized Shannon entropy to prevent degenerate splits.

Here is the complete summary of evaluated configurations (sorted by $K$ and $D_{Top20}$):

| Strategy | K | Entropy (Balance) | Variance Reduction | Mean KS | Divergence (V3) | Divergence (Top20) | Max Drift | Drift Feature |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Univariate_G_API (Best Proxy K=2)** | 2 | 1.0000 | 0.0875 | 0.2985 | 0.1123 | **0.1896** | 0.5612 | `V_rollmin_LST_modis_kobs30` |
| **Clustering_Dynamic (K=2)** | 2 | 0.9969 | 0.0718 | 0.3028 | 0.1172 | **0.1849** | 0.6307 | `V_rollmin_LST_modis_kobs30` |
| **Seasonal_Binary** | 2 | 0.9999 | 0.1005 | 0.3378 | 0.1223 | **0.1845** | 0.6287 | `V_rollmin_LST_modis_kobs30` |
| **Clustering_V3_Full** | 2 | 0.9976 | 0.1100 | 0.3214 | 0.1033 | 0.1709 | 0.6747 | `V_rollmin_LST_modis_kobs30` |
| **Univariate_LST_Lag30** | 2 | 1.0000 | 0.1756 | 0.3683 | 0.0949 | 0.1655 | 0.6606 | `V_ema_LST_modis_kobs30` |
| **Static_Elevation** | 2 | 0.9967 | 0.0892 | 0.3278 | 0.1048 | 0.1227 | 0.5896 | `K_aspect_cos` |
| **Univariate_NDMI** | 2 | 1.0000 | 0.0466 | 0.1872 | 0.0680 | 0.0993 | 0.4015 | `V_rollmin_LST_modis_kobs30` |
| **Clustering_Hybrid** | 2 | 0.9328 | 0.0002 | 0.1054 | 0.0772 | 0.0963 | 0.4573 | `V_rollrng_F_NDVI_kobs30` |
| **Static_Clay_Content** | 2 | 0.9798 | 0.0001 | 0.0923 | 0.0658 | 0.0837 | 0.4688 | `J_aspect_deg` |
| **Univariate_SMAP_Lag1** | 2 | 1.0000 | 0.0017 | 0.0651 | 0.0454 | 0.0366 | 0.3587 | `SMAP_sm_pm_interp_rollrange30` |
| | | | | | | | | |
| **Clustering_V3_Full (Best K=3)** | 3 | 0.9663 | 0.0681 | 0.2091 | 0.1459 | **0.2202** | 0.8079 | `V_ema_LST_modis_kobs30` |
| **Clustering_Dynamic (K=3)** | 3 | 0.9688 | 0.0908 | 0.2697 | 0.1447 | **0.2123** | 0.6668 | `V_rollmin_LST_modis_kobs30` |
| **Univariate_G_API (K=3)** | 3 | 1.0000 | 0.1443 | 0.3039 | 0.1208 | **0.1980** | 0.6419 | `V_rollmin_LST_modis_kobs30` |
| **Static_Elevation** | 3 | 0.9996 | 0.0089 | 0.1733 | 0.1329 | 0.1707 | 0.9278 | `J_aspect_deg` |
| **Clustering_Hybrid** | 3 | 0.8650 | 0.0049 | 0.1381 | 0.1211 | 0.1606 | 0.6301 | `D_cos_DOY` |
| **Univariate_NDMI** | 3 | 1.0000 | 0.0365 | 0.1528 | 0.1016 | 0.1509 | 0.5632 | `V_ema_LST_modis_kobs30` |
| **Seasonal_3Class (Max VR)** | 3 | 0.9790 | 0.2687 | 0.4035 | 0.1079 | 0.1447 | 0.4719 | `V_rollmax_G_API_kobs30` |
| **Univariate_LST_Lag30** | 3 | 1.0000 | 0.2382 | 0.3297 | 0.0973 | 0.1393 | 0.5633 | `V_ema_LST_modis_kobs30` |
| **Static_Clay_Content** | 3 | 0.9792 | 0.0007 | 0.0867 | 0.0913 | 0.1089 | 0.7927 | `slope` |
| **Univariate_SMAP_Lag1** | 3 | 1.0000 | 0.0010 | 0.0651 | 0.0830 | 0.0763 | 0.5331 | `J_aspect_deg` |

---

## 2. Key Insights

### 1. Coarse SMAP Satellite Soil Moisture is a Poor Gating Proxy
- Across all tests, **SMAP (`Univariate_SMAP_Lag1`) performed worst**.
- Its target variance reduction was near-zero (~0.001) and feature relationship divergence was extremely low ($D_{Top20} = 0.036$ for $K=2$ and $0.076$ for $K=3$).
- *Why*: Point-scale soil moisture in Washington’s mountainous/complex terrain is highly heterogeneous. Coarse-resolution (~9km to 36km) SMAP pixels do not represent local station moisture variations well.

### 2. Best Binary Strategy ($K=2$): Univariate API Grouping
- Binning on **Antecedent Precipitation Index (`Univariate_G_API`)** yields the highest relationship divergence ($D_{Top20} = 0.1896$) and perfect class balance (entropy = 1.000).
- This divides observations into **dry days vs. wet days**. Features related to thermal dynamics (`V_rollmin_LST_modis_kobs30`) drift in correlation by more than **0.56** between these two states, showing that moisture behaves differently on rainy days vs. dry days.
- **Seasonal_Binary** and **Clustering_Dynamic** are close runners-up, showing that seasonal/meteorological boundaries provide strong physical groupings.

### 3. Best 3-Class Strategy ($K=3$): Multivariate Clustering on Full V3 Features
- **K-Means on all V3 Features (`Clustering_V3_Full`)** achieves the highest overall feature divergence ($D_{Top20} = 0.2202$).
- **Clustering_Dynamic** (clustering strictly on SMAP, API, and LST) is a very close second ($D_{Top20} = 0.2123$) and provides higher target variance reduction (0.0908 vs 0.0681).
- **Seasonal_3Class** provides the highest target variance reduction (0.2687) because soil moisture is strongly bimodal by season. However, its relationship divergence ($D_{Top20} = 0.1447$) is much lower. Since our primary goal is to adapt different feature sets to each regime, **Clustering_V3_Full** or **Clustering_Dynamic** are the superior choices.

---

## 3. Specialist Regime Breakdown (`Clustering_V3_Full`)

Evaluating the clusters generated by the K-Means full V3 model reveals distinct, physically interpretable regimes that naturally dictate which features should be selected.

### Cluster 0: The Wet / Topography-Driven Regime
* **Target Statistics**: Mean SM = **0.2454** (highest), std = 0.0883, N = 5,533.
* **Physical Interpretation**: This cluster captures wet conditions and locations. In this regime, the soil is generally wet and close to field capacity, meaning that local weather/precipitation fluctuations are less critical. Instead, pointing-scale soil moisture is dominated by **local terrain orientation and solar radiation exposure**.
* **Top 5 Correlated Features**:
  1. `J_aspect_deg` (0.5153) - Aspect/orientation of the slope.
  2. `lia_std_asc_deg` (0.4785) - Local Incidence Angle standard deviation (topography marker).
  3. `K_aspect_cos` (0.4107) - Cosine of aspect.
  4. `slope` (0.3480) - Steepness of the slope.
  5. `D_sin_DOY` (0.2725) - Seasonality.
* **Feature Recommendation**: Focus on **static topography, slope, aspect, and location features**.

### Cluster 1: The Transition / Precipitation-Driven Regime
* **Target Statistics**: Mean SM = **0.1920**, std = 0.1096, N = 3,400.
* **Physical Interpretation**: This cluster captures the transitional seasons/months where soil moisture is actively fluctuating. In this regime, the soil moisture is highly sensitive to immediate meteorological events.
* **Top 5 Correlated Features**:
  1. `D_sin_DOY` (0.6527) - Strong temporal seasonal control.
  2. `C_lag_LST_modis_kobs30` (0.6514) - Local temperature dynamics.
  3. `V_rollmean_LST_modis_kobs30` (0.5970) - Smoothed surface temperature.
  4. `V_rollmax_G_API_kobs30` (0.5751) - Rolling antecedent precipitation.
  5. `G_API` (0.5511) - Direct antecedent precipitation.
* **Feature Recommendation**: Focus heavily on **dynamic precipitation (API), rolling weather statistics, and seasonal indices (DOY, LST)**.

### Cluster 2: The Dry / Thermal-Driven Regime
* **Target Statistics**: Mean SM = **0.1838** (lowest), std = 0.1142, N = 6,771.
* **Physical Interpretation**: This cluster captures dry summer months and dry locations. Here, because precipitation is low, soil moisture depletion is primarily driven by **evapotranspiration** (represented by Land Surface Temperature) and solar exposure (aspect).
* **Top 5 Correlated Features**:
  1. `V_ema_LST_modis_kobs30` (0.5194) - EMA of Land Surface Temperature.
  2. `V_rollmean_LST_modis_kobs30` (0.5170) - Mean Land Surface Temperature.
  3. `C_lag_LST_modis_kobs30` (0.4982) - Lagged Land Surface Temperature.
  4. `V_rollmin_LST_modis_kobs30` (0.4810) - Min Land Surface Temperature.
  5. `V_rollmax_LST_modis_kobs14` (0.4633) - Short-term max LST.
* **Feature Recommendation**: Focus on **surface temperature profiles (LST MODIS averages, rollings, EMAs) and aspect**.

---

## 4. Diagnostic Figures

The diagnostic figures have been generated and saved under the experiment directory:
- **[gating_quality_metrics_comparison.png](./gating_quality_metrics_comparison.png)**: Bar plot comparing Divergence vs. Variance Reduction across all strategies for $K=2$ and $K=3$.
- **[best_gating_target_distributions.png](./best_gating_target_distributions.png)**: Target density overlay curves for the best binary (`Univariate_G_API`) and 3-class (`Clustering_V3_Full`) strategies.
- **[best_k3_correlation_drift_heatmap.png](./best_k3_correlation_drift_heatmap.png)**: Heatmap showing how the top 20 V3 feature correlations drift across the three clusters for `Clustering_V3_Full`, demonstrating the change in feature sensitivity.
