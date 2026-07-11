# Features Analysis: Models V1, V2, V3, V4, and V5 Comparison

This document provides a detailed comparative analysis of the features used by **Model V1 (40 features)**, **Model V2 (40 features)**, **Model V3 (47 features)**, **Model V4 (50 features)**, and **Model V5 (32 features)** on the Washington-only `derived_8.2` dataset. The goal is to provide a physical and statistical explanation of these feature sets beyond simple pipeline selection metrics.

---

## 1. Feature Selection Configurations & Pipeline Context

The five feature sets represent the chronological and architectural evolution of the soil moisture modeling pipeline:

1. **Model V1 (40 Features)**:
   - **Pipeline Stages**: Mutual Information (MI) pre-filtering $\rightarrow$ ElasticNet linear selection $\rightarrow$ Bootstrap Stability Selection.
   - **Parameters**: `mi_k = 300`, `elasticnet_k = 60`, `stability_min_freq = 0.6`.
   - **Context**: The baseline pipeline stage. **No SMAP soil moisture data** was integrated yet. The pipeline also had **no autoregressive lags (`C_lag_`)**, **no temporal differences (`A_d_`)**, and **no temporal gradients (`A_grad_`)**. It relied heavily on raw observations and seasonal anomalies.

2. **Model V2 (40 Features)**:
   - **Pipeline Stages**: MI pre-filtering $\rightarrow$ ElasticNet linear selection $\rightarrow$ Bootstrap Stability Selection.
   - **Parameters**: `mi_k = 300`, `elasticnet_k = 60`, `stability_min_freq = 0.6`.
   - **Context**: The **Temporal Refinement stage**. This pipeline integrated coarse-scale SMAP soil moisture as a primary physical prior, and introduced structured temporal differences (`A_d_`), linear gradients (`A_grad_`), and autoregressive lags (`C_lag_`) calculated over local observation (`kobs`) windows to smooth out sensor noise and model temporal memory.

3. **Model V3 (47 Features)**:
   - **Pipeline Stages**: MI pre-filtering $\rightarrow$ ElasticNet linear selection $\rightarrow$ Bootstrap Stability Selection.
   - **Parameters**: `mi_k = 300`, `elasticnet_k = 60`, `stability_min_freq = 0.6`.
   - **Context**: The **Expanded Feature stage** (highest performer, $R^2 = 0.6474$). This pipeline added advanced spectral Fourier features (`D_fft_`) to capture temperature periodicities and multi-temporal Sentinel-1 roughness proxies (`E_rough_`) to decouple canopy/surface roughness from soil moisture dielectric changes.

4. **Model V4 (50 Features)**:
   - **Pipeline Stages**: **No MI stage** $\rightarrow$ ElasticNet linear selection $\rightarrow$ Bootstrap Stability Selection.
   - **Parameters**: `elasticnet_k = 80`, `stability_min_freq = 0.01` (keeps all features with bootstrap frequency above 1%, sorting to select the top 50).
   - **Context**: Skips the Mutual Information pre-filtering step, sending all 499 candidate features directly to the L1-regularized ElasticNet stage.

5. **Model V5 (32 Features)**:
   - **Pipeline Stages**: **No MI stage** $\rightarrow$ ElasticNet linear selection $\rightarrow$ Bootstrap Stability Selection.
   - **Parameters**: `elasticnet_k = 100`, `stability_min_freq = 0.6` (restores strict stability frequency threshold of 60%).
   - **Context**: Skips the MI stage and filters bootstrap outputs strictly at 60% frequency. *V5 features are exactly the top 32 most stable features from the V4 selection.*

---

## 2. Category Summary Table

The table below breaks down the number of features selected in each category across the five models, as well as their Union (all unique features across all configurations):

| Feature Category | V1 (40) | V2 (40) | V3 (47) | V4 (50) | V5 (32) | Union (92) | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Inputs (RAW)** | **9** | **4** | **5** | **0** | **0** | **9** | Coordinates, raw bands, raw LST/radar, DOY |
| **Static GIS & Soil Attributes** | **4** | **6** | **6** | **0** | **0** | **6** | HWSD soil textures (clay/sand), aspect, land cover code |
| **Bioclimatic Variables (`J_bio_`)** | **2** | **5** | **3** | **0** | **0** | **6** | Long-term regional precipitation/temp statistics |
| **Seasonal / Calendar (`D_`, `sin/cos_year`)** | **9** | **3** | **4** | **0** | **0** | **10** | Climatology anomalies (sa/z-scores), calendar cycles |
| **Spectral Fourier (`D_fft_`)** | **0** | **0** | **2** | **0** | **0** | **2** | Dominant frequency and spectral entropy of LST |
| **Hydrologic Inputs (`G_`)** | **1** | **2** | **2** | **0** | **0** | **2** | Raw API, Days Since Last Rain |
| **Local Incidence Angle (`lia_`)** | **1** | **1** | **1** | **2** | **2** | **2** | Radar orbital look geometries (mean/std) |
| **Surface Roughness Proxy (`E_rough_`)** | **0** | **0** | **2** | **2** | **1** | **4** | Sentinel-1 multi-temporal backscatter variance |
| **Temporal Anomalous Spike (`I_`)** | **0** | **0** | **0** | **1** | **1** | **1** | Quality control flag for non-physical radar jumps |
| **SMAP Interpolations (`SMAP_`)** | **0** | **2** | **2** | **5** | **3** | **6** | Coarse-resolution SMAP soil moisture indicators |
| **Temporal Difference (`A_d_`)** | **0** | **3** | **3** | **7** | **5** | **8** | Changes in satellite/weather values over $k$-obs |
| **Temporal Gradient (`A_grad_`)** | **0** | **2** | **3** | **7** | **5** | **7** | Moving slopes (rate of change) of physical values |
| **Autoregressive Lag (`C_lag_`)** | **0** | **2** | **2** | **5** | **4** | **5** | Lagged historical values shifted back by $k$-obs |
| **Rolling Operators (`V_roll*`, `V_ema_`)** | **14** | **10** | **12** | **21** | **11** | **40** | Moving average, minimum, maximum, range, CV, std dev |

---

## 3. Key Intersections & Shared Baselines

### A. The Baseline Calibration Feature (Intersection of V1 to V5)
There is only **one single feature** that is shared across every single version of the model:
* **`lia_std_asc_deg`** (Local Incidence Angle standard deviation during ascending passes).
* **Physical Significance**: Radar backscatter (VV, VH) is highly sensitive to the local terrain slope relative to the satellite's look direction. Every pipeline configuration—whether spatial or purely temporal, whether using raw data or lags—MUST include this calibration factor to correct Sentinel-1 radar intensity signals across Washington's complex topography.

### B. The Spatial Baseline (Intersection of V1, V2, and V3)
There are **15 features** shared across V1, V2, and V3. These represent the spatial and calendar baseline coordinates of the model:
* **Geospatial & Topography**: `latitude`, `slope`, `J_aspect_deg`, `K_aspect_cos`
* **Static Soil**: `J_soil_texture_usda_b0` (topsoil), `J_soil_texture_usda_b200` (deep drainage)
* **Regional Climate**: `J_bio_bio15` (precipitation seasonality)
* **Calendar Coordinates**: `DOY`, `D_cos_DOY`, `D_sin_DOY`, `sin_year`
* **Baseline Satellites & Weather**: `s2_b8` (NIR greenness), `V_rollmean_LST_modis_kobs30` (average temperature), `G_API` (antecedent precipitation memory)

*Note: All 15 of these baseline features were dropped in V4 and V5 because skipping the MI stage allowed ElasticNet to wipe out constant/calendar features.*

### C. The Temporal Backbone (Intersection of V2, V3, V4, and V5)
When we exclude the baseline V1 (which lacks temporal operators) and focus on SMAP-era models, **10 features** represent the core temporal backbone:
* `A_d_E_SAR_diff_kobs30` & `A_d_E_SAR_ratio_kobs30` (Long-term radar changes)
* `A_grad_E_SAR_diff_kobs30` & `A_grad_E_SAR_ratio_kobs30` (Long-term radar trends)
* `C_lag_F_NDVI_kobs30` & `C_lag_LST_modis_kobs30` (Historical vegetation and temperature lags)
* `SMAP_sm_pm_interp_rollrange30` & `V_rollmin_LST_modis_kobs30` (Regional moisture bounds & seasonal cold extremes)
* `V_rollrng_G_API_kobs7` & `lia_std_asc_deg` (Weather storm intensity & look angle correction)

---

## 4. Pipeline Evolution & Performance Transitions

### A. Model V1 $\rightarrow$ Model V2: The Transition to Lags, Trends, and SMAP ($R^2$: $0.6091 \rightarrow 0.6347$)
The transition from V1 to V2 represents a major architectural redesign of the data pipeline:

* **What V1 Lacked**: V1 had **zero** autoregressive lags (`C_lag_`), temporal differences (`A_d_`), temporal gradients (`A_grad_`), or SMAP soil moisture data.
* **V1's Climatological Anomalies**: V1 relied on raw satellite feeds (`LST_modis`, `E_SAR_ratio`, `F_NDMI`) and **seasonal anomalies** (`D_sa_*` and `D_z_*` standard deviations). Anomalies subtract the multi-year daily climatological mean from the current observation.
* **Why V2 is Superior**: 
  1. **SMAP Soil Moisture**: Introducing coarse-scale SMAP data provided the model with a direct physical prior of soil water content.
  2. **Replacing Anomalies with `kobs` Trends**: Seasonal anomalies are computationally intensive and sensitive to missing data or short historical records. By replacing them with differences (`A_d_`), gradients (`A_grad_`), and lags (`C_lag_`) calculated over observation (`kobs`) windows, the pipeline smoothed out raw satellite noise and modeled temporal memory in a robust, data-driven way.
  * **Result**: This transition yielded a significant **$+0.0256$ absolute $R^2$ improvement** ($0.6091 \rightarrow 0.6347$).

### B. Model V2 $\rightarrow$ Model V3: Decoupling Roughness & LST Periodicities ($R^2$: $0.6347 \rightarrow 0.6474$)
V3 expanded the candidate feature space from 450+ to 499 features by introducing advanced spectral and radar transformations:

* **Fourier Spectral LST (`D_fft_*`)**: Added dominant frequency and spectral entropy of temperature over 30 observations, capturing periodic solar cycles and temperature complexity.
* **Sentinel-1 Roughness Proxies (`E_rough_*`)**: Measures the multi-temporal variance of radar backscatter. This captures vegetation and surface roughness, allowing the tree-based model to decouple radar intensity changes caused by surface roughness from those caused by soil moisture.
  * **Result**: Adding these features yielded another **$+0.0127$ absolute $R^2$ improvement** ($0.6347 \rightarrow 0.6474$).

### C. Model V3 $\rightarrow$ Model V4/V5: The Spatial Collapse ($R^2$: $0.6474 \rightarrow 0.6088 / 0.5957$)
V4 and V5 evaluated the pipeline behavior when skipping the Mutual Information (MI) pre-filtering stage.
* **The L1 Lasso Trap**: Because static/spatial features (soil sand/clay fractions, latitude, elevation, land cover, bioclimates) are constant over time at individual stations, they explain *none* of the daily temporal variance. In a high-dimensional 499-column space, the L1 regularization in ElasticNet drives all constant features to **exactly zero**, favoring temporal variations.
* **Loss of Spatial Context**: Because V4 and V5 contain zero spatial features, they cannot differentiate between station characteristics (e.g. clay forests vs sandy croplands) under the same weather. This collapsed the global model into a purely temporal model, dropping performance back down to V1-like levels.

---

## 5. Comprehensive Feature Catalog

Below is the complete catalog of all 92 unique features across all five models, along with their category and physical/mathematical interpretation:

### A. Raw Inputs, GIS & Static Environmental Features

| Feature Name | Model(s) | Category | Physical / Mathematical Interpretation |
| :--- | :---: | :---: | :--- |
| `latitude` | V1, V2, V3 | RAW | Geographic latitude. Captures solar insolation limits and broad temperature gradients. |
| `slope` | V1, V2, V3 | RAW | Terrain slope angle. Controls the rate of surface water runoff vs infiltration. |
| `s2_b4` | V1, V3 | RAW | Sentinel-2 Band 4 (Red reflectance). Sensitive to soil color and vegetation chlorophyll. |
| `s2_b8` | V1, V2, V3 | RAW | Sentinel-2 Band 8 (Near-Infrared reflectance). Sensitive to canopy density and leaf structure. |
| `DOY` | V1, V2, V3 | RAW | Day of Year (1 to 365). Represents raw annual calendar progression. |
| `LST_modis` | V1 | RAW | Raw MODIS Land Surface Temperature. Volatile daily temperature reference. |
| `E_SAR_ratio` | V1 | RAW | Raw Sentinel-1 backscatter ratio (VV/VH). Volatile raw radar backscatter proxy. |
| `F_NDMI` | V1 | RAW | Raw Normalized Difference Moisture Index. Volatile raw vegetation moisture reference. |
| `F_MSI` | V1 | RAW | Raw Moisture Stress Index. Reflects canopy water stress. |
| `J_aspect_deg` | V1, V2, V3 | Static GIS | Aspect angle (0–360°). Determines solar radiation exposure. |
| `K_aspect_cos` | V1, V2, V3 | Static GIS | Cosine of aspect. Solves angular discontinuity (1 = North, -1 = South). |
| `J_lc_code` | V2, V3 | Static GIS | Copernicus land cover class. Dictates root depth and transpiration characteristics. |
| `J_soil_texture_usda_b0` | V1, V2, V3 | Static Soil | USDA soil texture class of topsoil (0cm). Determines topsoil water retention. |
| `J_soil_texture_usda_b10` | V2, V3 | Static Soil | USDA soil texture class of subsoil (10cm). Controls sub-surface percolation. |
| `J_soil_texture_usda_b200` | V1, V2, V3 | Static Soil | USDA soil texture class of deep soil (200cm). Governs deep drainage limits. |
| `J_bio_bio03` | V1 | Bioclimatic | Isothermality (diurnal range / annual range). Tracks local temperature stability. |
| `J_bio_bio12` | V2 | Bioclimatic | Annual Precipitation. Represents the long-term annual moisture baseline. |
| `J_bio_bio13` | V2 | Bioclimatic | Precipitation of the Wettest Month. Represents wet-season baseline. |
| `J_bio_bio15` | V1, V2, V3 | Bioclimatic | Precipitation Seasonality (CV of monthly rain). Tracks annual rainfall consistency. |
| `J_bio_bio16` | V2, V3 | Bioclimatic | Precipitation of the Wettest Quarter. Tracks extreme seasonal rainfall limits. |
| `J_bio_bio19` | V2, V3 | Bioclimatic | Precipitation of the Coldest Quarter. Tracks winter precipitation (often snowpack). |

### B. Climatological Anomalies, Cycles & Hydrologic Context

| Feature Name | Model(s) | Category | Physical / Mathematical Interpretation |
| :--- | :---: | :---: | :--- |
| `D_cos_DOY` / `D_sin_DOY` | V1, V2, V3 | Seasonal | Harmonic day of year transforms. Models smooth annual solar cycle. |
| `sin_year` | V1, V2, V3 | Seasonal | Sine of fractional year. Models inter-annual cycles. |
| `cos_year` | V3 | Seasonal | Cosine of fractional year. Complements `sin_year` to form continuous calendar coordinates. |
| `D_sa_LST_modis` | V1 | Seasonal | Seasonally adjusted MODIS LST anomaly. (Current LST - Daily climatological mean). |
| `D_sa_E_SAR_ratio` | V1 | Seasonal | Seasonally adjusted radar ratio anomaly. |
| `D_sa_F_NDMI` | V1 | Seasonal | Seasonally adjusted NDMI anomaly. |
| `D_z_LST_modis` | V1 | Seasonal | Z-scored MODIS LST anomaly. Standardized temperature deviation from climate mean. |
| `D_z_E_SAR_ratio` | V1 | Seasonal | Z-scored radar ratio anomaly. |
| `D_z_F_NDMI` | V1 | Seasonal | Z-scored NDMI anomaly. Standardized vegetation moisture deviation. |
| `D_fft_dom_LST_modis_kobs30` | V3 | Spectral | Dominant frequency from FFT of LST. Captures temperature seasonality cycles. |
| `D_fft_ent_LST_modis_kobs30` | V3 | Spectral | Spectral entropy of LST. Measures complexity of local temperature dynamics. |
| `G_API` | V1, V2, V3 | Hydrologic | Antecedent Precipitation Index. Tracks decayed rain storage. |
| `G_DSLR` | V2, V3 | Hydrologic | Days Since Last Rain. Measures length of dry spells to govern soil drydown. |
| `SMAP_x_year` | V2, V3 | Interaction | Interaction between SMAP and year. Models long-term satellite calibration drift. |

### C. Surface Roughness & Geometric Correction Features

| Feature Name | Model(s) | Category | Physical / Mathematical Interpretation |
| :--- | :---: | :---: | :--- |
| `lia_std_asc_deg` | V1-V5 | LIA | Std dev of ascending pass LIA. Calibrates topography-induced radar distortions (universal). |
| `lia_mean_asc_deg` | V4, V5 | LIA | Mean ascending pass Local Incidence Angle. Captures orbital viewing baseline. |
| `E_rough_s1_vh_kobs7` | V3 | Roughness | 7-obs variance of cross-polarized VH backscatter. Proxies canopy roughness. |
| `E_rough_s1_vh_kobs14` | V3 | Roughness | 14-obs variance of VH backscatter. Proxies long-term canopy/vegetation roughness. |
| `E_rough_s1_vv_kobs7` | V4 | Roughness | 7-obs variance of co-polarized VV backscatter. Proxies short-term ground roughness. |
| `E_rough_s1_vv_kobs14` | V4, V5 | Roughness | 14-obs variance of VV backscatter. Proxies long-term ground roughness. |

### D. SMAP and Quality Control Features (V4/V5 Heavy)

| Feature Name | Model(s) | Category | Physical / Mathematical Interpretation |
| :--- | :---: | :---: | :--- |
| `SMAP_sm_pm_interp_rollrange30` | V2, V3, V4, V5 | SMAP | 30-obs range (max-min) of evening SMAP. Seasonal regional soil moisture bounds. |
| `SMAP_sm_pm_interp_rollstd30` | V4, V5 | SMAP | 30-obs std dev of evening SMAP. Regional soil moisture volatility. |
| `SMAP_sm_pm_interp_rollmean7` | V4 | SMAP | 7-obs rolling mean of evening SMAP. Recent average regional soil moisture. |
| `SMAP_sm_pm_interp_lag1` | V4 | SMAP | Previous evening's SMAP soil moisture. Coarse temporal prior. |
| `SMAP_ampm_diff_interp` | V4, V5 | SMAP | Diurnal difference (AM - PM) of SMAP. Captures soil drying under solar heating. |
| `I_ts_spike_s1_vv` | V4, V5 | Spike | Binary indicator for spikes in VV backscatter. Filters out sensor noise. |

### E. Autoregressive Lags, Differences, and Gradients

| Feature Name | Model(s) | Category | Physical / Mathematical Interpretation |
| :--- | :---: | :---: | :--- |
| `C_lag_F_NDVI_kobs30` | V2, V3, V4, V5 | Lag | NDVI 30 observations ago. Historical vegetation baseline prior. |
| `C_lag_LST_modis_kobs30` | V2, V3, V4, V5 | Lag | MODIS LST 30 observations ago. Represents thermal history. |
| `C_lag_E_SAR_ratio_kobs30` | V4, V5 | Lag | Radar ratio 30 observations ago. Represents historical vegetation-soil state. |
| `C_lag_G_API_kobs1` | V4, V5 | Lag | Previous day's weather API. Immediate antecedent precipitation memory. |
| `C_lag_F_NDVI_kobs12` | V4 | Lag | NDVI 12 observations ago. Mid-term vegetation baseline prior. |
| `A_d_E_SAR_diff_kobs30` | V2, V3, V4, V5 | Difference | 30-obs difference in radar difference. Seasonal radar change trend. |
| `A_d_E_SAR_ratio_kobs30` | V2, V3, V4, V5 | Difference | 30-obs difference in radar ratio. Seasonal canopy-dielectric shifts. |
| `A_d_SMAP_sm_interp_kobs5` | V2, V3 | Difference | 5-obs difference in SMAP. Captures short-term (1-2 weeks) regional shifts. |
| `A_d_SMAP_sm_interp_kobs30` | V4, V5 | Difference | 30-obs difference in SMAP. Long-term (6-month) regional soil trends. |
| `A_d_F_NDMI_kobs30` | V4, V5 | Difference | 30-obs difference in NDMI. Tracks seasonal canopy moisture stress changes. |
| `A_d_LST_modis_kobs30` | V4, V5 | Difference | 30-obs difference in MODIS LST. Long-term thermal shifts. |
| `A_d_LST_modis_kobs14` | V4 | Difference | 14-obs difference in MODIS LST. Sub-seasonal temperature shift. |
| `A_d_E_SAR_ratio_kobs14` | V4 | Difference | 14-obs difference in radar ratio. Sub-seasonal canopy shifts. |
| `A_grad_E_SAR_diff_kobs30` | V2, V3, V4, V5 | Gradient | 30-obs slope of radar difference. Rate of change of radar backscattering. |
| `A_grad_E_SAR_ratio_kobs30` | V2, V3, V4, V5 | Gradient | 30-obs slope of radar ratio. Rate of change of vegetation density/water content. |
| `A_grad_SMAP_sm_interp_kobs30` | V3, V4, V5 | Gradient | 30-obs slope of SMAP. General rate of change of regional soil moisture. |
| `A_grad_F_NDMI_kobs30` | V4, V5 | Gradient | 30-obs slope of NDMI. Rate of change of vegetation moisture stress. |
| `A_grad_LST_modis_kobs30` | V4, V5 | Gradient | 30-obs slope of MODIS LST. Seasonal warming/cooling rate. |
| `A_grad_E_SAR_ratio_kobs14` | V4 | Gradient | 14-obs slope of radar ratio. Sub-seasonal vegetation rate of change. |
| `A_grad_LST_modis_kobs14` | V4 | Gradient | 14-obs slope of LST. Sub-seasonal warming/cooling rate. |

### F. Rolling Statistical Operators (Volatilities, Extremes, and Averages)

| Feature Name | Model(s) | Category | Physical / Mathematical Interpretation |
| :--- | :---: | :---: | :--- |
| `V_rollmin_LST_modis_kobs30` | V2, V3, V4, V5 | Rolling | 30-obs minimum of LST. Captures typical coldest temperature baseline. |
| `V_rollrng_G_API_kobs7` | V1-V5 | Rolling | 7-obs range of weather API. Measures short-term storm intensity. |
| `V_rollrng_E_SAR_ratio_kobs30` | V2, V3, V4 | Rolling | 30-obs range of radar ratio. Measures seasonal radar envelope width. |
| `V_rollmax_G_API_kobs30` | V2, V3, V4 | Rolling | 30-obs maximum of weather API. Tracks peak rainfall index over ~6 months. |
| `V_ema_LST_modis_kobs30` | V2, V3, V4 | Rolling | 30-obs EMA of LST. Exponentially decayed historical temperature. |
| `V_rollmean_LST_modis_kobs30` | V1, V2, V3 | Rolling | 30-obs moving average of LST. Average temperature seasonal trend. |
| `V_rollmin_s2_b11_kobs30` | V2, V3 | Rolling | 30-obs minimum of SWIR Band 11. Reflects peak soil/canopy water absorption. |
| `V_rollmax_LST_modis_kobs14` | V2, V3 | Rolling | 14-obs maximum of LST. Captures recent hot temperature extremes. |
| `V_rollmax_F_NDVI_kobs14` | V2, V3 | Rolling | 14-obs maximum of NDVI. Captures recent peak vegetation density. |
| `V_rollrng_F_NDVI_kobs30` | V2, V3 | Rolling | 30-obs range of NDVI. Seasonal amplitude of vegetation growth. |
| `V_rollrng_s2_b11_kobs30` | V3 | Rolling | 30-obs range of SWIR Band 11. Tracks surface moisture variability. |
| `V_rollmin_F_NDMI_kobs14` | V3 | Rolling | 14-obs minimum of NDMI. Represents peak vegetation water stress. |
| `V_ema_E_SAR_diff_kobs30` | V1 | Rolling | 30-obs EMA of radar difference (VV-VH). Captures smoothed historical radar levels. |
| `V_ema_E_SAR_ratio_kobs30` | V1 | Rolling | 30-obs EMA of radar ratio (VV/VH). |
| `V_ema_F_NDVI_kobs30` | V1 | Rolling | 30-obs EMA of NDVI. Smoothed historical vegetation baseline. |
| `V_rollmax_E_SAR_ratio_kobs30` | V1 | Rolling | 30-obs rolling maximum of radar ratio. Tracks peak seasonal vegetation/moisture. |
| `V_rollmax_F_NDMI_kobs7` | V1 | Rolling | 7-obs rolling maximum of NDMI. Recent peak canopy water content. |
| `V_rollmax_F_NDMI_kobs30` | V1 | Rolling | 30-obs rolling maximum of NDMI. Seasonal peak canopy water content. |
| `V_rollmax_s2_b11_kobs30` | V1 | Rolling | 30-obs rolling maximum of SWIR Band 11. Tracks peak dry-season ground reflectance. |
| `V_rollmean_E_SAR_diff_kobs30` | V1 | Rolling | 30-obs moving average of radar difference. |
| `V_rollmean_E_SAR_ratio_kobs30` | V1 | Rolling | 30-obs moving average of radar ratio. |
| `V_rollmean_F_NDMI_kobs30` | V1 | Rolling | 30-obs moving average of NDMI. |
| `V_rollmin_F_NDVI_kobs30` | V1 | Rolling | 30-obs moving minimum of NDVI. Tracks dry/winter leaf-off greenness baseline. |
| `V_rollmin_s2_b12_kobs30` | V1 | Rolling | 30-obs minimum of SWIR Band 12. Tracks peak water absorption. |
| `C_smm_E_SAR_diff_alpha0.85_n5` | V1 | Rolling | Smoothed moving average of radar difference. Double exponentially smoothed. |
| `V_rollcv_E_SAR_diff_kobs30` | V4, V5 | Rolling | 30-obs CV of radar difference. Normalized radar volatility. |
| `V_rollcv_G_API_kobs30` | V4, V5 | Rolling | 30-obs CV of weather API. Normalized precipitation volatility. |
| `V_rollmax_E_SAR_ratio_kobs30` | V4, V5 | Rolling | 30-obs rolling maximum of radar ratio. Represents peak wetness/greenness. |
| `V_rollrng_F_NDMI_kobs30` | V4, V5 | Rolling | 30-obs range of NDMI. Extreme swing of vegetation water content. |
| `V_rollrng_G_API_kobs30` | V4, V5 | Rolling | 30-obs range of weather API. Total seasonal water input variation. |
| `V_rollstd_G_API_kobs30` | V4, V5 | Rolling | 30-obs std dev of weather API. Seasonal weather volatility. |
| `V_rollrng_E_SAR_diff_kobs30` | V4, V5 | Rolling | 30-obs range of radar difference. Extreme radar response swings. |
| `V_rollstd_G_API_kobs7` | V4, V5 | Rolling | 7-obs std dev of weather API. Short-term rainfall volatility. |
| `V_rollstd_G_API_kobs14` | V4, V5 | Rolling | 14-obs std dev of weather API. Mid-term rainfall volatility. |
| `V_rollrng_G_API_kobs14` | V4 | Rolling | 14-obs range of weather API. Mid-term storm intensity. |
| `V_rollcv_G_API_kobs14` | V4 | Rolling | 14-obs CV of weather API. |
| `V_rollmean_F_NDVI_kobs30` | V4 | Rolling | 30-obs moving average of NDVI. Average vegetation density. |
| `V_rollcv_G_API_kobs7` | V4 | Rolling | 7-obs CV of weather API. |
| `V_rollcv_E_SAR_diff_kobs14` | V4 | Rolling | 14-obs CV of radar difference. |
| `V_rollmax_G_API_kobs7` | V4 | Rolling | 7-obs maximum of weather API. Peak storm index in the last 2 weeks. |
| `V_rollrng_E_SAR_diff_kobs14` | V4 | Rolling | 14-obs range of radar difference. |
