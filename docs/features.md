# MDR Soil Moisture Dataset: Feature Reference Guide

This document provides a comprehensive catalog and reference guide for the features available in the Spatio-temporal Soil Moisture modeling dataset. It is based on **`derived_8.0`** as the baseline reference, which contains **499 columns** (1 target variable, 2 metadata columns, 14 raw physical inputs, and 482 engineered/derived variables).

---

## 1. Overview & Dataset Structure

The `derived_8.0` dataset is designed for spatio-temporal soil moisture estimation using a combination of in-situ measurements, satellite remote sensing, weather models, and static environmental features. 

### Data Splits & Versioning
As documented in [split_meta.json](../data/splits/derived_8.0/split_meta.json):
* **Source Split**: Derived from `derived_7.0` (post-2016 stable SMAP-era split).
* **Train Years**: 2017–2020 (6,868 rows)
* **Val Years**: 2021–2022 (2,720 rows)
* **Test Years**: 2023–2025 (4,016 rows)
* **LIA Features**: Integrates station-level pass-specific Local Incidence Angle (LIA) statistics. See [smap_resplit.py](../data/splits/derived_8.0/smap_resplit.py) for the dataset merging script.

### Targets and Metadata
* **`soil_moisture_5cm`** (Target): Daily in-situ volumetric soil water content (m³/m³) measured at a depth of 5cm.
* **`station_id`** (Metadata): Unique string identifier for the weather/SNOTEL monitoring station.
* **`date`** (Metadata): Daily observation timestamp formatted as `YYYY-MM-DD`.

### Raw Data Sources & Resolution Scales

| Data Source Provider | GEE / Provider Catalog ID | Native Spatial Resolution | Pipeline Extraction Scale (`scale=`) | Temporal Resolution / Cadence | Mapped Raw / Key Features |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **In-Situ Monitoring Networks** | USCRN / USDA ISMN SNOTEL | **Point** | Point (`lat, lon`) | Daily (aggregated from 1-hr / daily) | `soil_moisture_5cm` (Target), `station_id` |
| **Open-Meteo Weather API** | ERA5-Land Reanalysis | **~11 km** ($0.1^\circ$ grid) | Point interpolation | Hourly $\to$ Daily | `precip_mm`, `rain_mm`, `G_API`, `G_rain_*` |
| **Copernicus Sentinel-1 SAR** | `COPERNICUS/S1_GRD` | **10 m** (IW Mode GRD) | `scale=30` (30 m) | 6–12 Days | `s1_vv`, `s1_vh`, `lia_*`, `E_rough_*`, `E_SAR_*` |
| **Copernicus Sentinel-2 MSI** | `COPERNICUS/S2_SR_HARMONIZED` | **10 m** (B4, B8) / **20 m** (B11, B12) | `scale=20` (20 m) | 5 Days | `s2_b4`, `s2_b8`, `s2_b11`, `s2_b12`, `F_NDVI`, `F_NDMI` |
| **NASA MODIS LST** | `MODIS/061/MOD11A1` | **1 km** (1000 m / 926.6 m) | `scale=1000` (1 km) | Daily | `LST_modis` |
| **NASA MODIS NDVI** | `MODIS/061/MOD13A3` | **1 km** (1000 m) | `scale=250` (250 m buffer) | Monthly (16-day composite) | `NDVI_modis` |
| **NASA SMAP Radiometer** | `NASA/SMAP/SPL3SMP_E/005` & `006` | **9 km** (EASE-Grid 2.0) | `scale=9000` (9 km) | Daily (AM ~6:00 AM, PM ~6:00 PM) | `SMAP_sm_am`, `SMAP_sm_pm`, `SMAP_sm_interp` |
| **USGS SRTM DEM** | `USGS/SRTMGL1_003` | **30 m** (1-Arc Second) | `scale=30` (30 m) | Static (2000 baseline) | `elev`, `slope`, `aspect`, `J_elev_m`, `J_slope_deg`, `J_aspect_deg` |
| **FAO HWSD v1.2** | Harmonized World Soil Database | **~1 km** (30-Arc Seconds) | Point lookup | Static | `J_clay_wfrac_b*`, `J_sand_wfrac_b*`, `J_soil_texture_usda_b*` |
| **WorldClim v2.1** | WorldClim 2.1 Climate Normals | **~1 km** (30-Arc Seconds) | Point lookup | Static (1970–2000 Climatological Normals) | `J_bio_bio01` to `J_bio_bio19` |
| **ESA WorldCover / NLCD** | Land Cover Classification | **10 m – 30 m** | Point lookup | Static / Annual | `J_lc_code` |

---

## 2. Feature Families (Prefix Conventions)

Features follow a strict prefix naming convention representing the mathematical transformation and the source data family. The system classifies columns into the families outlined below. (Refer to the helper grouping module in [groups.py](../Modeling/Src/soilmoist_fl/Features/groups.py)).

| Prefix/Family | Category | Description | Count |
| :--- | :--- | :--- | :---: |
| **`RAW`** | Physical Raw Inputs | Direct observations from satellite bands, meteorological measurements, or elevation profiles. | 14 |
| **`A_d_`** | Temporal Differences | First-order differences over a specified observation window. | 54 |
| **`C_lag_`** | Autoregressive Lags | Historical values shifted backward in the observation sequence. | 54 |
| **`A_grad_`** | Temporal Gradients | The rate of change (linear slope) of a feature over a rolling observation window. | 27 |
| **`V_rollstd_`** | Rolling Standard Deviation | Statistical volatility / variance of a feature over a window. | 27 |
| **`V_rollrng_`** | Rolling Range | Difference between rolling maximum and minimum (`max - min`). | 27 |
| **`V_rollcv_`** | Rolling Coefficient of Variation | Normalized dispersion metric, computed as `standard deviation / mean`. | 27 |
| **`V_rollmean_`** | Rolling Mean | Moving average of a feature over a specified observation window. | 27 |
| **`V_rollmin_`** | Rolling Minimum | Lowest value observed in the feature window. | 27 |
| **`V_rollmax_`** | Rolling Maximum | Highest value observed in the feature window. | 27 |
| **`V_ema_`** | Exponential Moving Average | Weighted moving average where exponentially more weight is given to recent observations. | 27 |
| **`C_smm_`** | Smoothed Moving Average | Double exponentially smoothed moving average. | 9 |
| **`A_pct_`** | Percent Change | Percentage rate of change relative to the preceding observation. | 9 |
| **`E_rough_`** | Surface Roughness Proxies | Multi-temporal variance estimates of Sentinel-1 backscatter to proxy terrain roughness. | 4 |
| **`E_`** | SAR Physics | Physical descriptors of SAR (Sentinel-1) signal ratios and differences. | 3 |
| **`I_`** | Temporal Anomalous Spikes | Indicators flags identifying sudden non-physical jumps in time series. | 1 |
| **`H_`** | Cross-Correlations | Moving Pearson correlation between two time series variables. | 4 |
| **`D_sa_`** | Seasonally Adjusted Anomalies | Residual values after subtracting the local daily climatological mean. | 3 |
| **`D_z_`** | Z-Scored Seasonal Anomalies | Standardized seasonal anomalies (climatology subtracted, divided by standard deviation). | 3 |
| **`D_fft_`** | Fourier / Spectral | Rolling FFT dominant frequency and spectral entropy features. | 6 |
| **`D_`** | Trigonometric Calendar | Sine and cosine transformations of calendar dates. | 2 |
| **`G_`** | Hydrologic & Gap-Filled | Antecedent Precipitation Indices (API), gap-filling flags, and cumulative rainfall. | 6 |
| **`lia_`** | Local Incidence Angle | Radar orbital look geometries (mean and standard deviation of LIA). | 4 |
| **`J_bio_`** | Bioclimatic variables | WorldClim variables BIO01 to BIO19 representing regional climate. | 19 |
| **`J_`** | Static Soil & GIS Attributes | HWSD physical parameters (sand, clay, soil texture class) and DEM statistics. | 24 |
| **`K_`** | Engineered Static Attributes | Trigonometric transformations of terrain parameters and sand-clay ratios. | 6 |
| **`Other`** | Miscellaneous | Interpolated satellite values and interaction features. | 55 |

---

## 3. Combinatorial Feature Breakdown (Math: 499 Columns)

Every single column in the dataset is generated through a rigorous combinatorial formulation. By understanding these groupings, you can ensure that you do not introduce redundant features.

### A. Metadata, Targets & Raw Inputs (17 columns)
* **Metadata & Targets (3)**: `station_id`, `date`, `soil_moisture_5cm`
* **Raw Geospatial, Weather & Satellite Inputs (14)**:
  * Spatial coordinates (2): `longitude`, `latitude` (Point location)
  * Terrain attributes (3): `elev`, `slope`, `aspect` (30 m spatial resolution via USGS SRTM DEM)
  * Time variables (1): `DOY` (Day of Year)
  * Meteorological (1): `precip_mm` (~11 km spatial resolution via Open-Meteo ERA5-Land)
  * Sentinel-1 backscatter (2): `s1_vv`, `s1_vh` (10 m native spatial resolution, C-band SAR)
  * Sentinel-2 raw spectral bands (4): `s2_b4` (Red, 10 m), `s2_b8` (NIR, 10 m), `s2_b11` (SWIR-1, 20 m), `s2_b12` (SWIR-2, 20 m)
  * MODIS raw temperature (1): `LST_modis` (1 km spatial resolution via MOD11A1)

### B. Core Time-Series Transforms (362 columns)
The temporal operators are systematically applied to **9 base variables**:
`s2_b11`, `s2_b12`, `F_NDVI`, `F_NDMI`, `LST_modis`, `E_SAR_ratio`, `E_SAR_diff`, `SMAP_sm_interp`, `G_API`.

* **Autoregressive Lags (`C_lag_`)**: Applied at $k \in \{1, 2, 5, 6, 12, 30\}$
  $$\text{Count: } 9 \text{ bases} \times 6 \text{ lags} = 54 \text{ columns}$$
* **Temporal Differences (`A_d_`)**: First-order difference $\Delta x = x_t - x_{t-k}$ applied at $k \in \{1, 2, 5, 7, 14, 30\}$
  $$\text{Count: } 9 \text{ bases} \times 6 \text{ windows} = 54 \text{ columns}$$
* **Temporal Gradients (`A_grad_`)**: Regression slope of the variable over window $k \in \{7, 14, 30\}$
  $$\text{Count: } 9 \text{ bases} \times 3 \text{ windows} = 27 \text{ columns}$$
* **Percent Change (`A_pct_`)**: Normalized rate of change $(x_t - x_{t-1}) / x_{t-1}$
  $$\text{Count: } 9 \text{ bases} \times 1 = 9 \text{ columns}$$
* **Smoothed Moving Averages (`C_smm_`)**: Double moving averages smoothed with parameter $\alpha=0.85$ and window size $n=5$
  $$\text{Count: } 9 \text{ bases} \times 1 = 9 \text{ columns}$$
* **Rolling Volatilities & Moving Averages (`V_roll*`, `V_ema_`)**: Rolling statistical transforms applied over $k \in \{7, 14, 30\}$:
  * Rolling Standard Deviation (`V_rollstd_`)
  * Rolling Range (`V_rollrng_`)
  * Rolling Coefficient of Variation (`V_rollcv_`)
  * Rolling Mean (`V_rollmean_`)
  * Rolling Minimum (`V_rollmin_`)
  * Rolling Maximum (`V_rollmax_`)
  * Exponential Moving Average (`V_ema_`)
  $$\text{Count: } 9 \text{ bases} \times 7 \text{ operators} \times 3 \text{ windows} = 189 \text{ columns}$$

### C. Advanced Temporal & Spectral Dynamics (20 columns)
* **Roughness Proxies (`E_rough_`)**: Rolling standard deviation of Sentinel-1 backscatter on both `s1_vv` and `s1_vh` over $k \in \{7, 14\}$ observations.
  $$\text{Count: } 2 \text{ channels} \times 2 \text{ windows} = 4 \text{ columns}$$
* **Cross-Correlations (`H_corr_`)**: Moving Pearson correlation between:
  1. `E_SAR_ratio` and `F_NDMI` (at $k \in \{7, 14\}$)
  2. `LST_modis` and `F_NDMI` (at $k \in \{7, 14\}$)
  $$\text{Count: } 2 \text{ pairs} \times 2 \text{ windows} = 4 \text{ columns}$$
* **Seasonal Anomalies (`D_sa_`, `D_z_`)**: Seasonally-adjusted and climatology-standardized z-scores computed for `F_NDMI`, `E_SAR_ratio`, and `LST_modis`.
  $$\text{Count: } 3 \text{ bases} \times 2 \text{ methods} = 6 \text{ columns}$$
* **Spectral Fourier Features (`D_fft_`)**: Rolling Fast Fourier Transform dominant frequency and spectral entropy computed over $k=30$ observations for `F_NDMI`, `E_SAR_ratio`, and `LST_modis`.
  $$\text{Count: } 3 \text{ bases} \times 2 \text{ spectral attributes} = 6 \text{ columns}$$

### D. SMAP Active Passive Interpolations (43 columns)
Separate morning (AM) and evening (PM) SMAP passes (9 km native spatial resolution on EASE-Grid 2.0 via SPL3SMP_E) are interpolated and derived as follows:
* **Base variables (3)**: `SMAP_sm_am_interp`, `SMAP_sm_pm_interp`, `SMAP_sm_interp`
* **Diurnal difference (1)**: `SMAP_ampm_diff_interp`
* **Derived indicators (39)**: For each of the 3 base variables, the following 13 metrics are computed:
  * Presence Mask: `_mask`
  * Historical Lags: `_lag1`, `_lag7`, `_lag30`
  * First-order difference: `_diff1`
  * Slope gradient: `_grad7`
  * Rate changes: `_pctchg`
  * Rolling features: `_rollmean7`, `_rollstd7`, `_rollrange7`, `_rollmean30`, `_rollstd30`, `_rollrange30`
  * Moving average: `_ema02`
  $$\text{Count: } 3 \text{ bases} \times 13 \text{ derived} = 39 \text{ columns}$$

### E. Static Soils, GIS & Bioclimatic variables (49 columns)
* **GIS Terrain Parameters (3)**: `J_aspect_deg`, `J_slope_deg`, `J_elev_m` (30 m spatial resolution via USGS SRTM DEM).
* **Engineered Terrain parameters (4)**: Sine and cosine transformations to solve angular discontinuities:
  * Aspect: `K_aspect_sin`, `K_aspect_cos`
  * Slope: `K_slope_sin`, `K_slope_cos`
* **HWSD Soil fractions at 6 depths (12)**: Clay weight fraction (`J_clay_wfrac_b*`) and Sand weight fraction (`J_sand_wfrac_b*`) at depths $\{0, 10, 30, 60, 100, 200\}$ cm (~1 km spatial resolution via FAO HWSD v1.2).
* **USDA Soil Classes at 6 depths (6)**: USDA Soil texture category code (`J_soil_texture_usda_b*`) at depths $\{0, 10, 30, 60, 100, 200\}$ cm (~1 km spatial resolution via FAO HWSD v1.2).
* **Soil Fraction Composites (5)**:
  * Land Cover code: `J_lc_code` (10 m – 30 m spatial resolution via ESA WorldCover / NLCD)
  * Soil mixture sum: `J_clay_plus_sand_b0`, `K_clay_plus_sand_b0`
  * Soil fraction ratios: `J_sand_clay_ratio_b0`, `K_sand_clay_ratio_b0`
* **WorldClim Climatological Seasonality (19)**: Long-term climate variables `J_bio_bio01` to `J_bio_bio19` (~1 km spatial resolution via WorldClim v2.1 30-arc-second climate normals).

### F. Calendar Cycles & Interaction Features (8 columns)
* **Temporal baseline (4)**: `year`, `year_frac`, `sin_year`, `cos_year` (cyclical calendar seasonality).
* **Interactions (2)**: `API_x_year`, `SMAP_x_year` (cross-multiplication interactions with long term years).
* **Sine/Cosine DOY (2)**: `D_sin_DOY`, `D_cos_DOY`.

---

## 4. Understanding Observation-Based Windows (`kobs`)

> [!IMPORTANT]
> **Observation-Based vs. Calendar-Based Time Windows**
> Features ending with the suffix **`_kobs<k>`** (e.g., `_kobs7`, `_kobs14`, `_kobs30`) represent calculations over the last $k$ **available observations**, not calendar days.

### Why `kobs` is Used:
Optical satellite data (Sentinel-2) is heavily obscured by clouds, and radar satellite data (Sentinel-1) operates on a 6-to-12-day revisit cycle. Consequently, a 30-calendar-day window might contain only 1 or 2 valid observation points. 
* By structuring windows around **observations** (steps in the database series), the statistics (mean, standard deviation, lags) are always calculated over a constant number of data points, ensuring mathematical stability during model training.

---

## 5. Local Incidence Angle (LIA) Features

Radar backscatter intensity is highly sensitive to the local terrain slope relative to the satellite's look direction. LIA features correct for these geometric distortions using statistics derived over the historical Sentinel-1 collection. See the extraction script at [fetch_lia.py](../data/splits/derived_8.0/LIA/fetch_lia.py).

* **`lia_mean_asc_deg` / `lia_std_asc_deg`**: The mean and standard deviation of local incidence angles during the satellite's **ascending** orbit passes (looking eastwards).
* **`lia_mean_desc_deg` / `lia_std_desc_deg`**: The mean and standard deviation of local incidence angles during the satellite's **descending** orbit passes (looking westwards).
* **`lia_mean_all_deg` / `lia_std_all_deg`**: The mean and standard deviation of local incidence angles aggregated across all orbit passes.

---

## 9. Feature Hygiene & Avoiding Redundant Features

To maintain dataset integrity and avoid model degradation due to multicollinearity (which increases overfitting risk and slows training), follow these feature hygiene guidelines before engineering new features:

### 1. Cross-Reference Before Coding
Check if your proposed feature is already present in the combinatorial mapping:
* **Rolling statistics**: Do not build a custom rolling mean or standard deviation on `NDVI`, `NDMI`, `SAR`, `LST`, or `SMAP`. They are already computed for windows $7$, $14$, and $30$ under `V_rollmean_`, `V_rollstd_`, etc.
* **Lags & Differences**: Lag values up to 30 steps (`C_lag_`) and differences (`A_d_`) are pre-computed. Do not manually shift the target or predictors unless testing a model architecture that specifically handles sequence steps natively.
* **Terrain Transformations**: Do not compute $\sin$ or $\cos$ values of aspects and slopes. They are already provided as `K_aspect_sin`, `K_aspect_cos`, `K_slope_sin`, and `K_slope_cos`.

### 2. Differentiate `kobs` and Calendar Days
If your task requires calendar-based hydrologic memory (e.g. cumulative rainfall over the past 3 days, 7 days, or 30 calendar days), use the **`G_rain_sum_3d`**, **`G_rain_sum_7d`**, and **`G_rain_sum_30d`** columns. These are calculated over true calendar days because weather precipitation data is daily and continuous.
* *Do not recreate rolling sums of precipitation using `kobs` indices*, as gaps in satellite observations would skew the precipitation sums.

### 3. Check Static GIS variables
Before extracting elevation, slope, aspect, USDA soil classification, clay/sand weights, or regional climates (annual precipitation, mean temperatures), check the **`J_`** and **`J_bio_`** groups. These static variables are already mapped at the station locations. Adding duplicate terrain parameters from a different digital elevation model (DEM) will introduce correlation issues without adding predictive signal.

### 4. Code Symbol Entry Points
* Feature creation logic is defined in [feature_pipe.py](../src/pipeline/pipes/feature_pipe.py).
* Filtering, sanity checking, and preprocessing of features are performed in [preprocess.py](../Modeling/Src/soilmoist_fl/Features/preprocess.py).
* Automated feature grouping by prefixes is maintained in [groups.py](../Modeling/Src/soilmoist_fl/Features/groups.py). Refer to `infer_family()` when writing any new family tag classifiers.
