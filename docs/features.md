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

## 3. Core Base Variables (Time-Series)

The temporal operators (difference, lag, gradient, rolling statistics) are applied systematically to a set of **9 primary base variables**:

1. **`s2_b11`**: Sentinel-2 Shortwave Infrared 1 (SWIR-1, ~1610 nm). Sensitive to soil moisture and leaf water content.
2. **`s2_b12`**: Sentinel-2 Shortwave Infrared 2 (SWIR-2, ~2190 nm). Sensitive to soil mineral composition and moisture.
3. **`F_NDVI`**: Gap-filled/Smoothed Normalized Difference Vegetation Index. Computed from Sentinel-2 NIR (`s2_b8`) and Red (`s2_b4`) bands:
   $$\text{NDVI} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}$$
4. **`F_NDMI`**: Gap-filled/Smoothed Normalized Difference Moisture Index. Computed from Sentinel-2 NIR (`s2_b8`) and SWIR-1 (`s2_b11`) bands:
   $$\text{NDMI} = \frac{\text{B8} - \text{B11}}{\text{B8} + \text{B11}}$$
5. **`LST_modis`**: Land Surface Temperature (Kelvin) retrieved from MODIS thermal bands.
6. **`E_SAR_ratio`**: Cross-polarization backscatter ratio derived from Sentinel-1:
   $$\text{SAR\_ratio} = \frac{\text{s1\_vv}}{\text{s1\_vh}}$$
7. **`E_SAR_diff`**: Polarization backscatter difference derived from Sentinel-1:
   $$\text{SAR\_diff} = \text{s1\_vv} - \text{s1\_vh}$$
8. **`SMAP_sm_interp`**: Multi-daily soil moisture estimate interpolated from Soil Moisture Active Passive (SMAP) L-band radiometry.
9. **`G_API`**: Antecedent Precipitation Index, modeling soil moisture decay ("hydrologic memory") from daily rainfall:
   $$\text{API}_t = \text{API}_{t-1} \cdot k + \text{precip}_t \quad (\text{with decay } k \approx 0.85 \text{ to } 0.90)$$

---

## 4. Understanding Observation-Based Windows (`kobs`)

> [!IMPORTANT]
> **Observation-Based vs. Calendar-Based Time Windows**
> Features ending with the suffix **`_kobs<k>`** (e.g., `_kobs7`, `_kobs14`, `_kobs30`) represent calculations over the last $k$ **available observations**, not calendar days.

### Why `kobs` is Used:
Optical satellite data (Sentinel-2) is heavily obscured by clouds, and radar satellite data (Sentinel-1) operates on a 6-to-12-day revisit cycle. Consequently, a 30-calendar-day window might contain only 1 or 2 valid observation points. 
* By structuring windows around **observations** (steps in the database series), the statistics (mean, standard deviation, lags) are always calculated over a constant number of data points, ensuring mathematical stability during model training.

### Configured Window Sizes:
* **Lags (`C_lag_`)**: $k \in \{1, 2, 5, 6, 12, 30\}$
* **Differences (`A_d_`)**: $k \in \{1, 2, 5, 7, 14, 30\}$
* **Gradients and Rolling Stats (`A_grad_`, `V_roll*`, `V_ema_`)**: $k \in \{7, 14, 30\}$ (analogous to short, medium, and long memory).

---

## 5. Raw Physical Features (`RAW`)

These are the baseline physical inputs ingested directly by the pipeline before any windowed engineering:

* **Coordinates & Elevation**: `longitude`, `latitude`, and `elev` (meters above sea level).
* **Local Terrain**: `slope` (terrain inclination in degrees) and `aspect` (terrain orientation in degrees).
* **Precipitation**: `precip_mm` (raw daily total precipitation).
* **Day of Year**: `DOY` (Julian calendar day, 1–366).
* **Sentinel-1 SAR Backscatter**: `s1_vv` and `s1_vh` (vertical-vertical and vertical-horizontal backscatter coefficients in dB).
* **Sentinel-2 Optical Bands**: `s2_b4` (Red), `s2_b8` (NIR), `s2_b11` (SWIR-1), and `s2_b12` (SWIR-2).
* **Land Surface Temperature**: `LST_modis` (MODIS thermal observation).

---

## 6. Static GIS & Soil Properties (`J_` and `K_`)

Static features capture the spatial context of SNOTEL and weather stations, allowing the models (particularly gating routers and spatial specialists) to generalize across different mountain sites in Washington state.

### HWSD Soil Physical Fractions (`J_` and `K_`)
Soil characteristics are extracted from the Harmonized World Soil Database (HWSD) at multiple depth layers: surface (`b0`), 10cm (`b10`), 30cm (`b30`), 60cm (`b60`), 100cm (`b100`), and 200cm (`b200`).
* **`J_clay_wfrac_b<depth>`**: Clay weight fraction in the soil layer.
* **`J_sand_wfrac_b<depth>`**: Sand weight fraction in the soil layer.
* **`J_soil_texture_usda_b<depth>`**: USDA numerical soil texture classification code.
* **`J_clay_plus_sand_b0`**: Sum of clay and sand fractions at the surface.
* **`J_sand_clay_ratio_b0`**: Ratio of sand weight to clay weight.
* **`J_lc_code`**: ESA Land Cover classification code.
* **`K_clay_plus_sand_b0` & `K_sand_clay_ratio_b0`**: Engineered versions of the surface soil fractions.

### Terrain Transformations (`K_`)
To prevent boundary discontinuities during modeling, slope and aspect angles are decomposed into sine and cosine components:
* **`K_slope_sin` / `K_slope_cos`**: $\sin(\text{slope})$ and $\cos(\text{slope})$
* **`K_aspect_sin` / `K_aspect_cos`**: $\sin(\text{aspect})$ and $\cos(\text{aspect})$

### WorldClim Bioclimatic variables (`J_bio_`)
These 19 variables capture long-term climatology patterns (annual trends, seasonality, and extreme temperature/precipitation parameters) associated with each station's location:

| Feature Name | Bioclimatic Variable (Description) |
| :--- | :--- |
| **`J_bio_bio01`** | Annual Mean Temperature |
| **`J_bio_bio02`** | Mean Diurnal Range (Mean of monthly max - min temp) |
| **`J_bio_bio03`** | Isothermality (BIO2 / BIO7) × 100 |
| **`J_bio_bio04`** | Temperature Seasonality (standard deviation × 100) |
| **`J_bio_bio05`** | Max Temperature of Warmest Month |
| **`J_bio_bio06`** | Min Temperature of Coldest Month |
| **`J_bio_bio07`** | Temperature Annual Range (BIO5 - BIO6) |
| **`J_bio_bio08`** | Mean Temperature of Wettest Quarter |
| **`J_bio_bio09`** | Mean Temperature of Driest Quarter |
| **`J_bio_bio10`** | Mean Temperature of Warmest Quarter |
| **`J_bio_bio11`** | Mean Temperature of Coldest Quarter |
| **`J_bio_bio12`** | Annual Precipitation |
| **`J_bio_bio13`** | Precipitation of Wettest Month |
| **`J_bio_bio14`** | Precipitation of Driest Month |
| **`J_bio_bio15`** | Precipitation Seasonality (Coefficient of Variation) |
| **`J_bio_bio16`** | Precipitation of Wettest Quarter |
| **`J_bio_bio17`** | Precipitation of Driest Quarter |
| **`J_bio_bio18`** | Precipitation of Warmest Quarter |
| **`J_bio_bio19`** | Precipitation of Coldest Quarter |

---

## 7. Local Incidence Angle (LIA) Features

Radar backscatter intensity is highly sensitive to the local terrain slope relative to the satellite's look direction. LIA features correct for these geometric distortions using statistics derived over the historical Sentinel-1 collection. See the extraction script at [fetch_lia.py](../data/splits/derived_8.0/LIA/fetch_lia.py).

* **`lia_mean_asc_deg` / `lia_std_asc_deg`**: The mean and standard deviation of local incidence angles during the satellite's **ascending** orbit passes (looking eastwards).
* **`lia_mean_desc_deg` / `lia_std_desc_deg`**: The mean and standard deviation of local incidence angles during the satellite's **descending** orbit passes (looking westwards).
* **`lia_mean_all_deg` / `lia_std_all_deg`**: The mean and standard deviation of local incidence angles aggregated across all orbit passes.

---

## 8. Miscellaneous & Interaction Features (`Other`)

* **`F_MSI`**: Moisture Stress Index computed as SWIR-1 / NIR:
  $$\text{MSI} = \frac{\text{s2\_b11}}{\text{s2\_b8}}$$
* **`SMAP_sm_am_interp` / `SMAP_sm_pm_interp`**: Separate morning (AM) and evening (PM) satellite soil moisture passes.
* **`SMAP_ampm_diff_interp`**: The daily diurnal soil moisture variation observed by SMAP:
  $$\text{SMAP\_ampm\_diff} = \text{SMAP\_sm\_am\_interp} - \text{SMAP\_sm\_pm\_interp}$$
* **`API_x_year`**: An interaction term crossing `G_API` with the year fraction, allowing the model to adapt rainfall memory response depending on long-term inter-annual trends.
* **`SMAP_x_year`**: An interaction term crossing `SMAP_sm_interp` with the year fraction.
* **`sin_year` / `cos_year`**: Sine and cosine of the year fraction, providing a smooth seasonal cyclicity signal.
