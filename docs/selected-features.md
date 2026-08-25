# Selected Features Reference — `derived_8.0` (38) vs `derived_8.4` (54)

This document lists the exact feature sets used by the two canonical model families, with family assignment and a brief physical/mathematical description for each feature.

- **Sources of truth:** `derived_8.4` backbone is the 54-feature set from `notebooks/experiment/derived_8.4-feature-selection-2.0/artifacts/selected_features.json:336-391` (6-round greedy optimizing `unweighted_pooled_test_r2_2023_2025`), copied byte-identical to `notebooks/experiment/derived_8.4-eval-1.1/selected_features.json:2-56` and `notebooks/experiment/derived_8.4-eval-1.1/config.yaml:28-82` (`shared_backbone_54` / `Global_Single_54`). The legacy 50-feature `OVERALL_SELECTED_FEATURES_V0` at `data/splits/derived_8.4/dataset_metadata.py:7-57` (and `data/splits/derived_8.3/dataset_metadata.py:7`) is the `Baseline_V0_50` baseline, **not** the 54. See `docs/plans/20260812-clustering-methodlogy-audit.md:12` for leakage provenance.
- **`derived_8.0` 38** is Jakob's hand-selected `MDR-v25` set at `notebooks/training/MDR-v25.ipynb:Cell 7` (`FEATURE_COLS`, 38), mirrored at `Models/Temporal/lstm/train_v20.py:65-104` (`ANCHORS:jakob_feature_count=38`), described in `writeup/sections/shared/features.tex:65`. Distinct from the failed auto-selected 50 at `notebooks/experiment/derived_8.0-optimization-1.0/selected_features.json:1-51`. See `notebooks/experiment/derived_8.0-optimization-1.0/README.md:5-8`.
- **Pool:** both sets are subsets of the 499-column combinatorial pool (`docs/features.md:3`, `data/splits/derived_8.0/split_meta.json` / `data/splits/derived_8.4/split_meta.json`). Family taxonomy from `docs/features.md:46-74` and `Modeling/Src/soilmoist_fl/Features/groups.py:infer_family`; creation logic `src/pipeline/pipes/feature_pipe.py`.
- **Window convention:** suffix `_kobsK` = computed over last K valid *observations* (not calendar days) to handle S1 6–12 d revisit + cloud gaps; calendar hydrology uses true-day `G_rain_sum_3d/7d` (`docs/features.md:163-172,195`).

---

## 1. `derived_8.4` — 54 Features (`shared_backbone_54`)

| # | Feature | Family | Description |
|---|---------|--------|-------------|
| 1 | `precip_mm` | RAW (Weather, `docs/features.md:29`) | Daily ERA5-Land precipitation at station point (mm, ~11 km). |
| 2 | `s2_b4` | RAW (Sentinel-2 `docs/features.md:31`) | Sentinel-2 Red band B4 (665 nm) surface reflectance, 20 m, cloud-masked. |
| 3 | `s2_b8` | RAW (Sentinel-2) | Sentinel-2 NIR band B8 (842 nm) SR, 10 m; with B4 forms NDVI. |
| 4 | `SMAP_sm_pm_interp` | Interpolated SMAP `docs/features.md:130` | SMAP L3 PM (6 pm) soil moisture interpolated to daily (9 km → point, gap-filled). |
| 5 | `D_sin_DOY` | Calendar `docs/features.md:68` | `sin(2π·DOY/365)` — seasonal cycle phase. |
| 6 | `D_cos_DOY` | Calendar | `cos(2π·DOY/365)` — seasonal cycle quadrature. |
| 7 | `E_SAR_ratio` | SAR physics `docs/features.md:62` | Sentinel-1 physics descriptor `s1_vh/s1_vv` (vegetation/soil-moisture sensitive). |
| 8 | `G_API` | Hydrologic `docs/features.md:69` | Antecedent Precipitation Index — exponentially decayed hydrologic memory. |
| 9 | `G_DSLR` | Hydrologic | Days Since Last Rain — dry-down duration. |
| 10 | `G_rain_sum_3d` | Hydrologic (calendar, `docs/features.md:196`) | Calendar 3-day cumulative rainfall (mm, true days not `kobs`). |
| 11 | `G_rain_sum_7d` | Hydrologic (calendar) | Calendar 7-day cumulative rainfall. |
| 12 | `SMAP_sm_pm_interp_lag7` | SMAP lag `docs/features.md:135` | PM SMAP value shifted back 7 days (`lag7`). |
| 13 | `SMAP_sm_pm_interp_lag30` | SMAP lag | PM SMAP shifted back 30 days. |
| 14 | `SMAP_sm_pm_interp_rollrange7` | SMAP rolling `docs/features.md:139` | `rollmax7 − rollmin7` of PM SMAP — 7-day range. |
| 15 | `SMAP_sm_pm_interp_rollmean30` | SMAP rolling | 30-day rolling mean of PM SMAP. |
| 16 | `SMAP_sm_pm_interp_rollrange30` | SMAP rolling | 30-day rolling range of PM SMAP. |
| 17 | `SMAP_sm_interp_rollrange7` | SMAP rolling | 7-day rolling range of combined (AM+PM) interpolated SMAP. |
| 18 | `SMAP_ampm_diff_interp` | SMAP diurnal `docs/features.md:132` | `PM − AM` interpolated SMAP — diurnal dry-down signal. |
| 19 | `V_rollrng_G_API_kobs7` | V_rollrng `docs/features.md:53` | Rolling range of API over last 7 observations (`kobs`, not days). |
| 20 | `V_rollmax_F_NDMI_kobs30` | V_rollmax `docs/features.md:57` | Rolling max of NDMI (S2 SWIR-based moisture index) over 30 obs. |
| 21 | `A_d_E_SAR_ratio_kobs30` | A_d `docs/features.md:49` | First difference `x_t − x_{t−30obs}` of SAR ratio. |
| 22 | `V_rollmax_E_SAR_ratio_kobs7` | V_rollmax | Rolling max of SAR ratio over 7 obs. |
| 23 | `V_rollmin_E_SAR_ratio_kobs30` | V_rollmin `docs/features.md:56` | Rolling min of SAR ratio over 30 obs. |
| 24 | `V_rollmax_E_SAR_ratio_kobs30` | V_rollmax | Rolling max of SAR ratio over 30 obs. |
| 25 | `V_rollmin_LST_modis_kobs30` | V_rollmin | Rolling min of MODIS LST (1 km) over 30 obs — cold extreme. |
| 26 | `V_ema_LST_modis_kobs30` | V_ema `docs/features.md:58` | Exponential moving average of LST over 30 obs (recent-weighted). |
| 27 | `V_rollmax_F_NDVI_kobs14` | V_rollmax | Rolling max of NDVI (`(s2_b8−s2_b4)/(s2_b8+s2_b4)`) over 14 obs. |
| 28 | `V_rollmax_F_NDVI_kobs30` | V_rollmax | Rolling max of NDVI over 30 obs. |
| 29 | `V_ema_F_NDVI_kobs30` | V_ema | EMA of NDVI over 30 obs. |
| 30 | `C_lag_F_NDVI_kobs30` | C_lag `docs/features.md:50` | NDVI value lagged 30 obs. |
| 31 | `A_grad_E_SAR_diff_kobs30` | A_grad `docs/features.md:51` | Linear slope (gradient) of SAR difference `s1_vh − s1_vv` over 30 obs. |
| 32 | `V_rollmax_E_SAR_diff_kobs14` | V_rollmax | Rolling max of SAR difference over 14 obs. |
| 33 | `V_rollrng_E_SAR_diff_kobs30` | V_rollrng | Rolling range of SAR difference over 30 obs. |
| 34 | `V_rollmax_E_SAR_diff_kobs30` | V_rollmax | Rolling max of SAR difference over 30 obs. |
| 35 | `A_grad_s2_b11_kobs30` | A_grad | Slope of Sentinel-2 SWIR1 B11 (1610 nm) over 30 obs — moisture trend. |
| 36 | `V_rollrng_s2_b11_kobs30` | V_rollrng | Rolling range of B11 over 30 obs. |
| 37 | `V_rollmin_s2_b11_kobs30` | V_rollmin | Rolling min of B11 over 30 obs. |
| 38 | `V_rollmin_s2_b12_kobs30` | V_rollmin | Rolling min of B12 (SWIR2, 2190 nm) over 30 obs. |
| 39 | `A_d_SMAP_sm_interp_kobs30` | A_d | First difference of combined SMAP over 30 obs — wetting/drying step. |
| 40 | `V_rollmin_SMAP_sm_interp_kobs14` | V_rollmin | Rolling min of combined SMAP over 14 obs. |
| 41 | `V_rollmin_SMAP_sm_interp_kobs30` | V_rollmin | Rolling min of combined SMAP over 30 obs. |
| 42 | `E_rough_s1_vh_kobs14` | E_rough `docs/features.md:61` | Surface roughness proxy: rolling std of `s1_vh` over 14 obs. |
| 43 | `J_aspect_deg` | Static GIS `docs/features.md:71` | SRTM-derived aspect (0–360°) at station, 30 m. |
| 44 | `J_bio_bio02` | Bioclim `docs/features.md:72` | WorldClim BIO02: Mean Diurnal Temperature Range (1970–2000, ~1 km). |
| 45 | `J_bio_bio13` | Bioclim | WorldClim BIO13: Precipitation of Wettest Month. |
| 46 | `J_lc_code` | Land cover `docs/features.md:73` | ESA WorldCover / NLCD land-cover class code (10–30 m). |
| 47 | `J_soil_texture_usda_b0` | HWSD `docs/features.md:72` | FAO HWSD USDA texture class at 0 cm depth (~1 km). |
| 48 | `sin_year` | Calendar `docs/features.md:72` | `sin(2π·year_frac)` — multi-year cyclic trend. |
| 49 | `cos_year` | Calendar | `cos(2π·year_frac)` — multi-year quadrature. |
| 50 | `SMAP_x_year` | Interaction `docs/features.md:74` | `SMAP_sm_interp × year` — long-term sensor drift interaction. |
| 51 | `D_z_F_NDMI` | Seasonal Z-anomaly `docs/features.md:66` | Z-scored NDMI anomaly (climatology mean subtracted / std). |
| 52 | `D_z_LST_modis` | Seasonal Z-anomaly | Z-scored LST anomaly — thermal deviation from seasonal norm. |
| 53 | `D_fft_dom_LST_modis_kobs30` | Spectral `docs/features.md:67` | Dominant FFT frequency of LST over 30 obs — periodicity. |
| 54 | `D_fft_ent_LST_modis_kobs30` | Spectral | Spectral entropy of LST over 30 obs — signal complexity/noise. |

MoE winner `derived_8.4-feature-selection-2.0/artifacts/selected_features.json:392-405` (`delta_c0_0_c1_10`, pooled R² 0.815) reuses this 54 as base; `cluster_1` = 64 via 10 deltas: `V_rollmin_F_NDMI_kobs30;V_rollmean_G_API_kobs14;J_bio_bio04;J_bio_bio07;DOY;SMAP_sm_am_interp;SMAP_sm_am_interp_lag1;C_lag_F_NDMI_kobs30;J_bio_bio06;SMAP_sm_pm_interp_lag1`.

---

## 2. `derived_8.0` — 38 Features (Jakob `MDR-v25`)

| # | Feature | Family | Description |
|---|---------|--------|-------------|
| 1 | `SMAP_sm_pm_interp_ema02` | SMAP EMA `docs/features.md:140` | EMA of PM SMAP with α=0.2 — smoothed PM moisture memory. |
| 2 | `V_rollmin_LST_modis_kobs30` | V_rollmin `docs/features.md:56` | Rolling min of MODIS LST over 30 obs — cold extreme. |
| 3 | `D_sin_DOY` | Calendar `docs/features.md:68` | `sin(2π·DOY/365)` — seasonal cycle phase. |
| 4 | `G_rain_sum_3d` | Hydrologic (calendar) | Calendar 3-day cumulative rainfall (mm). |
| 5 | `V_ema_G_API_kobs7` | V_ema `docs/features.md:58` | EMA of API over 7 obs — recent-weighted wetness memory. |
| 6 | `V_rollmin_G_API_kobs30` | V_rollmin | Rolling min of API over 30 obs — driest recent spell. |
| 7 | `G_rain_sum_7d` | Hydrologic (calendar) | Calendar 7-day cumulative rainfall. |
| 8 | `C_lag_LST_modis_kobs30` | C_lag `docs/features.md:50` | LST lagged 30 obs. |
| 9 | `C_lag_G_API_kobs1` | C_lag | API lagged 1 obs — immediate antecedent state. |
| 10 | `V_ema_G_API_kobs14` | V_ema | EMA of API over 14 obs. |
| 11 | `V_rollmean_G_API_kobs14` | V_rollmean `docs/features.md:55` | Rolling mean of API over 14 obs. |
| 12 | `G_API` | Hydrologic `docs/features.md:69` | Antecedent Precipitation Index — exponentially decayed memory. |
| 13 | `G_DSLR` | Hydrologic | Days Since Last Rain — dry-down duration. |
| 14 | `SMAP_ampm_diff_interp` | SMAP diurnal `docs/features.md:132` | `PM − AM` interpolated SMAP — diurnal dry-down signal. |
| 15 | `V_rollmax_G_API_kobs30` | V_rollmax `docs/features.md:57` | Rolling max of API over 30 obs — wettest recent spell. |
| 16 | `V_ema_G_API_kobs30` | V_ema | EMA of API over 30 obs. |
| 17 | `V_rollmean_s2_b11_kobs7` | V_rollmean | Rolling mean of Sentinel-2 SWIR1 B11 over 7 obs. |
| 18 | `V_ema_LST_modis_kobs7` | V_ema | EMA of LST over 7 obs. |
| 19 | `V_rollmean_G_API_kobs7` | V_rollmean | Rolling mean of API over 7 obs. |
| 20 | `C_lag_s2_b11_kobs30` | C_lag | B11 (SWIR1) lagged 30 obs. |
| 21 | `A_d_E_SAR_diff_kobs14` | A_d `docs/features.md:49` | First difference `x_t − x_{t−14obs}` of SAR difference. |
| 22 | `C_lag_LST_modis_kobs6` | C_lag | LST lagged 6 obs. |
| 23 | `A_d_LST_modis_kobs14` | A_d | First difference of LST over 14 obs. |
| 24 | `A_d_SMAP_sm_interp_kobs14` | A_d | First difference of combined SMAP over 14 obs. |
| 25 | `V_rollstd_SMAP_sm_interp_kobs30` | V_rollstd `docs/features.md:52` | Rolling std (volatility) of combined SMAP over 30 obs. |
| 26 | `SMAP_sm_interp_grad7` | SMAP gradient `docs/features.md:137` | 7-day slope (linear trend) of combined SMAP. |
| 27 | `year_frac` | Calendar `docs/features.md:72` | Fractional year `year + DOY/365` — continuous time axis. |
| 28 | `sin_year` | Calendar | `sin(2π·year_frac)` — multi-year cyclic trend. |
| 29 | `cos_year` | Calendar | `cos(2π·year_frac)` — multi-year quadrature. |
| 30 | `API_x_year` | Interaction `docs/features.md:74` | `API × year` — API drift / non-stationarity interaction. |
| 31 | `SMAP_x_year` | Interaction | `SMAP_sm_interp × year` — SMAP drift interaction. |
| 32 | `slope` | RAW terrain `docs/features.md:35` | SRTM slope (°) at station, 30 m. |
| 33 | `elev` | RAW terrain | SRTM elevation (m) at station, 30 m. |
| 34 | `K_slope_sin` | Engineered terrain `docs/features.md:73` | `sin(slope)` — linearized slope effect. |
| 35 | `K_slope_cos` | Engineered terrain | `cos(slope)` — linearized slope effect. |
| 36 | `K_aspect_cos` | Engineered terrain | `cos(aspect)` — north-south exposure (avoids 0/360° discontinuity). |
| 37 | `J_clay_wfrac_b0` | HWSD `docs/features.md:72` | Clay weight fraction at 0 cm depth, FAO HWSD (~1 km). |
| 38 | `J_sand_wfrac_b0` | HWSD | Sand weight fraction at 0 cm depth, FAO HWSD (~1 km). |

Static subset `Models/Temporal/lstm/train_v20.py:106-114`: `slope, elev, K_slope_sin, K_slope_cos, K_aspect_cos, J_clay_wfrac_b0, J_sand_wfrac_b0` (7); remainder 31 time-varying.

---

## 3. Comparison

### 3a. Overlap Summary (10 / 82 unique)

- **Overlap (10):** `D_sin_DOY, G_API, G_DSLR, G_rain_sum_3d, G_rain_sum_7d, SMAP_ampm_diff_interp, SMAP_x_year, V_rollmin_LST_modis_kobs30, sin_year, cos_year` — core hydrologic + calendar + one SMAP diurnal + one LST volatility signal retained across eras.
- **Only in 54 (44):** `precip_mm, s2_b4, s2_b8, SMAP_sm_pm_interp, D_cos_DOY, E_SAR_ratio, SMAP_sm_pm_interp_lag7/lag30/rollrange7/rollmean30/rollrange30, SMAP_sm_interp_rollrange7, A_d_E_SAR_ratio_kobs30, A_grad_E_SAR_diff_kobs30, V_rollmax/rollmin/rollrng_E_SAR_*, E_rough_s1_vh_kobs14, V_rollmax_F_NDMI_kobs30, C_lag_F_NDVI_kobs30, V_ema_F_NDVI_kobs30, V_rollmax_F_NDVI_kobs14/30, A_grad_s2_b11_kobs30, V_rollrng/min_s2_b11/b12, A_d_SMAP_sm_interp_kobs30, V_rollmin_SMAP_sm_interp_kobs14/30, J_aspect_deg, J_bio_bio02/13, J_lc_code, J_soil_texture_usda_b0, D_z_F_NDMI, D_z_LST_modis, D_fft_dom/ent_LST_modis_kobs30, V_rollrng_G_API_kobs7, V_ema_LST_modis_kobs30`.
- **Only in 38 (28):** `V_ema_G_API_kobs7/14/30, V_rollmin_G_API_kobs30, V_rollmax_G_API_kobs30, V_rollmean_G_API_kobs7/14, C_lag_G_API_kobs1, C_lag_LST_modis_kobs30/kobs6, V_ema_LST_modis_kobs7, A_d_LST_modis_kobs14, V_rollmean_s2_b11_kobs7, C_lag_s2_b11_kobs30, A_d_E_SAR_diff_kobs14, A_d_SMAP_sm_interp_kobs14, V_rollstd_SMAP_sm_interp_kobs30, SMAP_sm_interp_grad7, SMAP_sm_pm_interp_ema02, year_frac, API_x_year, slope, elev, K_slope_sin/cos, K_aspect_cos, J_clay_wfrac_b0, J_sand_wfrac_b0`.

### 3b. Family-Level Shift

| Family | 38 | 54 | Overlap | Interpretation |
|---|---:|---:|---:|---|
| RAW terrain/weather/S2/SMAP | 3 (`slope`, `elev`, `precip` via `G_API`) | 4 (`precip_mm`, `s2_b4`, `s2_b8`, `SMAP_sm_pm_interp`) | 0 | 54 adds raw optical (Red/NIR) + raw precip + raw PM SMAP; 38 keeps raw terrain `slope`, `elev`. |
| Hydrologic `G_` (API/DSLR/rain) | 4 (`G_API`, `G_DSLR`, `G_rain_sum_3d`, `G_rain_sum_7d`) | 4 (same) | 4 | Hydrologic core fully retained. |
| SMAP lags/rolls/EMA/grad | 5 (`SMAP_sm_pm_interp_ema02`, `A_d_SMAP_sm_interp_kobs14`, `V_rollstd_SMAP_sm_interp_kobs30`, `SMAP_sm_interp_grad7`, `SMAP_ampm_diff_interp`) | 8 (`SMAP_sm_pm_interp_lag7`, `SMAP_sm_pm_interp_lag30`, `SMAP_sm_pm_interp_rollrange7`, `SMAP_sm_pm_interp_rollmean30`, `SMAP_sm_pm_interp_rollrange30`, `SMAP_sm_interp_rollrange7`, `A_d_SMAP_sm_interp_kobs30`, `V_rollmin_SMAP_sm_interp_kobs14`, `V_rollmin_SMAP_sm_interp_kobs30`, `SMAP_ampm_diff_interp`) | 1 (`SMAP_ampm_diff_interp`) | 54 shifts from EMA/std/grad to lag + range/mean/rollmin family. |
| SAR `E_`/`E_rough`/`A_d`/`A_grad`/`V_*_E_SAR` | 1 (`A_d_E_SAR_diff_kobs14`) | 10 (`E_SAR_ratio`, `A_d_E_SAR_ratio_kobs30`, `A_grad_E_SAR_diff_kobs30`, `V_rollmax_E_SAR_ratio_kobs7`, `V_rollmax_E_SAR_ratio_kobs30`, `V_rollmin_E_SAR_ratio_kobs30`, `V_rollmax_E_SAR_diff_kobs14`, `V_rollmax_E_SAR_diff_kobs30`, `V_rollrng_E_SAR_diff_kobs30`, `E_rough_s1_vh_kobs14`) | 0 | 54 massively expands SAR physics + volatility. |
| Vegetation `F_NDVI`/`F_NDMI`/`s2_b11`/`s2_b12` | 2 (`V_rollmean_s2_b11_kobs7`, `C_lag_s2_b11_kobs30`) | 10 (`V_rollmax_F_NDMI_kobs30`, `C_lag_F_NDVI_kobs30`, `V_ema_F_NDVI_kobs30`, `V_rollmax_F_NDVI_kobs14`, `V_rollmax_F_NDVI_kobs30`, `A_grad_s2_b11_kobs30`, `V_rollrng_s2_b11_kobs30`, `V_rollmin_s2_b11_kobs30`, `V_rollmin_s2_b12_kobs30`) | 0 | 54 adds full NDVI/NDMI/SWIR volatility; 38 minimal. |
| LST `V_*`/`C_lag`/`A_d` | 5 (`C_lag_LST_modis_kobs30`, `C_lag_LST_modis_kobs6`, `V_ema_LST_modis_kobs7`, `V_rollmin_LST_modis_kobs30`, `A_d_LST_modis_kobs14`) | 2 (`V_rollmin_LST_modis_kobs30`, `V_ema_LST_modis_kobs30`) | 1 (`V_rollmin_LST_modis_kobs30`) | 38 richer LST lags/EMA; 54 pruned to 2. |
| `G_API` dynamics `V_ema`/`V_rollmean`/`V_rollmin`/`V_rollmax`/`V_rollrng`/`C_lag` | 8 (`V_ema_G_API_kobs7`, `V_ema_G_API_kobs14`, `V_ema_G_API_kobs30`, `V_rollmean_G_API_kobs7`, `V_rollmean_G_API_kobs14`, `V_rollmin_G_API_kobs30`, `V_rollmax_G_API_kobs30`, `C_lag_G_API_kobs1`) | 1 (`V_rollrng_G_API_kobs7`) | 0 | Biggest shift: 54 drops 7/8 G_API variants for single volatility. |
| `V_roll*` total (all bases) | 7 | 17 | 1 | 54 trades breadth for SAR/vegetation/SMAP. |
| Calendar `D_sin`/`D_cos`/`sin_year`/`cos_year`/`year_frac`/`DOY` | 4 (`D_sin_DOY`, `sin_year`, `cos_year`, `year_frac`) | 4 (`D_sin_DOY`, `D_cos_DOY`, `sin_year`, `cos_year`) | 3 | 54 swaps `year_frac` → `D_cos_DOY`. |
| Interaction `*_x_year` | 2 (`API_x_year`, `SMAP_x_year`) | 1 (`SMAP_x_year`) | 1 | 54 drops `API_x_year`. |
| Static `J_`/`K_` (HWSD/WorldClim/terrain) | 7 (`slope`, `elev`, `K_slope_sin`, `K_slope_cos`, `K_aspect_cos`, `J_clay_wfrac_b0`, `J_sand_wfrac_b0`) | 5 (`J_aspect_deg`, `J_bio_bio02`, `J_bio_bio13`, `J_lc_code`, `J_soil_texture_usda_b0`) | 0 | 38 = topographic + texture fractions; 54 = bioclim + land cover + texture class. |
| Seasonal/Spectral `D_z`/`D_fft` | 0 | 4 (`D_z_F_NDMI`, `D_z_LST_modis`, `D_fft_dom_LST_modis_kobs30`, `D_fft_ent_LST_modis_kobs30`) | 0 | 54 adds Z-anomalies + Fourier; 38 none. |

### 3c. Full Presence Matrix (82 unique features, sorted alphabetically)

| Feature | In 38 | In 54 | Family | Description |
|---|:---:|:---:|---|---|
| `A_d_E_SAR_diff_kobs14` | ✓ |  | A_d | First difference of SAR difference `s1_vh−s1_vv` over 14 obs. |
| `A_d_E_SAR_ratio_kobs30` |  | ✓ | A_d | First difference of SAR ratio `s1_vh/s1_vv` over 30 obs. |
| `A_d_LST_modis_kobs14` | ✓ |  | A_d | First difference of MODIS LST over 14 obs. |
| `A_d_SMAP_sm_interp_kobs14` | ✓ |  | A_d | First difference of combined SMAP over 14 obs. |
| `A_d_SMAP_sm_interp_kobs30` |  | ✓ | A_d | First difference of combined SMAP over 30 obs — wetting/drying step. |
| `A_grad_E_SAR_diff_kobs30` |  | ✓ | A_grad | Linear slope of SAR difference over 30 obs. |
| `A_grad_s2_b11_kobs30` |  | ✓ | A_grad | Slope of Sentinel-2 SWIR1 B11 (1610 nm) over 30 obs. |
| `API_x_year` | ✓ |  | Interaction | `API × year` — API drift / non-stationarity interaction. |
| `C_lag_F_NDVI_kobs30` |  | ✓ | C_lag | NDVI lagged 30 obs. |
| `C_lag_G_API_kobs1` | ✓ |  | C_lag | API lagged 1 obs — immediate antecedent state. |
| `C_lag_LST_modis_kobs30` | ✓ |  | C_lag | LST lagged 30 obs. |
| `C_lag_LST_modis_kobs6` | ✓ |  | C_lag | LST lagged 6 obs. |
| `C_lag_s2_b11_kobs30` | ✓ |  | C_lag | B11 (SWIR1) lagged 30 obs. |
| `D_cos_DOY` |  | ✓ | Calendar | `cos(2π·DOY/365)` — seasonal quadrature. |
| `D_fft_dom_LST_modis_kobs30` |  | ✓ | Spectral | Dominant FFT frequency of LST over 30 obs — periodicity. |
| `D_fft_ent_LST_modis_kobs30` |  | ✓ | Spectral | Spectral entropy of LST over 30 obs — signal complexity. |
| `D_sin_DOY` | ✓ | ✓ | Calendar | `sin(2π·DOY/365)` — seasonal cycle phase. |
| `D_z_F_NDMI` |  | ✓ | Seasonal Z-anomaly | Z-scored NDMI anomaly (climatology mean/std). |
| `D_z_LST_modis` |  | ✓ | Seasonal Z-anomaly | Z-scored LST anomaly — thermal deviation. |
| `E_rough_s1_vh_kobs14` |  | ✓ | E_rough | Surface roughness proxy: rolling std of `s1_vh` over 14 obs. |
| `E_SAR_ratio` |  | ✓ | SAR physics | Sentinel-1 physics descriptor `s1_vh/s1_vv`. |
| `G_API` | ✓ | ✓ | Hydrologic | Antecedent Precipitation Index — exponentially decayed memory. |
| `G_DSLR` | ✓ | ✓ | Hydrologic | Days Since Last Rain — dry-down duration. |
| `G_rain_sum_3d` | ✓ | ✓ | Hydrologic (calendar) | Calendar 3-day cumulative rainfall (mm). |
| `G_rain_sum_7d` | ✓ | ✓ | Hydrologic (calendar) | Calendar 7-day cumulative rainfall. |
| `J_aspect_deg` |  | ✓ | Static GIS | SRTM-derived aspect (0–360°), 30 m. |
| `J_bio_bio02` |  | ✓ | Bioclim | WorldClim BIO02: Mean Diurnal Range. |
| `J_bio_bio13` |  | ✓ | Bioclim | WorldClim BIO13: Precipitation of Wettest Month. |
| `J_clay_wfrac_b0` | ✓ |  | HWSD | Clay weight fraction at 0 cm, FAO HWSD (~1 km). |
| `J_lc_code` |  | ✓ | Land cover | ESA WorldCover / NLCD land-cover class code. |
| `J_sand_wfrac_b0` | ✓ |  | HWSD | Sand weight fraction at 0 cm, FAO HWSD (~1 km). |
| `J_soil_texture_usda_b0` |  | ✓ | HWSD | FAO HWSD USDA texture class at 0 cm. |
| `K_aspect_cos` | ✓ |  | Engineered terrain | `cos(aspect)` — north-south exposure. |
| `K_slope_cos` | ✓ |  | Engineered terrain | `cos(slope)` — linearized slope. |
| `K_slope_sin` | ✓ |  | Engineered terrain | `sin(slope)` — linearized slope. |
| `SMAP_ampm_diff_interp` | ✓ | ✓ | SMAP diurnal | `PM − AM` interpolated SMAP — diurnal dry-down. |
| `SMAP_sm_interp_grad7` | ✓ |  | SMAP gradient | 7-day slope (linear trend) of combined SMAP. |
| `SMAP_sm_interp_rollrange7` |  | ✓ | SMAP rolling | 7-day rolling range of combined SMAP. |
| `SMAP_sm_pm_interp` |  | ✓ | Interpolated SMAP | SMAP L3 PM soil moisture interpolated to daily. |
| `SMAP_sm_pm_interp_ema02` | ✓ |  | SMAP EMA | EMA of PM SMAP with α=0.2 — smoothed PM moisture. |
| `SMAP_sm_pm_interp_lag30` |  | ✓ | SMAP lag | PM SMAP shifted back 30 days. |
| `SMAP_sm_pm_interp_lag7` |  | ✓ | SMAP lag | PM SMAP shifted back 7 days. |
| `SMAP_sm_pm_interp_rollmean30` |  | ✓ | SMAP rolling | 30-day rolling mean of PM SMAP. |
| `SMAP_sm_pm_interp_rollrange30` |  | ✓ | SMAP rolling | 30-day rolling range of PM SMAP. |
| `SMAP_sm_pm_interp_rollrange7` |  | ✓ | SMAP rolling | 7-day rolling range of PM SMAP. |
| `SMAP_x_year` | ✓ | ✓ | Interaction | `SMAP_sm_interp × year` — long-term drift interaction. |
| `V_ema_F_NDVI_kobs30` |  | ✓ | V_ema | EMA of NDVI over 30 obs. |
| `V_ema_G_API_kobs14` | ✓ |  | V_ema | EMA of API over 14 obs. |
| `V_ema_G_API_kobs30` | ✓ |  | V_ema | EMA of API over 30 obs. |
| `V_ema_G_API_kobs7` | ✓ |  | V_ema | EMA of API over 7 obs — recent-weighted wetness. |
| `V_ema_LST_modis_kobs30` |  | ✓ | V_ema | EMA of LST over 30 obs (recent-weighted). |
| `V_ema_LST_modis_kobs7` | ✓ |  | V_ema | EMA of LST over 7 obs. |
| `V_rollmax_E_SAR_diff_kobs14` |  | ✓ | V_rollmax | Rolling max of SAR difference over 14 obs. |
| `V_rollmax_E_SAR_diff_kobs30` |  | ✓ | V_rollmax | Rolling max of SAR difference over 30 obs. |
| `V_rollmax_E_SAR_ratio_kobs30` |  | ✓ | V_rollmax | Rolling max of SAR ratio over 30 obs. |
| `V_rollmax_E_SAR_ratio_kobs7` |  | ✓ | V_rollmax | Rolling max of SAR ratio over 7 obs. |
| `V_rollmax_F_NDMI_kobs30` |  | ✓ | V_rollmax | Rolling max of NDMI over 30 obs. |
| `V_rollmax_F_NDVI_kobs14` |  | ✓ | V_rollmax | Rolling max of NDVI over 14 obs. |
| `V_rollmax_F_NDVI_kobs30` |  | ✓ | V_rollmax | Rolling max of NDVI over 30 obs. |
| `V_rollmax_G_API_kobs30` | ✓ |  | V_rollmax | Rolling max of API over 30 obs — wettest recent spell. |
| `V_rollmean_G_API_kobs14` | ✓ |  | V_rollmean | Rolling mean of API over 14 obs. |
| `V_rollmean_G_API_kobs7` | ✓ |  | V_rollmean | Rolling mean of API over 7 obs. |
| `V_rollmean_s2_b11_kobs7` | ✓ |  | V_rollmean | Rolling mean of B11 (SWIR1) over 7 obs. |
| `V_rollmin_E_SAR_ratio_kobs30` |  | ✓ | V_rollmin | Rolling min of SAR ratio over 30 obs. |
| `V_rollmin_G_API_kobs30` | ✓ |  | V_rollmin | Rolling min of API over 30 obs — driest recent spell. |
| `V_rollmin_LST_modis_kobs30` | ✓ | ✓ | V_rollmin | Rolling min of LST over 30 obs — cold extreme. |
| `V_rollmin_SMAP_sm_interp_kobs14` |  | ✓ | V_rollmin | Rolling min of combined SMAP over 14 obs. |
| `V_rollmin_SMAP_sm_interp_kobs30` |  | ✓ | V_rollmin | Rolling min of combined SMAP over 30 obs. |
| `V_rollmin_s2_b11_kobs30` |  | ✓ | V_rollmin | Rolling min of B11 over 30 obs. |
| `V_rollmin_s2_b12_kobs30` |  | ✓ | V_rollmin | Rolling min of B12 (SWIR2) over 30 obs. |
| `V_rollrng_E_SAR_diff_kobs30` |  | ✓ | V_rollrng | Rolling range of SAR difference over 30 obs. |
| `V_rollrng_G_API_kobs7` |  | ✓ | V_rollrng | Rolling range of API over 7 obs. |
| `V_rollrng_s2_b11_kobs30` |  | ✓ | V_rollrng | Rolling range of B11 over 30 obs. |
| `V_rollstd_SMAP_sm_interp_kobs30` | ✓ |  | V_rollstd | Rolling std (volatility) of combined SMAP over 30 obs. |
| `cos_year` | ✓ | ✓ | Calendar | `cos(2π·year_frac)` — multi-year quadrature. |
| `elev` | ✓ |  | RAW terrain | SRTM elevation (m), 30 m. |
| `precip_mm` |  | ✓ | RAW weather | Daily ERA5-Land precipitation (mm, ~11 km). |
| `s2_b4` |  | ✓ | RAW S2 | Sentinel-2 Red B4 (665 nm) SR, 20 m. |
| `s2_b8` |  | ✓ | RAW S2 | Sentinel-2 NIR B8 (842 nm) SR, 10 m. |
| `sin_year` | ✓ | ✓ | Calendar | `sin(2π·year_frac)` — multi-year cyclic trend. |
| `slope` | ✓ |  | RAW terrain | SRTM slope (°), 30 m. |
| `year_frac` | ✓ |  | Calendar | Fractional year `year + DOY/365` — continuous time axis. |

Family shift summary: 54 trades 38's heavy G_API dynamics (8 G_API variants → 1) for broader SAR/S2/vegetation coverage, adds raw S2 red/NIR, `E_SAR_ratio`, seasonal `D_z` + spectral `D_fft`, and bioclim/land-cover statics (`J_bio02/13, J_lc_code, J_soil_texture`) in place of topographic `elev/slope/K_*` and texture fractions (`J_clay/sand`).
