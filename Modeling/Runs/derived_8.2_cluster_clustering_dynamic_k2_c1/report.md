# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_clustering_dynamic_k2_c1
- Generated: 2026-07-14T16:23:17
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 47 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | C_lag_s2_b11_kobs30 |
| 2 | D_cos_DOY |
| 3 | D_sin_DOY |
| 4 | G_API |
| 5 | J_aspect_deg |
| 6 | J_bio_bio13 |
| 7 | J_bio_bio19 |
| 8 | J_clay_plus_sand_b0 |
| 9 | J_soil_texture_usda_b0 |
| 10 | J_soil_texture_usda_b10 |
| 11 | J_soil_texture_usda_b200 |
| 12 | K_aspect_cos |
| 13 | K_clay_plus_sand_b0 |
| 14 | V_rollrng_LST_modis_kobs30 |
| 15 | cos_year |
| 16 | latitude |
| 17 | lia_std_asc_deg |
| 18 | sin_year |
| 19 | slope |
| 20 | V_rollrng_F_NDVI_kobs14 |
| 21 | J_bio_bio12 |
| 22 | SMAP_sm_pm_interp_rollmean30 |
| 23 | G_DSLR |
| 24 | s2_b4 |
| 25 | SMAP_x_year |
| 26 | SMAP_sm_interp_rollrange30 |
| 27 | V_rollmean_E_SAR_diff_kobs30 |
| 28 | V_rollrng_E_SAR_ratio_kobs14 |
| 29 | V_rollrng_E_SAR_ratio_kobs30 |
| 30 | API_x_year |
| 31 | V_rollrng_G_API_kobs30 |
| 32 | V_ema_s2_b11_kobs30 |
| 33 | SMAP_ampm_diff_interp |
| 34 | V_rollmin_G_API_kobs30 |
| 35 | V_rollrng_SMAP_sm_interp_kobs30 |
| 36 | D_sa_F_NDMI |
| 37 | SMAP_sm_pm_interp_grad7 |
| 38 | J_bio_bio10 |
| 39 | SMAP_sm_am_interp_rollrange30 |
| 40 | A_grad_s2_b12_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
