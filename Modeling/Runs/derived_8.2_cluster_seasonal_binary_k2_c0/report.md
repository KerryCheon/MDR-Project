# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_seasonal_binary_k2_c0
- Generated: 2026-07-14T16:27:54
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 49 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_E_SAR_ratio_kobs30 |
| 2 | A_grad_E_SAR_ratio_kobs30 |
| 3 | D_cos_DOY |
| 4 | D_sin_DOY |
| 5 | J_aspect_deg |
| 6 | J_bio_bio02 |
| 7 | J_bio_bio16 |
| 8 | J_bio_bio19 |
| 9 | J_lc_code |
| 10 | J_soil_texture_usda_b200 |
| 11 | K_aspect_cos |
| 12 | K_slope_cos |
| 13 | V_rollmin_G_API_kobs30 |
| 14 | V_rollmin_LST_modis_kobs30 |
| 15 | V_rollrng_F_NDVI_kobs30 |
| 16 | lia_std_asc_deg |
| 17 | s2_b4 |
| 18 | s2_b8 |
| 19 | sin_year |
| 20 | D_sa_LST_modis |
| 21 | J_bio_bio15 |
| 22 | A_grad_SMAP_sm_interp_kobs30 |
| 23 | V_ema_LST_modis_kobs30 |
| 24 | V_rollmin_G_API_kobs14 |
| 25 | V_rollmin_LST_modis_kobs14 |
| 26 | A_d_E_SAR_diff_kobs30 |
| 27 | A_d_SMAP_sm_interp_kobs30 |
| 28 | G_DSLR |
| 29 | A_grad_E_SAR_diff_kobs30 |
| 30 | G_API |
| 31 | SMAP_sm_pm_interp_rollrange30 |
| 32 | SMAP_sm_pm_interp_lag30 |
| 33 | V_rollmin_s2_b11_kobs30 |
| 34 | cos_year |
| 35 | API_x_year |
| 36 | C_lag_F_NDVI_kobs30 |
| 37 | V_rollmax_LST_modis_kobs30 |
| 38 | J_bio_bio12 |
| 39 | E_rough_s1_vh_kobs14 |
| 40 | V_rollmin_s2_b12_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
