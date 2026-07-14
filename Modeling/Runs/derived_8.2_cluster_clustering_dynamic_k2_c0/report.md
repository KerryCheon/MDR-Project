# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_clustering_dynamic_k2_c0
- Generated: 2026-07-14T16:21:14
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
| 1 | A_d_E_SAR_diff_kobs30 |
| 2 | A_d_E_SAR_ratio_kobs30 |
| 3 | A_grad_E_SAR_diff_kobs30 |
| 4 | A_grad_E_SAR_ratio_kobs30 |
| 5 | C_lag_F_NDVI_kobs30 |
| 6 | D_cos_DOY |
| 7 | D_sin_DOY |
| 8 | G_rain_sum_7d |
| 9 | J_aspect_deg |
| 10 | J_bio_bio12 |
| 11 | J_bio_bio13 |
| 12 | J_bio_bio16 |
| 13 | J_bio_bio19 |
| 14 | J_lc_code |
| 15 | J_soil_texture_usda_b200 |
| 16 | K_aspect_cos |
| 17 | K_slope_cos |
| 18 | SMAP_sm_pm_interp_grad7 |
| 19 | SMAP_sm_pm_interp_rollrange30 |
| 20 | V_ema_LST_modis_kobs30 |
| 21 | V_rollmax_LST_modis_kobs30 |
| 22 | V_rollmin_LST_modis_kobs30 |
| 23 | V_rollmin_s2_b11_kobs30 |
| 24 | V_rollrng_F_NDVI_kobs30 |
| 25 | lia_std_asc_deg |
| 26 | sin_year |
| 27 | API_x_year |
| 28 | G_API |
| 29 | V_rollrng_F_NDMI_kobs30 |
| 30 | V_rollmin_G_API_kobs30 |
| 31 | V_rollmean_F_NDVI_kobs30 |
| 32 | D_fft_dom_LST_modis_kobs30 |
| 33 | SMAP_sm_pm_interp_rollmean30 |
| 34 | V_rollrng_E_SAR_ratio_kobs30 |
| 35 | G_rain_sum_3d |
| 36 | V_rollmax_G_API_kobs30 |
| 37 | V_ema_F_NDVI_kobs30 |
| 38 | V_rollmin_G_API_kobs14 |
| 39 | J_bio_bio02 |
| 40 | V_rollrng_E_SAR_diff_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
