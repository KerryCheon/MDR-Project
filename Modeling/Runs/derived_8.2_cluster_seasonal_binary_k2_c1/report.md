# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_seasonal_binary_k2_c1
- Generated: 2026-07-14T16:29:05
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 34 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_F_NDMI_kobs30 |
| 2 | A_grad_F_NDMI_kobs30 |
| 3 | D_cos_DOY |
| 4 | D_sin_DOY |
| 5 | E_rough_s1_vv_kobs14 |
| 6 | G_API |
| 7 | J_aspect_deg |
| 8 | J_bio_bio03 |
| 9 | J_bio_bio13 |
| 10 | J_bio_bio19 |
| 11 | J_soil_texture_usda_b0 |
| 12 | J_soil_texture_usda_b10 |
| 13 | K_aspect_cos |
| 14 | V_rollmax_s2_b12_kobs14 |
| 15 | V_rollrng_LST_modis_kobs30 |
| 16 | cos_year |
| 17 | latitude |
| 18 | lia_std_asc_deg |
| 19 | sin_year |
| 20 | slope |
| 21 | G_rain_sum_7d |
| 22 | SMAP_sm_pm_interp_rollmean30 |
| 23 | V_rollrng_F_NDMI_kobs14 |
| 24 | G_rain_sum_3d |
| 25 | A_d_LST_modis_kobs30 |
| 26 | A_grad_LST_modis_kobs30 |
| 27 | A_d_E_SAR_ratio_kobs30 |
| 28 | A_grad_E_SAR_ratio_kobs30 |
| 29 | V_rollrng_E_SAR_diff_kobs30 |
| 30 | SMAP_sm_pm_interp_grad7 |
| 31 | J_lc_code |
| 32 | C_lag_s2_b12_kobs12 |
| 33 | precip_mm |
| 34 | V_rollrng_E_SAR_diff_kobs14 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
