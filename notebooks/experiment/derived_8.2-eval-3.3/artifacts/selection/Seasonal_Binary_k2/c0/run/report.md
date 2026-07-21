# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-19T21:01:50
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_E_SAR_diff_kobs30 |
| 2 | A_d_E_SAR_ratio_kobs30 |
| 3 | A_grad_E_SAR_diff_kobs30 |
| 4 | A_grad_E_SAR_ratio_kobs30 |
| 5 | A_grad_SMAP_sm_interp_kobs30 |
| 6 | DOY |
| 7 | G_rain_sum_30d |
| 8 | J_bio_bio15 |
| 9 | J_bio_bio16 |
| 10 | J_bio_bio19 |
| 11 | K_aspect_cos |
| 12 | K_aspect_sin |
| 13 | SMAP_sm_pm_interp_rollrange30 |
| 14 | SMAP_x_year |
| 15 | V_rollmax_G_API_kobs7 |
| 16 | V_rollmax_LST_modis_kobs30 |
| 17 | V_rollmin_LST_modis_kobs30 |
| 18 | V_rollmin_s2_b11_kobs30 |
| 19 | V_rollrng_F_NDVI_kobs30 |
| 20 | V_rollstd_G_API_kobs30 |
| 21 | lia_mean_desc_deg |
| 22 | lia_std_asc_deg |
| 23 | slope |
| 24 | A_d_SMAP_sm_interp_kobs30 |
| 25 | SMAP_ampm_diff_interp |
| 26 | V_rollmin_LST_modis_kobs14 |
| 27 | V_rollrng_s2_b12_kobs30 |
| 28 | J_sand_clay_ratio_b0 |
| 29 | K_sand_clay_ratio_b0 |
| 30 | SMAP_sm_pm_interp_lag30 |
| 31 | E_rough_s1_vh_kobs7 |
| 32 | V_rollmin_G_API_kobs14 |
| 33 | V_rollmin_G_API_kobs30 |
| 34 | C_lag_F_NDVI_kobs30 |
| 35 | V_rollrng_F_NDMI_kobs30 |
| 36 | s2_b8 |
| 37 | E_rough_s1_vh_kobs14 |
| 38 | J_bio_bio12 |
| 39 | V_ema_F_NDVI_kobs30 |
| 40 | V_rollmax_F_NDVI_kobs14 |

## Metrics

_No metrics provided._
