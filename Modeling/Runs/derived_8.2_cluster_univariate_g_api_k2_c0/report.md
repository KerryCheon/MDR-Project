# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_univariate_g_api_k2_c0
- Generated: 2026-07-14T16:17:48
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 22 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_E_SAR_diff_kobs30 |
| 2 | A_grad_E_SAR_diff_kobs30 |
| 3 | C_lag_E_SAR_ratio_kobs30 |
| 4 | C_lag_F_NDVI_kobs30 |
| 5 | C_lag_LST_modis_kobs30 |
| 6 | D_sin_DOY |
| 7 | G_API |
| 8 | J_aspect_deg |
| 9 | SMAP_ampm_diff_interp |
| 10 | V_ema_G_API_kobs7 |
| 11 | V_rollmean_F_NDVI_kobs30 |
| 12 | V_rollmin_G_API_kobs7 |
| 13 | V_rollmin_LST_modis_kobs30 |
| 14 | V_rollrng_E_SAR_ratio_kobs30 |
| 15 | V_rollrng_F_NDMI_kobs30 |
| 16 | lia_mean_desc_deg |
| 17 | V_ema_F_NDVI_kobs30 |
| 18 | V_rollmin_G_API_kobs14 |
| 19 | SMAP_sm_pm_interp_rollrange30 |
| 20 | V_rollmax_E_SAR_ratio_kobs30 |
| 21 | J_lc_code |
| 22 | V_ema_G_API_kobs14 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
