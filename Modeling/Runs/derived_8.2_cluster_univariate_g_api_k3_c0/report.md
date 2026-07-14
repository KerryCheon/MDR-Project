# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_univariate_g_api_k3_c0
- Generated: 2026-07-14T16:35:13
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 25 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | C_lag_E_SAR_ratio_kobs30 |
| 2 | C_lag_F_NDVI_kobs30 |
| 3 | C_lag_LST_modis_kobs30 |
| 4 | D_sin_DOY |
| 5 | G_API |
| 6 | J_aspect_deg |
| 7 | J_lc_code |
| 8 | V_ema_F_NDVI_kobs30 |
| 9 | V_rollmean_F_NDVI_kobs30 |
| 10 | V_rollmin_G_API_kobs14 |
| 11 | V_rollmin_G_API_kobs7 |
| 12 | V_rollrng_F_NDMI_kobs30 |
| 13 | lia_mean_desc_deg |
| 14 | V_rollmax_E_SAR_ratio_kobs30 |
| 15 | V_rollmin_LST_modis_kobs30 |
| 16 | V_rollrng_E_SAR_ratio_kobs30 |
| 17 | SMAP_ampm_diff_interp |
| 18 | A_d_E_SAR_diff_kobs30 |
| 19 | A_grad_E_SAR_diff_kobs30 |
| 20 | V_ema_G_API_kobs14 |
| 21 | V_rollmax_F_NDVI_kobs30 |
| 22 | V_rollmin_G_API_kobs30 |
| 23 | SMAP_sm_pm_interp_rollrange30 |
| 24 | V_rollstd_F_NDMI_kobs30 |
| 25 | V_rollstd_E_SAR_ratio_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
