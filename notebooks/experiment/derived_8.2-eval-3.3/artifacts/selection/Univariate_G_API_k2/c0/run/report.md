# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-19T21:00:41
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 20 |
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
| 7 | J_aspect_deg |
| 8 | SMAP_ampm_diff_interp |
| 9 | SMAP_sm_pm_interp_rollrange30 |
| 10 | V_ema_F_NDVI_kobs30 |
| 11 | V_rollmax_G_API_kobs7 |
| 12 | V_rollmean_F_NDVI_kobs30 |
| 13 | V_rollmin_G_API_kobs14 |
| 14 | V_rollmin_G_API_kobs7 |
| 15 | V_rollmin_LST_modis_kobs30 |
| 16 | V_rollrng_E_SAR_ratio_kobs30 |
| 17 | V_rollrng_F_NDMI_kobs30 |
| 18 | lia_mean_desc_deg |
| 19 | V_rollmax_E_SAR_ratio_kobs30 |
| 20 | G_rain_sum_30d |

## Metrics

_No metrics provided._
