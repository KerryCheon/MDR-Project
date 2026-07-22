# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T22:30:28
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
| 1 | C_lag_E_SAR_ratio_kobs30 |
| 2 | C_lag_F_NDVI_kobs30 |
| 3 | C_lag_LST_modis_kobs30 |
| 4 | D_sin_DOY |
| 5 | V_ema_F_NDVI_kobs30 |
| 6 | V_ema_G_API_kobs7 |
| 7 | V_ema_LST_modis_kobs30 |
| 8 | V_rollmax_G_API_kobs7 |
| 9 | V_rollmean_F_NDVI_kobs30 |
| 10 | V_rollmean_G_API_kobs7 |
| 11 | V_rollmean_LST_modis_kobs30 |
| 12 | V_rollmin_G_API_kobs7 |
| 13 | V_rollmin_LST_modis_kobs30 |
| 14 | V_rollrng_F_NDMI_kobs30 |
| 15 | V_rollmin_G_API_kobs30 |
| 16 | A_d_E_SAR_diff_kobs30 |
| 17 | A_grad_E_SAR_diff_kobs30 |
| 18 | V_rollmin_G_API_kobs14 |
| 19 | V_rollmax_F_NDVI_kobs30 |
| 20 | G_rain_sum_30d |

## Metrics

_No metrics provided._
