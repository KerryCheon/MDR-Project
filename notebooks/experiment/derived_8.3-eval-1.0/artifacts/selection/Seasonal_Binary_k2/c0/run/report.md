# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T22:30:41
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 15 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | D_sin_DOY |
| 2 | V_ema_G_API_kobs7 |
| 3 | V_rollmin_G_API_kobs14 |
| 4 | V_rollmin_G_API_kobs30 |
| 5 | V_rollmin_LST_modis_kobs30 |
| 6 | V_rollrng_F_NDMI_kobs30 |
| 7 | C_lag_F_NDVI_kobs30 |
| 8 | C_lag_LST_modis_kobs30 |
| 9 | V_rollmean_F_NDVI_kobs30 |
| 10 | C_lag_E_SAR_ratio_kobs30 |
| 11 | V_ema_LST_modis_kobs30 |
| 12 | V_rollmax_F_NDVI_kobs30 |
| 13 | A_d_E_SAR_diff_kobs30 |
| 14 | A_grad_E_SAR_diff_kobs30 |
| 15 | SMAP_sm_pm_interp_rollrange30 |

## Metrics

_No metrics provided._
