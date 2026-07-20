# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:25:02
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 55 |
| Stages | correlation, xgb_importance, family_coverage, stability |
| Top-k target | 55 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_G_API_kobs1 |
| 3 | C_lag_LST_modis_kobs30 |
| 4 | C_lag_SMAP_sm_interp_kobs6 |
| 5 | DOY |
| 6 | G_API |
| 7 | G_rain_sum_3d |
| 8 | G_rain_sum_7d |
| 9 | J_bio_bio05 |
| 10 | J_bio_bio09 |
| 11 | SMAP_sm_pm_interp_lag30 |
| 12 | SMAP_sm_pm_interp_lag7 |
| 13 | SMAP_x_year |
| 14 | V_ema_G_API_kobs30 |
| 15 | V_ema_G_API_kobs7 |
| 16 | V_rollcv_F_NDMI_kobs30 |
| 17 | V_rollmax_E_SAR_ratio_kobs7 |
| 18 | V_rollmin_E_SAR_diff_kobs30 |
| 19 | V_rollmin_F_NDVI_kobs30 |
| 20 | V_rollmin_G_API_kobs30 |
| 21 | V_rollmin_LST_modis_kobs30 |
| 22 | V_rollmin_s2_b12_kobs30 |
| 23 | slope |
| 24 | year |
| 25 | C_lag_E_SAR_diff_kobs30 |
| 26 | C_lag_E_SAR_ratio_kobs30 |
| 27 | V_ema_G_API_kobs14 |
| 28 | V_rollmax_E_SAR_ratio_kobs30 |
| 29 | V_rollmax_F_NDMI_kobs30 |
| 30 | V_rollmean_LST_modis_kobs30 |
| 31 | V_rollmin_E_SAR_diff_kobs14 |
| 32 | V_rollmin_G_API_kobs14 |
| 33 | V_rollmin_LST_modis_kobs14 |
| 34 | V_rollmin_s2_b12_kobs14 |
| 35 | aspect |
| 36 | A_grad_LST_modis_kobs30 |
| 37 | SMAP_sm_am_interp_lag30 |
| 38 | V_rollmax_E_SAR_ratio_kobs14 |
| 39 | V_rollmax_F_NDVI_kobs30 |
| 40 | V_rollmax_G_API_kobs14 |

## Metrics

_No metrics provided._
