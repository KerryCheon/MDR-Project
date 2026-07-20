# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:35:24
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 65 |
| Stages | correlation, xgb_importance, family_coverage, stability |
| Top-k target | 65 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | A_grad_LST_modis_kobs30 |
| 3 | C_lag_E_SAR_ratio_kobs30 |
| 4 | C_lag_G_API_kobs1 |
| 5 | C_lag_LST_modis_kobs30 |
| 6 | C_lag_SMAP_sm_interp_kobs6 |
| 7 | DOY |
| 8 | G_API |
| 9 | G_rain_sum_3d |
| 10 | G_rain_sum_7d |
| 11 | J_bio_bio05 |
| 12 | J_bio_bio09 |
| 13 | SMAP_sm_pm_interp_lag30 |
| 14 | SMAP_sm_pm_interp_lag7 |
| 15 | SMAP_x_year |
| 16 | V_ema_G_API_kobs30 |
| 17 | V_ema_G_API_kobs7 |
| 18 | V_rollcv_F_NDMI_kobs30 |
| 19 | V_rollmax_E_SAR_ratio_kobs14 |
| 20 | V_rollmax_E_SAR_ratio_kobs30 |
| 21 | V_rollmax_E_SAR_ratio_kobs7 |
| 22 | V_rollmax_F_NDMI_kobs30 |
| 23 | V_rollmax_G_API_kobs14 |
| 24 | V_rollmin_E_SAR_diff_kobs30 |
| 25 | V_rollmin_F_NDVI_kobs30 |
| 26 | V_rollmin_G_API_kobs30 |
| 27 | V_rollmin_LST_modis_kobs30 |
| 28 | V_rollmin_LST_modis_kobs7 |
| 29 | V_rollmin_s2_b12_kobs30 |
| 30 | aspect |
| 31 | slope |
| 32 | year |
| 33 | C_lag_E_SAR_diff_kobs30 |
| 34 | C_lag_SMAP_sm_interp_kobs12 |
| 35 | D_sin_DOY |
| 36 | SMAP_sm_am_interp_lag30 |
| 37 | V_ema_G_API_kobs14 |
| 38 | V_rollmax_E_SAR_diff_kobs30 |
| 39 | V_rollmax_F_NDVI_kobs30 |
| 40 | V_rollmean_E_SAR_diff_kobs30 |

## Metrics

_No metrics provided._
