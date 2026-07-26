# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-26T16:04:35
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
| 2 | C_lag_F_NDMI_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_LST_modis_kobs30 |
| 5 | C_lag_SMAP_sm_interp_kobs1 |
| 6 | DOY |
| 7 | D_cos_DOY |
| 8 | D_sin_DOY |
| 9 | G_rain_sum_30d |
| 10 | G_rain_sum_3d |
| 11 | G_rain_sum_7d |
| 12 | J_bio_bio02 |
| 13 | J_bio_bio04 |
| 14 | SMAP_sm_pm_interp_lag30 |
| 15 | SMAP_sm_pm_interp_rollmean30 |
| 16 | SMAP_x_year |
| 17 | V_ema_G_API_kobs14 |
| 18 | V_ema_G_API_kobs30 |
| 19 | V_ema_G_API_kobs7 |
| 20 | V_rollmax_E_SAR_diff_kobs14 |
| 21 | V_rollmax_F_NDMI_kobs30 |
| 22 | V_rollmax_F_NDVI_kobs30 |
| 23 | V_rollmax_G_API_kobs30 |
| 24 | V_rollmax_LST_modis_kobs30 |
| 25 | V_rollmean_E_SAR_ratio_kobs30 |
| 26 | V_rollmean_G_API_kobs30 |
| 27 | V_rollmean_LST_modis_kobs30 |
| 28 | V_rollmin_E_SAR_diff_kobs30 |
| 29 | V_rollmin_F_NDVI_kobs14 |
| 30 | V_rollmin_F_NDVI_kobs30 |
| 31 | V_rollmin_F_NDVI_kobs7 |
| 32 | V_rollmin_G_API_kobs30 |
| 33 | V_rollmin_LST_modis_kobs30 |
| 34 | V_rollmin_SMAP_sm_interp_kobs14 |
| 35 | V_rollmin_SMAP_sm_interp_kobs30 |
| 36 | V_rollmin_s2_b11_kobs30 |
| 37 | V_rollrng_E_SAR_diff_kobs30 |
| 38 | V_rollrng_F_NDVI_kobs30 |
| 39 | elev |
| 40 | latitude |

## Metrics

_No metrics provided._
