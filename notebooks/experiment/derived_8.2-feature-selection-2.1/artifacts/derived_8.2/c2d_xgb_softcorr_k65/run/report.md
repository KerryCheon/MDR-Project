# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:36:40
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
| 2 | C_lag_E_SAR_diff_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_G_API_kobs1 |
| 5 | C_lag_LST_modis_kobs30 |
| 6 | DOY |
| 7 | D_cos_DOY |
| 8 | D_sin_DOY |
| 9 | G_rain_sum_30d |
| 10 | G_rain_sum_3d |
| 11 | G_rain_sum_7d |
| 12 | J_aspect_deg |
| 13 | J_bio_bio04 |
| 14 | J_bio_bio05 |
| 15 | J_bio_bio09 |
| 16 | K_aspect_cos |
| 17 | SMAP_sm_pm_interp_lag1 |
| 18 | SMAP_sm_pm_interp_rollmean30 |
| 19 | V_ema_G_API_kobs14 |
| 20 | V_ema_G_API_kobs30 |
| 21 | V_rollmax_E_SAR_ratio_kobs30 |
| 22 | V_rollmax_F_NDMI_kobs30 |
| 23 | V_rollmax_F_NDVI_kobs14 |
| 24 | V_rollmax_F_NDVI_kobs30 |
| 25 | V_rollmax_G_API_kobs30 |
| 26 | V_rollmax_LST_modis_kobs30 |
| 27 | V_rollmax_SMAP_sm_interp_kobs30 |
| 28 | V_rollmean_F_NDVI_kobs30 |
| 29 | V_rollmean_LST_modis_kobs30 |
| 30 | V_rollmin_E_SAR_diff_kobs30 |
| 31 | V_rollmin_F_NDVI_kobs30 |
| 32 | V_rollmin_G_API_kobs14 |
| 33 | V_rollmin_G_API_kobs30 |
| 34 | V_rollmin_LST_modis_kobs30 |
| 35 | V_rollmin_s2_b11_kobs30 |
| 36 | V_rollrng_F_NDVI_kobs30 |
| 37 | aspect |
| 38 | cos_year |
| 39 | elev |
| 40 | latitude |

## Metrics

_No metrics provided._
