# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T18:51:33
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 55 |
| Stages | xgb_importance, family_coverage, stability |
| Top-k target | 55 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_F_NDVI_kobs30 |
| 3 | C_lag_LST_modis_kobs30 |
| 4 | D_cos_DOY |
| 5 | D_sin_DOY |
| 6 | J_bio_bio04 |
| 7 | J_slope_deg |
| 8 | V_ema_G_API_kobs30 |
| 9 | V_ema_LST_modis_kobs30 |
| 10 | V_rollmax_F_NDVI_kobs30 |
| 11 | V_rollmax_G_API_kobs30 |
| 12 | V_rollmax_LST_modis_kobs30 |
| 13 | V_rollmean_G_API_kobs30 |
| 14 | V_rollmean_LST_modis_kobs30 |
| 15 | V_rollmin_G_API_kobs30 |
| 16 | V_rollmin_LST_modis_kobs30 |
| 17 | aspect |
| 18 | elev |
| 19 | latitude |
| 20 | longitude |
| 21 | DOY |
| 22 | G_API |
| 23 | V_ema_G_API_kobs14 |
| 24 | V_rollmax_F_NDMI_kobs30 |
| 25 | V_rollmin_G_API_kobs14 |
| 26 | slope |
| 27 | V_ema_G_API_kobs7 |
| 28 | V_rollmax_E_SAR_ratio_kobs30 |
| 29 | V_rollmean_E_SAR_ratio_kobs30 |
| 30 | V_rollmin_F_NDVI_kobs7 |
| 31 | V_rollrng_F_NDVI_kobs30 |
| 32 | K_slope_sin |
| 33 | V_rollmin_s2_b11_kobs30 |
| 34 | G_rain_sum_30d |
| 35 | SMAP_x_year |
| 36 | V_ema_E_SAR_ratio_kobs30 |
| 37 | V_rollmin_SMAP_sm_interp_kobs30 |
| 38 | C_lag_G_API_kobs1 |
| 39 | V_ema_F_NDVI_kobs30 |
| 40 | V_ema_LST_modis_kobs14 |

## Metrics

_No metrics provided._
