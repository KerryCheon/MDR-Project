# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T18:47:30
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
| 2 | C_lag_LST_modis_kobs30 |
| 3 | DOY |
| 4 | D_cos_DOY |
| 5 | D_sin_DOY |
| 6 | G_API |
| 7 | G_rain_sum_30d |
| 8 | J_bio_bio04 |
| 9 | K_slope_sin |
| 10 | SMAP_x_year |
| 11 | V_ema_G_API_kobs30 |
| 12 | V_ema_G_API_kobs7 |
| 13 | V_rollmax_E_SAR_ratio_kobs30 |
| 14 | V_rollmax_F_NDMI_kobs30 |
| 15 | V_rollmax_F_NDVI_kobs30 |
| 16 | V_rollmax_G_API_kobs30 |
| 17 | V_rollmax_LST_modis_kobs30 |
| 18 | V_rollmean_E_SAR_ratio_kobs30 |
| 19 | V_rollmean_F_NDVI_kobs30 |
| 20 | V_rollmean_G_API_kobs30 |
| 21 | V_rollmean_LST_modis_kobs30 |
| 22 | V_rollmin_G_API_kobs14 |
| 23 | V_rollmin_G_API_kobs30 |
| 24 | V_rollmin_LST_modis_kobs30 |
| 25 | V_rollmin_s2_b11_kobs30 |
| 26 | aspect |
| 27 | elev |
| 28 | latitude |
| 29 | longitude |
| 30 | C_lag_G_API_kobs1 |
| 31 | G_rain_sum_3d |
| 32 | V_ema_G_API_kobs14 |
| 33 | V_rollmin_E_SAR_ratio_kobs30 |
| 34 | V_rollmin_F_NDVI_kobs7 |
| 35 | V_rollmin_SMAP_sm_interp_kobs30 |
| 36 | V_rollrng_F_NDVI_kobs30 |
| 37 | slope |
| 38 | C_lag_F_NDMI_kobs30 |
| 39 | C_lag_F_NDVI_kobs30 |
| 40 | D_sa_F_NDMI |

## Metrics

_No metrics provided._
