# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T18:54:33
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
| 2 | C_lag_F_NDVI_kobs30 |
| 3 | C_lag_G_API_kobs1 |
| 4 | C_lag_LST_modis_kobs30 |
| 5 | DOY |
| 6 | D_cos_DOY |
| 7 | D_sin_DOY |
| 8 | G_API |
| 9 | G_rain_sum_30d |
| 10 | G_rain_sum_3d |
| 11 | G_rain_sum_7d |
| 12 | J_bio_bio04 |
| 13 | K_slope_sin |
| 14 | SMAP_sm_pm_interp_rollmean7 |
| 15 | SMAP_x_year |
| 16 | V_ema_G_API_kobs14 |
| 17 | V_ema_G_API_kobs30 |
| 18 | V_ema_G_API_kobs7 |
| 19 | V_rollmax_E_SAR_ratio_kobs30 |
| 20 | V_rollmax_F_NDMI_kobs30 |
| 21 | V_rollmax_F_NDVI_kobs30 |
| 22 | V_rollmax_G_API_kobs30 |
| 23 | V_rollmax_LST_modis_kobs30 |
| 24 | V_rollmax_s2_b11_kobs30 |
| 25 | V_rollmean_E_SAR_ratio_kobs30 |
| 26 | V_rollmean_F_NDVI_kobs30 |
| 27 | V_rollmean_G_API_kobs30 |
| 28 | V_rollmean_LST_modis_kobs30 |
| 29 | V_rollmin_E_SAR_ratio_kobs30 |
| 30 | V_rollmin_F_NDVI_kobs7 |
| 31 | V_rollmin_G_API_kobs14 |
| 32 | V_rollmin_G_API_kobs30 |
| 33 | V_rollmin_LST_modis_kobs30 |
| 34 | V_rollmin_s2_b11_kobs30 |
| 35 | V_rollrng_F_NDVI_kobs30 |
| 36 | aspect |
| 37 | elev |
| 38 | latitude |
| 39 | longitude |
| 40 | D_sa_F_NDMI |

## Metrics

_No metrics provided._
