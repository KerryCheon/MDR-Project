# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-26T16:03:44
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
| 2 | C_lag_F_NDMI_kobs30 |
| 3 | C_lag_LST_modis_kobs30 |
| 4 | D_cos_DOY |
| 5 | J_bio_bio02 |
| 6 | V_ema_G_API_kobs14 |
| 7 | V_ema_G_API_kobs30 |
| 8 | V_ema_LST_modis_kobs30 |
| 9 | V_rollmax_F_NDVI_kobs30 |
| 10 | V_rollmax_LST_modis_kobs30 |
| 11 | V_rollmean_LST_modis_kobs30 |
| 12 | V_rollmin_E_SAR_diff_kobs30 |
| 13 | V_rollmin_G_API_kobs30 |
| 14 | V_rollmin_LST_modis_kobs30 |
| 15 | V_rollmin_SMAP_sm_interp_kobs30 |
| 16 | latitude |
| 17 | lia_mean_asc_deg |
| 18 | longitude |
| 19 | D_sin_DOY |
| 20 | G_rain_sum_30d |
| 21 | SMAP_sm_pm_interp_rollmean30 |
| 22 | V_rollmax_F_NDMI_kobs30 |
| 23 | V_rollmean_F_NDMI_kobs30 |
| 24 | V_rollmean_G_API_kobs30 |
| 25 | V_rollrng_F_NDVI_kobs30 |
| 26 | DOY |
| 27 | SMAP_sm_pm_interp_lag30 |
| 28 | SMAP_x_year |
| 29 | V_rollmax_G_API_kobs30 |
| 30 | V_rollmax_G_API_kobs7 |
| 31 | V_ema_E_SAR_ratio_kobs14 |
| 32 | V_rollmax_F_NDVI_kobs14 |
| 33 | V_rollmin_s2_b11_kobs30 |
| 34 | C_lag_F_NDVI_kobs30 |
| 35 | V_rollmin_SMAP_sm_interp_kobs14 |
| 36 | V_ema_G_API_kobs7 |
| 37 | V_rollmean_F_NDVI_kobs30 |
| 38 | V_rollmin_F_NDVI_kobs7 |
| 39 | V_rollrng_E_SAR_diff_kobs30 |
| 40 | sin_year |

## Metrics

_No metrics provided._
