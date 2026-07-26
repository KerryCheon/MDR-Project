# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-26T16:02:41
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
| 2 | C_lag_F_NDMI_kobs30 |
| 3 | C_lag_LST_modis_kobs30 |
| 4 | D_cos_DOY |
| 5 | G_rain_sum_30d |
| 6 | J_bio_bio02 |
| 7 | J_bio_bio04 |
| 8 | SMAP_sm_pm_interp_lag30 |
| 9 | SMAP_sm_pm_interp_rollmean30 |
| 10 | SMAP_x_year |
| 11 | V_ema_G_API_kobs30 |
| 12 | V_ema_G_API_kobs7 |
| 13 | V_rollmax_E_SAR_diff_kobs14 |
| 14 | V_rollmax_F_NDMI_kobs30 |
| 15 | V_rollmax_F_NDVI_kobs30 |
| 16 | V_rollmax_LST_modis_kobs30 |
| 17 | V_rollmean_E_SAR_ratio_kobs30 |
| 18 | V_rollmean_LST_modis_kobs30 |
| 19 | V_rollmin_E_SAR_diff_kobs30 |
| 20 | V_rollmin_F_NDVI_kobs7 |
| 21 | V_rollmin_G_API_kobs30 |
| 22 | V_rollmin_LST_modis_kobs30 |
| 23 | V_rollmin_SMAP_sm_interp_kobs30 |
| 24 | V_rollrng_E_SAR_diff_kobs30 |
| 25 | V_rollrng_F_NDVI_kobs30 |
| 26 | latitude |
| 27 | lia_mean_asc_deg |
| 28 | C_lag_F_NDVI_kobs30 |
| 29 | DOY |
| 30 | D_sin_DOY |
| 31 | G_API |
| 32 | V_ema_G_API_kobs14 |
| 33 | V_rollmax_F_NDVI_kobs14 |
| 34 | V_rollmax_G_API_kobs30 |
| 35 | V_rollmean_G_API_kobs30 |
| 36 | V_rollmin_F_NDVI_kobs14 |
| 37 | V_rollmin_F_NDVI_kobs30 |
| 38 | V_rollmin_SMAP_sm_interp_kobs14 |
| 39 | G_rain_sum_3d |
| 40 | V_rollmin_s2_b11_kobs30 |

## Metrics

_No metrics provided._
