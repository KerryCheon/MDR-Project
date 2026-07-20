# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:10:51
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | correlation, mi, xgb_importance, family_coverage, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_E_SAR_diff_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_LST_modis_kobs30 |
| 5 | C_smm_G_API_alpha0.85_n5 |
| 6 | DOY |
| 7 | D_cos_DOY |
| 8 | D_sin_DOY |
| 9 | G_rain_sum_3d |
| 10 | G_rain_sum_7d |
| 11 | J_aspect_deg |
| 12 | J_bio_bio05 |
| 13 | J_bio_bio09 |
| 14 | J_bio_bio15 |
| 15 | K_aspect_cos |
| 16 | SMAP_sm_pm_interp_lag1 |
| 17 | V_ema_G_API_kobs30 |
| 18 | V_rollmax_F_NDMI_kobs30 |
| 19 | V_rollmax_F_NDVI_kobs30 |
| 20 | V_rollmax_LST_modis_kobs30 |
| 21 | V_rollmean_E_SAR_ratio_kobs30 |
| 22 | V_rollmean_F_NDVI_kobs30 |
| 23 | V_rollmin_E_SAR_diff_kobs30 |
| 24 | V_rollmin_G_API_kobs14 |
| 25 | V_rollmin_G_API_kobs30 |
| 26 | V_rollmin_LST_modis_kobs30 |
| 27 | aspect |
| 28 | cos_year |
| 29 | latitude |
| 30 | lia_mean_desc_deg |
| 31 | slope |
| 32 | year |
| 33 | J_bio_bio14 |
| 34 | V_rollmin_F_NDVI_kobs14 |
| 35 | V_rollmin_F_NDVI_kobs30 |
| 36 | elev |
| 37 | lia_std_desc_deg |
| 38 | C_lag_LST_modis_kobs6 |
| 39 | J_bio_bio08 |
| 40 | J_bio_bio19 |

## Metrics

_No metrics provided._
