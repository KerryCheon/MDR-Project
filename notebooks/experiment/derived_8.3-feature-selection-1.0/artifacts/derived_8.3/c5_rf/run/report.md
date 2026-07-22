# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T19:06:30
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | correlation, rf_importance, family_coverage, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_F_NDVI_kobs30 |
| 3 | C_lag_LST_modis_kobs30 |
| 4 | C_smm_G_API_alpha0.85_n5 |
| 5 | DOY |
| 6 | D_cos_DOY |
| 7 | D_sin_DOY |
| 8 | G_rain_sum_3d |
| 9 | G_rain_sum_7d |
| 10 | J_aspect_deg |
| 11 | J_bio_bio01 |
| 12 | J_sand_wfrac_b200 |
| 13 | K_aspect_cos |
| 14 | SMAP_sm_interp_lag1 |
| 15 | SMAP_sm_pm_interp_rollmean30 |
| 16 | SMAP_x_year |
| 17 | V_ema_E_SAR_diff_kobs30 |
| 18 | V_ema_G_API_kobs30 |
| 19 | V_rollmax_E_SAR_ratio_kobs30 |
| 20 | V_rollmax_F_NDMI_kobs30 |
| 21 | V_rollmax_F_NDVI_kobs30 |
| 22 | V_rollmax_LST_modis_kobs30 |
| 23 | V_rollmean_E_SAR_ratio_kobs30 |
| 24 | V_rollmean_F_NDVI_kobs30 |
| 25 | V_rollmin_E_SAR_ratio_kobs30 |
| 26 | V_rollmin_G_API_kobs14 |
| 27 | V_rollmin_G_API_kobs30 |
| 28 | V_rollmin_LST_modis_kobs30 |
| 29 | V_rollmin_SMAP_sm_interp_kobs30 |
| 30 | V_rollmin_s2_b11_kobs30 |
| 31 | V_rollstd_F_NDVI_kobs30 |
| 32 | elev |
| 33 | latitude |
| 34 | C_lag_E_SAR_ratio_kobs30 |
| 35 | C_lag_LST_modis_kobs6 |
| 36 | J_sand_clay_ratio_b0 |
| 37 | SMAP_sm_pm_interp_lag30 |
| 38 | V_rollmax_F_NDVI_kobs14 |
| 39 | slope |
| 40 | J_bio_bio04 |

## Metrics

_No metrics provided._
