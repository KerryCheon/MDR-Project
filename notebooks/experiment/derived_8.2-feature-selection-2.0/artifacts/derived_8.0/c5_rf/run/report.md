# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:07:12
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
| 2 | A_d_G_API_kobs2 |
| 3 | A_grad_F_NDMI_kobs30 |
| 4 | A_grad_LST_modis_kobs30 |
| 5 | C_lag_E_SAR_diff_kobs12 |
| 6 | C_lag_E_SAR_diff_kobs6 |
| 7 | C_lag_E_SAR_ratio_kobs30 |
| 8 | C_lag_LST_modis_kobs30 |
| 9 | C_smm_LST_modis_alpha0.85_n5 |
| 10 | DOY |
| 11 | D_sin_DOY |
| 12 | D_z_E_SAR_ratio |
| 13 | G_API |
| 14 | G_rain_sum_3d |
| 15 | G_rain_sum_7d |
| 16 | J_bio_bio05 |
| 17 | J_bio_bio09 |
| 18 | J_clay_wfrac_b200 |
| 19 | SMAP_x_year |
| 20 | V_ema_G_API_kobs14 |
| 21 | V_rollcv_F_NDMI_kobs30 |
| 22 | V_rollcv_G_API_kobs30 |
| 23 | V_rollcv_SMAP_sm_interp_kobs30 |
| 24 | V_rollmax_E_SAR_ratio_kobs30 |
| 25 | V_rollmax_F_NDVI_kobs30 |
| 26 | V_rollmax_G_API_kobs30 |
| 27 | V_rollmax_LST_modis_kobs30 |
| 28 | V_rollmax_SMAP_sm_interp_kobs30 |
| 29 | V_rollmin_E_SAR_diff_kobs30 |
| 30 | V_rollmin_F_NDVI_kobs30 |
| 31 | V_rollmin_G_API_kobs30 |
| 32 | V_rollmin_LST_modis_kobs30 |
| 33 | V_rollmin_s2_b11_kobs30 |
| 34 | V_rollrng_F_NDVI_kobs30 |
| 35 | precip_mm |
| 36 | s2_b8 |
| 37 | sin_year |
| 38 | year |
| 39 | V_rollcv_F_NDVI_kobs30 |
| 40 | V_rollmean_F_NDMI_kobs30 |

## Metrics

_No metrics provided._
