# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-26T16:01:17
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 12 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | C_lag_LST_modis_kobs30 |
| 2 | V_rollmin_LST_modis_kobs30 |
| 3 | A_d_E_SAR_ratio_kobs30 |
| 4 | A_d_E_SAR_diff_kobs30 |
| 5 | SMAP_sm_pm_interp_rollrange30 |
| 6 | C_lag_E_SAR_ratio_kobs30 |
| 7 | A_d_LST_modis_kobs14 |
| 8 | A_d_SMAP_sm_interp_kobs30 |
| 9 | A_grad_LST_modis_kobs14 |
| 10 | A_grad_SMAP_sm_interp_kobs30 |
| 11 | D_sa_F_NDMI |
| 12 | V_ema_LST_modis_kobs30 |

## Metrics

_No metrics provided._
