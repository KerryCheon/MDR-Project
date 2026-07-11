# Feature Selection Report

## Run Info
- Run ID: 2026-07-10_201657
- Generated: 2026-07-10T20:18:29
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 32 |
| Stages | elasticnet, stability |
| Top-k target | 60 |
| Score | 0.4547 |
| Mean R2 | 0.5946 |
| Std R2 | 0.1361 |
| Train-Val Gap | 0.1993 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_E_SAR_diff_kobs30 |
| 2 | A_d_E_SAR_ratio_kobs30 |
| 3 | A_d_F_NDMI_kobs30 |
| 4 | A_d_LST_modis_kobs30 |
| 5 | A_d_SMAP_sm_interp_kobs30 |
| 6 | A_grad_E_SAR_diff_kobs30 |
| 7 | A_grad_E_SAR_ratio_kobs30 |
| 8 | A_grad_F_NDMI_kobs30 |
| 9 | A_grad_LST_modis_kobs30 |
| 10 | A_grad_SMAP_sm_interp_kobs30 |
| 11 | C_lag_E_SAR_ratio_kobs30 |
| 12 | C_lag_F_NDVI_kobs30 |
| 13 | C_lag_G_API_kobs1 |
| 14 | C_lag_LST_modis_kobs30 |
| 15 | I_ts_spike_s1_vv |
| 16 | SMAP_ampm_diff_interp |
| 17 | SMAP_sm_pm_interp_rollrange30 |
| 18 | V_rollcv_E_SAR_diff_kobs30 |
| 19 | V_rollcv_G_API_kobs30 |
| 20 | V_rollmax_E_SAR_ratio_kobs30 |
| 21 | V_rollmin_LST_modis_kobs30 |
| 22 | V_rollrng_F_NDMI_kobs30 |
| 23 | V_rollrng_G_API_kobs30 |
| 24 | V_rollrng_G_API_kobs7 |
| 25 | V_rollstd_G_API_kobs30 |
| 26 | lia_mean_asc_deg |
| 27 | lia_std_asc_deg |
| 28 | V_rollrng_E_SAR_diff_kobs30 |
| 29 | V_rollstd_G_API_kobs7 |
| 30 | SMAP_sm_pm_interp_rollstd30 |
| 31 | E_rough_s1_vv_kobs14 |
| 32 | V_rollstd_G_API_kobs14 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

> Note: These models have not been tuned or optimized in any way

| split   |     n |   dropped_nonfinite |     r2 |   rmse |   rel_rmse |    mae |   bias_me | model   |   n_features |
|:--------|------:|--------------------:|-------:|-------:|-----------:|-------:|----------:|:--------|-------------:|
| train   | 15704 |                   0 | 0.4401 | 0.0812 |     0.3916 | 0.0665 |    0      | linear  |           32 |
| val     |  7149 |                   0 | 0.4022 | 0.0887 |     0.4413 | 0.0735 |    0.0144 | linear  |           32 |
| test    |  8902 |                   0 | 0.251  | 0.0911 |     0.4885 | 0.0737 |    0.0246 | linear  |           32 |
| train   | 15704 |                   0 | 0.9541 | 0.0232 |     0.1121 | 0.0166 |   -0      | xgb     |           32 |
| val     |  7149 |                   0 | 0.6839 | 0.0645 |     0.3209 | 0.0442 |    0.0039 | xgb     |           32 |
| test    |  8902 |                   0 | 0.5147 | 0.0734 |     0.3932 | 0.0563 |    0.0096 | xgb     |           32 |
| train   | 15704 |                   0 | 0.9876 | 0.0121 |     0.0583 | 0.0073 |    0      | rf      |           32 |
| val     |  7149 |                   0 | 0.6976 | 0.0631 |     0.3139 | 0.0427 |    0.004  | rf      |           32 |
| test    |  8902 |                   0 | 0.548  | 0.0708 |     0.3794 | 0.0541 |    0.0125 | rf      |           32 |
