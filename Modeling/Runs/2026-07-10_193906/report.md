# Feature Selection Report

## Run Info
- Run ID: 2026-07-10_193906
- Generated: 2026-07-10T19:40:45
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | elasticnet, stability |
| Top-k target | 50 |
| Score | 0.4373 |
| Mean R2 | 0.5952 |
| Std R2 | 0.1335 |
| Train-Val Gap | 0.2054 |

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
| 33 | V_rollrng_G_API_kobs14 |
| 34 | V_ema_LST_modis_kobs30 |
| 35 | A_d_E_SAR_ratio_kobs14 |
| 36 | A_grad_E_SAR_ratio_kobs14 |
| 37 | V_rollcv_G_API_kobs14 |
| 38 | V_rollmean_F_NDVI_kobs30 |
| 39 | V_rollcv_G_API_kobs7 |
| 40 | A_d_LST_modis_kobs14 |

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
| train   | 15704 |                   0 | 0.4488 | 0.0805 |     0.3886 | 0.0659 |    0      | linear  |           50 |
| val     |  7149 |                   0 | 0.4064 | 0.0884 |     0.4398 | 0.0733 |    0.0146 | linear  |           50 |
| test    |  8902 |                   0 | 0.2598 | 0.0906 |     0.4856 | 0.073  |    0.0258 | linear  |           50 |
| train   | 15704 |                   0 | 0.9629 | 0.0209 |     0.1008 | 0.0149 |   -0      | xgb     |           50 |
| val     |  7149 |                   0 | 0.6863 | 0.0643 |     0.3197 | 0.0439 |    0.0034 | xgb     |           50 |
| test    |  8902 |                   0 | 0.5236 | 0.0727 |     0.3895 | 0.0552 |    0.0119 | xgb     |           50 |
| train   | 15704 |                   0 | 0.9899 | 0.0109 |     0.0527 | 0.0064 |    0      | rf      |           50 |
| val     |  7149 |                   0 | 0.6928 | 0.0636 |     0.3164 | 0.0428 |    0.0049 | rf      |           50 |
| test    |  8902 |                   0 | 0.5481 | 0.0708 |     0.3794 | 0.0534 |    0.0134 | rf      |           50 |
