# Feature Selection Report

## Run Info
- Run ID: derived_8.2_regime_binary_regime_dry_v1
- Generated: 2026-07-14T14:06:28
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 27 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |
| Score | 0.1043 |
| Mean R2 | 0.2713 |
| Std R2 | 0.0632 |
| Train-Val Gap | 0.5420 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | D_sin_DOY |
| 2 | J_bio_bio02 |
| 3 | SMAP_sm_pm_interp_rollrange30 |
| 4 | SMAP_x_year |
| 5 | V_rollrng_F_NDMI_kobs30 |
| 6 | sin_year |
| 7 | G_DSLR |
| 8 | V_rollmin_E_SAR_ratio_kobs30 |
| 9 | V_rollstd_G_API_kobs30 |
| 10 | A_grad_E_SAR_diff_kobs30 |
| 11 | C_lag_E_SAR_ratio_kobs30 |
| 12 | latitude |
| 13 | A_d_E_SAR_diff_kobs30 |
| 14 | SMAP_sm_pm_interp_rollmean30 |
| 15 | C_lag_F_NDVI_kobs30 |
| 16 | V_rollrng_F_NDVI_kobs14 |
| 17 | J_lc_code |
| 18 | V_rollmax_LST_modis_kobs7 |
| 19 | C_lag_SMAP_sm_interp_kobs12 |
| 20 | V_rollrng_E_SAR_diff_kobs30 |
| 21 | SMAP_sm_pm_interp_lag30 |
| 22 | slope |
| 23 | SMAP_sm_am_interp_rollrange30 |
| 24 | V_rollmax_LST_modis_kobs14 |
| 25 | SMAP_sm_interp_rollrange30 |
| 26 | K_aspect_cos |
| 27 | V_rollrng_SMAP_sm_interp_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

> Note: These models have not been tuned or optimized in any way

| split   |    n |   dropped_nonfinite |      r2 |   rmse |   rel_rmse |    mae |   bias_me | model   |   n_features |
|:--------|-----:|--------------------:|--------:|-------:|-----------:|-------:|----------:|:--------|-------------:|
| train   | 5521 |                   0 |  0.514  | 0.0329 |     0.389  | 0.0265 |    0      | linear  |           27 |
| val     | 2939 |                   0 |  0.1834 | 0.0394 |     0.4925 | 0.0314 |    0.0138 | linear  |           27 |
| test    | 3357 |                   0 | -0.2505 | 0.0519 |     0.7491 | 0.0409 |    0.0269 | linear  |           27 |
| train   | 5521 |                   0 |  0.9563 | 0.0099 |     0.1167 | 0.0064 |   -0      | xgb     |           27 |
| val     | 2939 |                   0 |  0.3014 | 0.0365 |     0.4555 | 0.0278 |    0.0092 | xgb     |           27 |
| test    | 3357 |                   0 |  0.0704 | 0.0447 |     0.6459 | 0.0335 |    0.0196 | xgb     |           27 |
| train   | 5521 |                   0 |  0.9698 | 0.0082 |     0.097  | 0.0049 |   -0.0001 | rf      |           27 |
| val     | 2939 |                   0 |  0.3292 | 0.0357 |     0.4463 | 0.028  |    0.0094 | rf      |           27 |
| test    | 3357 |                   0 |  0.0639 | 0.0449 |     0.6481 | 0.0344 |    0.0181 | rf      |           27 |
