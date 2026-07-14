# Feature Selection Report

## Run Info
- Run ID: derived_8.2_regime_ternary_regime_transition_v1
- Generated: 2026-07-14T14:03:38
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |
| Score | 0.0514 |
| Mean R2 | 0.2105 |
| Std R2 | 0.0386 |
| Train-Val Gap | 0.4488 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | J_bio_bio06 |
| 3 | V_rollmin_G_API_kobs30 |
| 4 | G_rain_sum_3d |
| 5 | A_d_E_SAR_diff_kobs5 |
| 6 | A_d_E_SAR_ratio_kobs30 |
| 7 | A_d_E_SAR_ratio_kobs5 |
| 8 | A_d_E_SAR_ratio_kobs7 |
| 9 | A_d_LST_modis_kobs5 |
| 10 | A_d_s2_b11_kobs5 |
| 11 | A_grad_E_SAR_ratio_kobs14 |
| 12 | A_grad_E_SAR_ratio_kobs7 |
| 13 | A_grad_s2_b11_kobs14 |
| 14 | C_lag_E_SAR_diff_kobs30 |
| 15 | C_lag_E_SAR_diff_kobs6 |
| 16 | C_lag_E_SAR_ratio_kobs12 |
| 17 | C_lag_F_NDMI_kobs2 |
| 18 | C_lag_F_NDVI_kobs1 |
| 19 | C_lag_F_NDVI_kobs6 |
| 20 | C_lag_LST_modis_kobs1 |
| 21 | C_lag_s2_b11_kobs1 |
| 22 | C_lag_s2_b12_kobs12 |
| 23 | C_lag_s2_b12_kobs6 |
| 24 | DOY |
| 25 | F_MSI |
| 26 | F_NDVI |
| 27 | G_DSLR |
| 28 | J_aspect_deg |
| 29 | J_bio_bio02 |
| 30 | J_bio_bio10 |
| 31 | J_bio_bio11 |
| 32 | J_bio_bio14 |
| 33 | J_bio_bio16 |
| 34 | J_clay_wfrac_b0 |
| 35 | J_sand_wfrac_b10 |
| 36 | J_sand_wfrac_b60 |
| 37 | K_sand_clay_ratio_b0 |
| 38 | SMAP_sm_am_interp_lag30 |
| 39 | SMAP_sm_am_interp_lag7 |
| 40 | SMAP_sm_pm_interp_rollrange30 |

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
| train   | 4004 |                   0 |  0.2172 | 0.0221 |     0.1081 | 0.0183 |   -0      | linear  |           50 |
| val     | 1293 |                   0 |  0.1568 | 0.023  |     0.1097 | 0.0191 |    0.0031 | linear  |           50 |
| test    | 2471 |                   0 | -0.2525 | 0.0289 |     0.1407 | 0.0233 |    0.0125 | linear  |           50 |
| train   | 4004 |                   0 |  0.8701 | 0.009  |     0.044  | 0.0063 |    0      | xgb     |           50 |
| val     | 1293 |                   0 |  0.2289 | 0.0219 |     0.1049 | 0.0174 |    0.0036 | xgb     |           50 |
| test    | 2471 |                   0 | -0.2396 | 0.0288 |     0.1399 | 0.0228 |    0.011  | xgb     |           50 |
| train   | 4004 |                   0 |  0.8907 | 0.0083 |     0.0404 | 0.0056 |    0      | rf      |           50 |
| val     | 1293 |                   0 |  0.2458 | 0.0217 |     0.1038 | 0.0182 |   -0.0002 | rf      |           50 |
| test    | 2471 |                   0 |  0.0455 | 0.0253 |     0.1228 | 0.0206 |    0.0038 | rf      |           50 |
