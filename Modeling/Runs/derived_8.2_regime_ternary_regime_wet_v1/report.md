# Feature Selection Report

## Run Info
- Run ID: derived_8.2_regime_ternary_regime_wet_v1
- Generated: 2026-07-14T14:04:52
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 6 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |
| Score | 0.1163 |
| Mean R2 | 0.2193 |
| Std R2 | 0.0176 |
| Train-Val Gap | 0.4411 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | D_cos_DOY |
| 2 | V_rollrng_E_SAR_diff_kobs30 |
| 3 | latitude |
| 4 | slope |
| 5 | J_soil_texture_usda_b200 |
| 6 | D_sin_DOY |

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
| train   | 6179 |                   0 |  0.2772 | 0.0344 |     0.108  | 0.029  |   -0      | linear  |            6 |
| val     | 2917 |                   0 |  0.2409 | 0.0314 |     0.0983 | 0.0243 |   -0.006  | linear  |            6 |
| test    | 3074 |                   0 | -0.3427 | 0.04   |     0.1337 | 0.0337 |    0.0105 | linear  |            6 |
| train   | 6179 |                   0 |  0.8113 | 0.0176 |     0.0552 | 0.0132 |    0.0002 | xgb     |            6 |
| val     | 2917 |                   0 |  0.2193 | 0.0318 |     0.0997 | 0.0241 |   -0.0056 | xgb     |            6 |
| test    | 3074 |                   0 | -0.4184 | 0.0411 |     0.1374 | 0.0323 |    0.0076 | xgb     |            6 |
| train   | 6179 |                   0 |  0.8929 | 0.0132 |     0.0416 | 0.0094 |    0      | rf      |            6 |
| val     | 2917 |                   0 |  0.1978 | 0.0323 |     0.101  | 0.0241 |   -0.0053 | rf      |            6 |
| test    | 3074 |                   0 | -0.4212 | 0.0412 |     0.1375 | 0.0325 |    0.0071 | rf      |            6 |
