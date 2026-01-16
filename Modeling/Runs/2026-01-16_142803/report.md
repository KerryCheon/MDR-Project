# Feature Selection Report

## Run Info

- Run ID: 2026-01-16_142803
- Generated: 2026-01-16T14:31:14
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item              | Value                     |
| ----------------- | ------------------------- |
| Selected features | 40                        |
| Stages            | mi, elasticnet, stability |
| Top-k target      | 40                        |
| Score             | 0.6232                    |
| Mean R2           | 0.7132                    |
| Std R2            | 0.0545                    |
| Train-Val Gap     | 0.1136                    |

## Top Selected Features

| #   | Feature                      |
| --- | ---------------------------- |
| 1   | C_lag_E_SAR_diff_kobs12      |
| 2   | C_lag_E_SAR_diff_kobs30      |
| 3   | C_lag_E_SAR_ratio_kobs30     |
| 4   | C_lag_F_NDVI_kobs30          |
| 5   | C_lag_LST_modis_kobs12       |
| 6   | C_lag_LST_modis_kobs30       |
| 7   | DOY                          |
| 8   | D_sa_E_SAR_ratio             |
| 9   | D_sa_F_NDMI                  |
| 10  | D_z_E_SAR_ratio              |
| 11  | D_z_F_NDMI                   |
| 12  | E_SAR_diff                   |
| 13  | E_SAR_ratio                  |
| 14  | F_MSI                        |
| 15  | F_NDMI                       |
| 16  | V_ema_LST_modis_kobs30       |
| 17  | V_rollmax_E_SAR_diff_kobs14  |
| 18  | V_rollmax_E_SAR_diff_kobs30  |
| 19  | V_rollmax_F_NDVI_kobs30      |
| 20  | V_rollmax_G_API_kobs30       |
| 21  | V_rollmax_G_API_kobs7        |
| 22  | V_rollmax_LST_modis_kobs7    |
| 23  | V_rollmax_s2_b11_kobs30      |
| 24  | V_rollmean_G_API_kobs30      |
| 25  | V_rollmin_E_SAR_diff_kobs30  |
| 26  | V_rollmin_E_SAR_ratio_kobs30 |
| 27  | V_rollmin_F_NDMI_kobs30      |
| 28  | V_rollmin_G_API_kobs7        |
| 29  | V_rollmin_s2_b11_kobs30      |
| 30  | V_rollmin_s2_b12_kobs30      |
| 31  | s1_vh                        |
| 32  | s2_b8                        |
| 33  | A_d_LST_modis_kobs7          |
| 34  | A_grad_LST_modis_kobs14      |
| 35  | C_lag_E_SAR_diff_kobs6       |
| 36  | V_rollmax_E_SAR_diff_kobs7   |
| 37  | V_rollmax_E_SAR_ratio_kobs14 |
| 38  | V_rollmax_F_NDVI_kobs14      |
| 39  | s2_b12                       |
| 40  | D_sa_LST_modis               |

## Score Weights

| Metric    | Weight  |
| --------- | ------- |
| gap       | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2   | 1.0000  |
| std_r2    | -0.5000 |

## Metrics

| split |     n | dropped_nonfinite |     r2 |   rmse | rel_rmse |    mae | bias_me | model  | n_features |
| :---- | ----: | ----------------: | -----: | -----: | -------: | -----: | ------: | :----- | ---------: |
| train | 16972 |                 0 | 0.6749 | 0.0583 |   0.2992 | 0.0462 |       0 | linear |         40 |
| val   |  2919 |                 0 | 0.6587 | 0.0588 |   0.2667 | 0.0459 | -0.0194 | linear |         40 |
| test  |  2829 |                 0 | 0.5918 | 0.0598 |    0.276 | 0.0479 |  -0.007 | linear |         40 |
| train | 16972 |                 0 | 0.9786 |  0.015 |   0.0767 | 0.0103 | -0.0001 | xgb    |         40 |
| val   |  2919 |                 0 | 0.7677 | 0.0485 |     0.22 | 0.0371 | -0.0223 | xgb    |         40 |
| test  |  2829 |                 0 | 0.6769 | 0.0532 |   0.2455 | 0.0408 | -0.0189 | xgb    |         40 |
