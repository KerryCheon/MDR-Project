# Feature Selection Report

## Run Info

- Run ID: 2026-01-16_144317
- Generated: 2026-01-16T14:46:49
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
| Score             | 0.6353                    |
| Mean R2           | 0.7283                    |
| Std R2            | 0.0494                    |
| Train-Val Gap     | 0.1417                    |

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
| train | 16972 |                 0 | 0.6746 | 0.0583 |   0.2993 | 0.0463 |       0 | linear |         40 |
| val   |  2919 |                 0 | 0.6608 | 0.0587 |   0.2659 | 0.0458 | -0.0193 | linear |         40 |
| test  |  2829 |                 0 | 0.5945 | 0.0596 |   0.2751 | 0.0478 | -0.0069 | linear |         40 |
| train | 16972 |                 0 |  0.951 | 0.0226 |   0.1161 | 0.0162 | -0.0001 | xgb    |         40 |
| val   |  2919 |                 0 | 0.7466 | 0.0507 |   0.2298 | 0.0388 | -0.0245 | xgb    |         40 |
| test  |  2829 |                 0 | 0.6613 | 0.0544 |   0.2514 | 0.0421 |  -0.019 | xgb    |         40 |
| train | 16972 |                 0 | 0.9846 | 0.0127 |   0.0651 |  0.008 |       0 | rf     |         40 |
| val   |  2919 |                 0 | 0.7776 | 0.0475 |   0.2153 | 0.0357 | -0.0225 | rf     |         40 |
| test  |  2829 |                 0 |  0.695 | 0.0517 |   0.2385 |  0.039 | -0.0167 | rf     |         40 |
