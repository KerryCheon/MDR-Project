# SHAP Analysis (Feature Sets)

_Author: Jakob Balkovec_

I retrained the best-performing run (by test $R^2$ in `SELECTED_FEATURES.md`) for each feature set and ran SHAP on the full test split. This is meant to be readable, not overly academic.

> Note: The performance won't match exactly what was in `SELECTED_FEATURES.md` since I retrained the models from scratch (same hyperparameters, data splits, etc.) to ensure consistency, but did not apply any post processing (e.g., calibration, bootstrapping) techniques.

> Note: Each feature set includes a SHAP summary plot (overall impact distribution), a SHAP bar plot (ranked average impact), and SHAP dependence plots (feature relationship) of the top 3 performing features (these use a random sample of the test set for speed, still temporal)

**Overview of What you’re looking at:**

- Metrics are from the retrained models (same splits as the original runs)
- SHAP is computed on the full test set
- Relationship plots are SHAP dependence plots using a random sample of test rows (just for speed)
- For v7.1 I include both XGBoost and RandomForest, since both were used in the notebook
- Plots live in `Temporal/Pipeline/data/features/review/shap_plots/`

## Feature set 01

Best run (by test $R^2$): `Models/Temporal/v1/v1.0/mdr_ts_v2_3_20260108_115420`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.8791 | 0.0354 | 0.0248 |
| Val   | 0.6137 | 0.0668 | 0.0509 |
| Test  | 0.5208 | 0.0644 | 0.0498 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature   | Mean (SHAP) |
| ---- | --------- | ----------- |
| 1    | DOY       | 0.059176    |
| 2    | s1_vv     | 0.014511    |
| 3    | SAR_ratio | 0.010819    |
| 4    | s1_vh     | 0.008814    |
| 5    | precip_mm | 0.006762    |
| 6    | NDMI      | 0.006368    |
| 7    | NDVI      | 0.004806    |
| 8    | MSI       | 0.002917    |
| 9    | rain_mm   | 0.001936    |

![Feature set 01 XGB SHAP summary](shap_plots/feature_set_01_xgb_summary.png)

![Feature set 01 XGB SHAP bar](shap_plots/feature_set_01_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 01 XGB dependence](shap_plots/feature_set_01_xgb_DOY_dependence.png)

![Feature set 01 XGB dependence](shap_plots/feature_set_01_xgb_s1_vv_dependence.png)

![Feature set 01 XGB dependence](shap_plots/feature_set_01_xgb_SAR_ratio_dependence.png)

---

## Feature set 02

Best run (by test $R^2$): `Models/Temporal/v1/v1.2/mdr_ts_v1_2_20251223_113808`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.8486 | 0.0393 | 0.0269 |
| Val   | 0.5350 | 0.0721 | 0.0549 |
| Test  | 0.4270 | 0.0736 | 0.0591 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature         | Mean (SHAP) |
| ---- | --------------- | ----------- |
| 1    | air_temp_mean   | 0.050853    |
| 2    | slope           | 0.010582    |
| 3    | precip_mm       | 0.010575    |
| 4    | s1_vv           | 0.008115    |
| 5    | NDVI            | 0.008001    |
| 6    | SAR_ratio       | 0.007145    |
| 7    | elev            | 0.006451    |
| 8    | solar_radiation | 0.006024    |
| 9    | s1_vh           | 0.005473    |
| 10   | MSI             | 0.004705    |

![Feature set 02 XGB SHAP summary](shap_plots/feature_set_02_xgb_summary.png)

![Feature set 02 XGB SHAP bar](shap_plots/feature_set_02_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 02 XGB dependence](shap_plots/feature_set_02_xgb_air_temp_mean_dependence.png)

![Feature set 02 XGB dependence](shap_plots/feature_set_02_xgb_slope_dependence.png)

![Feature set 02 XGB dependence](shap_plots/feature_set_02_xgb_precip_mm_dependence.png)

---

## Feature set 03

Best run (by test $R^2$): `Models/Temporal/v2/v2.1/mdr_ts_v2_1_20260106_211015`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.9408 | 0.0246 | 0.0178 |
| Val   | 0.6806 | 0.0597 | 0.0444 |
| Test  | 0.6540 | 0.0572 | 0.0435 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature         | Mean (SHAP) |
| ---- | --------------- | ----------- |
| 1    | DOY             | 0.042773    |
| 2    | air_temp_mean   | 0.029362    |
| 3    | rh_mean         | 0.017001    |
| 4    | precip_mm       | 0.008263    |
| 5    | s1_vv           | 0.007499    |
| 6    | s1_vh           | 0.007445    |
| 7    | solar_radiation | 0.007054    |
| 8    | SAR_ratio       | 0.006265    |
| 9    | NDVI            | 0.005113    |
| 10   | NDMI            | 0.003214    |

![Feature set 03 XGB SHAP summary](shap_plots/feature_set_03_xgb_summary.png)

![Feature set 03 XGB SHAP bar](shap_plots/feature_set_03_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 03 XGB dependence](shap_plots/feature_set_03_xgb_DOY_dependence.png)

![Feature set 03 XGB dependence](shap_plots/feature_set_03_xgb_air_temp_mean_dependence.png)

![Feature set 03 XGB dependence](shap_plots/feature_set_03_xgb_rh_mean_dependence.png)

---

## Feature set 04

Best run (by test $R^2$): `Models/Temporal/v1/v1.1/mdr_ts_v1_1_20251223_113030`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.8747 | 0.0358 | 0.0250 |
| Val   | 0.5445 | 0.0713 | 0.0542 |
| Test  | 0.4316 | 0.0733 | 0.0586 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature         | Mean (SHAP) |
| ---- | --------------- | ----------- |
| 1    | air_temp_mean   | 0.047036    |
| 2    | rh_mean         | 0.011767    |
| 3    | slope           | 0.011720    |
| 4    | solar_radiation | 0.009039    |
| 5    | elev            | 0.008560    |
| 6    | precip_mm       | 0.008556    |
| 7    | s1_vv           | 0.008389    |
| 8    | NDVI            | 0.007410    |
| 9    | s1_vh           | 0.006823    |
| 10   | SAR_ratio       | 0.005562    |

![Feature set 04 XGB SHAP summary](shap_plots/feature_set_04_xgb_summary.png)

![Feature set 04 XGB SHAP bar](shap_plots/feature_set_04_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 04 XGB dependence](shap_plots/feature_set_04_xgb_air_temp_mean_dependence.png)

![Feature set 04 XGB dependence](shap_plots/feature_set_04_xgb_rh_mean_dependence.png)

![Feature set 04 XGB dependence](shap_plots/feature_set_04_xgb_slope_dependence.png)

---

## Feature set 05

Best run (by test $R^2$): `Models/Temporal/v1/v1.0/mdr_ts_v1_0_20251223_105722`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.9444 | 0.0238 | 0.0171 |
| Val   | 0.6907 | 0.0588 | 0.0431 |
| Test  | 0.6125 | 0.0605 | 0.0459 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature         | Mean (SHAP) |
| ---- | --------------- | ----------- |
| 1    | DOY             | 0.042119    |
| 2    | air_temp_mean   | 0.028511    |
| 3    | rh_mean         | 0.017802    |
| 4    | slope           | 0.013600    |
| 5    | elev            | 0.008621    |
| 6    | solar_radiation | 0.007526    |
| 7    | precip_mm       | 0.006148    |
| 8    | s1_vv           | 0.005418    |
| 9    | NDVI            | 0.004821    |
| 10   | s1_vh           | 0.003933    |

![Feature set 05 XGB SHAP summary](shap_plots/feature_set_05_xgb_summary.png)

![Feature set 05 XGB SHAP bar](shap_plots/feature_set_05_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 05 XGB dependence](shap_plots/feature_set_05_xgb_DOY_dependence.png)

![Feature set 05 XGB dependence](shap_plots/feature_set_05_xgb_air_temp_mean_dependence.png)

![Feature set 05 XGB dependence](shap_plots/feature_set_05_xgb_rh_mean_dependence.png)

---

## Feature set 06

Best run (by test $R^2$): `Models/Temporal/v2/v2.3/mdr_ts_v2_2_20260107_170214`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.9634 | 0.0195 | 0.0141 |
| Val   | 0.7650 | 0.0521 | 0.0397 |
| Test  | 0.6715 | 0.0533 | 0.0411 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature      | Mean (SHAP) |
| ---- | ------------ | ----------- |
| 1    | API          | 0.036759    |
| 2    | DOY          | 0.027018    |
| 3    | LST_modis    | 0.016951    |
| 4    | s2_b8        | 0.009027    |
| 5    | s1_vv        | 0.007170    |
| 6    | SAR_ratio_sa | 0.007100    |
| 7    | NDMI_sa      | 0.006819    |
| 8    | SAR_ratio    | 0.004949    |
| 9    | s1_vh        | 0.004866    |
| 10   | NDMI         | 0.004663    |

![Feature set 06 XGB SHAP summary](shap_plots/feature_set_06_xgb_summary.png)

![Feature set 06 XGB SHAP bar](shap_plots/feature_set_06_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 06 XGB dependence](shap_plots/feature_set_06_xgb_API_dependence.png)

![Feature set 06 XGB dependence](shap_plots/feature_set_06_xgb_DOY_dependence.png)

![Feature set 06 XGB dependence](shap_plots/feature_set_06_xgb_LST_modis_dependence.png)

---

## Feature set 07

Best run (by test $R^2$): `Models/Temporal/v3/v3.3/mdr_ts_v3_3_20260109_113817`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.8479 | 0.0397 | 0.0303 |
| Val   | 0.7771 | 0.0507 | 0.0391 |
| Test  | 0.7216 | 0.0491 | 0.0386 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature                      | Mean (SHAP) |
| ---- | ---------------------------- | ----------- |
| 1    | DOY                          | 0.024469    |
| 2    | C_smm_G_API_alpha0.85_n5     | 0.017969    |
| 3    | C_smm_LST_modis_alpha0.85_n5 | 0.016021    |
| 4    | G_API                        | 0.011978    |
| 5    | aspect                       | 0.009650    |
| 6    | s2_b8                        | 0.008646    |
| 7    | G_rain_sum_30d               | 0.005383    |
| 8    | s1_vv                        | 0.004558    |
| 9    | D_z_F_NDMI                   | 0.003882    |
| 10   | D_sa_F_NDMI                  | 0.003780    |

![Feature set 07 XGB SHAP summary](shap_plots/feature_set_07_xgb_summary.png)

![Feature set 07 XGB SHAP bar](shap_plots/feature_set_07_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 07 XGB dependence](shap_plots/feature_set_07_xgb_DOY_dependence.png)

![Feature set 07 XGB dependence](shap_plots/feature_set_07_xgb_C_smm_G_API_alpha0.85_n5_dependence.png)

![Feature set 07 XGB dependence](shap_plots/feature_set_07_xgb_C_smm_LST_modis_alpha0.85_n5_dependence.png)

---

## Feature set 08

Best run (by test $R^2$): `Models/Temporal/v3/v3.1/mdr_ts_v3_1_20260108_151234`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.9686 | 0.0180 | 0.0129 |
| Val   | 0.7755 | 0.0509 | 0.0395 |
| Test  | 0.6226 | 0.0571 | 0.0441 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature                      | Mean (SHAP) |
| ---- | ---------------------------- | ----------- |
| 1    | DOY                          | 0.026218    |
| 2    | C_smm_G_API_alpha0.85_n5     | 0.022854    |
| 3    | C_smm_LST_modis_alpha0.85_n5 | 0.018396    |
| 4    | aspect                       | 0.011868    |
| 5    | G_API                        | 0.008791    |
| 6    | s2_b8                        | 0.007421    |
| 7    | G_rain_sum_3d                | 0.006371    |
| 8    | E_SAR_diff                   | 0.003914    |
| 9    | G_rain_sum_30d               | 0.003726    |
| 10   | D_z_F_NDMI                   | 0.003659    |

![Feature set 08 XGB SHAP summary](shap_plots/feature_set_08_xgb_summary.png)

![Feature set 08 XGB SHAP bar](shap_plots/feature_set_08_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 08 XGB dependence](shap_plots/feature_set_08_xgb_DOY_dependence.png)

![Feature set 08 XGB dependence](shap_plots/feature_set_08_xgb_C_smm_G_API_alpha0.85_n5_dependence.png)

![Feature set 08 XGB dependence](shap_plots/feature_set_08_xgb_C_smm_LST_modis_alpha0.85_n5_dependence.png)

---

## Feature set 09

Best run (by test $R^2$): `Models/Temporal/v8/v8.1/mdr_ts_v8_1_20260121_105425`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.9714 | 0.0173 | 0.0123 |
| Val   | 0.7729 | 0.0480 | 0.0368 |
| Test  | 0.6742 | 0.0534 | 0.0417 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature                     | Mean (SHAP) |
| ---- | --------------------------- | ----------- |
| 1    | V_ema_LST_modis_kobs30      | 0.026414    |
| 2    | G_rain_sum_30d              | 0.018342    |
| 3    | DOY                         | 0.014421    |
| 4    | C_lag_LST_modis_kobs30      | 0.010545    |
| 5    | V_rollmin_E_SAR_diff_kobs30 | 0.009308    |
| 6    | V_rollmin_F_NDMI_kobs30     | 0.008718    |
| 7    | days_since_rain_event       | 0.007274    |
| 8    | rain_mm_impulse_0_7         | 0.006466    |
| 9    | s2_b8                       | 0.005532    |
| 10   | rain_event_impulse_0_7      | 0.004410    |

![Feature set 09 XGB SHAP summary](shap_plots/feature_set_09_xgb_summary.png)

![Feature set 09 XGB SHAP bar](shap_plots/feature_set_09_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 09 XGB dependence](shap_plots/feature_set_09_xgb_V_ema_LST_modis_kobs30_dependence.png)

![Feature set 09 XGB dependence](shap_plots/feature_set_09_xgb_G_rain_sum_30d_dependence.png)

![Feature set 09 XGB dependence](shap_plots/feature_set_09_xgb_DOY_dependence.png)

---

## Feature set 10

Best run (by test $R^2$): `Models/Temporal/v7/v7.1/mdr_ts_v7_1_20260117_161048`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.9760 | 0.0159 | 0.0113 |
| Val   | 0.7820 | 0.0470 | 0.0363 |
| Test  | 0.6992 | 0.0513 | 0.0397 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature                     | Mean (SHAP) |
| ---- | --------------------------- | ----------- |
| 1    | V_ema_LST_modis_kobs30      | 0.022706    |
| 2    | DOY                         | 0.015162    |
| 3    | V_rollmin_G_API_kobs7       | 0.011473    |
| 4    | V_rollmin_E_SAR_diff_kobs30 | 0.009893    |
| 5    | C_lag_LST_modis_kobs30      | 0.008203    |
| 6    | D_z_F_NDMI                  | 0.007831    |
| 7    | V_rollmax_G_API_kobs7       | 0.007680    |
| 8    | G_API                       | 0.007600    |
| 9    | V_rollmin_F_NDMI_kobs30     | 0.007489    |
| 10   | G_rain_sum_3d               | 0.006516    |

![Feature set 10 XGB SHAP summary](shap_plots/feature_set_10_xgb_summary.png)

![Feature set 10 XGB SHAP bar](shap_plots/feature_set_10_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 10 XGB dependence](shap_plots/feature_set_10_xgb_V_ema_LST_modis_kobs30_dependence.png)

![Feature set 10 XGB dependence](shap_plots/feature_set_10_xgb_DOY_dependence.png)

![Feature set 10 XGB dependence](shap_plots/feature_set_10_xgb_V_rollmin_G_API_kobs7_dependence.png)

### Model: RF

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.9742 | 0.0164 | 0.0106 |
| Val   | 0.7985 | 0.0452 | 0.0340 |
| Test  | 0.7225 | 0.0493 | 0.0376 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature                 | Mean (SHAP) |
| ---- | ----------------------- | ----------- |
| 1    | V_ema_LST_modis_kobs30  | 0.025015    |
| 2    | DOY                     | 0.011724    |
| 3    | V_rollmin_G_API_kobs7   | 0.009650    |
| 4    | C_lag_LST_modis_kobs30  | 0.007423    |
| 5    | V_rollmin_F_NDMI_kobs30 | 0.007173    |
| 6    | G_API                   | 0.006611    |
| 7    | V_rollmax_G_API_kobs7   | 0.006600    |
| 8    | G_rain_sum_30d          | 0.006358    |
| 9    | D_z_F_NDMI              | 0.005801    |
| 10   | C_lag_LST_modis_kobs12  | 0.005517    |

![Feature set 10 RF SHAP summary](shap_plots/feature_set_10_rf_summary.png)

![Feature set 10 RF SHAP bar](shap_plots/feature_set_10_rf_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 10 RF dependence](shap_plots/feature_set_10_rf_V_ema_LST_modis_kobs30_dependence.png)

![Feature set 10 RF dependence](shap_plots/feature_set_10_rf_DOY_dependence.png)

![Feature set 10 RF dependence](shap_plots/feature_set_10_rf_V_rollmin_G_API_kobs7_dependence.png)

---

## Feature set 11

Best run (by test $R^2$): `Models/Temporal/v3/v3.1/mdr_ts_v3_1_20260108_144926`

### Model: XGB

**Retrained metrics:**

| Split | $R^2$  | RMSE   | MAE    |
| ----- | ------ | ------ | ------ |
| Train | 0.9736 | 0.0166 | 0.0119 |
| Val   | 0.7782 | 0.0506 | 0.0390 |
| Test  | 0.6289 | 0.0566 | 0.0443 |

**Top features by mean |SHAP| (normalized):**

| Rank | Feature                   | Mean (SHAP) |
| ---- | ------------------------- | ----------- |
| 1    | DOY                       | 0.023681    |
| 2    | V_ema_G_API_kobs7         | 0.016551    |
| 3    | aspect                    | 0.011128    |
| 4    | C_lag_LST_modis_kobs5     | 0.009491    |
| 5    | V_rollmin_G_API_kobs7     | 0.007961    |
| 6    | s2_b8                     | 0.006835    |
| 7    | V_rollmin_LST_modis_kobs7 | 0.005539    |
| 8    | C_lag_G_API_kobs1         | 0.005048    |
| 9    | G_rain_sum_3d             | 0.004153    |
| 10   | D_z_F_NDMI                | 0.003691    |

![Feature set 11 XGB SHAP summary](shap_plots/feature_set_11_xgb_summary.png)

![Feature set 11 XGB SHAP bar](shap_plots/feature_set_11_xgb_bar.png)

**Relationships (SHAP dependence plots):**

![Feature set 11 XGB dependence](shap_plots/feature_set_11_xgb_DOY_dependence.png)

![Feature set 11 XGB dependence](shap_plots/feature_set_11_xgb_V_ema_G_API_kobs7_dependence.png)

![Feature set 11 XGB dependence](shap_plots/feature_set_11_xgb_aspect_dependence.png)

---
