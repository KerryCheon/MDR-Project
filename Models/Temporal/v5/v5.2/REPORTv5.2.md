# Report on v5.2.0 (Feature Ablation by Feature Family)

**Author: Jakob Balkovec**
**Date: Jan 15, 2026**

## Goal

The goal of this experiment was to understand which feature families actually add predictive signal beyond the raw inputs for the model.

I trained several (9) `XGBRegressor` models, each with a different set of feature families included:

- `raw_only`: only raw features
- `raw_A`: raw + family A
- `raw_B`: raw + family B
- `raw_C`: raw + family C
- `raw_D`: raw + family D
- `raw_E`: raw + family E
- `raw_F`: raw + family F
- `raw_G`: raw + family G
- `raw_H`: raw + family H
- `raw_I`: raw + family I

This let me measure:

- Performance uplift vs raw features
- Which types of features matter
- Which specific engineered features dominate within each family

## Experimental Setup

- Model: XGBoost (`reg:squarederror`)
- Depth: 3
- Regularization: lambda = 30, min_child_weight = 5
- Sampling: subsample = 1.0, colsample_bytree = 0.6
- Learning rate: 0.05
- Early stopping: 75 rounds
- Evaluation: temporal split (train / val / test)
- Importance metric: gain

For each family:

- Trained on raw + family
- Reported test metrics
- Extracted top 5 features from the family only (raw features excluded from reporting)

## Results

> Sorted by test R² uplift vs raw-only model

| name     | n_feat | best_iter | train_R2 | val_R2   | test_R2  | test_MAE | test_RMSE | test_R2_uplift_vs_raw |
| -------- | ------ | --------- | -------- | -------- | -------- | -------- | --------- | --------------------- |
| raw+C    | 66     | 180       | 0.891193 | 0.796591 | 0.681632 | 0.040118 | 0.052463  | 0.079456              |
| raw+A    | 90     | 508       | 0.896721 | 0.763101 | 0.667863 | 0.041556 | 0.053586  | 0.065687              |
| raw+B    | 178    | 182       | 0.898275 | 0.774850 | 0.666092 | 0.041196 | 0.053728  | 0.063916              |
| raw+G    | 16     | 614       | 0.905399 | 0.798573 | 0.659174 | 0.041240 | 0.054282  | 0.056998              |
| raw+E    | 17     | 153       | 0.809768 | 0.692876 | 0.610129 | 0.044448 | 0.058056  | 0.007953              |
| raw+F    | 13     | 169       | 0.813300 | 0.712534 | 0.609532 | 0.044106 | 0.058101  | 0.007356              |
| raw_only | 10     | 222       | 0.818618 | 0.708935 | 0.602176 | 0.044564 | 0.058646  | NaN                   |
| raw+D    | 22     | 173       | 0.819589 | 0.697613 | 0.600000 | 0.045488 | 0.058806  | -0.002176             |
| raw+I    | 11     | 234       | 0.821392 | 0.705402 | 0.597486 | 0.045397 | 0.058990  | -0.004690             |
| raw+H    | 14     | 379       | 0.833990 | 0.707535 | 0.594280 | 0.044992 | 0.059225  | -0.007895             |

Some key takeaways here:

- Family C (memory / lag features) is the clear winner...
- Families A and B also add strong signal
- Meteorology (G) punches above its weight with very few features
- Seasonality (D), events (I), and cross-signal coupling (H) hurt generalization

## Top Features by Family (Gain-Based)

> Note that gain is defined in terms of MSE reduction, so higher is better

### Family C: Memory & Lag (Best Performer)

Strong evidence that soil moisture has multi-scale temporal memory

| name  | rank | feature                   | gain     |
| ----- | ---- | ------------------------- | -------- |
| raw+C | 1    | C_smm_G_API_alpha0.85_n5  | 6.297982 |
| raw+C | 2    | C_lag_LST_modis_kobs12    | 4.748018 |
| raw+C | 3    | C_lag_LST_modis_kobs30    | 2.322512 |
| raw+C | 4    | C_lag_G_API_kobs6         | 1.539671 |
| raw+C | 5    | C_smm_F_NDMI_alpha0.85_n5 | 1.419127 |

**Interpretation**

- Exponentially decayed memory (SMM) is extremely effective
- Long lags (12–30 obs) matter more than short ones
- API and LST are the dominant drivers

---

### Family A: Short-Term Dynamics

Captures rate-of-change effects, especially after rainfall

| name  | rank | feature                | gain     |
| ----- | ---- | ---------------------- | -------- |
| raw+A | 1    | A_d_G_API_kobs1        | 0.546162 |
| raw+A | 2    | A_grad_LST_modis_kobs7 | 0.483721 |
| raw+A | 3    | A_d_G_API_kobs2        | 0.448454 |
| raw+A | 4    | A_d_G_API_kobs30       | 0.317390 |
| raw+A | 5    | A_grad_s2_b11_kobs30   | 0.232089 |

**Interpretation**

- Immediate precipitation changes matter most
- Thermal gradients also contribute, but less strongly than memory

---

### Family B — Smoothing & Volatility

High raw gain, but diminishing returns overall

| name  | rank | feature                    | gain      |
| ----- | ---- | -------------------------- | --------- |
| raw+B | 1    | V_ema_LST_modis_kobs30     | 10.807203 |
| raw+B | 2    | V_rollmin_G_API_kobs14     | 7.331640  |
| raw+B | 3    | V_rollmin_LST_modis_kobs30 | 7.220735  |
| raw+B | 4    | V_ema_G_API_kobs14         | 4.141772  |
| raw+B | 5    | V_rollmean_G_API_kobs7     | 2.440078  |

Interpretation

- Long-horizon smoothing is powerful
- But many features overlap, inflating dimensionality without proportional gains

---

### Family G: Meteorological Forcing

Very strong signal with very few features

| name  | rank | feature        | gain     |
| ----- | ---- | -------------- | -------- |
| raw+G | 1    | G_API          | 1.531489 |
| raw+G | 2    | G_rain_sum_30d | 0.357107 |
| raw+G | 3    | G_rain_sum_3d  | 0.131231 |
| raw+G | 4    | G_rain_sum_7d  | 0.101207 |
| raw+G | 5    | G_DSLR         | 0.048658 |

**Interpretation**

- API dominates everything
- Longer rainfall windows matter more than short ones

---

### Families E, F: Radar & Optical

Helpful, but secondary

- Radar (E): SAR difference and roughness add modest signal
- Optical (F): NDMI and MSI matter more than NDVI

---

### Families D, H, I: Weak or Harmful

- Seasonality anomalies overfit
- Cross-signal correlations are noisy
- Event timing features do not generalize well

These families likely need regularization, pruning, or redesign before reuse

## Final Set of Features

These are the final 40 features (40 to keep it consistent with what I've been doing):

**Handpicked**:

```python
[
'V_ema_LST_modis_kobs30',
'C_smm_G_API_alpha0.85_n5',
'V_rollmin_G_API_kobs14',
'V_rollmin_LST_modis_kobs30',
'C_lag_LST_modis_kobs12',
'V_ema_G_API_kobs14',
'C_lag_LST_modis_kobs30',
'V_rollmean_G_API_kobs7',
'V_rollmin_G_API_kobs30',
'V_ema_LST_modis_kobs14',
'V_ema_G_API_kobs7',
'V_rollmax_G_API_kobs14',
'C_lag_G_API_kobs6',
'C_smm_F_NDMI_alpha0.85_n5',
'C_lag_LST_modis_kobs6',
'C_lag_G_API_kobs1',
'G_API',
'V_rollmin_F_NDMI_kobs7',
'C_lag_G_API_kobs5',
'V_rollmax_G_API_kobs7',
'C_lag_G_API_kobs12',
'V_rollmin_E_SAR_diff_kobs30',
'C_lag_G_API_kobs2',
'V_rollmin_F_NDMI_kobs14',
'V_ema_F_NDMI_kobs30',
'V_rollcv_F_NDMI_kobs30',
'V_rollmin_E_SAR_diff_kobs14',
'V_rollmin_F_NDMI_kobs30',
'C_lag_F_NDMI_kobs1',
'C_smm_E_SAR_ratio_alpha0.85_n5',
'C_lag_F_NDMI_kobs2',
'C_smm_E_SAR_diff_alpha0.85_n5',
'V_rollmin_G_API_kobs7',
'V_rollmax_E_SAR_diff_kobs14',
'C_lag_E_SAR_diff_kobs2',
'C_lag_s2_b11_kobs6',
'A_d_G_API_kobs1',
'V_ema_G_API_kobs30',
'C_lag_F_NDMI_kobs12',
'V_rollmax_F_NDMI_kobs14'
]

```

**ElasticNet**:

```python
[
'E_SAR_ratio',
'D_sa_E_SAR_ratio',
'V_ema_LST_modis_kobs30',
'V_ema_E_SAR_ratio_kobs30',
'V_ema_E_SAR_ratio_kobs14',
'D_z_E_SAR_ratio',
'D_z_F_NDMI',
'V_rollmean_F_NDMI_kobs30',
'V_rollmax_s2_b12_kobs30',
'V_ema_F_NDMI_kobs30',
'V_rollmin_s2_b12_kobs30',
'V_rollmean_E_SAR_ratio_kobs30',
'V_rollcv_LST_modis_kobs30',
'V_rollmin_F_NDMI_kobs30',
'V_rollstd_LST_modis_kobs30',
's1_vh',
'V_rollstd_LST_modis_kobs14',
'V_ema_G_API_kobs30',
'V_rollcv_LST_modis_kobs14',
'V_rollmax_s2_b11_kobs30',
'V_ema_E_SAR_ratio_kobs7',
'V_rollmean_s2_b12_kobs30',
'C_lag_F_NDMI_kobs30',
'V_rollmax_F_NDVI_kobs30',
'C_lag_F_NDMI_kobs1',
'V_rollmean_LST_modis_kobs30',
'V_ema_s2_b12_kobs30',
'C_lag_E_SAR_ratio_kobs1',
'V_rollmin_s2_b11_kobs30',
'V_ema_s2_b11_kobs30',
'V_rollstd_s2_b12_kobs30',
'V_rollmax_E_SAR_ratio_kobs30',
'V_rollrng_F_NDVI_kobs30',
'V_rollmean_LST_modis_kobs14',
'D_fft_ent_F_NDMI_kobs30',
'D_sa_F_NDMI',
'V_rollmin_F_NDVI_kobs30',
'G_API',
'A_d_F_NDMI_kobs30',
'V_rollcv_s2_b11_kobs30'
]
```

**Common Features**

```python
C_lag_F_NDMI_kobs1
G_API
V_ema_F_NDMI_kobs30
V_ema_G_API_kobs30
V_ema_LST_modis_kobs30
V_rollmin_F_NDMI_kobs30
```

All in all, I think family V and family C are the clear winners...I think that's where we get most of our signal from.

## New XGBoost Model Results:

| split | R2       | MAE      | RMSE     |
| ----- | -------- | -------- | -------- |
| train | 0.877047 | 0.026388 | 0.035707 |
| val   | 0.768260 | 0.039763 | 0.051717 |
| test  | 0.651347 | 0.042455 | 0.054902 |

## ElasticNet Results

I conducted the same experiment using an `ElasticNet` model as well. The results are from the model fitted with the new 40 features (picked by ElasticNet)

| split | R2       | MAE      | RMSE     |
| ----- | -------- | -------- | -------- |
| train | 0.722787 | 0.042621 | 0.053616 |
| val   | 0.684616 | 0.047666 | 0.060332 |
| test  | 0.564117 | 0.049490 | 0.061387 |

> Note that these are not necessarily the "best" features, just the ones with the biggest impact. They were sorted based on the magnitude of the coefficient.

## My Final Conclusions

1. Temporal memory is the single most important factor
2. API-driven features consistently dominate across families
3. Many features add redundancy rather than signal
4. A compact, high-performing model should prioritize:
   - Family C (memory)
   - Select features from A, B, and G
5. Families D, H, and I should be excluded from the final model
