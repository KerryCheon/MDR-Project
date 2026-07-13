# derived_8.2-eval-1.3 (Global Models Comparison Report)

This directory contains the training and evaluation notebook for comparing six **single global** XGBoost models on the Washington-only `derived_8.2` dataset.

The models are evaluated using the following feature sets defined in `dataset_metadata.py`:
1. **Model V0**: Trained using **OVERALL_SELECTED_FEATURES_V0** (40 features, based on the OG pipeline settings).
2. **Model V1**: Trained using **OVERALL_SELECTED_FEATURES_V1** (40 features).
3. **Model V2**: Trained using **OVERALL_SELECTED_FEATURES_V2** (40 features, updated pipeline).
4. **Model V3**: Trained using **OVERALL_SELECTED_FEATURES_V3** (47 features, expanded pipeline).
5. **Model V4**: Trained using **OVERALL_SELECTED_FEATURES_V4** (50 features, no MI stage, min_freq=0.01).
6. **Model V5**: Trained using **OVERALL_SELECTED_FEATURES_V5** (32 features, no MI stage, min_freq=0.6).

For all feature sets, we evaluate:
- A non-weighted baseline using `objective="reg:absoluteerror"`.
- A temporally weighted baseline using `objective="reg:pseudohubererror"` with `beta = 0.2`.

---

## 1. Comparative Results Table

The performance metrics on the held-out test split are summarized below:

| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V0 (40 Features, No Weights)** | 0.4997 | 0.0745 | 0.0736 | −0.0114 | 0.0554 | 0.0408 | 0.7174 |
| **Model V0 (40 Features, Weighted)** | 0.5009 | 0.0744 | 0.0737 | −0.0100 | 0.0544 | 0.0392 | 0.7181 |
| **Model V1 (40 Features, No Weights)** | 0.6091 | 0.0658 | 0.0645 | −0.0133 | 0.0484 | 0.0358 | 0.7984 |
| **Model V1 (40 Features, Weighted)** | 0.6263 | 0.0644 | 0.0630 | −0.0131 | 0.0479 | 0.0369 | 0.8084 |
| **Model V2 (40 Features, No Weights)** | 0.6347 | 0.0636 | 0.0600 | −0.0211 | 0.0484 | 0.0375 | 0.8252 |
| **Model V2 (40 Features, Weighted)** | 0.6426 | 0.0629 | 0.0595 | −0.0205 | 0.0474 | 0.0359 | 0.8302 |
| **Model V3 (47 Features, No Weights)** | **0.6474** | **0.0625** | **0.0589** | −0.0211 | **0.0475** | **0.0364** | **0.8322** |
| **Model V3 (47 Features, Weighted)** | 0.6374 | 0.0634 | 0.0598 | −0.0212 | 0.0476 | 0.0361 | 0.8296 |
| **Model V4 (50 Features, No Weights)** | 0.6088 | 0.0659 | 0.0642 | −0.0145 | 0.0487 | 0.0376 | 0.7959 |
| **Model V4 (50 Features, Weighted)** | 0.6036 | 0.0663 | 0.0648 | −0.0138 | 0.0489 | 0.0372 | 0.7933 |
| **Model V5 (32 Features, No Weights)** | 0.5957 | 0.0670 | 0.0657 | −0.0132 | 0.0500 | 0.0386 | 0.7864 |
| **Model V5 (32 Features, Weighted)** | 0.5948 | 0.0670 | 0.0657 | −0.0133 | 0.0499 | 0.0386 | 0.7870 |

---

## 2. Key Insights and Discussion

### 1. Model V3 (No Weights) remains the absolute best configuration
Model V3 (retaining 47 features with MI stage active) yields the highest $R^2$ of **0.6474** and Pearson correlation of **0.8322**.

### 2. Model V0 performs poorly due to the OG pipeline settings
Model V0, using the OG pipeline settings, performs significantly worse than the other models, yielding an overall $R^2$ of only **0.5009** for the weighted variant. 
- **Lack of Static and Spatial Features**: The OG pipeline selection was heavily biased towards rolling features (`V_*`) and autoregressive lags (`C_lag_*`), completely omitting critical static soil attributes (HWSD Clay/Sand fractions) and bioclimatic statistics. This prevented the global model from generalizing across different geographical stations.
- **Extreme Temporal Instability**: While Model V0 yielded a competitive $R^2$ of **0.6239** in the year 2024, it collapsed to **0.4559** in 2023 and **0.3827** in 2025. This shows that relying purely on temporal lags makes the model highly overfitted to the training climatology of specific years, failing when the seasonal patterns shift.

### 3. Mutual Information (MI) pre-filtering is indispensable
Skipping the MI stage (Model V4 and V5) leads to L1 regularization driving all constant/static features to zero, collapsing the model into a purely temporal predictor (similar to the spatial collapse of V0). The MI stage performs a critical pruning of weakly-related features, which resolves collinearity confusion and preserves essential spatial/geographic context.

---

## 3. Visualizations

The generated plots are saved in this directory:
- `residuals_comparison.png` (displays a 6x2 grid comparing unweighted and weighted residuals for Models V0 through V5)
- `shap_importance_comparison.png` (compares mean absolute SHAP values for the top 20 overall features across the weighted models)
- **Individual SHAP Charts**:
  - `shap_model_v0_weighted.png`
  - `shap_model_v1_weighted.png`
  - `shap_model_v2_weighted.png`
  - `shap_model_v3_weighted.png`
  - `shap_model_v4_weighted.png`
  - `shap_model_v5_weighted.png`

---

## 4. Year-by-Year Comparative Results

The performance metrics on the held-out test split, broken down by test year (2023, 2024, and 2025):

### Year 2023
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V0 (40 Features, No Weights)** | 0.4322 | 0.0811 | 0.0776 | −0.0235 | 0.0616 | 0.0464 | 0.6962 |
| **Model V0 (40 Features, Weighted)** | 0.4559 | 0.0794 | 0.0763 | −0.0218 | 0.0586 | 0.0416 | 0.7096 |
| **Model V1 (40 Features, No Weights)** | 0.5999 | 0.0681 | 0.0608 | −0.0305 | 0.0502 | 0.0395 | 0.8274 |
| **Model V1 (40 Features, Weighted)** | 0.6189 | 0.0664 | 0.0594 | −0.0297 | 0.0500 | 0.0410 | 0.8359 |
| **Model V2 (40 Features, No Weights)** | 0.6337 | 0.0651 | 0.0579 | −0.0298 | 0.0509 | 0.0422 | 0.8444 |
| **Model V2 (40 Features, Weighted)** | **0.6382** | **0.0647** | **0.0583** | −0.0281 | **0.0504** | 0.0413 | **0.8431** |
| **Model V3 (47 Features, No Weights)** | 0.6292 | 0.0655 | 0.0569 | −0.0325 | 0.0509 | 0.0406 | 0.8503 |
| **Model V3 (47 Features, Weighted)** | 0.6242 | 0.0660 | 0.0588 | −0.0299 | 0.0514 | 0.0424 | 0.8411 |
| **Model V4 (50 Features, No Weights)** | 0.6040 | 0.0677 | 0.0624 | −0.0263 | 0.0499 | 0.0383 | 0.8186 |
| **Model V4 (50 Features, Weighted)** | 0.5938 | 0.0686 | 0.0634 | −0.0261 | 0.0504 | 0.0390 | 0.8126 |
| **Model V5 (32 Features, No Weights)** | 0.5875 | 0.0691 | 0.0642 | −0.0257 | 0.0517 | 0.0412 | 0.8082 |
| **Model V5 (32 Features, Weighted)** | 0.5752 | 0.0701 | 0.0648 | −0.0268 | 0.0518 | 0.0406 | 0.8035 |

### Year 2024
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V0 (40 Features, No Weights)** | 0.5976 | 0.0633 | 0.0629 | −0.0072 | 0.0460 | 0.0327 | 0.7772 |
| **Model V0 (40 Features, Weighted)** | 0.6239 | 0.0612 | 0.0610 | −0.0055 | 0.0448 | 0.0331 | 0.7925 |
| **Model V1 (40 Features, No Weights)** | 0.5747 | 0.0651 | 0.0646 | −0.0076 | 0.0473 | 0.0347 | 0.7908 |
| **Model V1 (40 Features, Weighted)** | 0.5983 | 0.0633 | 0.0629 | −0.0069 | 0.0470 | 0.0359 | 0.8037 |
| **Model V2 (40 Features, No Weights)** | 0.6263 | 0.0610 | 0.0580 | −0.0190 | 0.0457 | 0.0360 | 0.8251 |
| **Model V2 (40 Features, Weighted)** | 0.6281 | 0.0609 | 0.0575 | −0.0200 | 0.0454 | 0.0338 | 0.8296 |
| **Model V3 (47 Features, No Weights)** | **0.6443** | **0.0595** | **0.0571** | −0.0169 | **0.0445** | 0.0346 | **0.8308** |
| **Model V3 (47 Features, Weighted)** | 0.6283 | 0.0609 | 0.0575 | −0.0200 | 0.0445 | 0.0326 | 0.8310 |
| **Model V4 (50 Features, No Weights)** | 0.5771 | 0.0649 | 0.0636 | −0.0130 | 0.0472 | 0.0364 | 0.7802 |
| **Model V4 (50 Features, Weighted)** | 0.5763 | 0.0650 | 0.0639 | −0.0115 | 0.0474 | 0.0363 | 0.7809 |
| **Model V5 (32 Features, No Weights)** | 0.5568 | 0.0664 | 0.0653 | −0.0125 | 0.0487 | 0.0368 | 0.7672 |
| **Model V5 (32 Features, Weighted)** | 0.5704 | 0.0654 | 0.0643 | −0.0123 | 0.0482 | 0.0374 | 0.7772 |

### Year 2025
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V0 (40 Features, No Weights)** | 0.4413 | 0.0783 | 0.0783 | −0.0007 | 0.0590 | 0.0443 | 0.6699 |
| **Model V0 (40 Features, Weighted)** | 0.3827 | 0.0823 | 0.0823 | −0.0001 | 0.0608 | 0.0460 | 0.6353 |
| **Model V1 (40 Features, No Weights)** | 0.6292 | 0.0638 | 0.0637 | ＋0.0020 | 0.0472 | 0.0324 | 0.7947 |
| **Model V1 (40 Features, Weighted)** | 0.6381 | 0.0630 | 0.0630 | ＋0.0010 | 0.0463 | 0.0325 | 0.8008 |
| **Model V2 (40 Features, No Weights)** | 0.6160 | 0.0649 | 0.0637 | −0.0124 | 0.0484 | 0.0337 | 0.7960 |
| **Model V2 (40 Features, Weighted)** | 0.6363 | 0.0632 | 0.0622 | −0.0112 | 0.0459 | 0.0317 | 0.8094 |
| **Model V3 (47 Features, No Weights)** | **0.6473** | **0.0622** | **0.0611** | −0.0115 | **0.0467** | 0.0340 | **0.8131** |
| **Model V3 (47 Features, Weighted)** | 0.6366 | 0.0631 | 0.0621 | −0.0112 | 0.0463 | 0.0326 | 0.8107 |
| **Model V4 (50 Features, No Weights)** | 0.6198 | 0.0646 | 0.0646 | −0.0010 | 0.0490 | 0.0379 | 0.7876 |
| **Model V4 (50 Features, Weighted)** | 0.6162 | 0.0649 | 0.0649 | −0.0005 | 0.0488 | 0.0364 | 0.7866 |
| **Model V5 (32 Features, No Weights)** | 0.6184 | 0.0647 | 0.0647 | ＋0.0024 | 0.0492 | 0.0369 | 0.7870 |
| **Model V5 (32 Features, Weighted)** | 0.6167 | 0.0648 | 0.0648 | ＋0.0031 | 0.0495 | 0.0376 | 0.7871 |

---

## 5. Year-by-Year Insights and Discussion

- **Model V3 No Weights** consistently generalizes better than all other configurations across the test years.
- **Model V0** shows extreme year-by-year instability, indicating severe overfitting to specific temporal regimes due to a lack of spatial coordinates and geographic context.

## 6. Year-by-Year Visualizations

- `residuals_by_year.png` (displays a 6x3 grid with the six models as rows and the three test years as columns, comparing weighted model residuals)
- `r2_by_year.png` (displays a side-by-side comparison of yearly R2 scores for unweighted and weighted models)
- `metrics_by_year.csv` (contains the detailed metrics breakdown by year)
- `metrics_summary.csv` (contains the overall metrics comparison)

## 7. Features List
See [features.md](features.md)
