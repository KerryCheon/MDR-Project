# derived_8.2-eval-1.1 (Global Models Comparison Report)

This directory contains the training and evaluation notebook for comparing four **single global** XGBoost models on the Washington-only `derived_8.2` dataset.

Both configurations use the modeling techniques from the **MDR-v25** baseline and are evaluated using the following feature sets defined in `dataset_metadata.py`:
1. **Model V1**: Trained using **OVERALL_SELECTED_FEATURES_V1** (40 features).
2. **Model V2**: Trained using **OVERALL_SELECTED_FEATURES_V2** (40 features, updated pipeline).
3. **Model V3**: Trained using **OVERALL_SELECTED_FEATURES_V3** (47 features, expanded pipeline).
4. **Model V4**: Trained using **OVERALL_SELECTED_FEATURES_V4** (50 features, no MI stage, expanded ElasticNet k).

For all feature sets, we evaluate:
- A non-weighted baseline using `objective="reg:absoluteerror"`.
- A temporally weighted baseline using `objective="reg:pseudohubererror"` with `beta = 0.2`.

---

## 1. Comparative Results Table

The performance metrics on the held-out test split are summarized below:

| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.6091 | 0.0658 | 0.0645 | −0.0133 | 0.0484 | 0.0358 | 0.7984 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | 0.6263 | 0.0644 | 0.0630 | −0.0131 | 0.0479 | 0.0369 | 0.8084 |
| **Model V2 (40 Features, No Weights)** | 0.6347 | 0.0636 | 0.0600 | −0.0211 | 0.0484 | 0.0375 | 0.8252 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | 0.6426 | 0.0629 | 0.0595 | −0.0205 | 0.0474 | 0.0359 | 0.8302 |
| **Model V3 (47 Features, No Weights)** | **0.6474** | **0.0625** | **0.0589** | −0.0211 | **0.0475** | **0.0364** | **0.8322** |
| **Model V3 (47 Features, Weighted, $\beta=0.2$)** | 0.6374 | 0.0634 | 0.0598 | −0.0212 | 0.0476 | 0.0361 | 0.8296 |
| **Model V4 (50 Features, No Weights)** | 0.6088 | 0.0659 | 0.0642 | −0.0145 | 0.0487 | 0.0376 | 0.7959 |
| **Model V4 (50 Features, Weighted, $\beta=0.2$)** | 0.6036 | 0.0663 | 0.0648 | −0.0138 | 0.0489 | 0.0372 | 0.7933 |

---

## 2. Key Insights and Discussion

### 1. Expanded Feature Set V3 (47 Features) Remains the Best Configuration
Model V3 (No Weights) achieves the overall highest $R^2$ of **0.6474** and the highest Pearson correlation of **0.8322**. 

### 2. Skipping the MI Stage (Model V4) Leads to Performance Degradation
Model V4, which was trained on the 50 features selected by skipping the Mutual Information (MI) stage, performed significantly worse than Model V3 ($R^2$ of **0.6088** vs. **0.6474**).
- **Collinearity Confusion**: Skipping the MI stage left all 400+ features in the candidate pool for ElasticNet and Stability Selection. Because of the massive collinearity (e.g. many overlapping rolling statistics), the Lasso/ElasticNet penalty shrunk the coefficients of important joint-predictive features to zero or distributed frequencies widely, leading to an unstable and suboptimal final feature subset.
- **Importance of Pruning**: This highlights that the Mutual Information (MI) stage is crucial. Pre-pruning the feature space to 300 features restricts the search space and resolves the collinearity issues, enabling the downstream ElasticNet selector to identify more robust features.

---

## 3. Visualizations

The generated scatter plots of residuals against true soil moisture are saved in this directory:
- `residuals_comparison.png` (displays residuals for V1, V2, V3, and V4 Weighted side-by-side)

---

## 4. Year-by-Year Comparative Results

The performance metrics on the held-out test split, broken down by test year (2023, 2024, and 2025):

### Year 2023
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.5999 | 0.0681 | 0.0608 | −0.0305 | 0.0502 | 0.0395 | 0.8274 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | 0.6189 | 0.0664 | 0.0594 | −0.0297 | 0.0500 | 0.0410 | 0.8359 |
| **Model V2 (40 Features, No Weights)** | 0.6337 | 0.0651 | 0.0579 | −0.0298 | 0.0509 | 0.0422 | 0.8444 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | **0.6382** | **0.0647** | 0.0583 | −0.0281 | **0.0504** | 0.0413 | 0.8431 |
| **Model V3 (47 Features, No Weights)** | 0.6292 | 0.0655 | **0.0569** | −0.0325 | 0.0509 | **0.0406** | **0.8503** |
| **Model V3 (47 Features, Weighted, $\beta=0.2$)** | 0.6242 | 0.0660 | 0.0588 | −0.0299 | 0.0514 | 0.0424 | 0.8411 |
| **Model V4 (50 Features, No Weights)** | 0.6040 | 0.0677 | 0.0624 | −0.0263 | 0.0499 | 0.0383 | 0.8186 |
| **Model V4 (50 Features, Weighted, $\beta=0.2$)** | 0.5938 | 0.0686 | 0.0634 | −0.0261 | 0.0504 | 0.0390 | 0.8126 |

### Year 2024
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.5747 | 0.0651 | 0.0646 | −0.0076 | 0.0473 | 0.0347 | 0.7908 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | 0.5983 | 0.0633 | 0.0629 | −0.0069 | 0.0470 | 0.0359 | 0.8037 |
| **Model V2 (40 Features, No Weights)** | 0.6263 | 0.0610 | 0.0580 | −0.0190 | 0.0457 | 0.0360 | 0.8251 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | 0.6281 | 0.0609 | 0.0575 | −0.0200 | 0.0454 | 0.0338 | 0.8296 |
| **Model V3 (47 Features, No Weights)** | **0.6443** | **0.0595** | **0.0571** | −0.0169 | **0.0445** | 0.0346 | 0.8308 |
| **Model V3 (47 Features, Weighted, $\beta=0.2$)** | 0.6283 | 0.0609 | 0.0575 | −0.0200 | **0.0445** | **0.0326** | **0.8310** |
| **Model V4 (50 Features, No Weights)** | 0.5771 | 0.0649 | 0.0636 | −0.0130 | 0.0472 | 0.0364 | 0.7802 |
| **Model V4 (50 Features, Weighted, $\beta=0.2$)** | 0.5763 | 0.0650 | 0.0639 | −0.0115 | 0.0474 | 0.0363 | 0.7809 |

### Year 2025
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.6292 | 0.0638 | 0.0637 | +0.0020 | 0.0472 | 0.0324 | 0.7947 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | 0.6381 | 0.0630 | 0.0630 | +0.0010 | 0.0463 | 0.0325 | 0.8008 |
| **Model V2 (40 Features, No Weights)** | 0.6160 | 0.0649 | 0.0637 | −0.0124 | 0.0484 | 0.0337 | 0.7960 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | 0.6363 | 0.0632 | 0.0622 | −0.0112 | 0.0459 | **0.0317** | 0.8094 |
| **Model V3 (47 Features, No Weights)** | **0.6473** | **0.0622** | **0.0611** | −0.0115 | 0.0467 | 0.0340 | **0.8131** |
| **Model V3 (47 Features, Weighted, $\beta=0.2$)** | 0.6366 | 0.0631 | 0.0621 | −0.0112 | **0.0463** | 0.0326 | 0.8107 |
| **Model V4 (50 Features, No Weights)** | 0.6198 | 0.0646 | 0.0646 | −0.0010 | 0.0490 | 0.0379 | 0.7876 |
| **Model V4 (50 Features, Weighted, $\beta=0.2$)** | 0.6162 | 0.0649 | 0.0649 | −0.0005 | 0.0488 | 0.0364 | 0.7866 |

---

## 5. Year-by-Year Insights and Discussion

- **Model V3 No Weights** consistently achieves the best generalization in both 2024 ($R^2$ of **0.6443**) and 2025 ($R^2$ of **0.6473**).
- **Model V4**'s performance remains mediocre across all years, reflecting that the feature set selection was suboptimal due to lack of pre-filtering (skipping the MI stage).

---

## 6. Year-by-Year Visualizations

- `residuals_by_year.png` (displays a 3x4 grid comparing residuals of Weighted Models V1, V2, V3, and V4 for 2023, 2024, and 2025)
- `r2_by_year.png` (displays a line chart of yearly R2 scores for all eight configurations)
- `metrics_by_year.csv` (contains the detailed metrics breakdown by year)
- `metrics_summary.csv` (contains the overall metrics comparison)
