# derived_8.2-eval-1.0 (Global Models Comparison Report)

This directory contains the training and evaluation notebook for comparing three **single global** XGBoost models on the Washington-only `derived_8.2` dataset.

Both configurations use the modeling techniques from the **MDR-v25** baseline and are evaluated using the following feature sets defined in `dataset_metadata.py`:
1. **Model V1**: Trained using **OVERALL_SELECTED_FEATURES_V1** (40 features).
2. **Model V2**: Trained using **OVERALL_SELECTED_FEATURES_V2** (40 features, updated pipeline).
3. **Model V3**: Trained using **OVERALL_SELECTED_FEATURES_V3** (47 features, expanded pipeline).

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
| **Model V3 (47 Features, No Weights)** | **0.6474** | **0.0625** | **0.0589** | −0.0211 | **0.0475** | 0.0364 | **0.8322** |
| **Model V3 (47 Features, Weighted, $\beta=0.2$)** | 0.6374 | 0.0634 | 0.0598 | −0.0212 | 0.0476 | 0.0361 | 0.8296 |

---

## 2. Key Insights and Discussion

### 1. Expanded Feature Set V3 (47 Features) Achieves Best Overall Performance
Model V3 (No Weights) achieves the overall highest $R^2$ of **0.6474** (a $+0.0048$ absolute improvement over the previous best Model V2 Weighted, and $+0.0127$ over Model V2 No Weights). It also records the lowest overall RMSE of **0.0625** and the highest Pearson correlation of **0.8322**. This confirms that increasing the target features count (to 47 features retained from a target of 50 in stability selection) successfully expands the model's predictive capacity.

### 2. Divergent Impact of Temporal Recency Weighting on Model V3
Unlike Models V1 and V2, where temporal recency weighting ($\beta=0.2$) consistently improved performance on the test split:
- For Model V3, applying temporal weights *decreased* overall performance ($R^2$ dropped from **0.6474** to **0.6374**, and Pearson correlation dropped from **0.8322** to **0.8296**).
- This indicates that with a larger, more expressive feature set (47 features), the model benefits more from utilizing the full volume of historical training data without downweighting older records. Applying recency weights restricts the effective dataset size and focus, leading to slightly worse overall test set generalization for the larger feature set.

---

## 3. Visualizations

The generated scatter plots of residuals against true soil moisture are saved in this directory:
- `residuals_comparison.png` (displays residuals for V1 Weighted, V2 Weighted, and V3 Weighted side-by-side)

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

### Year 2024
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.5747 | 0.0651 | 0.0646 | −0.0076 | 0.0473 | 0.0347 | 0.7908 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | 0.5983 | 0.0633 | 0.0629 | −0.0069 | 0.0470 | 0.0359 | 0.8037 |
| **Model V2 (40 Features, No Weights)** | 0.6263 | 0.0610 | 0.0580 | −0.0190 | 0.0457 | 0.0360 | 0.8251 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | 0.6281 | 0.0609 | 0.0575 | −0.0200 | 0.0454 | 0.0338 | 0.8296 |
| **Model V3 (47 Features, No Weights)** | **0.6443** | **0.0595** | **0.0571** | −0.0169 | **0.0445** | 0.0346 | 0.8308 |
| **Model V3 (47 Features, Weighted, $\beta=0.2$)** | 0.6283 | 0.0609 | 0.0575 | −0.0200 | **0.0445** | **0.0326** | **0.8310** |

### Year 2025
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.6292 | 0.0638 | 0.0637 | +0.0020 | 0.0472 | 0.0324 | 0.7947 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | 0.6381 | 0.0630 | 0.0630 | +0.0010 | 0.0463 | 0.0325 | 0.8008 |
| **Model V2 (40 Features, No Weights)** | 0.6160 | 0.0649 | 0.0637 | −0.0124 | 0.0484 | 0.0337 | 0.7960 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | 0.6363 | 0.0632 | 0.0622 | −0.0112 | 0.0459 | **0.0317** | 0.8094 |
| **Model V3 (47 Features, No Weights)** | **0.6473** | **0.0622** | **0.0611** | −0.0115 | 0.0467 | 0.0340 | **0.8131** |
| **Model V3 (47 Features, Weighted, $\beta=0.2$)** | 0.6366 | 0.0631 | 0.0621 | −0.0112 | **0.0463** | 0.0326 | 0.8107 |

---

## 5. Year-by-Year Insights and Discussion

### 1. Superior Year-by-Year Generalization of Model V3 No Weights
Model V3 (No Weights) is the best overall performing configuration in both 2024 and 2025:
- **2024**: $R^2$ reaches **0.6443** (compared to $0.6281$ for V2 Weighted).
- **2025**: $R^2$ reaches **0.6473** (compared to $0.6363$ for V2 Weighted).
- In **2023**, Model V2 (Weighted) still holds the highest $R^2$ ($0.6382$), but Model V3 (No Weights) records the highest Pearson correlation ($0.8503$) and lowest ubRMSE ($0.0569$).

This consistently high yearly performance shows that the expanded 47 features provide a more generalized model that performs robustly across different weather patterns represented in separate future years.

### 2. Recency Weighting Benefits Vary by Year and Feature Capacity
While recency weighting helped the smaller feature sets (V1 and V2) generalize better by biasing them to temporal shifts:
- For Model V3, the unweighted model was consistently superior to the weighted model in 2023 ($R^2$ $0.6292$ vs $0.6242$), 2024 ($0.6443$ vs $0.6283$), and 2025 ($0.6473$ vs $0.6366$).
- The data suggest that when a model has sufficient features to capture complex interactions, adding temporal weights actually limits the training variety and reduces overall year-by-year performance.

---

## 6. Year-by-Year Visualizations

- `residuals_by_year.png` (displays a 3x3 grid comparing residuals of Model V1 (Weighted), Model V2 (Weighted), and Model V3 (Weighted) for 2023, 2024, and 2025)
- `r2_by_year.png` (displays a line chart of yearly R2 scores for all six model configurations with horizontal dashed reference lines indicating overall test set R2 scores)
- `metrics_by_year.csv` (contains the detailed metrics breakdown by year)
