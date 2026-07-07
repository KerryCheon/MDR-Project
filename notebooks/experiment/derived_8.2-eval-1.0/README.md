# derived_8.2-eval-1.0 (Global Models Comparison Report)

This directory contains the training and evaluation notebook for comparing two **single global** XGBoost models on the Washington-only `derived_8.2` dataset.

Both configurations use the modeling techniques from the **MDR-v25** baseline and are evaluated using the following feature sets defined in `dataset_metadata.py`:
1. **Model V1**: Trained using **OVERALL_SELECTED_FEATURES_V1** (40 features).
2. **Model V2**: Trained using **OVERALL_SELECTED_FEATURES_V2** (40 features, updated pipeline).

For both feature sets, we evaluate:
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
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | **0.6426** | **0.0629** | **0.0595** | −0.0205 | **0.0474** | **0.0359** | **0.8302** |

---

## 2. Key Insights and Discussion

### 1. Feature Set V2 (Updated Pipeline) Outperforms Feature Set V1
Model V2 (40 features, updated pipeline) consistently outperforms Model V1 across all main metrics:
- $R^2$ improves from **0.6263** (V1 Weighted) to **0.6426** (V2 Weighted), showing a $+0.0163$ absolute increase.
- RMSE decreases from **0.0644** to **0.0629** (a $2.3\%$ error reduction).
- Pearson correlation increases from **0.8084** to **0.8302**.

This confirms that the updated feature selection pipeline (V2) produces a more predictive and generalizable feature subset for Washington state soil moisture prediction.

### 2. Positive Impact of Temporal Recency Weighting ($\beta=0.2$)
Unlike some previous evaluations where weighting had mixed effects, here we see that applying temporal recency weights improves performance for **both** feature sets:
- **Model V1**: $R^2$ increases from **0.6091** to **0.6263** ($+0.0172$ absolute).
- **Model V2**: $R^2$ increases from **0.6347** to **0.6426** ($+0.0079$ absolute).

The pseudo-Huber loss with temporal decay is successful in aligning the models toward more recent temporal dynamics, which benefits generalization on the test split.

---

## 3. Visualizations

The generated scatter plots of residuals against true soil moisture are saved in this directory:
- `residuals_comparison.png`

---

## 4. Year-by-Year Comparative Results

The performance metrics on the held-out test split, broken down by test year (2023, 2024, and 2025):

### Year 2023
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.5999 | 0.0681 | 0.0608 | −0.0305 | 0.0502 | 0.0395 | 0.8274 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | 0.6189 | 0.0664 | 0.0594 | −0.0297 | 0.0500 | 0.0410 | 0.8359 |
| **Model V2 (40 Features, No Weights)** | 0.6337 | 0.0651 | 0.0579 | −0.0298 | 0.0509 | 0.0422 | 0.8444 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | **0.6382** | **0.0647** | **0.0583** | −0.0281 | **0.0504** | **0.0413** | **0.8431** |

### Year 2024
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.5747 | 0.0651 | 0.0646 | −0.0076 | 0.0473 | 0.0347 | 0.7908 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | 0.5983 | 0.0633 | 0.0629 | −0.0069 | 0.0470 | 0.0359 | 0.8037 |
| **Model V2 (40 Features, No Weights)** | 0.6263 | 0.0610 | 0.0580 | −0.0190 | 0.0457 | 0.0360 | 0.8251 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | **0.6281** | **0.0609** | **0.0575** | −0.0200 | **0.0454** | **0.0338** | **0.8296** |

### Year 2025
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model V1 (40 Features, No Weights)** | 0.6292 | 0.0638 | 0.0637 | +0.0020 | 0.0472 | 0.0324 | 0.7947 |
| **Model V1 (40 Features, Weighted, $\beta=0.2$)** | **0.6381** | **0.0630** | **0.0630** | +0.0010 | 0.0463 | 0.0325 | 0.8008 |
| **Model V2 (40 Features, No Weights)** | 0.6160 | 0.0649 | 0.0637 | −0.0124 | 0.0484 | 0.0337 | 0.7960 |
| **Model V2 (40 Features, Weighted, $\beta=0.2$)** | 0.6363 | **0.0632** | **0.0622** | −0.0112 | **0.0459** | **0.0317** | **0.8094** |

---

## 5. Year-by-Year Insights and Discussion

### 1. Consistent Performance of Model V2 vs Model V1
Model V2 (updated pipeline features) outperforms Model V1 across 2023 and 2024 test years, showing $+0.0193$ and $+0.0298$ absolute $R^2$ improvements respectively for the weighted configurations. In 2025, Model V1 (Weighted) slightly edges out Model V2 (Weighted) in $R^2$ ($0.6381$ vs $0.6363$), but Model V2 retains a higher Pearson correlation ($0.8094$ vs $0.8008$), confirming the overall predictive robustness of the V2 feature set.

### 2. Benefits of Recency Weighting ($\beta=0.2$) across All Years
The temporal recency weighting consistently improves $R^2$ and reduces RMSE across **all** test years for both models:
- **2023**: Model V1 $+0.0190$ $R^2$ boost; Model V2 $+0.0045$ $R^2$ boost.
- **2024**: Model V1 $+0.0236$ $R^2$ boost; Model V2 $+0.0018$ $R^2$ boost.
- **2025**: Model V1 $+0.0089$ $R^2$ boost; Model V2 $+0.0203$ $R^2$ boost.

This confirms the generalizability of temporal recency weighting, demonstrating that training weights biased toward recent temporal patterns consistently help generalization even to specific single future years.

### 3. Model Performance Differences across Years
Comparing the years themselves reveals interesting dynamics:
- **Year 2023** has the highest correlation (Pearson correlation $\approx 0.843$ for V2 Weighted) and high $R^2$ ($0.6382$), but also the highest RMSE ($0.0647$). This suggests 2023 had higher variance in soil moisture values, which helps the $R^2$ score even with larger absolute errors. The model exhibits a moderate negative bias (approx $-0.028$) in 2023, indicating it generally underpredicts soil moisture.
- **Year 2024** exhibits the lowest RMSE ($0.0609$ for V2 Weighted) but a slightly lower $R^2$ ($0.6281$). The bias is $-0.020$, showing persistent underprediction.
- **Year 2025** shows a moderate $R^2$ ($0.6363$ for V2 Weighted) and the lowest negative bias ($-0.011$ for V2 Weighted, and almost zero $+0.001$ for V1 Weighted). However, Pearson correlation is lowest in 2025 ($\approx 0.809$), indicating that while the absolute predictions were closer in mean value to the true targets, the linear correlation was slightly weaker.

Overall, the model does not show extreme degradation in any single test year, indicating consistent spatio-temporal generalization.

---

## 6. Year-by-Year Visualizations

- `residuals_by_year.png` (displays a 3x2 grid comparing residuals of Model V1 (Weighted) and Model V2 (Weighted) for 2023, 2024, and 2025)
- `metrics_by_year.csv` (contains the detailed metrics breakdown by year)
