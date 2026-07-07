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
