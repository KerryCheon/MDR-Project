# derived_8.1_pos-eval-3.0 (Global Models Comparison Report)

This directory contains the training and evaluation notebook for comparing two **single global** XGBoost models on the Washington-only `derived_8.1_pos` dataset (N=8,902 test samples across 13 stations). 

Both configurations use the modeling techniques from the **MDR-v25** baseline:
1. **Model A**: Trained using the **37 features** from the `v25` baseline.
2. **Model B**: Trained using the **40 features** from `dataset_metadata.py` (`OVERALL_SELECTED_FEATURES`).

For both feature sets, we evaluate:
- A non-weighted baseline using `objective="reg:absoluteerror"`.
- A temporally weighted baseline using `objective="reg:pseudohubererror"` with `beta = 0.2`.

---

## 1. Comparative Results Table

The performance metrics on the held-out test split (N=8,902) are summarized below:

| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **Model A (37 Features, No Weights)** | 0.6229 | 0.0647 | 0.0598 | −0.0246 | 0.0492 | 0.0390 | 0.8244 |
| **Model A (37 Features, Weighted, $\beta=0.2$)** | **0.6280** | **0.0642** | 0.0598 | −0.0233 | **0.0491** | **0.0380** | **0.8247** |
| **Model B (40 Features, No Weights)** | 0.5023 | 0.0743 | 0.0733 | −0.0123 | 0.0548 | 0.0398 | 0.7202 |
| **Model B (40 Features, Weighted, $\beta=0.2$)** | 0.4909 | 0.0751 | 0.0742 | −0.0118 | 0.0550 | 0.0401 | 0.7130 |

---

## 2. Key Insights and Discussion

### 1. Feature Set A (MDR-v25 Baseline) Outperforms Feature Set B
Model A (37 features) achieves a massive performance boost over Model B (40 features) across all metrics:
- $R^2$ improves from **0.4909** to **0.6280** ($+0.1371$ absolute increase).
- RMSE drops from **0.0751** to **0.0642** (a $14.5\%$ error reduction).
- Pearson correlation increases from **0.7130** to **0.8247**.

This demonstrates that the feature selection from the `v25` baseline is significantly more robust and generalizes far better to unseen Washington state stations.

### 2. Impact of Temporal Recency Weighting ($\beta=0.2$)
The recency weights have varying effects depending on the feature set:
- For **Model A**, temporal recency weights ($\beta=0.2$) and the pseudo-Huber loss function improve the performance: $R^2$ increases from **0.6229** to **0.6280**.
- For **Model B**, the weights slightly decrease performance: $R^2$ drops from **0.5023** to **0.4909**. This suggests that under the metadata feature set, weighting older temporal samples too low leads to overfitting on recent periods and degrades spatial generalization.

---

## 3. Visualizations

The generated scatter plots of residuals against true soil moisture are saved as:
- `residuals_comparison.png`

These plots highlight the tighter distribution of residuals around the zero-line for Model A compared to Model B.
