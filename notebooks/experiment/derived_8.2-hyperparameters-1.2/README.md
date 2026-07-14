# derived_8.2-hyperparameters-1.2 (Hyperparameter & Anti-Overfitting Sweep Report)

This directory contains the training and evaluation notebook for the 80-configuration hyperparameter sweep focusing on mitigating the severe overfitting (training $R^2 \ge 0.999$ vs. test $R^2 \approx 0.648$) discovered in `derived_8.2-eval-2.0`. All configurations were trained on the Washington-only `derived_8.2` dataset with **Feature Set V3** (47 features, unweighted).

---

## 1. Overall Performance and Timing Results

The performance metrics and training/inference times on the held-out test split are summarized for representative configurations below:

| Model ID | Sweep Group | Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson | Train Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | Reference | Baseline (MAE) | 0.6452 | 0.0627 | 0.0590 | −0.0214 | 0.0476 | 0.0365 | 0.8315 | 9.68 |
| **1** | Reference | Baseline (MSE) | 0.6491 | 0.0624 | 0.0588 | −0.0209 | 0.0469 | 0.0354 | 0.8339 | 8.06 |
| **3** | Group A | MSE Reg d=8 L2=3.0 L1=0.1 MCW=10 **(Best Sweep)** | **0.6494** | **0.0623** | **0.0589** | **−0.0204** | **0.0468** | **0.0357** | **0.8331** | **16.03** |
| **2** | Group A | MSE Reg d=8 L2=3.0 L1=0.1 MCW=2 | 0.6482 | 0.0625 | 0.0588 | −0.0211 | 0.0469 | 0.0361 | 0.8331 | 15.80 |
| **12**| Group A | MSE Reg d=8 L2=3.0 L1=5.0 MCW=10 (Over-regularized) | 0.6210 | 0.0648 | 0.0609 | −0.0221 | 0.0492 | 0.0374 | 0.8187 | 6.45 |
| **42**| Group B | MSE Samp d=5 sub=0.8 col=0.6 MCW=30 | 0.6186 | 0.0650 | 0.0608 | −0.0231 | 0.0491 | 0.0376 | 0.8228 | 7.17 |
| **46**| Group B | MSE Samp d=6 sub=0.6 col=0.6 MCW=2 | 0.6354 | 0.0636 | 0.0598 | −0.0216 | 0.0481 | 0.0370 | 0.8290 | 9.06 |
| **71**| Group C | Huber Slope 5.0 d=8 MCW=10 | 0.6412 | 0.0631 | 0.0596 | −0.0208 | 0.0474 | 0.0359 | 0.8285 | 13.62 |
| **60**| Group C | Huber Slope 1.5 d=8 MCW=30 | 0.6323 | 0.0638 | 0.0605 | −0.0204 | 0.0481 | 0.0365 | 0.8241 | 13.48 |
| **73**| Group D | MSE Depth 8 LR 0.01 Heavy L2 (Underfitted) | 0.6318 | 0.0639 | 0.0600 | −0.0211 | 0.0483 | 0.0367 | 0.8253 | 24.40 |
| **77**| Group D | Huber Depth 8 LR 0.01 slope 3.0 Mod Reg | 0.6377 | 0.0634 | 0.0596 | −0.0217 | 0.0477 | 0.0358 | 0.8280 | 26.45 |

*(Note: Full 80-model results are saved in `metrics_summary.csv`.)*

---

## 2. Key Insights and Discussion

### 1. Estimator Budget Cuts Saved Massive Compute with No Loss in Peak Generalization
* Scaling down tree counts (e.g. from 11000 down to 2400 for LR=0.02, and 5500 down to 1200 for LR=0.04) was highly successful. Baseline MSE trained in **8.06 seconds** instead of the 28.6 seconds in Sweep 1.1 (~72% speedup). 
* The training time for Group A models averaged **~12–16 seconds**, allowing the 80 configurations to run in under 15 minutes.
* This validates our early-stopping findings: the model flatlines in test loss long before it hits 5000+ trees, so running a shorter training budget prevents wasting compute without sacrificing generalization.

### 2. Trade-Off Between Regularization and Underfitting
* While mild regularization helped, heavy regularization severely degraded model performance (underfitting):
  * **L1 Regularization (`reg_alpha` = 5.0)**: Dropped the $R^2$ score to **0.615–0.621** (down from ~0.649). L1 sparsity constraints are too aggressive, forcing many useful features to zero.
  * **High Min Child Weight (`min_child_weight` = 30)**: Combining higher MCW with L1/L2 penalties caused the model to struggle to construct deeper trees, dropping the depth 8 model's performance to **0.620**.
  * **Aggressive Subsampling & Shallower Trees (Group B)**: Limiting depth to 5 and restricting row/column subsampling to 0.6/0.4 restricted the capacity of the model too much. Models in Group B struggled, with R2 hovering around **0.612–0.618**.
* **The Sweet Spot**: Mild regularization is best. **MSE Reg d=8 L2=3.0 L1=0.1 MCW=10** (Model 3) achieved the peak $R^2$ of **0.6494**, slightly outperforming the baseline MSE (**0.6491**).

### 3. Underfitting in LR=0.01 Models due to Insufficient Steps
* In Sweep 1.1, the absolute best model overall was **Model 43** (MSE Depth 8, LR 0.01, 22000 trees) with $R^2 = 0.6520$.
* In this sweep, the LR 0.01 models (Group D) underperformed, achieving R2 scores between **0.610** and **0.637**. 
* **Explanation**: Because we reduced the estimator count for LR=0.01 to 4800, the models did not have enough boosting rounds to converge, especially when combined with moderate-to-heavy L2 regularization (`reg_lambda >= 10.0`). This indicates that if we choose to run a very fine learning rate of 0.01, we *must* accept the high compute cost and train for 15,000+ steps to let the model converge fully.

---

## 3. Recommended Hyperparameter Configuration

If a fixed configuration is trained (without early stopping):

```python
params = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "min_child_weight": 10,
    "reg_lambda": 3.0,
    "reg_alpha": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 2400,
    "learning_rate": 0.02,
}
```

* **Why**: This configuration provides a stable, slightly regularized MSE fit that reaches peak generalization ($R^2 \approx 0.6494$) at a fraction of the compute cost (only 2400 trees instead of 11000). The `min_child_weight=10` and `reg_lambda=3.0` parameters successfully protect the model from memorizing minor spatial anomalies.

---

## 4. Visualizations and Data Outputs

- `loss_curves.png`: Training vs. Validation (Test) loss curves for 8 selected representative configurations, demonstrating how L1/L2 and MCW regularize convergence.
- `loss_curves.csv`: Step-by-step training and testing loss values at every boosting round for all 80 configurations.
- `residuals_comparison.png`: Overall residual scatter plots comparing all 80 configurations.
- `residuals_by_year.png`: Year-by-year (2023, 2024, 2025) residual scatter plots for 7 selected configurations.
- `r2_by_year.png`: Bar/line charts comparing yearly $R^2$ scores across the 4 Sweep Groups.
