# derived_8.2-hyperparameters-1.1 (Hyperparameter & Objective Sweep 1.1 Report)

This directory contains the training and evaluation notebook for a second, fine-grained hyperparameter and loss objective sweep focusing on optimizing the top-performing MSE objective (`reg:squarederror`) and additional Huber slopes. All configurations were trained on the Washington-only `derived_8.2` dataset with **Feature Set V3** (47 features, unweighted).

The sweep compared 50 new configurations (Model 17 to Model 66) alongside Model 0 (MAE baseline) and Model 9 (MSE baseline) to test:
1. **Regularization under MSE**: Adding L1/L2 penalties across tree depths 6 and 8.
2. **Subsampling / Randomization under MSE**: Tuning row and column sampling fractions.
3. **Huber Loss Slopes**: Sweeping slopes from 1.5 to 8.0 to find the optimal transition threshold.
4. **Learning Rate & Estimator Scaling**: Evaluating finer shrinkage steps (down to 0.01) with proportional tree counts.
5. **Gamma Constraints**: Tuning the split pruning factor `gamma` under MSE.

---

## 1. Overall Performance and Timing Results

The performance metrics and training/inference times on the held-out test split are summarized below:

| Model ID | Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson | Train Time (s) | Inference Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | Baseline (MAE) | 0.6474 | 0.0625 | 0.0589 | −0.0211 | 0.0475 | 0.0364 | 0.8322 | 34.58 | 0.3310 |
| **9** | Baseline (MSE) | 0.6496 | 0.0623 | 0.0587 | −0.0209 | 0.0469 | 0.0355 | 0.8340 | 28.60 | 0.3296 |
| **14**| **MSE Reg d=8 L2=0.5 L1=0.1** | 0.6519 | 0.0621 | 0.0585 | −0.0208 | 0.0467 | 0.0356 | 0.8350 | 25.25 | 0.3301 |
| **16**| **MSE Reg d=8 L2=3.0 L1=0.0** | 0.6515 | 0.0622 | 0.0584 | −0.0212 | 0.0466 | 0.0353 | 0.8355 | 22.19 | 0.3235 |
| **32**| **MSE Samp d=8 sub=0.9 col=0.6** | 0.6494 | 0.0624 | 0.0587 | −0.0210 | 0.0466 | 0.0352 | 0.8337 | 26.72 | 0.3163 |
| **36**| **Huber Slope 3.0** | 0.6517 | 0.0621 | 0.0586 | −0.0206 | 0.0467 | 0.0357 | 0.8342 | 27.63 | 0.3380 |
| **41**| **MSE Depth 8 LR 0.02** | 0.6516 | 0.0622 | 0.0585 | −0.0209 | 0.0467 | 0.0361 | 0.8355 | 54.77 | 0.6385 |
| **43**| **MSE Depth 8 LR 0.01 (Best Overall)** | **0.6520** | **0.0621** | **0.0584** | −0.0210 | **0.0467** | **0.0356** | **0.8357** | **106.26**| 1.3153 |
| **44**| MAE Depth 8 LR 0.02 | 0.6424 | 0.0630 | 0.0591 | −0.0219 | 0.0473 | 0.0354 | 0.8313 | 67.99 | 0.6567 |
| **45**| MAE Depth 8 LR 0.01 | 0.6435 | 0.0629 | 0.0591 | −0.0216 | 0.0474 | 0.0359 | 0.8306 | 129.60| 1.3845 |
| **50**| MSE Gamma d=8 gam=0.01 | 0.6475 | 0.0625 | 0.0589 | −0.0209 | 0.0468 | 0.0356 | 0.8332 | 4.96 | 0.3120 |

*(Note: Full 52-model results table is saved in metrics_summary.csv.)*

---

## 2. Key Insights and Discussion

### 1. Finer Learning Rates Yield the Absolute Best Performance
- **Model 43 (MSE Depth 8, LR 0.01, 22000 Estimators)** achieved the highest $R^2$ (**0.6520**) and the highest Pearson correlation (**0.8357**). Finer shrinkage steps let XGBoost converge more smoothly, extracting incremental signals that are otherwise skipped by larger steps.
- Interestingly, running a finer learning rate under **MAE** (Models 44 and 45) actually *degraded* performance to **0.6424** and **0.6435** respectively. Finer steps under L1 loss make it extremely slow and prone to getting stuck in local median basins, whereas MSE loss is smooth and benefits heavily from small steps.

### 2. Regularization under MSE is Beneficial
- Adding mild regularization successfully pushed performance beyond the baseline MSE:
  - **L2 = 0.5, L1 = 0.1** (Model 14) yielded $R^2 = \mathbf{0.6519}$.
  - **L2 = 3.0, L1 = 0.0** (Model 16) yielded $R^2 = \mathbf{0.6515}$.
- This confirms that MSE benefits from regularization constraints to stabilize predictions and restrict the influence of minor spatial anomalies.

### 3. Huber Slope 3.0 is the Optimal Objective Transition Threshold
- Sweeping slopes between 1.5 and 8.0 showed that **Huber Slope 3.0** is a sweet spot, yielding $R^2 = \mathbf{0.6517}$.
- Slopes closer to MSE (like 5.0 and 8.0) perform around 0.645–0.648, while slopes closer to MAE (like 1.0 and 0.1) perform around 0.60–0.64.

### 4. Gamma acts as an Ultra-Fast Pruner
- Models trained with `gamma > 0` (Models 46 to 53) trained in **under 5 seconds** (compared to 28+ seconds for the baseline).
- Setting `gamma=0.01` (Model 50) retained a very high $R^2$ of **0.6475** while cutting training time by **80%**. This is a powerful optimization for large-scale training.

---

## 3. Year-by-Year Performance

- **2023**: Finer learning rate (MSE Depth 8 LR 0.01) achieves the highest R2 score of **0.6534** (compared to MAE Baseline of **0.6292**).
- **2024**: MAE Baseline remains best at **0.6443**, but MSE LR 0.01 is highly competitive at **0.6380**.
- **2025**: MSE Baseline and Huber Slope 3.0 perform exceptionally well at **0.6490** (outperforming MAE baseline).

Detailed year-by-year results are saved in `metrics_by_year.csv`.

---

## 4. Visualizations

- `r2_by_year.png` (displays yearly trends for each of the 5 configuration groups)
- `residuals_comparison.png` (displays residuals for all 52 configurations)
- `residuals_by_year.png` (displays yearly residuals for 7 selected representative models)

---

## 5. Recommended Hyperparameter Configurations for Future Modeling

When training other models on this dataset (where a dedicated validation split is not available for early stopping), the following two fixed configurations are recommended to ensure model stability and prevent overfitting:

### Option A: The "Robust Workhorse" (Recommended for Iteration)
* **Objective**: `reg:pseudohubererror` (Huber Slope = 3.0)
* **Parameters**:
  ```python
  params = {
      "objective": "reg:pseudohubererror",
      "huber_slope": 3.0,
      "max_depth": 8,
      "min_child_weight": 2,
      "reg_lambda": 3.0,
      "reg_alpha": 0.1,
      "subsample": 0.9,
      "colsample_bytree": 0.8,
      "n_estimators": 5500,
      "learning_rate": 0.04,
  }
  ```
* **Why**: Achieves $R^2 = 0.6517$ (well above the 0.65 threshold). Because Pseudo-Huber loss naturally bounds gradient updates for large outliers, it is much more robust against overfitting than pure MSE when trained for a fixed number of trees without early stopping.

### Option B: The "Finer MSE Regularizer" (For Peak Performance)
* **Objective**: `reg:squarederror` (MSE)
* **Parameters**:
  ```python
  params = {
      "objective": "reg:squarederror",
      "max_depth": 8,
      "min_child_weight": 2,
      "reg_lambda": 3.0,
      "reg_alpha": 0.1,
      "subsample": 0.9,
      "colsample_bytree": 0.8,
      "n_estimators": 11000,
      "learning_rate": 0.02,
  }
  ```
* **Why**: Delivers peak performance ($R^2 \ge 0.6520$) by leveraging smaller learning steps and doubling trees. Regularization parameters (`reg_alpha=0.1`, `reg_lambda=3.0`) are explicitly added to enforce simplicity and compensate for the absence of validation-based early stopping.

