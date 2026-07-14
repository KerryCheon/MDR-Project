# derived_8.2-eval-2.0 (Model Training and Overfitting Evaluation)

This directory contains the training and evaluation notebook for diagnosing overfitting in the top two hyperparameter configurations identified in `derived_8.2-hyperparameters-1.1`. Both configurations were trained on the Washington-only `derived_8.2` dataset with **Feature Set V3** (47 features) and **without temporal recency weighting (unweighted)**.

To evaluate overfitting and optimization progression, we:
1. Tracked the training and validation (test) losses at each boosting round to plot loss curves.
2. Compared performance metrics (such as $R^2$, RMSE, ubRMSE) on both the training set (`trainval`) and the test split.
3. Plotted overall and year-by-year residual distributions to examine error structures.

---

## 1. Overall Performance and Overfitting Diagnosis

The training and testing metrics on the Washington-only `derived_8.2` split are summarized below:

| Configuration | Split | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| **Option A (Robust Workhorse)** | **Train** | 0.9990 | 0.0034 | 0.0034 | +0.0000 | 0.0021 | 0.0013 | 0.9995 |
| | **Test** | 0.6468 | 0.0626 | 0.0590 | −0.0210 | 0.0470 | 0.0356 | 0.8323 |
| **Option B (Finer MSE Regularizer)** | **Train** | 0.9991 | 0.0034 | 0.0034 | +0.0000 | 0.0021 | 0.0013 | 0.9995 |
| | **Test** | 0.6487 | 0.0624 | 0.0588 | −0.0210 | 0.0469 | 0.0361 | 0.8333 |

*(Note: Data saved in `metrics_summary.csv`.)*

### Key Insight: Severe Overfitting Discovered
Both models suffer from **extreme overfitting**:
- The **training $R^2$ is $\ge 0.9990$**, and training RMSE is **$0.0034$**.
- The **test $R^2$ is $\approx 0.647$–$0.649$**, and test RMSE is **$0.0624$–$0.0626$**.
- The test RMSE is **nearly 20 times larger** than the training RMSE. The models basically memorize the training dataset (nearly perfect fits with $R^2 = 0.999$), leaving a large generalization gap.
- Option B (Finer MSE Regularizer) has slightly higher test $R^2$ (**0.6487**) than Option A (**0.6468**), but both exhibit identical levels of training set memorization.

---

## 2. Training Loss Curves

The training vs. validation (test) loss curves are saved in `loss_curves.png`.

- **Option A (Robust Workhorse, 5500 trees, learning rate = 0.04)**: The training Pseudo-Huber loss decays exponentially and approaches 0 after 2000 rounds. However, the validation (test) Pseudo-Huber loss plateaus around **round 1000** and remains flat. This means **4,500 rounds (~82% of the training budget)** were wastefully trained without yielding any generalization benefit.
- **Option B (Finer MSE Regularizer, 11000 trees, learning rate = 0.02)**: The training RMSE decays exponentially towards 0. The validation (test) RMSE plateaus early and shows no further improvement beyond **round 2000**. This means **9,000 rounds (~82% of the training budget)** were wastefully trained without improvement.

### Implications of "Flatline Overfitting"
While validation performance does not degrade (there is no U-shaped deterioration in loss), this "flatline overfitting" has critical practical consequences:
1. **Severely Wasteful Compute**: Roughly 82% of the trees in both models are trained redundantly. Option B requires over 100 seconds to train on GPU, which could be cut down to ~20 seconds by capping the tree count.
2. **No Generalization Penalty**: Because validation error does not worsen, GBDT shrinkage and regularization effectively protect the model from generating high-variance predictions on unseen samples, even when trained long past convergence.

---

## 3. Year-by-Year Performance

The year-by-year metrics on the held-out test split (2023, 2024, and 2025) are detailed below:

### Year 2023
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| Option A (Robust Workhorse) | 0.6381 | 0.0647 | 0.0568 | −0.0311 | 0.0504 | 0.0407 | 0.8511 |
| Option B (Finer MSE Regularizer) | **0.6504** | **0.0636** | **0.0560** | −0.0301 | **0.0497** | 0.0409 | **0.8550** |

### Year 2024
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| Option A (Robust Workhorse) | 0.6291 | 0.0608 | 0.0575 | −0.0198 | 0.0447 | 0.0336 | 0.8277 |
| Option B (Finer MSE Regularizer) | **0.6352** | **0.0603** | **0.0566** | −0.0208 | **0.0442** | 0.0337 | **0.8321** |

### Year 2025
| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| Option A (Robust Workhorse) | **0.6503** | **0.0619** | **0.0612** | −0.0093 | **0.0454** | 0.0326 | **0.8139** |
| Option B (Finer MSE Regularizer) | 0.6331 | 0.0634 | 0.0627 | −0.0095 | 0.0466 | 0.0335 | 0.8043 |

*(Note: Data saved in `metrics_by_year.csv`.)*

- **2023 & 2024**: Option B (Finer MSE Regularizer) outperforms Option A by about 0.006–0.012 in $R^2$ score.
- **2025**: Option A (Robust Workhorse) performs better, yielding $R^2 = 0.6503$ compared to Option B's $0.6331$. This suggests that the Pseudo-Huber loss objective is indeed more robust to temporal anomalies / shifts in 2025.

---

## 4. Visualizations and Data Outputs

- `loss_curves.png`: Training vs. Validation (Test) loss curves for both configurations.
- `loss_curves.csv`: Step-by-step training and testing loss values at every boosting round.
- `residuals_comparison.png`: Overall residual scatter plots comparing Option A and Option B.
- `residuals_by_year.png`: Year-by-year (2023, 2024, 2025) residual scatter plots.
- `r2_by_year.png`: Bar chart comparing yearly $R^2$ scores.
