# Mixture of Experts (MoE) Evaluation Suite - derived_8.2-eval-3.0

This directory details the experimental setup and results for the Mixture of Experts (MoE) evaluation suite on the Washington-only `derived_8.2` soil moisture dataset. 

We evaluate 11 different models (including unweighted global baseline, learned gating MoE models, and unsupervised clustering-based specialist models) using the step-shrinked SOTA XGBoost hyperparameters from `derived_8.2-hyperparameters-1.3-lite`.

---

## 1. Summary of Overall Performance

The overall test performance metrics across the held-out test split for all 11 models are summarized below:

| Model ID | Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| **Model 1** | **Baseline V3 (Global Model)** | **0.6551** | **0.0618** | **0.0583** | **−0.0207** | **0.0464** | **0.0352** | **0.8373** |
| Model 9 | Clustering V3 Full K=3 (MoE) | 0.6336 | 0.0637 | 0.0602 | −0.0209 | 0.0477 | 0.0366 | 0.8234 |
| Model 10 | Clustering Dynamic K=3 (MoE) | 0.6284 | 0.0642 | 0.0613 | −0.0190 | 0.0477 | 0.0368 | 0.8164 |
| Model 7 | Clustering Dynamic K=2 (MoE) | 0.6273 | 0.0643 | 0.0608 | −0.0208 | 0.0475 | 0.0355 | 0.8205 |
| Model 8 | Seasonal Binary K=2 (MoE) | 0.5970 | 0.0669 | 0.0644 | −0.0179 | 0.0497 | 0.0384 | 0.7951 |
| Model 4 | Trained Gating K=2 (V3 Features) | 0.5778 | 0.0684 | 0.0640 | −0.0242 | 0.0498 | 0.0363 | 0.8100 |
| Model 2 | Trained Gating K=2 (Specialist Features) | 0.5712 | 0.0690 | 0.0669 | −0.0169 | 0.0487 | 0.0345 | 0.7916 |
| Model 5 | Trained Gating K=3 (V3 Features) | 0.5663 | 0.0693 | 0.0649 | −0.0243 | 0.0500 | 0.0356 | 0.8112 |
| Model 11 | Univariate G_API K=3 (MoE) | 0.5465 | 0.0709 | 0.0688 | −0.0171 | 0.0521 | 0.0381 | 0.7662 |
| Model 3 | Trained Gating K=3 (Specialist Features) | 0.5419 | 0.0713 | 0.0674 | −0.0231 | 0.0501 | 0.0345 | 0.8000 |
| Model 6 | Univariate G_API K=2 (MoE) | 0.5395 | 0.0715 | 0.0694 | −0.0172 | 0.0511 | 0.0364 | 0.7626 |

---

## 2. Year-by-Year Performance Breakdown

The performance breakdown ($R^2$ scores) on the held-out test split for each test year (2023, 2024, and 2025):

| Model Name | Overall $R^2$ | Year 2023 $R^2$ | Year 2024 $R^2$ | Year 2025 $R^2$ |
|---|---|---|---|---|
| **Model 1: Baseline V3** | **0.6551** | **0.6581** | **0.6403** | 0.6396 |
| Model 2: Trained Gating K=2 (Spec) | 0.5712 | 0.4918 | 0.5715 | 0.6460 |
| Model 3: Trained Gating K=3 (Spec) | 0.5419 | 0.4539 | 0.5477 | 0.6200 |
| Model 4: Trained Gating K=2 (V3) | 0.5778 | 0.4997 | 0.5662 | **0.6648** |
| Model 5: Trained Gating K=3 (V3) | 0.5663 | 0.4921 | 0.5648 | 0.6357 |
| Model 6: Univariate G_API K=2 | 0.5395 | 0.4814 | 0.5249 | 0.5991 |
| **Model 7: Clustering Dynamic K=2** | 0.6273 | 0.5996 | 0.6112 | 0.6539 |
| Model 8: Seasonal Binary K=2 | 0.5970 | 0.5782 | 0.5787 | 0.6109 |
| **Model 9: Clustering V3 Full K=3** | **0.6336** | 0.5986 | **0.6319** | 0.6543 |
| **Model 10: Clustering Dynamic K=3** | 0.6284 | 0.5948 | 0.6168 | **0.6581** |
| Model 11: Univariate G_API K=3 | 0.5465 | 0.5105 | 0.5402 | 0.5666 |

---

## 3. Gating Router Classification Performance

The trained gating routers (XGBoost Classifiers) are evaluated by comparing their predictions against the true target-sliced regimes on the held-out test split:

* **Ternary threshold**: Dry ($y < 0.16$), Transition ($0.16 \le y < 0.25$), Wet ($y \ge 0.25$)
* **Binary threshold**: Dry ($y < 0.16$), Wet ($y \ge 0.16$)

| Gating Router | K | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|---|
| Model 2 & 4 (Binary Gating) | 2 | 0.8730 | 0.8654 | 0.8636 | 0.8645 |
| Model 3 & 5 (Ternary Gating) | 3 | 0.7407 | 0.7307 | 0.7130 | 0.7013 |

---

## 4. Key Findings and Discussion

### 1. Global Baseline V3 is Still Superior Overall
Model 1 (the single global model using baseline features `OVERALL_SELECTED_FEATURES_V3`) remains the SOTA model overall with an $R^2$ of **0.6551**. However, it suffers from a slight regression in Year 2025 compared to the MoE models.

### 2. Failure of Learned Gating Routers (The Mean Fallback Trap)
Learned gating MoE models (Models 2, 3, 4, 5) degrade significantly in overall performance ($R^2$ drops from $0.6551$ to the $0.54$–$0.57$ range). 
* **The Gating Error Penalty**: Because specialists are trained on partitioned ranges of target soil moisture, a specialist has a narrow target distribution in its training set. If the learned gating classifier makes even a minor routing error at test time (e.g. routing a sample with true soil moisture of 0.3 to the "Dry" specialist), that specialist will predict a value near its training mean (clipping the prediction to $\approx 0.08$). This introduces massive prediction errors.
* **Feature Set Independence**: Even when all specialists use the full V3 feature set (Models 4 & 5), the performance is only slightly better ($0.5778$ and $0.5663$) than when using specialist features (Models 2 & 3). This confirms that the bottleneck of learned target gating is the routing classifier error rate rather than feature selection.

### 3. Success of Unsupervised Clustering-Based MoE
Unsupervised clustering gating (which is deterministic and does not rely on target-threshold slicing) achieves strong, robust performance:
* **Clustering V3 Full K=3** (Model 9) achieves the best MoE performance ($R^2 = 0.6336$).
* **Clustering Dynamic K=3** (Model 10) and **Clustering Dynamic K=2** (Model 7) also perform very well ($R^2 = 0.6284$ and $0.6273$, respectively).
* **Robustness to Year 2025**: Interestingly, both Model 9, Model 10, and Model 7 **outperform the baseline model in the year 2025** (e.g., Model 10 achieves $R^2 = 0.6581$ vs. Baseline's $0.6396$). This suggests that clustering on meteorological and physical attributes (rather than slicing on the target) creates specialists that generalize better to unseen temporal shifts/anomalies.

---

## 5. Visualizations and Saved Files

* **Diagnostic Plots**: Saved as `diagnostics_{model_name}.png` in the directory, representing the $2 \times 4$ grid (True vs Pred & True vs Residuals overall and yearly).
* **R2 Line Chart Comparison**: Saved as `r2_performance_over_years.png` showing the $R^2$ line charts of all 11 models over the test years.
* **Trained Weights**: Saved in `models/` directory for fast loading on re-runs.
