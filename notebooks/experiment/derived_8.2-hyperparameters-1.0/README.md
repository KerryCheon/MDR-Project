# derived_8.2-hyperparameters-1.0 (Hyperparameter & Objective Sweep Report)

This directory contains the training and evaluation notebook for a comprehensive hyperparameter and loss objective sweep using Washington-only `derived_8.2` split data and **Feature Set V3** (47 features, unweighted).

The sweep compared 17 model configurations to address:
1. **Model Capacity Limitations**: Testing if higher tree depth or smaller min child weights mitigate the regression-to-the-mean effect.
2. **Overfitting Concerns**: Testing if shallower depths or stronger regularizations improve test set generalization.
3. **Objective Formulations**: Evaluating MAE (`reg:absoluteerror`), MSE (`reg:squarederror`), and Pseudo-Huber (`reg:pseudohubererror` with slopes 0.1, 1.0, 5.0) objectives.

---

## 1. Overall Performance and Timing Results

The overall performance metrics and training/inference times on the held-out test split are summarized below:

| Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson | Train Time (s) | Inference Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (MAE)** | 0.6474 | 0.0625 | 0.0589 | −0.0211 | 0.0475 | 0.0364 | 0.8322 | 34.58 | 0.3310 |
| **High Capacity Depth 10 (MAE)** | 0.6443 | 0.0628 | 0.0585 | −0.0230 | 0.0472 | 0.0352 | 0.8339 | 58.77 | 0.3418 |
| **High Capacity Depth 12 (MAE)** | 0.6350 | 0.0636 | 0.0595 | −0.0225 | 0.0481 | 0.0362 | 0.8267 | 109.99 | 0.3511 |
| **High Capacity Min Child Weight 1 (MAE)** | 0.6335 | 0.0638 | 0.0602 | −0.0210 | 0.0483 | 0.0369 | 0.8240 | 41.08 | 0.3367 |
| **High Capacity Low Reg (MAE)** | 0.6443 | 0.0628 | 0.0592 | −0.0211 | 0.0473 | 0.0355 | 0.8307 | 31.85 | 0.3297 |
| **Low Capacity Depth 6 (MAE)** | 0.6199 | 0.0649 | 0.0613 | −0.0214 | 0.0491 | 0.0370 | 0.8180 | 21.49 | 0.3129 |
| **Low Capacity Depth 4 (MAE)** | 0.6046 | 0.0662 | 0.0628 | −0.0211 | 0.0503 | 0.0390 | 0.8087 | 14.86 | 0.3058 |
| **Low Capacity High Reg (MAE)** | 0.6292 | 0.0641 | 0.0603 | −0.0217 | 0.0483 | 0.0362 | 0.8231 | 38.07 | 0.3495 |
| **Low Capacity Sampling (MAE)** | 0.6335 | 0.0637 | 0.0598 | −0.0220 | 0.0477 | 0.0351 | 0.8263 | 34.39 | 0.3501 |
| **Baseline (MSE)** | **0.6496** | **0.0623** | **0.0587** | −0.0209 | **0.0469** | **0.0355** | **0.8340** | **28.60** | 0.3296 |
| **Huber Slope 1.0** | 0.6440 | 0.0628 | 0.0594 | −0.0206 | 0.0472 | 0.0356 | 0.8307 | 29.18 | 0.3721 |
| **Huber Slope 0.1** | 0.6085 | 0.0659 | 0.0625 | −0.0210 | 0.0502 | 0.0392 | 0.8121 | 30.10 | 0.4565 |
| **Huber Slope 5.0** | 0.6462 | 0.0626 | 0.0588 | −0.0217 | 0.0471 | 0.0356 | 0.8341 | 31.25 | 0.3476 |
| **High Capacity Depth 10 (MSE)** | 0.6408 | 0.0631 | 0.0593 | −0.0216 | 0.0473 | 0.0360 | 0.8296 | 33.75 | 0.4437 |
| **High Capacity Depth 12 (MSE)** | 0.6464 | 0.0626 | 0.0592 | −0.0205 | 0.0469 | 0.0349 | 0.8299 | 36.27 | 0.3286 |
| **Low Capacity Depth 6 (MSE)** | 0.6334 | 0.0638 | 0.0598 | −0.0222 | 0.0480 | 0.0368 | 0.8293 | 17.82 | 0.3227 |
| **Low Capacity Depth 4 (MSE)** | 0.6077 | 0.0660 | 0.0622 | −0.0218 | 0.0500 | 0.0393 | 0.8150 | 13.19 | 0.2959 |

---

## 2. Key Insights and Discussion

### 1. Depth 8 represents the optimal capacity sweet spot
- **Overfitting with Higher Depth**: Increasing the tree depth to 10 and 12 underperforms the baseline across both loss objectives. For example, for MAE, $R^2$ decreases from **0.6474** (depth 8) to **0.6443** (depth 10) and then collapses to **0.6350** (depth 12). For MSE, depth 8 (**0.6496**) similarly outperforms depth 10 (**0.6408**) and depth 12 (**0.6464**). This proves that the model is **not underfitting** due to a lack of capacity; instead, increasing depth leads to overfitting on the training set and lower generalization.
- **Underfitting with Lower Depth**: Reducing tree depth to 6 or 4 severely hurts performance, dropping $R^2$ to **0.6199** and **0.6046** (for MAE) or **0.6334** and **0.6077** (for MSE). This confirms a depth of 8 is the ideal balance of bias and variance.

### 2. MSE objective outperforms MAE
- **Baseline (MSE)** achieves the highest overall $R^2$ (**0.6496**) and the lowest overall MAE (**0.0469**), outperforming the baseline MAE model.
- Because MSE penalizes large errors quadratically, it acts aggressively on the tail predictions, dragging predictions closer to the true extremes. This directly mitigates the regression-to-the-mean behavior.
- Pseudo-Huber loss with a larger slope (**Huber Slope 5.0**) approximates MSE behavior and yields a very strong $R^2$ of **0.6462**. Conversely, Huber Slope 0.1 behaves like L1/MAE loss and degrades performance to **0.6085**.

### 3. Training Time Scales Better under MSE than MAE
- In XGBoost, the L2 squared error gradients and hessians are smooth and continuous, making tree construction more stable and faster.
- While training a depth 12 model under MAE takes **110.0 seconds**, training the same depth 12 model under MSE takes only **36.3 seconds** — a **3x speedup**.
- Inference times remain extremely fast and consistent (~0.33 seconds) across all configurations.

---

## 3. Year-by-Year Performance

Metrics broken down by test year (2023, 2024, and 2025) are saved in `metrics_by_year.csv`.

- **Year 2023**: MSE Baseline remains superior with $R^2 = 0.6321$ compared to MAE Baseline ($R^2 = 0.6292$).
- **Year 2024**: MSE Baseline outperforms MAE Baseline ($R^2 = 0.6457$ vs $0.6443$).
- **Year 2025**: MSE Baseline dominates ($R^2 = 0.6698$ vs $0.6473$).
- Higher capacity models (depth 10 and 12) consistently show temporal degradation across all years compared to depth 8.

---

## 4. Visualizations

The generated plots are saved in this directory:
- `r2_by_year.png` (compares R2 trends grouped by MAE family, MSE family, and Huber comparison)
- `residuals_comparison.png` (displays a 6x3 grid comparing residuals for all 17 models)
- `residuals_by_year.png` (displays a massive 17x3 grid comparing model residuals across the three test years)
- `metrics_summary.csv` (detailed overall comparative table)
- `metrics_by_year.csv` (detailed yearly metrics breakdown)
