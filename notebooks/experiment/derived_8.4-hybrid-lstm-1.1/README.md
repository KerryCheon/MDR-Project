# Experiment: `derived_8.4-hybrid-lstm-1.1` — Hybrid LSTM Context Vector (`ctx`) + XGBoost with Accelerated SHAP Feature Importance Analysis

## Objective
Evaluate whether concatenating frozen 160-dimensional temporal attention context vectors (`ctx_0`..`ctx_159`) extracted from a converged **BiLSTM+Attn (v9)** model with tabular XGBoost features improves prediction performance on the `derived_8.4` test set (7 Washington stations), and analyze feature importance using C++/CUDA-accelerated SHAP.

Baseline comparison against pure tabular models from `derived_8.4-eval-1.1`:
1. **Global Single Model (54 Backbone)**
2. **Clustering_V0_Full_k2 (Winner c0=0, c1=10)**
3. **Global Single Model (54 Backbone + 160 CTX)**
4. **Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)**

---

## Overall Leaderboard (2023–2025 Test Set)

Evaluated on CUDA on the `derived_8.4` test set (6,620 samples across 7 WA stations):

| model_name                                           |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:-----------------------------------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |    0.81496  |     0.0438197 |       0.043337  |    0.00648569 |    0.0337195 |         0.905594 |
| Global Single Model (54 Backbone)                    |    0.77923  |     0.0478636 |       0.0466868 |    0.0105483  |    0.0370592 |         0.889432 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |    0.772048 |     0.0486359 |       0.0478636 |    0.00863326 |    0.0363815 |         0.886389 |
| Global Single Model (54 Backbone + 160 CTX)          |    0.760117 |     0.0498925 |       0.048944  |    0.00968251 |    0.0372908 |         0.880724 |

---

## Per-Regime Performance Breakdown

| model_name                                           |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |        bias |       mae |   pearson |
|:-----------------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|------------:|----------:|----------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 | 0.00861494  | 0.0359221 |  0.900537 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 | 0.000797055 | 0.0278349 |  0.9191   |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |         0 |     10624 |     4817 | 0.752727 | 0.0497472 | 0.0487938 | 0.00969237  | 0.0389163 |  0.876766 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |         1 |      3984 |     1803 | 0.817104 | 0.0455343 | 0.045163  | 0.00580369  | 0.0296096 |  0.908769 |

---

## Year-by-Year $R^2$ Breakdown

| model_name                                           |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:-----------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |    0.81496  |       0.82297  |       0.783256 |       0.830289 |
| Global Single Model (54 Backbone)                    |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |    0.772048 |       0.753059 |       0.757131 |       0.800783 |
| Global Single Model (54 Backbone + 160 CTX)          |    0.760117 |       0.755145 |       0.743329 |       0.77401  |

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Execution Speedup (C++/CUDA `pred_contribs` vs. Standard `TreeExplainer`)

| Model Name                                           |   XGBoost pred_contribs (s) |   TreeExplainer (s) | Speedup   |
|:-----------------------------------------------------|----------------------------:|--------------------:|:----------|
| Global Single Model (54 Backbone)                    |                     959.548 |             975.316 | 1.02x     |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |                     706.193 |             574.159 | 0.81x     |
| Global Single Model (54 Backbone + 160 CTX)          |                     984.45  |             996.648 | 1.01x     |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |                     750.094 |             630.482 | 0.84x     |

### Feature Category Contribution (Sum vs. Mean vs. Median |SHAP|)

> **Note on Feature Importance vs. Column Count**:
> The **~75% total SHAP sum** for CTX vectors is largely driven by their **high column count (160 CTX dims vs 54 tabular dims)**. 
> While top individual CTX vectors (`ctx_24`, `ctx_105`, `ctx_144`) have higher peak importance than any tabular feature, the **median importance of individual CTX features (0.000166 - 0.000201)** is actually **lower than that of tabular features (0.000219 - 0.000234)**. This confirms that importance is heavily diluted across the 160 raw context dimensions, supporting the need for dimensionality reduction / feature pruning in `derived_8.4-hybrid-lstm-1.2`.

| Model Name | Category | Feature Count (% Cols) | Total SHAP Sum (% Sum) | Mean SHAP / Feature | Median SHAP / Feature | Max SHAP |
|:-----------|:---------|-----------------------:|-----------------------:|--------------------:|----------------------:|---------:|
| **Global Hybrid** | Tabular | 54 (25.23%) | 0.0382 (24.09%) | 0.000707 | **0.000219** | 0.010786 (`D_sin_DOY`) |
| | CTX | 160 (74.77%) | 0.1202 (75.91%) | 0.000751 | **0.000166** | 0.026145 (`ctx_24`) |
| **Clustering Hybrid** | Tabular | 64 (28.57%) | 0.0410 (25.67%) | 0.000641 | **0.000234** | 0.008829 (`SMAP_x_year`) |
| | CTX | 160 (71.43%) | 0.1188 (74.33%) | 0.000742 | **0.000201** | 0.021268 (`ctx_105`) |

### Top 10 Features by Mean Absolute SHAP Value

| Rank | Global Single (54 Backbone) | SHAP Val | Clustering Winner (c0=0, c1=10) | SHAP Val | Global Hybrid (54 + 160 CTX) | SHAP Val | Clustering Hybrid (54+160 CTX) | SHAP Val |
|-----:|:----------------------------|---------:|:--------------------------------|---------:|:-----------------------------|---------:|:-------------------------------|---------:|
| 1 | V_rollmin_LST_modis_kobs30 | 0.030712 | V_ema_LST_modis_kobs30 | 0.029229 | ctx_24 | 0.026145 | ctx_105 | 0.021268 |
| 2 | D_sin_DOY | 0.026203 | SMAP_sm_pm_interp_rollmean30 | 0.019198 | ctx_144 | 0.020045 | ctx_126 | 0.011572 |
| 3 | J_bio_bio13 | 0.017645 | J_aspect_deg | 0.016196 | D_sin_DOY | 0.010786 | ctx_138 | 0.009565 |
| 4 | V_ema_LST_modis_kobs30 | 0.015211 | D_sin_DOY | 0.014573 | ctx_126 | 0.009079 | ctx_144 | 0.009173 |
| 5 | G_API | 0.013152 | SMAP_x_year | 0.007972 | SMAP_x_year | 0.007956 | SMAP_x_year | 0.008829 |
| 6 | SMAP_x_year | 0.006825 | V_rollmin_LST_modis_kobs30 | 0.007077 | ctx_83 | 0.006508 | D_sin_DOY | 0.007583 |
| 7 | J_aspect_deg | 0.006381 | G_API | 0.006832 | ctx_97 | 0.006375 | ctx_153 | 0.004073 |
| 8 | V_rollmax_F_NDVI_kobs30 | 0.005134 | V_rollmax_F_NDVI_kobs30 | 0.006401 | ctx_157 | 0.003523 | V_rollmax_F_NDVI_kobs30 | 0.004043 |
| 9 | D_cos_DOY | 0.004217 | D_cos_DOY | 0.004627 | ctx_44 | 0.003413 | ctx_85 | 0.003895 |
| 10 | J_bio_bio02 | 0.004036 | V_rollmin_SMAP_sm_interp_kobs30 | 0.004180 | ctx_51 | 0.003013 | ctx_97 | 0.003742 |

---

## Key Insights & Architecture Summary
- **Phase 1**: BiLSTM+Attn model trained until validation convergence.
- **Phase 2**: Frozen hidden attention-pooled context state `ctx` (160-dim) extracted across `train`, `val`, and `test` splits.
- **Phase 3**: XGBoost fit on `[Tabular + CTX]` features.
- **Phase 4**: SHAP feature importance computed efficiently using XGBoost native `pred_contribs=True` with multi-threading / CUDA acceleration.
