# Experiment: `derived_8.4-hybrid-lstm-1.3` — Hybrid LSTM Context Vectors + Pre-ReLU Head + XGBoost with Accelerated SHAP

## Objective
Evaluate whether concatenating frozen temporal context vectors extracted from a converged **BiLSTM+Attn (v9)** model with tabular XGBoost features improves prediction performance on the `derived_8.4` test set (7 Washington stations). This experiment adds a new representation:
- **80-dim pre-ReLU head**: Intermediate representation after head `Linear(160→80)` **BEFORE** ReLU — testing whether the ReLU bottleneck activation matters

Compared to v1.2 which had:
- **160-dim `ctx`**: Full attention-pooled hidden state
- **80-dim `head_hidden`**: Intermediate after head `Linear(160→80)→ReLU`

Models evaluated (9 rows):
1. **Global Single Model (54 Backbone)** — Pure tabular baseline
2. **Clustering_V0_Full_k2 (Winner c0=0, c1=10)** — Pure tabular MoE baseline
3. **Global Single Model (54 Backbone + 160 CTX)** — Hybrid global (214 features)
4. **Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)** — Hybrid MoE (214/224 features)
5. **Global Single Model (54 Backbone + 80 CTX-head)** — Hybrid global (134 features, post-ReLU)
6. **Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head)** — Hybrid MoE (134/144 features, post-ReLU)
7. **Global Single Model (54 Backbone + 80 pre-ReLU)** — Hybrid global (134 features, **pre-ReLU**)
8. **Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 pre-ReLU)** — Hybrid MoE (134/144 features, **pre-ReLU**)
9. **BiLSTM+Attn (LSTM-only)** — Pure LSTM baseline (no XGBoost)

LSTM weights reused from v1.2 (non-deterministic training skipped).

---

## Overall Leaderboard (2023–2025 Test Set)

Evaluated on CUDA on the `derived_8.4` test set (6,620 samples across 7 WA stations):

| model_name                                               |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:---------------------------------------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| Global Single Model (54 Backbone)                        |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |    0.772048 |     0.048636  |       0.0478636 |    0.00863328 |    0.0363815 |         0.886389 |
| Global Single Model (54 Backbone + 160 CTX)              |    0.760117 |     0.0498925 |       0.048944  |    0.00968252 |    0.0372909 |         0.880724 |
| Global Single Model (54 Backbone + 80 CTX-head)          |    0.759103 |     0.0499979 |       0.0490199 |    0.00984044 |    0.0373411 |         0.880411 |
| Global Single Model (54 Backbone + 80 pre-ReLU)          |    0.757562 |     0.0501575 |       0.0489749 |    0.0108278  |    0.0374757 |         0.880893 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |    0.751557 |     0.0507749 |       0.049719  |    0.010301   |    0.0379416 |         0.877271 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 pre-ReLU) |    0.750457 |     0.0508872 |       0.0495888 |    0.0114217  |    0.0381421 |         0.877616 |
| BiLSTM+Attn (LSTM-only)                                  |    0.618635 |     0.062908  |       0.0628947 |   -0.00129683 |    0.046418  |       nan        |

---

## Per-Regime Performance Breakdown

| model_name                                               |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |        bias |       mae |   pearson |
|:---------------------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|------------:|----------:|----------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 | 0.00861491  | 0.0359221 |  0.900537 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 | 0.000797068 | 0.0278349 |  0.9191   |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |         0 |     10624 |     4817 | 0.752726 | 0.0497472 | 0.0487938 | 0.00969239  | 0.0389163 |  0.876766 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |         1 |      3984 |     1803 | 0.817104 | 0.0455344 | 0.045163  | 0.0058037   | 0.0296096 |  0.908769 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |         0 |     10624 |     4817 | 0.729051 | 0.0520742 | 0.0508075 | 0.0114158   | 0.0406314 |  0.866454 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |         1 |      3984 |     1803 | 0.804077 | 0.0471281 | 0.0465558 | 0.00732255  | 0.0307554 |  0.902486 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 pre-ReLU) |         0 |     10624 |     4817 | 0.724888 | 0.0524728 | 0.0507246 | 0.0134317   | 0.0410932 |  0.866148 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 pre-ReLU) |         1 |      3984 |     1803 | 0.810198 | 0.046386  | 0.0459895 | 0.00605182  | 0.0302579 |  0.905217 |

---

## Year-by-Year $R^2$ Breakdown

| model_name                                               |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:---------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| Global Single Model (54 Backbone)                        |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |    0.772048 |       0.753058 |       0.757131 |       0.800783 |
| Global Single Model (54 Backbone + 160 CTX)              |    0.760117 |       0.755145 |       0.743329 |       0.77401  |
| Global Single Model (54 Backbone + 80 CTX-head)          |    0.759103 |       0.759648 |       0.738143 |       0.770645 |
| Global Single Model (54 Backbone + 80 pre-ReLU)          |    0.757562 |       0.758435 |       0.738293 |       0.767015 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |    0.751557 |       0.747124 |       0.735497 |       0.763817 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 pre-ReLU) |    0.750457 |       0.73814  |       0.734672 |       0.771553 |
| BiLSTM+Attn (LSTM-only)                                  |    0.618635 |     nan        |     nan        |     nan        |

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Computation Time (C++/CUDA `pred_contribs`)

| Model Name                                               |   pred_contribs (s) |
|:---------------------------------------------------------|--------------------:|
| Global Single Model (54 Backbone)                        |              1.1859 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |              1.042  |
| Global Single Model (54 Backbone + 160 CTX)              |              1.2634 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |              1.1317 |
| Global Single Model (54 Backbone + 80 CTX-head)          |              1.1354 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |              1.0518 |
| Global Single Model (54 Backbone + 80 pre-ReLU)          |              1.1948 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 pre-ReLU) |              1.0751 |

### Feature Category Contribution (Tabular vs. LSTM Context Vectors)

| Model Name                                               |   Tabular SHAP Sum |   Tabular Mean abs(SHAP) |   Tabular Median abs(SHAP) |   CTX SHAP Sum |   CTX Mean abs(SHAP) |   CTX Median abs(SHAP) | Tabular % Share   | CTX % Share   |
|:---------------------------------------------------------|-------------------:|-------------------------:|---------------------------:|---------------:|---------------------:|-----------------------:|:------------------|:--------------|
| Global Single Model (54 Backbone)                        |             0.1726 |                   0.0032 |                     0.0011 |         0      |               0      |                 0      | 100.00%           | 0.00%         |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |             0.1713 |                   0.0027 |                     0.0009 |         0      |               0      |                 0      | 100.00%           | 0.00%         |
| Global Single Model (54 Backbone + 160 CTX)              |             0.0382 |                   0.0007 |                     0.0002 |         0.1202 |               0.0008 |                 0.0002 | 24.09%            | 75.91%        |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |             0.041  |                   0.0006 |                     0.0002 |         0.1188 |               0.0007 |                 0.0002 | 25.67%            | 74.33%        |
| Global Single Model (54 Backbone + 80 CTX-head)          |             0.0522 |                   0.001  |                     0.0005 |         0.1019 |               0.0013 |                 0.0002 | 33.88%            | 66.12%        |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |             0.0552 |                   0.0009 |                     0.0004 |         0.097  |               0.0012 |                 0.0002 | 36.25%            | 63.75%        |
| Global Single Model (54 Backbone + 80 pre-ReLU)          |             0.0477 |                   0.0009 |                     0.0004 |         0.108  |               0.0014 |                 0.0003 | 30.61%            | 69.39%        |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 pre-ReLU) |             0.0497 |                   0.0008 |                     0.0003 |         0.1013 |               0.0013 |                 0.0004 | 32.93%            | 67.07%        |

### Top 10 Features by Mean Absolute SHAP Value (averaged across all 8 XGBoost models)

| Feature                      |   Mean abs(SHAP) |
|:-----------------------------|-----------------:|
| D_sin_DOY                    |           0.0124 |
| SMAP_x_year                  |           0.0093 |
| V_ema_LST_modis_kobs30       |           0.0059 |
| V_rollmin_LST_modis_kobs30   |           0.0053 |
| hh_43                        |           0.0053 |
| hp_43                        |           0.005  |
| hh_38                        |           0.004  |
| SMAP_sm_pm_interp_rollmean30 |           0.0038 |
| ctx_144                      |           0.0037 |
| ctx_24                       |           0.0036 |

---

## Key Insights & Architecture Summary
- **Phase 1**: BiLSTM+Attn model (reused from v1.2 — same `best_lstm_model.pt` checkpoint).
- **Phase 2**: Three frozen representations extracted:
  - `ctx` (160-dim): Attention-pooled hidden state (reused from v1.2)
  - `head_hidden` (80-dim): After head `Linear(160→80)→ReLU` (reused from v1.2)
  - `head_pre_relu` (80-dim): After head `Linear(160→80)` BEFORE ReLU (**new in v1.3**)
- **Phase 3**: XGBoost fit on `[Tabular + CTX]`, `[Tabular + head_hidden]`, and `[Tabular + pre-ReLU]` features.
- **Phase 4**: SHAP feature importance computed efficiently using XGBoost native `pred_contribs=True` with CUDA acceleration.
