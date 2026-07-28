# Experiment: `derived_8.4-hybrid-lstm-1.0` — Hybrid LSTM Context Vector (`ctx`) + XGBoost

## Objective
Evaluate whether concatenating frozen 160-dimensional temporal attention context vectors (`ctx_0`..`ctx_159`) extracted from a converged **BiLSTM+Attn (v9)** model with tabular XGBoost features improves prediction performance on the `derived_8.4` test set (7 Washington stations).

Baseline comparison against pure tabular models from `derived_8.4-eval-1.1`:
1. **Global Single Model (54 Backbone)**
2. **Clustering_V0_Full_k2 (Winner c0=0, c1=10)**

---

## Overall Leaderboard (2023–2025 Test Set)

Evaluated on CUDA on the `derived_8.4` test set (6,620 samples across 7 WA stations):

| model_name                                           |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:-----------------------------------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| Global Single Model (54 Backbone)                    |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |    0.772048 |     0.048636  |       0.0478636 |    0.00863328 |    0.0363815 |         0.886389 |
| Global Single Model (54 Backbone + 160 CTX)          |    0.760117 |     0.0498925 |       0.048944  |    0.00968252 |    0.0372909 |         0.880724 |

---

## Per-Regime Performance Breakdown

| model_name                                           |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |        bias |       mae |   pearson |
|:-----------------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|------------:|----------:|----------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 | 0.00861491  | 0.0359221 |  0.900537 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 | 0.000797068 | 0.0278349 |  0.9191   |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |         0 |     10624 |     4817 | 0.752726 | 0.0497472 | 0.0487938 | 0.00969239  | 0.0389163 |  0.876766 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |         1 |      3984 |     1803 | 0.817104 | 0.0455344 | 0.045163  | 0.0058037   | 0.0296096 |  0.908769 |

---

## Year-by-Year $R^2$ Breakdown

| model_name                                           |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:-----------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| Global Single Model (54 Backbone)                    |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX) |    0.772048 |       0.753058 |       0.757131 |       0.800783 |
| Global Single Model (54 Backbone + 160 CTX)          |    0.760117 |       0.755145 |       0.743329 |       0.77401  |

---

## Key Insights & Architecture Summary
- **Phase 1**: BiLSTM+Attn model trained until validation convergence.
- **Phase 2**: Frozen hidden attention-pooled context state `ctx` (160-dim) extracted across `train`, `val`, and `test` splits.
- **Phase 3**: XGBoost fit on `[Tabular + CTX]` features.
