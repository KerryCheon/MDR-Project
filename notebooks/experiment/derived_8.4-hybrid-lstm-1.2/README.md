# Experiment: `derived_8.4-hybrid-lstm-1.2` — Hybrid LSTM Context Vector (`ctx` + `head_hidden`) + XGBoost with Accelerated SHAP

## Objective
Evaluate whether concatenating frozen temporal context vectors extracted from a converged **BiLSTM+Attn (v9)** model with tabular XGBoost features improves prediction performance on the `derived_8.4` test set (7 Washington stations). This experiment compares two CTX representations:
- **160-dim `ctx`**: Full attention-pooled hidden state (BiLSTM output)
- **80-dim `head_hidden`**: Intermediate representation after head `Linear(160→80)→ReLU`

Hypothesis: d=160 may be too large, diluting CTX feature importance. The 80-dim bottleneck may provide a more compressed, higher-signal representation.

Models evaluated:
1. **Global Single Model (54 Backbone)** — Pure tabular baseline
2. **Clustering_V0_Full_k2 (Winner c0=0, c1=10)** — Pure tabular MoE baseline
3. **Global Single Model (54 Backbone + 160 CTX)** — Hybrid global (214 features)
4. **Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)** — Hybrid MoE (214/224 features)
5. **Global Single Model (54 Backbone + 80 CTX-head)** — Hybrid global (134 features)
6. **Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head)** — Hybrid MoE (134/144 features)

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
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |    0.751557 |     0.0507749 |       0.049719  |    0.010301   |    0.0379416 |         0.877271 |

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

---

## Year-by-Year $R^2$ Breakdown

| model_name                                               |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:---------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| Global Single Model (54 Backbone)                        |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |    0.772048 |       0.753058 |       0.757131 |       0.800783 |
| Global Single Model (54 Backbone + 160 CTX)              |    0.760117 |       0.755145 |       0.743329 |       0.77401  |
| Global Single Model (54 Backbone + 80 CTX-head)          |    0.759103 |       0.759648 |       0.738143 |       0.770645 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |    0.751557 |       0.747124 |       0.735497 |       0.763817 |

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Computation Time (C++/CUDA `pred_contribs`)

| Model Name                                               |   pred_contribs (s) |
|:---------------------------------------------------------|--------------------:|
| Global Single Model (54 Backbone)                        |              1.2074 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |              1.0489 |
| Global Single Model (54 Backbone + 160 CTX)              |              1.2776 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |              1.142  |
| Global Single Model (54 Backbone + 80 CTX-head)          |              1.1367 |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |              1.0646 |

### Feature Category Contribution (Tabular vs. LSTM Context Vectors `ctx` / `head_hidden`)

| Model Name                                               |   Tabular SHAP Sum |   CTX SHAP Sum | Tabular % Share   | CTX % Share   |
|:---------------------------------------------------------|-------------------:|---------------:|:------------------|:--------------|
| Global Single Model (54 Backbone)                        |             0.1726 |         0      | 100.00%           | 0.00%         |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)               |             0.1713 |         0      | 100.00%           | 0.00%         |
| Global Single Model (54 Backbone + 160 CTX)              |             0.0382 |         0.1202 | 24.09%            | 75.91%        |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)     |             0.041  |         0.1188 | 25.67%            | 74.33%        |
| Global Single Model (54 Backbone + 80 CTX-head)          |             0.0522 |         0.1019 | 33.88%            | 66.12%        |
| Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head) |             0.0552 |         0.097  | 36.25%            | 63.75%        |

### Top 10 Features by Mean Absolute SHAP Value

```
Global Single Model (54 Backbone)_feature  Global Single Model (54 Backbone)_shap_val Clustering_V0_Full_k2 (Winner c0=0, c1=10)_feature  Clustering_V0_Full_k2 (Winner c0=0, c1=10)_shap_val Global Single Model (54 Backbone + 160 CTX)_feature  Global Single Model (54 Backbone + 160 CTX)_shap_val Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)_feature  Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)_shap_val Global Single Model (54 Backbone + 80 CTX-head)_feature  Global Single Model (54 Backbone + 80 CTX-head)_shap_val Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head)_feature  Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head)_shap_val
               V_rollmin_LST_modis_kobs30                                    0.030712                             V_ema_LST_modis_kobs30                                             0.029229                                              ctx_24                                              0.026145                                                      ctx_105                                                       0.021268                                                   hh_43                                                  0.030010                                                             hh_7                                                           0.022753
                                D_sin_DOY                                    0.026203                       SMAP_sm_pm_interp_rollmean30                                             0.019198                                             ctx_144                                              0.020045                                                      ctx_126                                                       0.011572                                                   hh_38                                                  0.014274                                                            hh_38                                                           0.017991
                              J_bio_bio13                                    0.017645                                       J_aspect_deg                                             0.016196                                           D_sin_DOY                                              0.010786                                                      ctx_138                                                       0.009565                                                   hh_23                                                  0.011929                                                            hh_43                                                           0.012253
                   V_ema_LST_modis_kobs30                                    0.015211                                          D_sin_DOY                                             0.014573                                             ctx_126                                              0.009079                                                      ctx_144                                                       0.009173                                               D_sin_DOY                                                  0.010531                                                      SMAP_x_year                                                           0.011416
                                    G_API                                    0.013152                                        SMAP_x_year                                             0.007972                                         SMAP_x_year                                              0.007956                                                  SMAP_x_year                                                       0.008829                                             SMAP_x_year                                                  0.010024                                                        D_sin_DOY                                                           0.009583
                              SMAP_x_year                                    0.006825                         V_rollmin_LST_modis_kobs30                                             0.007077                                              ctx_83                                              0.006508                                                    D_sin_DOY                                                       0.007583                                                   hh_52                                                  0.008689                                                            hh_19                                                           0.005990
                             J_aspect_deg                                    0.006381                                              G_API                                             0.006832                                              ctx_97                                              0.006375                                                      ctx_153                                                       0.004073                                                   hh_61                                                  0.004867                                                            hh_52                                                           0.003959
                  V_rollmax_F_NDVI_kobs30                                    0.005134                            V_rollmax_F_NDVI_kobs30                                             0.006401                                             ctx_157                                              0.003523                                      V_rollmax_F_NDVI_kobs30                                                       0.004043                                                   hh_77                                                  0.004551                                                            hh_23                                                           0.003317
                                D_cos_DOY                                    0.004217                                          D_cos_DOY                                             0.004627                                              ctx_44                                              0.003413                                                       ctx_85                                                       0.003895                                                    hh_4                                                  0.003730                                                            hh_77                                                           0.003028
                              J_bio_bio02                                    0.004036                    V_rollmin_SMAP_sm_interp_kobs30                                             0.004180                                              ctx_51                                              0.003013                                                       ctx_97                                                       0.003742                                                    hh_7                                                  0.002510                                                             hh_1                                                           0.002820
```

---

## Key Insights & Architecture Summary
- **Phase 1**: BiLSTM+Attn model trained until validation convergence.
- **Phase 2**: Frozen representations extracted across `train`, `val`, and `test` splits:
  - `ctx` (160-dim): Attention-pooled hidden state
  - `head_hidden` (80-dim): Intermediate after head `Linear(160→80)→ReLU`
- **Phase 3**: XGBoost fit on `[Tabular + CTX]` and `[Tabular + head_hidden]` features.
- **Phase 4**: SHAP feature importance computed efficiently using XGBoost native `pred_contribs=True` with multi-threading / CUDA acceleration.
