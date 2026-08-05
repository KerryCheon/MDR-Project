# Experiment: `derived_8.4-hybrid-lstm-2.0` — Strict Static/Temporal Split (Static → XGBoost, Temporal → LSTM)

## Objective
In `derived_8.4-hybrid-lstm-1.6`, the hybrid XGBoost received the 54-feature backbone
(49 of which are temporal/rolling) PLUS the LSTM context that was computed from largely
the same temporal features; SHAP showed the LSTM context dominating (56–86% of SHAP
share) and diluting the tabular set. This experiment implements a **strict split** to
remove that overlap:
- **XGBoost direct input = STATIC features only** (18, all constant
  per station: 5 from the 54-feature backbone + 13 station attributes formerly in the
  LSTM input).
- **LSTM input = TEMPORAL features only** (79: the 45 temporal
  features 1.6 fed the LSTM + the 34 temporal backbone features 1.6 did not feed it),
  retrained so the frozen context vector carries ALL temporal dynamics.

Everything else matches 1.6: V21 BiLSTM+Attn, `hidden_size ∈ {40, 20, 16, 8, 4}`
(1 seed each, seq_len=30), raw (non-PCA) `ctx`/`head_hidden`/`head_pre_relu`, Global +
Clustering strategies, same XGBoost hyperparameters. Cluster-1 temporal additions are
intentionally omitted (they would violate the static-only design). Leaderboard rows from
`derived_8.4-hybrid-lstm-1.6` are appended as `[1.6]` references.

Per hidden size H the representations are:
- `ctx` (2H-dim): attention-pooled bidirectional hidden state
- `head_hidden` (H-dim): after head `Linear(2H→H)→ReLU`
- `head_pre_relu` (H-dim): after head `Linear(2H→H)` BEFORE ReLU

---

## Feature Split: What Went Where

The strict split is a disjoint partition of the feature universe between the two models
(audited in the setup cell: 18 static + 79 temporal,
no overlap, and every static feature is constant per station).

### XGBoost — direct (tabular) input: 18 static features

The 5 static members of the 54-feature backbone (constant per station):

```text
J_aspect_deg, J_bio_bio02, J_bio_bio13, J_lc_code, J_soil_texture_usda_b0
```

The 13 station attributes moved out of the 1.6 LSTM input
(longitude, latitude, elevation, slope, aspect, soil texture, terrain transforms):

```text
longitude, latitude, elev, slope, aspect, J_clay_wfrac_b0, J_sand_wfrac_b0, K_slope_sin, K_slope_cos, K_aspect_cos, K_sand_clay_ratio_b0, K_clay_plus_sand_b0, K_aspect_sin
```

XGBoost receives **no temporal/rolling features directly** — all temporal signal arrives
through the frozen LSTM context vectors (`ctx` / `head_hidden` / `head_pre_relu`).

### LSTM — sequence input: 79 temporal/rolling features

The 45 temporal features carried over from 1.6's LSTM input:

```text
SMAP_sm_pm_interp_ema02, V_rollmin_LST_modis_kobs30, D_sin_DOY, G_rain_sum_3d, V_ema_G_API_kobs7, V_rollmin_G_API_kobs30, G_rain_sum_7d, C_lag_LST_modis_kobs30, C_lag_G_API_kobs1, V_ema_G_API_kobs14, V_rollmean_G_API_kobs14, G_API, G_DSLR, SMAP_ampm_diff_interp, V_rollmax_G_API_kobs30, V_ema_G_API_kobs30, V_rollmean_s2_b11_kobs7, V_ema_LST_modis_kobs7, V_rollmean_G_API_kobs7, C_lag_s2_b11_kobs30, A_d_E_SAR_diff_kobs14, C_lag_LST_modis_kobs6, A_d_LST_modis_kobs14, A_d_SMAP_sm_interp_kobs14, V_rollstd_SMAP_sm_interp_kobs30, SMAP_sm_interp_grad7, year_frac, sin_year, cos_year, API_x_year, SMAP_x_year, precip_mm, s1_vv, s1_vh, s2_b4, s2_b8, s2_b11, s2_b12, LST_modis, F_NDVI, F_NDMI, E_SAR_ratio, SMAP_sm_am_interp, SMAP_sm_pm_interp, SMAP_sm_interp_mask
```

The 34 temporal backbone features that 1.6 did NOT feed the LSTM
(added so the frozen context carries ALL temporal dynamics):

```text
D_cos_DOY, SMAP_sm_pm_interp_lag7, SMAP_sm_pm_interp_lag30, SMAP_sm_pm_interp_rollrange7, SMAP_sm_pm_interp_rollmean30, SMAP_sm_pm_interp_rollrange30, SMAP_sm_interp_rollrange7, V_rollrng_G_API_kobs7, V_rollmax_F_NDMI_kobs30, A_d_E_SAR_ratio_kobs30, V_rollmax_E_SAR_ratio_kobs7, V_rollmin_E_SAR_ratio_kobs30, V_rollmax_E_SAR_ratio_kobs30, V_ema_LST_modis_kobs30, V_rollmax_F_NDVI_kobs14, V_rollmax_F_NDVI_kobs30, V_ema_F_NDVI_kobs30, C_lag_F_NDVI_kobs30, A_grad_E_SAR_diff_kobs30, V_rollmax_E_SAR_diff_kobs14, V_rollrng_E_SAR_diff_kobs30, V_rollmax_E_SAR_diff_kobs30, A_grad_s2_b11_kobs30, V_rollrng_s2_b11_kobs30, V_rollmin_s2_b11_kobs30, V_rollmin_s2_b12_kobs30, A_d_SMAP_sm_interp_kobs30, V_rollmin_SMAP_sm_interp_kobs14, V_rollmin_SMAP_sm_interp_kobs30, E_rough_s1_vh_kobs14, D_z_F_NDMI, D_z_LST_modis, D_fft_dom_LST_modis_kobs30, D_fft_ent_LST_modis_kobs30
```

The LSTM receives **no static features** — station identity is supplied exclusively by
XGBoost's static inputs.


---

## Overall Leaderboard (2023–2025 Test Set)

| model_name                                                                    |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:------------------------------------------------------------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                        |   0.81496   |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| [1.6] Global Single (54 Backbone)                                             |   0.77923   |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 CTX [H20])         |   0.754996  |     0.0504222 |       0.0479503 |    0.0155939  |    0.0389788 |         0.884291 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Head Hidden [H20]) |   0.752932  |     0.0506342 |       0.0478914 |    0.0164388  |    0.0389331 |         0.882679 |
| [1.6] Global Single (54 Backbone + 40 CTX [H20])                              |   0.752359  |     0.0506929 |       0.0473342 |    0.0181452  |    0.0392618 |         0.885759 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Pre-ReLU [H20])    |   0.745201  |     0.0514203 |       0.0481712 |    0.0179885  |    0.0393917 |         0.881278 |
| [1.6] Global Single (54 Backbone + 20 Pre-ReLU [H20])                         |   0.742619  |     0.0516802 |       0.0478167 |    0.0196063  |    0.0403473 |         0.883001 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Pre-ReLU [H8])      |   0.742443  |     0.0516979 |       0.0479801 |    0.0192505  |    0.0405788 |         0.882238 |
| [1.6] Global Single (54 Backbone + 20 Head Hidden [H20])                      |   0.739913  |     0.0519511 |       0.0483196 |    0.0190823  |    0.0404485 |         0.880377 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Head Hidden [H8])   |   0.738107  |     0.0521312 |       0.0484172 |    0.0193246  |    0.0408708 |         0.879872 |
| [1.6] Global Single (54 Backbone + 8 Head Hidden [H8])                        |   0.733909  |     0.0525473 |       0.0476188 |    0.0222186  |    0.0417019 |         0.884759 |
| [1.6] Global Single (54 Backbone + 8 Pre-ReLU [H8])                           |   0.733817  |     0.0525565 |       0.0475882 |    0.0223057  |    0.0416023 |         0.885011 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 CTX [H8])          |   0.724513  |     0.0534671 |       0.0475959 |    0.0243589  |    0.0426123 |         0.885432 |
| [1.6] Global Single (54 Backbone + 16 Pre-ReLU [H16])                         |   0.722498  |     0.0536622 |       0.0485191 |    0.0229246  |    0.0416668 |         0.879314 |
| [1.6] Global Single (54 Backbone + 16 Head Hidden [H16])                      |   0.719788  |     0.0539236 |       0.0487581 |    0.0230306  |    0.0416721 |         0.878026 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Head Hidden [H16]) |   0.714247  |     0.0544542 |       0.050301  |    0.0208583  |    0.0415138 |         0.869637 |
| [1.6] Global Single (54 Backbone + 16 CTX [H8])                               |   0.714201  |     0.0544586 |       0.047933  |    0.0258488  |    0.043389  |         0.884963 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Pre-ReLU [H16])    |   0.712879  |     0.0545843 |       0.0505483 |    0.020599   |    0.0418048 |         0.868255 |
| [1.6] Global Single (54 Backbone + 32 CTX [H16])                              |   0.708794  |     0.0549713 |       0.0503754 |    0.0220036  |    0.0424845 |         0.869217 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 32 CTX [H16])         |   0.703916  |     0.0554298 |       0.0516102 |    0.0202199  |    0.0423911 |         0.862772 |
| [1.6] BiLSTM+Attn H16 (LSTM-only, V21)                                        |   0.703207  |     0.0554961 |       0.0540537 |    0.0125706  |    0.0433533 |       nan        |
| [1.6] BiLSTM+Attn H8 (LSTM-only, V21)                                         |   0.696313  |     0.056137  |       0.0553832 |    0.00916818 |    0.0455778 |       nan        |
| [1.6] BiLSTM+Attn H20 (LSTM-only, V21)                                        |   0.69309   |     0.0564341 |       0.0542288 |    0.0156219  |    0.0439822 |       nan        |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Head Hidden [H4])   |   0.691703  |     0.0565615 |       0.0497662 |    0.0268798  |    0.0430417 |         0.873326 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 CTX [H4])           |   0.684478  |     0.0572204 |       0.0498143 |    0.0281551  |    0.0440026 |         0.872963 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 CTX [H20])                  |   0.683728  |     0.0572884 |       0.0563586 |    0.0102791  |    0.0447836 |         0.835253 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Pre-ReLU [H4])      |   0.679918  |     0.0576324 |       0.0501511 |    0.0283965  |    0.0437607 |         0.871252 |
| [1.6] Global Single (54 Backbone + 8 CTX [H4])                                |   0.678224  |     0.0577847 |       0.0495127 |    0.029792   |    0.0450094 |         0.875225 |
| Global Single (18 Static + 40 CTX [H20])                                      |   0.675267  |     0.0580496 |       0.0571682 |    0.0100775  |    0.0450441 |         0.830063 |
| [1.6] Global Single (54 Backbone + 4 Pre-ReLU [H4])                           |   0.667166  |     0.0587692 |       0.0499439 |    0.0309746  |    0.045567  |         0.873094 |
| [1.6] Global Single (54 Backbone + 4 Head Hidden [H4])                        |   0.66573   |     0.0588959 |       0.050457  |    0.0303779  |    0.0456513 |         0.870037 |
| Global Single (18 Static + 20 Head Hidden [H20])                              |   0.653126  |     0.0599959 |       0.0583456 |    0.0139747  |    0.0469889 |         0.828608 |
| BiLSTM+Attn H20 (LSTM-only, temporal-only input)                              |   0.647652  |     0.0604675 |       0.0597615 |    0.00921305 |    0.0477251 |       nan        |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Head Hidden [H20])          |   0.645649  |     0.0606391 |       0.059536  |    0.0115136  |    0.047666  |         0.82297  |
| Global Single (18 Static + 20 Pre-ReLU [H20])                                 |   0.644306  |     0.0607539 |       0.0586762 |    0.0157525  |    0.0480703 |         0.823658 |
| Global Single (18 Static + 8 CTX [H4])                                        |   0.643888  |     0.0607896 |       0.0594023 |    0.0129128  |    0.0464751 |         0.820247 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Pre-ReLU [H40])    |   0.643591  |     0.0608149 |       0.0504216 |    0.0340018  |    0.0470259 |         0.868914 |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Head Hidden [H40]) |   0.642315  |     0.0609237 |       0.0506675 |    0.0338305  |    0.0470773 |         0.86753  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Pre-ReLU [H20])             |   0.641828  |     0.0609651 |       0.058969  |    0.0154728  |    0.0479493 |         0.821985 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 CTX [H4])                    |   0.64085   |     0.0610483 |       0.0599813 |    0.0113643  |    0.0465941 |         0.8167   |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 CTX [H40])         |   0.632925  |     0.0617182 |       0.0512324 |    0.0344149  |    0.0484144 |         0.864818 |
| [1.6] Global Single (54 Backbone + 40 Head Hidden [H40])                      |   0.628863  |     0.0620587 |       0.0494023 |    0.0375593  |    0.0493859 |         0.8756   |
| [1.6] Global Single (54 Backbone + 40 Pre-ReLU [H40])                         |   0.626165  |     0.0622839 |       0.0495051 |    0.037796   |    0.0494006 |         0.874962 |
| Global Single (18 Static + 4 Pre-ReLU [H4])                                   |   0.612016  |     0.0634516 |       0.0619865 |    0.0135564  |    0.0480967 |         0.8043   |
| [1.6] Global Single (54 Backbone + 80 CTX [H40])                              |   0.610982  |     0.0635362 |       0.0499429 |    0.0392753  |    0.0509103 |         0.873116 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Pre-ReLU [H4])               |   0.607391  |     0.0638287 |       0.0625713 |    0.0126067  |    0.0483598 |         0.801717 |
| Global Single (18 Static + 4 Head Hidden [H4])                                |   0.604655  |     0.0640507 |       0.061928  |    0.0163529  |    0.0489035 |         0.80317  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 32 CTX [H16])                  |   0.603175  |     0.0641705 |       0.0602104 |    0.0221937  |    0.0494602 |         0.81046  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Head Hidden [H4])            |   0.599106  |     0.0644986 |       0.0623177 |    0.0166305  |    0.0491697 |         0.801169 |
| Global Single (18 Static + 16 Pre-ReLU [H16])                                 |   0.596667  |     0.0646946 |       0.0616245 |    0.0196929  |    0.049998  |         0.803832 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 CTX [H8])                   |   0.595012  |     0.0648271 |       0.0582372 |    0.0284778  |    0.0499489 |         0.821113 |
| Global Single (18 Static + 8 Pre-ReLU [H8])                                   |   0.593869  |     0.0649185 |       0.0569818 |    0.0311045  |    0.0512895 |         0.829071 |
| Global Single (18 Static + 16 CTX [H8])                                       |   0.593     |     0.064988  |       0.0579291 |    0.029456   |    0.0504632 |         0.824156 |
| Global Single (18 Static + 8 Head Hidden [H8])                                |   0.591504  |     0.0651073 |       0.0589478 |    0.0276427  |    0.0511691 |         0.820083 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Pre-ReLU [H16])             |   0.59015   |     0.0652151 |       0.0623864 |    0.0189988  |    0.0507463 |         0.80055  |
| Global Single (18 Static + 32 CTX [H16])                                      |   0.589501  |     0.0652668 |       0.0607663 |    0.0238161  |    0.0499996 |         0.807403 |
| BiLSTM+Attn H4 (LSTM-only, temporal-only input)                               |   0.583751  |     0.0657222 |       0.0627283 |    0.0196105  |    0.0521541 |       nan        |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Pre-ReLU [H40])             |   0.580565  |     0.0659733 |       0.0555798 |    0.0355437  |    0.052034  |         0.839148 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 80 CTX [H40])                  |   0.578488  |     0.0661364 |       0.0543095 |    0.0377427  |    0.0513517 |         0.846075 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Head Hidden [H40])          |   0.576638  |     0.0662814 |       0.0552448 |    0.0366229  |    0.0530295 |         0.840419 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Pre-ReLU [H8])               |   0.576241  |     0.0663125 |       0.058618  |    0.0310045  |    0.0515886 |         0.817852 |
| Global Single (18 Static + 40 Pre-ReLU [H40])                                 |   0.568666  |     0.0669026 |       0.0559038 |    0.0367521  |    0.0527146 |         0.836701 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Head Hidden [H16])          |   0.567681  |     0.0669789 |       0.0626594 |    0.0236637  |    0.0516868 |         0.799193 |
| BiLSTM+Attn H8 (LSTM-only, temporal-only input)                               |   0.561618  |     0.0674469 |       0.0586474 |    0.0333103  |    0.0542442 |       nan        |
| Global Single (18 Static + 16 Head Hidden [H16])                              |   0.559643  |     0.0675987 |       0.0630676 |    0.0243323  |    0.0521465 |         0.795642 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Head Hidden [H8])            |   0.557435  |     0.0677679 |       0.0616524 |    0.0281331  |    0.0526825 |         0.801861 |
| Global Single (18 Static + 40 Head Hidden [H40])                              |   0.555932  |     0.067883  |       0.0566592 |    0.0373875  |    0.0541275 |         0.831285 |
| Global Single (18 Static + 80 CTX [H40])                                      |   0.555646  |     0.0679048 |       0.0546222 |    0.0403419  |    0.0539064 |         0.844873 |
| [1.6] BiLSTM+Attn H40 (LSTM-only, V21)                                        |   0.552283  |     0.0681612 |       0.0587193 |    0.0346122  |    0.0545359 |       nan        |
| BiLSTM+Attn H40 (LSTM-only, temporal-only input)                              |   0.533581  |     0.0695703 |       0.0578401 |    0.0386594  |    0.0557352 |       nan        |
| [1.6] BiLSTM+Attn H4 (LSTM-only, V21)                                         |   0.526266  |     0.0701137 |       0.0591161 |    0.0376992  |    0.0546092 |       nan        |
| BiLSTM+Attn H16 (LSTM-only, temporal-only input)                              |   0.52541   |     0.070177  |       0.063133  |    0.0306438  |    0.0550372 |       nan        |
| Global Single (18 Static)                                                     |   0.0240994 |     0.100633  |       0.100206  |    0.00925712 |    0.0829626 |         0.209137 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0)                                 |   0.0240393 |     0.100636  |       0.100209  |    0.00925651 |    0.0829646 |         0.209105 |

---

## Per-Regime Performance Breakdown

| model_name                                                           |   cluster |   n_train |   n_test |         r2 |      rmse |    ubrmse |       bias |       mae |   pearson |
|:---------------------------------------------------------------------|----------:|----------:|---------:|-----------:|----------:|----------:|-----------:|----------:|----------:|
| Global Single (18 Static)                                            |         0 |     14608 |     6620 |  0.0240994 | 0.100633  | 0.100206  | 0.00925712 | 0.0829626 |  0.209137 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0)                        |         0 |     10624 |     4817 | -0.0212638 | 0.101099  | 0.100597  | 0.0100681  | 0.0811736 |  0.139199 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0)                        |         1 |      3984 |     1803 |  0.128677  | 0.0993863 | 0.0991332 | 0.00708825 | 0.0877496 |  0.369085 |
| Global Single (18 Static + 80 CTX [H40])                             |         0 |     14608 |     6620 |  0.555646  | 0.0679048 | 0.0546222 | 0.0403419  | 0.0539064 |  0.844873 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 80 CTX [H40])         |         0 |     10624 |     4817 |  0.58047   | 0.0647979 | 0.0537918 | 0.0361277  | 0.0506231 |  0.844058 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 80 CTX [H40])         |         1 |      3984 |     1803 |  0.572857  | 0.0695863 | 0.0554386 | 0.0420572  | 0.0532985 |  0.856764 |
| Global Single (18 Static + 40 Head Hidden [H40])                     |         0 |     14608 |     6620 |  0.555932  | 0.067883  | 0.0566592 | 0.0373875  | 0.0541275 |  0.831285 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Head Hidden [H40]) |         0 |     10624 |     4817 |  0.572511  | 0.0654097 | 0.0552232 | 0.0350545  | 0.053157  |  0.835199 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Head Hidden [H40]) |         1 |      3984 |     1803 |  0.58541   | 0.0685562 | 0.0550839 | 0.0408131  | 0.052689  |  0.85721  |
| Global Single (18 Static + 40 Pre-ReLU [H40])                        |         0 |     14608 |     6620 |  0.568666  | 0.0669026 | 0.0559038 | 0.0367521  | 0.0527146 |  0.836701 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Pre-ReLU [H40])    |         0 |     10624 |     4817 |  0.579897  | 0.0648422 | 0.0556495 | 0.0332812  | 0.0517165 |  0.83487  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Pre-ReLU [H40])    |         1 |      3984 |     1803 |  0.58119   | 0.0689042 | 0.0549382 | 0.0415882  | 0.0528821 |  0.858576 |
| Global Single (18 Static + 40 CTX [H20])                             |         0 |     14608 |     6620 |  0.675267  | 0.0580496 | 0.0571682 | 0.0100775  | 0.0450441 |  0.830063 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 CTX [H20])         |         0 |     10624 |     4817 |  0.68785   | 0.0558934 | 0.0555961 | 0.00575734 | 0.043751  |  0.839814 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 CTX [H20])         |         1 |      3984 |     1803 |  0.673286  | 0.0608585 | 0.0566022 | 0.0223596  | 0.0475422 |  0.857259 |
| Global Single (18 Static + 20 Head Hidden [H20])                     |         0 |     14608 |     6620 |  0.653126  | 0.0599959 | 0.0583456 | 0.0139747  | 0.0469889 |  0.828608 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Head Hidden [H20]) |         0 |     10624 |     4817 |  0.654259  | 0.058824  | 0.0581964 | 0.00856989 | 0.0462041 |  0.836169 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Head Hidden [H20]) |         1 |      3984 |     1803 |  0.624535  | 0.0652412 | 0.0622968 | 0.0193783  | 0.0515716 |  0.811967 |
| Global Single (18 Static + 20 Pre-ReLU [H20])                        |         0 |     14608 |     6620 |  0.644306  | 0.0607539 | 0.0586762 | 0.0157525  | 0.0480703 |  0.823658 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Pre-ReLU [H20])    |         0 |     10624 |     4817 |  0.670915  | 0.0573896 | 0.0566474 | 0.00919983 | 0.0458594 |  0.838438 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Pre-ReLU [H20])    |         1 |      3984 |     1803 |  0.572408  | 0.0696228 | 0.0617125 | 0.0322321  | 0.0535328 |  0.819256 |
| Global Single (18 Static + 32 CTX [H16])                             |         0 |     14608 |     6620 |  0.589501  | 0.0652668 | 0.0607663 | 0.0238161  | 0.0499996 |  0.807403 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 32 CTX [H16])         |         0 |     10624 |     4817 |  0.632963  | 0.0606086 | 0.0583003 | 0.0165672  | 0.0470703 |  0.821055 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 32 CTX [H16])         |         1 |      3984 |     1803 |  0.532012  | 0.0728374 | 0.0626061 | 0.0372258  | 0.0558455 |  0.810923 |
| Global Single (18 Static + 16 Head Hidden [H16])                     |         0 |     14608 |     6620 |  0.559643  | 0.0675987 | 0.0630676 | 0.0243323  | 0.0521465 |  0.795642 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Head Hidden [H16]) |         0 |     10624 |     4817 |  0.603717  | 0.062977  | 0.0608371 | 0.0162773  | 0.0489022 |  0.811932 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Head Hidden [H16]) |         1 |      3984 |     1803 |  0.481701  | 0.0766526 | 0.0631844 | 0.0433976  | 0.0591263 |  0.805452 |
| Global Single (18 Static + 16 Pre-ReLU [H16])                        |         0 |     14608 |     6620 |  0.596667  | 0.0646946 | 0.0616245 | 0.0196929  | 0.049998  |  0.803832 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Pre-ReLU [H16])    |         0 |     10624 |     4817 |  0.610285  | 0.062453  | 0.0611259 | 0.0128064  | 0.0489058 |  0.81212  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Pre-ReLU [H16])    |         1 |      3984 |     1803 |  0.541729  | 0.0720773 | 0.0627044 | 0.0355428  | 0.0556634 |  0.811756 |
| Global Single (18 Static + 16 CTX [H8])                              |         0 |     14608 |     6620 |  0.593     | 0.064988  | 0.0579291 | 0.029456   | 0.0504632 |  0.824156 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 CTX [H8])          |         0 |     10624 |     4817 |  0.609255  | 0.0625354 | 0.0569392 | 0.0258574  | 0.0485762 |  0.826067 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 CTX [H8])          |         1 |      3984 |     1803 |  0.560498  | 0.0705858 | 0.0610216 | 0.0354785  | 0.0536161 |  0.82678  |
| Global Single (18 Static + 8 Head Hidden [H8])                       |         0 |     14608 |     6620 |  0.591504  | 0.0651073 | 0.0589478 | 0.0276427  | 0.0511691 |  0.820083 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Head Hidden [H8])   |         0 |     10624 |     4817 |  0.603418  | 0.0630008 | 0.0590149 | 0.0220532  | 0.0496143 |  0.819082 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Head Hidden [H8])   |         1 |      3984 |     1803 |  0.447973  | 0.0791074 | 0.0654881 | 0.0443767  | 0.0608799 |  0.790635 |
| Global Single (18 Static + 8 Pre-ReLU [H8])                          |         0 |     14608 |     6620 |  0.593869  | 0.0649185 | 0.0569818 | 0.0311045  | 0.0512895 |  0.829071 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Pre-ReLU [H8])      |         0 |     10624 |     4817 |  0.624229  | 0.0613255 | 0.0558296 | 0.0253745  | 0.0481538 |  0.830949 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Pre-ReLU [H8])      |         1 |      3984 |     1803 |  0.46209   | 0.0780893 | 0.063069  | 0.0460461  | 0.0607652 |  0.820577 |
| Global Single (18 Static + 8 CTX [H4])                               |         0 |     14608 |     6620 |  0.643888  | 0.0607896 | 0.0594023 | 0.0129128  | 0.0464751 |  0.820247 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 CTX [H4])           |         0 |     10624 |     4817 |  0.661881  | 0.058172  | 0.0579386 | 0.00520577 | 0.0444462 |  0.828603 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 CTX [H4])           |         1 |      3984 |     1803 |  0.590428  | 0.06814   | 0.0622031 | 0.027818   | 0.0523324 |  0.811599 |
| Global Single (18 Static + 4 Head Hidden [H4])                       |         0 |     14608 |     6620 |  0.604655  | 0.0640507 | 0.061928  | 0.0163529  | 0.0489035 |  0.80317  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Head Hidden [H4])   |         0 |     10624 |     4817 |  0.617826  | 0.0618458 | 0.0608936 | 0.0108105  | 0.0469548 |  0.809966 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Head Hidden [H4])   |         1 |      3984 |     1803 |  0.554042  | 0.0711023 | 0.0634035 | 0.0321797  | 0.0550874 |  0.803361 |
| Global Single (18 Static + 4 Pre-ReLU [H4])                          |         0 |     14608 |     6620 |  0.612016  | 0.0634516 | 0.0619865 | 0.0135564  | 0.0480967 |  0.8043   |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Pre-ReLU [H4])      |         0 |     10624 |     4817 |  0.628664  | 0.0609626 | 0.0606571 | 0.00609473 | 0.0463664 |  0.811935 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Pre-ReLU [H4])      |         1 |      3984 |     1803 |  0.556325  | 0.0709201 | 0.0642603 | 0.0300045  | 0.0536853 |  0.799011 |

---

## Year-by-Year $R^2$ Breakdown

| model_name                                                                    |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:------------------------------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                        |   0.81496   |    0.822971    |     0.783256   |      0.83029   |
| [1.6] Global Single (54 Backbone)                                             |   0.77923   |    0.750748    |     0.770077   |      0.813582  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 CTX [H20])         |   0.754996  |    0.723       |     0.783944   |      0.755301  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Head Hidden [H20]) |   0.752932  |    0.722345    |     0.785618   |      0.747873  |
| [1.6] Global Single (54 Backbone + 40 CTX [H20])                              |   0.752359  |    0.705015    |     0.804493   |      0.747682  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Pre-ReLU [H20])    |   0.745201  |    0.699411    |     0.790732   |      0.744938  |
| [1.6] Global Single (54 Backbone + 20 Pre-ReLU [H20])                         |   0.742619  |    0.691929    |     0.803262   |      0.733159  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Pre-ReLU [H8])      |   0.742443  |    0.751796    |     0.738287   |      0.726747  |
| [1.6] Global Single (54 Backbone + 20 Head Hidden [H20])                      |   0.739913  |    0.699826    |     0.797115   |      0.72143   |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Head Hidden [H8])   |   0.738107  |    0.750748    |     0.730679   |      0.72165   |
| [1.6] Global Single (54 Backbone + 8 Head Hidden [H8])                        |   0.733909  |    0.735918    |     0.734906   |      0.721394  |
| [1.6] Global Single (54 Backbone + 8 Pre-ReLU [H8])                           |   0.733817  |    0.729631    |     0.741729   |      0.721717  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 CTX [H8])          |   0.724513  |    0.708628    |     0.73265    |      0.725439  |
| [1.6] Global Single (54 Backbone + 16 Pre-ReLU [H16])                         |   0.722498  |    0.671147    |     0.762436   |      0.733371  |
| [1.6] Global Single (54 Backbone + 16 Head Hidden [H16])                      |   0.719788  |    0.663657    |     0.762067   |      0.733821  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Head Hidden [H16]) |   0.714247  |    0.668481    |     0.745048   |      0.727298  |
| [1.6] Global Single (54 Backbone + 16 CTX [H8])                               |   0.714201  |    0.702467    |     0.718691   |      0.713526  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Pre-ReLU [H16])    |   0.712879  |    0.673307    |     0.748864   |      0.713626  |
| [1.6] Global Single (54 Backbone + 32 CTX [H16])                              |   0.708794  |    0.643775    |     0.762972   |      0.721111  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 32 CTX [H16])         |   0.703916  |    0.664496    |     0.742016   |      0.702107  |
| [1.6] BiLSTM+Attn H16 (LSTM-only, V21)                                        |   0.703207  |  nan           |   nan          |    nan         |
| [1.6] BiLSTM+Attn H8 (LSTM-only, V21)                                         |   0.696313  |  nan           |   nan          |    nan         |
| [1.6] BiLSTM+Attn H20 (LSTM-only, V21)                                        |   0.69309   |  nan           |   nan          |    nan         |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Head Hidden [H4])   |   0.691703  |    0.721211    |     0.724064   |      0.615138  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 CTX [H4])           |   0.684478  |    0.718008    |     0.702525   |      0.617007  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 CTX [H20])                  |   0.683728  |    0.662239    |     0.710934   |      0.671082  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Pre-ReLU [H4])      |   0.679918  |    0.704727    |     0.711287   |      0.609367  |
| [1.6] Global Single (54 Backbone + 8 CTX [H4])                                |   0.678224  |    0.720122    |     0.677446   |      0.619253  |
| Global Single (18 Static + 40 CTX [H20])                                      |   0.675267  |    0.660171    |     0.707335   |      0.650155  |
| [1.6] Global Single (54 Backbone + 4 Pre-ReLU [H4])                           |   0.667166  |    0.713927    |     0.675228   |      0.593519  |
| [1.6] Global Single (54 Backbone + 4 Head Hidden [H4])                        |   0.66573   |    0.718179    |     0.668153   |      0.590955  |
| Global Single (18 Static + 20 Head Hidden [H20])                              |   0.653126  |    0.658166    |     0.646387   |      0.641872  |
| BiLSTM+Attn H20 (LSTM-only, temporal-only input)                              |   0.647652  |  nan           |   nan          |    nan         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Head Hidden [H20])          |   0.645649  |    0.647648    |     0.640891   |      0.635722  |
| Global Single (18 Static + 20 Pre-ReLU [H20])                                 |   0.644306  |    0.640146    |     0.662177   |      0.619334  |
| Global Single (18 Static + 8 CTX [H4])                                        |   0.643888  |    0.579134    |     0.69996    |      0.651796  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Pre-ReLU [H40])    |   0.643591  |    0.578502    |     0.687738   |      0.663549  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Head Hidden [H40]) |   0.642315  |    0.588043    |     0.688412   |      0.647773  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Pre-ReLU [H20])             |   0.641828  |    0.646199    |     0.657833   |      0.6087    |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 CTX [H4])                    |   0.64085   |    0.584018    |     0.690982   |      0.645277  |
| [1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 CTX [H40])         |   0.632925  |    0.586969    |     0.656359   |      0.65059   |
| [1.6] Global Single (54 Backbone + 40 Head Hidden [H40])                      |   0.628863  |    0.580307    |     0.670934   |      0.631166  |
| [1.6] Global Single (54 Backbone + 40 Pre-ReLU [H40])                         |   0.626165  |    0.573688    |     0.669796   |      0.631397  |
| Global Single (18 Static + 4 Pre-ReLU [H4])                                   |   0.612016  |    0.527509    |     0.671586   |      0.63831   |
| [1.6] Global Single (54 Backbone + 80 CTX [H40])                              |   0.610982  |    0.566764    |     0.627398   |      0.63274   |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Pre-ReLU [H4])               |   0.607391  |    0.537782    |     0.657182   |      0.625815  |
| Global Single (18 Static + 4 Head Hidden [H4])                                |   0.604655  |    0.513505    |     0.675726   |      0.627145  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 32 CTX [H16])                  |   0.603175  |    0.533247    |     0.680131   |      0.595238  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Head Hidden [H4])            |   0.599106  |    0.508       |     0.673083   |      0.61851   |
| Global Single (18 Static + 16 Pre-ReLU [H16])                                 |   0.596667  |    0.56271     |     0.660616   |      0.559512  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 CTX [H8])                   |   0.595012  |    0.554408    |     0.64022    |      0.58385   |
| Global Single (18 Static + 8 Pre-ReLU [H8])                                   |   0.593869  |    0.52927     |     0.644695   |      0.605002  |
| Global Single (18 Static + 16 CTX [H8])                                       |   0.593     |    0.538291    |     0.645166   |      0.59132   |
| Global Single (18 Static + 8 Head Hidden [H8])                                |   0.591504  |    0.548123    |     0.624683   |      0.595213  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Pre-ReLU [H16])             |   0.59015   |    0.55699     |     0.645172   |      0.560583  |
| Global Single (18 Static + 32 CTX [H16])                                      |   0.589501  |    0.514567    |     0.675564   |      0.577985  |
| BiLSTM+Attn H4 (LSTM-only, temporal-only input)                               |   0.583751  |  nan           |   nan          |    nan         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Pre-ReLU [H40])             |   0.580565  |    0.506315    |     0.5879     |      0.644992  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 80 CTX [H40])                  |   0.578488  |    0.494309    |     0.567281   |      0.672504  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Head Hidden [H40])          |   0.576638  |    0.485501    |     0.60664    |      0.638332  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Pre-ReLU [H8])               |   0.576241  |    0.5227      |     0.635195   |      0.565983  |
| Global Single (18 Static + 40 Pre-ReLU [H40])                                 |   0.568666  |    0.510915    |     0.566348   |      0.62299   |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Head Hidden [H16])          |   0.567681  |    0.518988    |     0.630547   |      0.547676  |
| BiLSTM+Attn H8 (LSTM-only, temporal-only input)                               |   0.561618  |  nan           |   nan          |    nan         |
| Global Single (18 Static + 16 Head Hidden [H16])                              |   0.559643  |    0.505542    |     0.625355   |      0.542847  |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Head Hidden [H8])            |   0.557435  |    0.520887    |     0.617771   |      0.525465  |
| Global Single (18 Static + 40 Head Hidden [H40])                              |   0.555932  |    0.490098    |     0.566095   |      0.606975  |
| Global Single (18 Static + 80 CTX [H40])                                      |   0.555646  |    0.485899    |     0.531526   |      0.644768  |
| [1.6] BiLSTM+Attn H40 (LSTM-only, V21)                                        |   0.552283  |  nan           |   nan          |    nan         |
| BiLSTM+Attn H40 (LSTM-only, temporal-only input)                              |   0.533581  |  nan           |   nan          |    nan         |
| [1.6] BiLSTM+Attn H4 (LSTM-only, V21)                                         |   0.526266  |  nan           |   nan          |    nan         |
| BiLSTM+Attn H16 (LSTM-only, temporal-only input)                              |   0.52541   |  nan           |   nan          |    nan         |
| Global Single (18 Static)                                                     |   0.0240994 |    2.96008e-05 |     0.00953046 |      0.0324991 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0)                                 |   0.0240393 |   -5.62556e-05 |     0.00949606 |      0.0324416 |

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Computation Time (C++/CUDA `pred_contribs`)

| Model Name                                                           |   pred_contribs (s) |
|:---------------------------------------------------------------------|--------------------:|
| Global Single (18 Static)                                            |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0)                        |                   0 |
| Global Single (18 Static + 80 CTX [H40])                             |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 80 CTX [H40])         |                   0 |
| Global Single (18 Static + 40 Head Hidden [H40])                     |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Head Hidden [H40]) |                   0 |
| Global Single (18 Static + 40 Pre-ReLU [H40])                        |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Pre-ReLU [H40])    |                   0 |
| Global Single (18 Static + 40 CTX [H20])                             |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 CTX [H20])         |                   0 |
| Global Single (18 Static + 20 Head Hidden [H20])                     |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Head Hidden [H20]) |                   0 |
| Global Single (18 Static + 20 Pre-ReLU [H20])                        |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Pre-ReLU [H20])    |                   0 |
| Global Single (18 Static + 32 CTX [H16])                             |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 32 CTX [H16])         |                   0 |
| Global Single (18 Static + 16 Head Hidden [H16])                     |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Head Hidden [H16]) |                   0 |
| Global Single (18 Static + 16 Pre-ReLU [H16])                        |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Pre-ReLU [H16])    |                   0 |
| Global Single (18 Static + 16 CTX [H8])                              |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 CTX [H8])          |                   0 |
| Global Single (18 Static + 8 Head Hidden [H8])                       |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Head Hidden [H8])   |                   0 |
| Global Single (18 Static + 8 Pre-ReLU [H8])                          |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Pre-ReLU [H8])      |                   0 |
| Global Single (18 Static + 8 CTX [H4])                               |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 CTX [H4])           |                   0 |
| Global Single (18 Static + 4 Head Hidden [H4])                       |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Head Hidden [H4])   |                   0 |
| Global Single (18 Static + 4 Pre-ReLU [H4])                          |                   0 |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Pre-ReLU [H4])      |                   0 |

### Feature Category Contribution (Tabular vs. LSTM Representations)

| Model Name                                                           |   Tabular SHAP Sum |   Tabular Mean abs(SHAP) |   Tabular Median abs(SHAP) |   Repr SHAP Sum |   Repr Mean abs(SHAP) |   Repr Median abs(SHAP) | Tabular % Share   | Repr % Share   |
|:---------------------------------------------------------------------|-------------------:|-------------------------:|---------------------------:|----------------:|----------------------:|------------------------:|:------------------|:---------------|
| Global Single (18 Static)                                            |             0.0378 |                   0.0021 |                     0.0001 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0)                        |             0.0295 |                   0.0016 |                     0      |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Global Single (18 Static + 80 CTX [H40])                             |             0.006  |                   0.0003 |                     0.0001 |          0.1239 |                0.0008 |                  0.0001 | 4.61%             | 95.39%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 80 CTX [H40])         |             0.0045 |                   0.0002 |                     0      |          0.128  |                0.0008 |                  0.0001 | 3.39%             | 96.61%         |
| Global Single (18 Static + 40 Head Hidden [H40])                     |             0.0115 |                   0.0006 |                     0.0003 |          0.1233 |                0.0008 |                  0      | 8.51%             | 91.49%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Head Hidden [H40]) |             0.0086 |                   0.0005 |                     0.0001 |          0.1272 |                0.0008 |                  0      | 6.32%             | 93.68%         |
| Global Single (18 Static + 40 Pre-ReLU [H40])                        |             0.0121 |                   0.0007 |                     0.0002 |          0.1275 |                0.0008 |                  0      | 8.69%             | 91.31%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 Pre-ReLU [H40])    |             0.0085 |                   0.0005 |                     0.0001 |          0.1269 |                0.0008 |                  0      | 6.25%             | 93.75%         |
| Global Single (18 Static + 40 CTX [H20])                             |             0.0107 |                   0.0006 |                     0.0002 |          0.1204 |                0.0008 |                  0      | 8.14%             | 91.86%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 40 CTX [H20])         |             0.0088 |                   0.0005 |                     0      |          0.1202 |                0.0008 |                  0      | 6.83%             | 93.17%         |
| Global Single (18 Static + 20 Head Hidden [H20])                     |             0.0142 |                   0.0008 |                     0.0003 |          0.1207 |                0.0008 |                  0      | 10.52%            | 89.48%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Head Hidden [H20]) |             0.0124 |                   0.0007 |                     0.0001 |          0.1256 |                0.0008 |                  0      | 9.00%             | 91.00%         |
| Global Single (18 Static + 20 Pre-ReLU [H20])                        |             0.0136 |                   0.0008 |                     0.0002 |          0.1177 |                0.0007 |                  0      | 10.37%            | 89.63%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 20 Pre-ReLU [H20])    |             0.0126 |                   0.0007 |                     0.0001 |          0.1216 |                0.0008 |                  0      | 9.38%             | 90.62%         |
| Global Single (18 Static + 32 CTX [H16])                             |             0.0068 |                   0.0004 |                     0.0002 |          0.1182 |                0.0007 |                  0      | 5.43%             | 94.57%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 32 CTX [H16])         |             0.0055 |                   0.0003 |                     0      |          0.1172 |                0.0007 |                  0      | 4.52%             | 95.48%         |
| Global Single (18 Static + 16 Head Hidden [H16])                     |             0.0097 |                   0.0005 |                     0.0003 |          0.1138 |                0.0007 |                  0      | 7.86%             | 92.14%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Head Hidden [H16]) |             0.0083 |                   0.0005 |                     0.0001 |          0.1189 |                0.0007 |                  0      | 6.52%             | 93.48%         |
| Global Single (18 Static + 16 Pre-ReLU [H16])                        |             0.0099 |                   0.0006 |                     0.0003 |          0.1114 |                0.0007 |                  0      | 8.17%             | 91.83%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 Pre-ReLU [H16])    |             0.0089 |                   0.0005 |                     0.0001 |          0.1186 |                0.0007 |                  0      | 7.00%             | 93.00%         |
| Global Single (18 Static + 16 CTX [H8])                              |             0.0117 |                   0.0007 |                     0.0002 |          0.1109 |                0.0007 |                  0      | 9.58%             | 90.42%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 16 CTX [H8])          |             0.0094 |                   0.0005 |                     0.0001 |          0.1101 |                0.0007 |                  0      | 7.85%             | 92.15%         |
| Global Single (18 Static + 8 Head Hidden [H8])                       |             0.0171 |                   0.0009 |                     0.0004 |          0.1028 |                0.0006 |                  0      | 14.23%            | 85.77%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Head Hidden [H8])   |             0.0145 |                   0.0008 |                     0.0003 |          0.1078 |                0.0007 |                  0      | 11.86%            | 88.14%         |
| Global Single (18 Static + 8 Pre-ReLU [H8])                          |             0.016  |                   0.0009 |                     0.0003 |          0.0967 |                0.0006 |                  0      | 14.21%            | 85.79%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 Pre-ReLU [H8])      |             0.0129 |                   0.0007 |                     0.0001 |          0.1034 |                0.0006 |                  0      | 11.06%            | 88.94%         |
| Global Single (18 Static + 8 CTX [H4])                               |             0.0132 |                   0.0007 |                     0.0004 |          0.1009 |                0.0006 |                  0      | 11.55%            | 88.45%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 8 CTX [H4])           |             0.0122 |                   0.0007 |                     0.0001 |          0.1019 |                0.0006 |                  0      | 10.66%            | 89.34%         |
| Global Single (18 Static + 4 Head Hidden [H4])                       |             0.0124 |                   0.0007 |                     0.0005 |          0.0919 |                0.0006 |                  0      | 11.85%            | 88.15%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Head Hidden [H4])   |             0.0108 |                   0.0006 |                     0.0003 |          0.094  |                0.0006 |                  0      | 10.27%            | 89.73%         |
| Global Single (18 Static + 4 Pre-ReLU [H4])                          |             0.0138 |                   0.0008 |                     0.0004 |          0.0905 |                0.0006 |                  0      | 13.21%            | 86.79%         |
| Clustering_V0_Full_k2 (18 Static, c0=0, c1=0 + 4 Pre-ReLU [H4])      |             0.0126 |                   0.0007 |                     0.0003 |          0.093  |                0.0006 |                  0      | 11.93%            | 88.07%         |

### Top 10 Features by Mean Absolute SHAP Value (averaged across all 32 XGBoost models)

| Feature      |   Mean abs(SHAP) |
|:-------------|-----------------:|
| hp_3         |           0.0056 |
| hh_3         |           0.0053 |
| J_aspect_deg |           0.0052 |
| hh_2         |           0.0052 |
| hp_2         |           0.0039 |
| hh_1         |           0.0033 |
| ctx_1        |           0.0032 |
| hp_1         |           0.0032 |
| ctx_0        |           0.003  |
| ctx_13       |           0.0026 |

---

## Why Performance Dropped vs 1.6 (Analysis)

The strict split removes the temporal overlap by design, but the leaderboard shows it
costs roughly 0.04–0.13 R² at every hidden size. The numbers indicate the "overlap"
removed was informative redundancy, not waste. Three compounding causes:

1. **The 49 temporal features were the model.** 1.6's tabular-only baselines scored
   **0.815** (Clustering) / **0.779** (Global) on the 54-feature backbone,
   while the 18-feature static-only baselines here collapse to **~0.024**. Static
   station attributes barely move daily soil-moisture dynamics, so in 2.0 every temporal
   signal must pass through the lossy LSTM context bottleneck instead of being read
   directly by XGBoost.
2. **The LSTM lost its station context.** The 13 static features moved out of the LSTM
   (slope, elev, lat/lon, aspect, soil texture, K_*) are what told the LSTM *which*
   station — i.e. its baseline moisture level. LSTM-only test R² fell from ~0.70 (1.6) to
   ~0.53–0.56 at H16/H8, and the 34 added rolling features (largely redundant derivatives
   of the same SMAP/NDVI/LST series) did not compensate. Validation RMSE barely moved, so
   this is an information-bottleneck issue, not a training failure.
3. **The hybrid is capped by its context.** The 18 static features add only ~+0.03 to +0.08
   over the LSTM-only model at each H, and the SHAP share of the context rose to
   **86–97%** (vs 57–87% in 1.6). The
   hybrids are effectively "LSTM context + a little station info", so their ceiling is what
   the context can encode.

### LSTM-only Regression (test R², same V21 architecture & hyperparameters)

|   H |   1.6 LSTM-only R² |   2.0 LSTM-only R² |   Δ LSTM-only |
|----:|-------------------:|-------------------:|--------------:|
|   4 |             0.5263 |             0.5838 |        0.0575 |
|   8 |             0.6963 |             0.5616 |       -0.1347 |
|  16 |             0.7032 |             0.5254 |       -0.1778 |
|  20 |             0.6931 |             0.6477 |       -0.0454 |
|  40 |             0.5523 |             0.5336 |       -0.0187 |

### Best Clustering + CTX Hybrid (test R²)

|   H |   1.6 best Clust+CTX R² |   2.0 best Clust+CTX R² |   Δ hybrid |
|----:|------------------------:|------------------------:|-----------:|
|   4 |                  0.6845 |                  0.6408 |    -0.0436 |
|   8 |                  0.7245 |                  0.595  |    -0.1295 |
|  16 |                  0.7039 |                  0.6032 |    -0.1007 |
|  20 |                  0.755  |                  0.6837 |    -0.0713 |
|  40 |                  0.6329 |                  0.5785 |    -0.0544 |


---

## Next Steps

The strict split proves the temporal features were load-bearing. To reduce overlap without
giving up performance:

1. **Restore the 13 station statics to the LSTM input** — keep the 18 static features in
   XGBoost *and* feed the station attributes back to the LSTM so it retains station
   identity. Cheapest test of whether the LSTM drop is caused by the removed statics.
2. **Allow a small set of raw temporal features back into XGBoost** — e.g. raw
   `SMAP_sm_pm_interp`, `precip_mm`, `F_NDVI` (not their rolling statistics), so XGBoost is
   not fully dependent on the context while the rolling/lagged derivatives stay in the LSTM.
3. **Use a wider / richer context** — `head_hidden` instead of `ctx`, concatenated
   representations, or a larger hidden size, to reduce the lossiness of the 2H-dim
   context bottleneck.
4. **Ablation** — retrain 1.6's LSTM on only its 45 temporal features (no statics, no 34
   additions) to isolate whether the LSTM drop comes from the removed statics or the added
   rolling features.


---

## Key Insights & Architecture Summary
- **Phase 0**: Feature-split audit — 18 static features verified
  constant per station and disjoint from 79 temporal LSTM features.
- **Phase 1**: V21 BiLSTM+Attn trained from scratch on `derived_8.4` for each hidden size
  `H ∈ {40, 20, 16, 8, 4}` (temporal-only input, seq_len=30, ReduceLROnPlateau, 1 seed).
- **Phase 2**: Three frozen raw representations extracted per hidden size:
  `ctx` (2H-dim), `head_hidden` (H-dim), `head_pre_relu` (H-dim).
- **Phase 3**: NO PCA — raw representations are used directly in the hybrid XGBoost models.
- **Phase 4**: XGBoost fit on `[Static + Repr]` for all 15 representation variants
  (5 hidden sizes × ctx/hh/hp) × 2 strategies (Global + Clustering) + 2 static baselines.
- **Phase 5**: SHAP feature importance via XGBoost native `pred_contribs=True` with CUDA acceleration.
- Reference rows from `derived_8.4-hybrid-lstm-1.6` (54-feature backbone hybrids) are marked `[1.6]` (37 rows).
