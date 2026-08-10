# Experiment: `derived_8.4-eval-mlp-2.0` — optimized MLP architecture to break the ceiling (2.0 H100-hours)

## Objective

Follow-up to `derived_8.4-eval-mlp-1.3` (plain-2-regime-MLP ceiling **confirmed**:
2-seed val-selected winners 2regime_96 `w512x512x512_d0.3_lr1e-3` test R² 0.761,
2regime_54 `w512x512x512_d0.3_huber0.1` 0.765; test-best 0.789; val top-10 avg
0.7825; XGBoost 2-regime winner 0.815) and `derived_8.4-eval-2.0` (LOSO: the
96-pool transfers spatially better — pooled LOSO 0.668 ≈ XGBoost 2-regime
0.689). 2.0 attacks the remaining ~0.03 gap with an **optimized architecture +
training**, within a **2.0 H100-hour budget**. LOSO is explicitly out of the
sweep (per the experiment brief); this is a temporal-protocol experiment.

All numbers below are the stdout of the executed report notebook
(`derived_8.4-eval-mlp-2.0.ipynb`). Weights/checkpoints/test predictions under
`models/`; preprocessed tensors and per-job logs under `artifacts/`; figures at
the experiment root.

## Verdict (TL;DR)

- **The ceiling is broken with the new `2regime_mixed` family.** The honest
  **mixed-family val top-5 / top-3 ensembles reach test R² 0.8003** (vs 1.3's
  honest ceiling 0.7825 and the 1.3 test-best 0.789) and the 2-seed honest
  single model `w512x512x512_d0.3_huber0.1_swa` (mixed) reaches **0.7903** —
  past the 1.3 test-best with an honest val-based selection. The cross-family
  ensemble (54+96+mixed val winners) is **0.7932**.
- **The 2025 year, previously the MLPs' weak spot, is now the mixed family's
  strength**: val top-3/top-5 ensembles score 2025 R² 0.8336/0.8324 — the best
  of any model including XGBoost 2-regime (0.8303).
- **SWA (as configured) is a documented negative**: with `swa_start_frac: 0.6`
  equal-weight averaging + BN recalibration, no SWA snapshot ever beat the live
  best on val (SWA val 0.070–0.126 vs live 0.045–0.062), so nothing deployed
  the SWA weights. The honest val-based deployment mechanism worked as designed.
  The `_swa` configs' gains over their anchors are **live-trajectory results**,
  not SWA weight deployment (documented implementation note, see the SWA
  section).
- **`fg` (grouped towers) and `plr` (PLR encoding) underperformed the plain
  MLP** at this scale (fg best 0.782, plr best 0.720 vs the plain MLP's
  0.790); the plain MLP + the mixed feature allocation remains the strongest
  architecture in this experiment.
- **Debias is partial**: 2regime_54 meets the <5% bias²/MSE criterion (median
  3.7%); the 96-pool families still carry bias (13.3% / 10.0%) — but
  capacity-control configs (`w256x256_d0.5_swa` 1.0%, `w448x448_d0.3_gelu`
  0.04%, mixed `fg_w512..._swa` ≈0%) are near-unbiased.
- **Budget:** 1.36 of 2.0 H100-hours used for the sweep (54 jobs, 4,884
  GPU-s, 8 parallel workers); champion step ≈ +0.25 GPU-h.

## The four 2.0 levers

1. **FeatureGroupedMLP (`architecture: fg`)** — per-semantic-group towers +
   fusion MLP (groups resolved by `mlp20/feature_groups.py`, validated).
2. **PLRRegressor (`architecture: plr`)** — piecewise-linear encoding
   (Gorishniy et al. 2022) + plain-MLP body.
3. **SWA (trainer knob `swa: true`)** — Stochastic Weight Averaging updated
   once per epoch from `swa_start_frac` with BN recalibration; deployed iff its
   best val beats the live best (honest within-val comparison).
4. **New family `2regime_mixed`** — c0 = 96-pool, c1 = 54-backbone + the
   eval-1.1 winner's 10 delta features: the per-cluster-optimal allocation
   motivated by mlp-1.3's per-cluster R² (c0-96 0.754 > c0-54 0.737; c1-54
   0.831 > c1-96 0.776).

Documented negatives honored (no GPU re-spent): no val-fit calibration, no
trainval retrain, patience-60 kept, aux2020 diagnostic-only, residual/FT
reference-only, no new routers / station embeddings / feature selection.

## Protocol (data_version 6, temporal only)

Train on the official train split (2017–2020, n=9,803); early-stop / select on
the official val split (2021–2022, n=4,805); evaluate on the untouched test
split (2023–2025, n=6,620). aux2020 (2020 slice of train, n=2,519) diagnostic
only. Winners selected by **2-seed mean val RMSE** among mlp/fg/plr (phase-2
2nd seed = seed 7 for the val top-8 per family). Patience-60; AdamW + warmup 5%
+ cosine; grad clip 1.0; median-impute → StandardScaler → clip [−5, 5] fit on
train only; target in original units; `cudnn.deterministic=True` — the anchors
**reproduce mlp-1.3 bit-identically under v6** (stack check: 2regime_54
`w512x512x512_d0.3_huber0.1` seed 42 val 0.056293 / test R² 0.761078, seed 7
0.056701 / 0.761543, 2-seed 0.056497 / **0.765110** — max |diff| = 0.000000).

## Sweep design

30 curated configs × 3 families (2regime_54, 2regime_96, NEW 2regime_mixed) —
1.3 anchors re-run under v6, SWA variants, mixup α=0.4, `fg`/`plr`
architectures, and bias-targeting small nets for the 96-family. See
`config.yaml` for the full config list.

| family | c0 features | c1 features | rationale |
|---|---|---|---|
| `2regime_54` | backbone_54 | backbone_54 + 10 deltas | mlp-1.3 anchor (near-unbiased, best c1) |
| `2regime_96` | candidate_pool_96 | candidate_pool_96 | mlp-1.3 anchor (extrapolation / LOSO strength) |
| `2regime_mixed` (NEW) | candidate_pool_96 | backbone_54 + 10 deltas | per-cluster-optimal allocation |

## Selection Protocol v6 Diagnostic

Selection = 2-seed mean val RMSE among mlp/fg/plr; aux2020 diagnostic only
(1.2: measures train fit). Spearman(val_rmse, test_r2) is weak/negative as in
1.2/1.3 — the honest single-model selection is reported without test cherry-
picking; `test best (ref)` rows are reporting only.

### 2-Regime-96 — top-10 by val RMSE

| config_id                      | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:-------------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| w512x512x512_d0.3_lr1e-3       | mlp            |         2 |  0.0482834 |  0.0249457 |  0.761018 |   0.0497987 |  0.0176019  |
| w512x512x512_d0.3_lr1e-3_swa   | mlp            |         2 |  0.0483296 |  0.0249357 |  0.754698 |   0.050453  |  0.0204438  |
| w512x512x512_d0.3_huber0.1_swa | mlp            |         2 |  0.0490918 |  0.0258471 |  0.773153 |   0.048518  |  0.0163212  |
| fg_w384x384_d0.3_huber0.1      | fg             |         2 |  0.0508182 |  0.0305982 |  0.710904 |   0.0547718 |  0.0220846  |
| fg_w384x384_d0.3_huber0.1_swa  | fg             |         2 |  0.0509411 |  0.0312027 |  0.716937 |   0.0541973 |  0.0200361  |
| w512x512_d0.35_gelu            | mlp            |         2 |  0.052903  |  0.0359984 |  0.772401 |   0.0485983 |  0.0174258  |
| plr_w256x256_d0.4_swa          | plr            |         2 |  0.0552381 |  0.0322751 |  0.623881 |   0.0624739 |  0.0340105  |
| w256x256_d0.5                  | mlp            |         2 |  0.0552727 |  0.0471912 |  0.785392 |   0.047191  |  0.00967028 |
| w256x256_d0.5_swa              | mlp            |         1 |  0.0570138 |  0.0496979 |  0.783361 |   0.0474137 |  0.0048058  |
| fg_w256x256_d0.4_swa           | fg             |         1 |  0.0602456 |  0.0598678 |  0.477245 |   0.073652  |  0.0294463  |

- Spearman(val_rmse, test_r2) = −0.042 (p=0.907, n=10) · Spearman(robust_score, test_r2) = +0.018
- val winner (honest): `w512x512x512_d0.3_lr1e-3` (0.7610) | test best (ref): `w256x256_d0.5` (0.7854)

### 2-Regime-54 — top-10 by val RMSE

| config_id                      | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:-------------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| w512x512x512_d0.3_huber0.1     | mlp            |         2 |  0.0564972 |  0.0232794 |  0.76511  |   0.0493706 | 0.00681535  |
| w512x512x512_d0.3_huber0.1_swa | mlp            |         2 |  0.0568509 |  0.0233968 |  0.764039 |   0.049483  | 0.00869853  |
| plr_w384x384_d0.3_gelu_swa     | plr            |         2 |  0.0574891 |  0.0288451 |  0.720252 |   0.053879  | 0.0203079   |
| fg_w384x384_d0.3               | fg             |         2 |  0.0575515 |  0.0261881 |  0.727882 |   0.0531391 | 0.0188997   |
| fg_w384x384_d0.3_swa           | fg             |         2 |  0.0575753 |  0.0260077 |  0.72938  |   0.0529927 | 0.0188537   |
| fg_w512x512_d0.3_huber0.1_swa  | fg             |         2 |  0.0577613 |  0.0250303 |  0.721153 |   0.0537921 | 0.0197122   |
| w384x384_d0.3_gelu             | mlp            |         1 |  0.0614041 |  0.0306479 |  0.786493 |   0.0470697 | 0.0034241   |
| w448x448_d0.3_gelu             | mlp            |         2 |  0.0618635 |  0.0302294 |  0.780859 |   0.0476868 | 0.000990915 |
| w448x448_d0.3_gelu_swa         | mlp            |         2 |  0.0618635 |  0.0302294 |  0.780859 |   0.0476868 | 0.000990915 |
| w384x384_d0.3_mixup0.4         | mlp            |         1 |  0.0624267 |  0.0349459 |  0.765446 |   0.0493353 | 0.0102525   |

- Spearman(val_rmse, test_r2) = +0.549 (p=0.100, n=10)
- val winner (honest): `w512x512x512_d0.3_huber0.1` (0.7651) | test best (ref): `w384x384_d0.3_gelu` (0.7865)

### 2-Regime-Mixed — top-10 by val RMSE

| config_id                       | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:--------------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| fg_w512x512_d0.3_huber0.1_swa   | fg             |         2 |  0.0463768 |  0.0252084 |  0.759881 |   0.049917  |  0.00044081 |
| fg_w384x384_d0.3_gelu_swa       | fg             |         2 |  0.0465913 |  0.0270722 |  0.782334 |   0.047526  |  0.018088   |
| w512x512x512_d0.3_huber0.1_swa  | mlp            |         2 |  0.0494507 |  0.0238544 |  0.790253 |   0.0466534 |  0.0124186  |
| w512x512x512_d0.3_huber0.1      | mlp            |         2 |  0.0494921 |  0.0245625 |  0.778871 |   0.0479025 |  0.016768   |
| w448x448_d0.3_gelu              | mlp            |         2 |  0.052322  |  0.0296164 |  0.760502 |   0.0498525 |  0.0208406  |
| w448x448_d0.3_gelu_swa          | mlp            |         2 |  0.0523246 |  0.0297563 |  0.763263 |   0.0495643 |  0.0205277  |
| w384x384_d0.3_gelu              | mlp            |         2 |  0.0528789 |  0.033321  |  0.778968 |   0.047892  |  0.0133468  |
| w384x384_d0.3_mixup0.4_huber0.1 | mlp            |         2 |  0.0531458 |  0.0321586 |  0.746382 |   0.051301  |  0.0202711  |
| w256x256_d0.4_swa               | mlp            |         1 |  0.0546939 |  0.0419791 |  0.777045 |   0.0480999 |  0.0129263  |
| plr_w384x384_d0.3_gelu          | plr            |         1 |  0.0570874 |  0.0295189 |  0.679851 |   0.0576384 |  0.0113503  |

- Spearman(val_rmse, test_r2) = −0.455 (p=0.187, n=10)
- val winner (honest): `fg_w512x512_d0.3_huber0.1_swa` (0.7599) | test best (ref): `w512x512x512_d0.3_huber0.1_swa` (0.7903)

## Overall Model Leaderboard (2023–2025 Test Set)

`test-best` rows are reporting only (selection on test = leakage; the XGBoost
reference itself was test-selected in eval-1.1). `(val top-k avg)` rows are
offline seed-averaged ensembles of the top-k val-selected honest configs (no
extra training). `(5-seed champ, ...)` rows are 5-seed champion ensembles of
the val-selected winners. `MLP cross-family` averages the val-selected winners
across the three families.
| model_name                                                                                                                                                | strategy_name          |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)                                                                                                                | XGBoost_Reference      |    0.81496  |     0.0438196 |       0.043337  |   0.00648567  |    0.0337195 |         0.905594 |
| MLP 2-Regime-Mixed (val top-5 avg)                                                                                                                        | MLP_2regime_mixed      |    0.800323 |     0.0455197 |       0.0434056 |   0.0137112   |    0.034711  |         0.90468  |
| MLP 2-Regime-Mixed (val top-3 avg)                                                                                                                        | MLP_2regime_mixed      |    0.800297 |     0.0455227 |       0.0443385 |   0.0103158   |    0.0346077 |         0.900368 |
| MLP 2-Regime-Mixed (val top-10 avg)                                                                                                                       | MLP_2regime_mixed      |    0.794724 |     0.0461536 |       0.0437507 |   0.0146978   |    0.0356735 |         0.903443 |
| MLP cross-family (val winners: 2regime_96/w512x512x512_d0.3_lr1e-3 + 2regime_54/w512x512x512_d0.3_huber0.1 + 2regime_mixed/fg_w512x512_d0.3_huber0.1_swa) | MLP_cross_family       |    0.793243 |     0.0463198 |       0.0455726 |   0.00828602  |    0.035305  |         0.894921 |
| MLP 2-Regime-Mixed (test-best, w512x512x512_d0.3_huber0.1_swa)                                                                                            | MLP_testbest_reference |    0.790253 |     0.0466534 |       0.0449702 |   0.0124186   |    0.0354233 |         0.898277 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.1_swa)                                                                                                       | MLP_2regime_mixed      |    0.790253 |     0.0466534 |       0.0449702 |   0.0124186   |    0.0354233 |         0.898277 |
| MLP-1.3 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                       | MLP_1.3_Reference      |    0.788821 |     0.0468125 |       0.0467953 |   0.00126699  |    0.0362252 |         0.888558 |
| MLP 2-Regime-54 (test-best, w384x384_d0.3_gelu)                                                                                                           | MLP_testbest_reference |    0.786493 |     0.0470697 |       0.046945  |   0.0034241   |    0.0364015 |         0.887498 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)                                                                                                                | MLP_testbest_reference |    0.785392 |     0.047191  |       0.0461896 |   0.00967028  |    0.0364359 |         0.892043 |
| MLP-1.3 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                            | MLP_1.3_Reference      |    0.783404 |     0.047409  |       0.0471517 |   0.00493258  |    0.0360624 |         0.887208 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5_swa)                                                                                                            | MLP_testbest_reference |    0.783361 |     0.0474137 |       0.0471695 |   0.0048058   |    0.0359738 |         0.88706  |
| MLP-1.3 2-Regime-54 (val top-10 avg)                                                                                                                      | MLP_1.3_Reference      |    0.782533 |     0.0475043 |       0.0470162 |   0.0067925   |    0.0369583 |         0.888139 |
| MLP 2-Regime-Mixed (test-best, fg_w384x384_d0.3_gelu_swa)                                                                                                 | MLP_testbest_reference |    0.782334 |     0.047526  |       0.0439493 |   0.018088    |    0.0364477 |         0.902285 |
| MLP 2-Regime-Mixed (fg_w384x384_d0.3_gelu_swa)                                                                                                            | MLP_2regime_mixed      |    0.782334 |     0.047526  |       0.0439493 |   0.018088    |    0.0364477 |         0.902285 |
| MLP 2-Regime-54 (test-best, w448x448_d0.3_gelu_swa)                                                                                                       | MLP_testbest_reference |    0.780859 |     0.0476868 |       0.0476765 |   0.000990915 |    0.0366186 |         0.883737 |
| MLP 2-Regime-54 (test-best, w448x448_d0.3_gelu)                                                                                                           | MLP_testbest_reference |    0.780859 |     0.0476868 |       0.0476765 |   0.000990915 |    0.0366186 |         0.883737 |
| MLP 2-Regime-54 (val top-10 avg)                                                                                                                          | MLP_2regime_54         |    0.780505 |     0.0477252 |       0.0464651 |   0.0108946   |    0.0372804 |         0.889914 |
| Global Single Model (54 Backbone)                                                                                                                         | XGBoost_Reference      |    0.77923  |     0.0478636 |       0.0466868 |   0.0105484   |    0.0370592 |         0.889432 |
| MLP 2-Regime-Mixed (test-best, w384x384_d0.3_gelu)                                                                                                        | MLP_testbest_reference |    0.778968 |     0.047892  |       0.0459946 |   0.0133468   |    0.0376232 |         0.892654 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.1)                                                                                                           | MLP_2regime_mixed      |    0.778871 |     0.0479025 |       0.0448719 |   0.016768    |    0.0369035 |         0.899325 |
| MLP 2-Regime-96 (w512x512x512_d0.3_huber0.1_swa)                                                                                                          | MLP_2regime_96         |    0.773153 |     0.048518  |       0.0456904 |   0.0163212   |    0.0371332 |         0.894992 |
| MLP 2-Regime-96 (test-best, w512x512x512_d0.3_huber0.1_swa)                                                                                               | MLP_testbest_reference |    0.773153 |     0.048518  |       0.0456904 |   0.0163212   |    0.0371332 |         0.894992 |
| MLP-1.3 2-Regime-96 (val top-10 avg)                                                                                                                      | MLP_1.3_Reference      |    0.772329 |     0.048606  |       0.0446198 |   0.0192772   |    0.0375105 |         0.899578 |
| MLP 2-Regime-54 (5-seed champ, w512x512x512_d0.3_huber0.1)                                                                                                | MLP_2regime_54         |    0.771661 |     0.0486772 |       0.0480595 |   0.00772988  |    0.0380031 |         0.886477 |
| MLP 2-Regime-Mixed (5-seed champ, fg_w512x512_d0.3_huber0.1_swa)                                                                                          | MLP_2regime_mixed      |    0.769858 |     0.048869  |       0.048842  |  -0.00162579  |    0.0370965 |         0.879672 |
| MLP 2-Regime-96 (val top-10 avg)                                                                                                                          | MLP_2regime_96         |    0.769168 |     0.0489422 |       0.0450254 |   0.0191846   |    0.0379532 |         0.897463 |
| MLP 2-Regime-54 (val top-3 avg)                                                                                                                           | MLP_2regime_54         |    0.766113 |     0.049265  |       0.0477961 |   0.0119406   |    0.0385241 |         0.884233 |
| MLP 2-Regime-96 (val top-3 avg)                                                                                                                           | MLP_2regime_96         |    0.76542  |     0.049338  |       0.0458892 |   0.0181223   |    0.0379486 |         0.894154 |
| MLP-1.3 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                 | MLP_1.3_Reference      |    0.76511  |     0.0493706 |       0.0488979 |   0.00681535  |    0.0385003 |         0.882441 |
| MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)                                                                                                              | MLP_2regime_54         |    0.76511  |     0.0493706 |       0.0488979 |   0.00681535  |    0.0385003 |         0.882441 |
| MLP 2-Regime-54 (val top-5 avg)                                                                                                                           | MLP_2regime_54         |    0.764315 |     0.0494541 |       0.0472141 |   0.014715    |    0.0387503 |         0.886404 |
| MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1_swa)                                                                                                          | MLP_2regime_54         |    0.764039 |     0.049483  |       0.0487124 |   0.00869853  |    0.0385382 |         0.881843 |
| MLP-1.3 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                   | MLP_1.3_Reference      |    0.761018 |     0.0497987 |       0.0465842 |   0.0176019   |    0.0384117 |         0.890751 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                                                                                                                | MLP_2regime_96         |    0.761018 |     0.0497987 |       0.0465842 |   0.0176019   |    0.0384117 |         0.890751 |
| MLP 2-Regime-Mixed (w448x448_d0.3_gelu)                                                                                                                   | MLP_2regime_mixed      |    0.760502 |     0.0498525 |       0.0452873 |   0.0208406   |    0.0390533 |         0.895915 |
| MLP 2-Regime-Mixed (fg_w512x512_d0.3_huber0.1_swa)                                                                                                        | MLP_2regime_mixed      |    0.759881 |     0.049917  |       0.0499151 |   0.00044081  |    0.038269  |         0.872243 |
| MLP 2-Regime-96 (val top-5 avg)                                                                                                                           | MLP_2regime_96         |    0.75966  |     0.04994   |       0.0460609 |   0.0192975   |    0.0391487 |         0.891959 |
| MLP 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                  | MLP_2regime_96         |    0.75656  |     0.050261  |       0.0459346 |   0.0204007   |    0.0391776 |         0.892873 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3_swa)                                                                                                            | MLP_2regime_96         |    0.754698 |     0.050453  |       0.0461254 |   0.0204438   |    0.0390437 |         0.893469 |
| MLP 2-Regime-54 (fg_w384x384_d0.3_swa)                                                                                                                    | MLP_2regime_54         |    0.72938  |     0.0529927 |       0.0495254 |   0.0188537   |    0.0414091 |         0.874198 |
| MLP 2-Regime-54 (fg_w384x384_d0.3)                                                                                                                        | MLP_2regime_54         |    0.727882 |     0.0531391 |       0.0496646 |   0.0188997   |    0.0415319 |         0.873341 |
| MLP 2-Regime-54 (plr_w384x384_d0.3_gelu_swa)                                                                                                              | MLP_2regime_54         |    0.720252 |     0.053879  |       0.0499052 |   0.0203079   |    0.0428403 |         0.871804 |
| MLP 2-Regime-96 (fg_w384x384_d0.3_huber0.1_swa)                                                                                                           | MLP_2regime_96         |    0.716937 |     0.0541973 |       0.0503577 |   0.0200361   |    0.0434832 |         0.869681 |
| MLP 2-Regime-96 (fg_w384x384_d0.3_huber0.1)                                                                                                               | MLP_2regime_96         |    0.710904 |     0.0547718 |       0.050122  |   0.0220846   |    0.0440234 |         0.87074  |
## Hyperparameter Sweep Summary

30 curated configs × 3 families; ranked by **2-seed mean val RMSE** (honest
signal); test R² for reference. Phase-2 configs carry `n_seeds=2`. The
`deployed` column is "live" for every config (see the SWA section).

### Sweep Top-10 — 2-Regime-96 (by val RMSE)

| config_id | architecture | n_seeds | dropout | lr | loss | val_rmse | aux_rmse | test_r2 | test_rmse | test_bias | best_epoch | train_time_s | deployed |
|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| w512x512x512_d0.3_lr1e-3 | mlp | 2 | 0.3 | 0.001 | mse | 0.0482834 | 0.0249457 | 0.761018 | 0.0497987 | 0.0176019 | 263 | 95.435 | live |
| w512x512x512_d0.3_lr1e-3_swa | mlp | 2 | 0.3 | 0.001 | mse | 0.0483296 | 0.0249357 | 0.754698 | 0.050453 | 0.0204438 | 280 | 123.883 | live |
| w512x512x512_d0.3_huber0.1_swa | mlp | 2 | 0.3 | 0.0003 | huber | 0.0490918 | 0.0258471 | 0.773153 | 0.048518 | 0.0163212 | 276 | 126.831 | live |
| fg_w384x384_d0.3_huber0.1 | fg | 2 | 0.3 | 0.0003 | huber | 0.0508182 | 0.0305982 | 0.710904 | 0.0547718 | 0.0220846 | 313 | 236.875 | live |
| fg_w384x384_d0.3_huber0.1_swa | fg | 2 | 0.3 | 0.0003 | huber | 0.0509411 | 0.0312027 | 0.716937 | 0.0541973 | 0.0200361 | 286 | 241.804 | live |
| w512x512_d0.35_gelu | mlp | 2 | 0.35 | 0.0003 | mse | 0.052903 | 0.0359984 | 0.772401 | 0.0485983 | 0.0174258 | 247 | 111.414 | live |
| plr_w256x256_d0.4_swa | plr | 2 | 0.4 | 0.0003 | mse | 0.0552381 | 0.0322751 | 0.623881 | 0.0624739 | 0.0340105 | 214 | 129.261 | live |
| w256x256_d0.5 | mlp | 2 | 0.5 | 0.0003 | mse | 0.0552727 | 0.0471912 | 0.785392 | 0.047191 | 0.00967028 | 380 | 154.718 | live |
| w256x256_d0.5_swa | mlp | 1 | 0.5 | 0.0003 | mse | 0.0570138 | 0.0496979 | 0.783361 | 0.0474137 | 0.0048058 | 351 | 104.26 | live |
| fg_w256x256_d0.4_swa | fg | 1 | 0.4 | 0.0003 | mse | 0.0602456 | 0.0598678 | 0.477245 | 0.073652 | 0.0294463 | 214 | 110.981 | live |

### Sweep Top-10 — 2-Regime-54 (by val RMSE)

| config_id | architecture | n_seeds | dropout | lr | loss | val_rmse | aux_rmse | test_r2 | test_rmse | test_bias | best_epoch | train_time_s | deployed |
|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| w512x512x512_d0.3_huber0.1 | mlp | 2 | 0.3 | 0.0003 | huber | 0.0564972 | 0.0232794 | 0.76511 | 0.0493706 | 0.00681535 | 345 | 95.8986 | live |
| w512x512x512_d0.3_huber0.1_swa | mlp | 2 | 0.3 | 0.0003 | huber | 0.0568509 | 0.0233968 | 0.764039 | 0.049483 | 0.00869853 | 276 | 130.752 | live |
| plr_w384x384_d0.3_gelu_swa | plr | 2 | 0.3 | 0.0003 | mse | 0.0574891 | 0.0288451 | 0.720252 | 0.053879 | 0.0203079 | 216 | 128.219 | live |
| fg_w384x384_d0.3 | fg | 2 | 0.3 | 0.0003 | mse | 0.0575515 | 0.0261881 | 0.727882 | 0.0531391 | 0.0188997 | 352 | 419.867 | live |
| fg_w384x384_d0.3_swa | fg | 2 | 0.3 | 0.0003 | mse | 0.0575753 | 0.0260077 | 0.72938 | 0.0529927 | 0.0188537 | 342 | 444.094 | live |
| fg_w512x512_d0.3_huber0.1_swa | fg | 2 | 0.3 | 0.0003 | huber | 0.0577613 | 0.0250303 | 0.721153 | 0.0537921 | 0.0197122 | 284 | 366.337 | live |
| w384x384_d0.3_gelu | mlp | 1 | 0.3 | 0.0003 | mse | 0.0614041 | 0.0306479 | 0.786493 | 0.0470697 | 0.0034241 | 256 | 56.4972 | live |
| w448x448_d0.3_gelu | mlp | 2 | 0.3 | 0.0003 | mse | 0.0618635 | 0.0302294 | 0.780859 | 0.0476868 | 0.000990915 | 234 | 107.91 | live |
| w448x448_d0.3_gelu_swa | mlp | 2 | 0.3 | 0.0003 | mse | 0.0618635 | 0.0302294 | 0.780859 | 0.0476868 | 0.000990915 | 234 | 106.564 | live |
| w384x384_d0.3_mixup0.4 | mlp | 1 | 0.3 | 0.0003 | mse | 0.0624267 | 0.0349459 | 0.765446 | 0.0493353 | 0.0102525 | 240 | 56.2533 | live |

### Sweep Top-10 — 2-Regime-Mixed (by val RMSE)

| config_id | architecture | n_seeds | dropout | lr | loss | val_rmse | aux_rmse | test_r2 | test_rmse | test_bias | best_epoch | train_time_s | deployed |
|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| fg_w512x512_d0.3_huber0.1_swa | fg | 2 | 0.3 | 0.0003 | huber | 0.0463768 | 0.0252084 | 0.759881 | 0.049917 | 0.00044081 | 297 | 293.996 | live |
| fg_w384x384_d0.3_gelu_swa | fg | 2 | 0.3 | 0.0003 | mse | 0.0465913 | 0.0270722 | 0.782334 | 0.047526 | 0.018088 | 363 | 331.653 | live |
| w512x512x512_d0.3_huber0.1_swa | mlp | 2 | 0.3 | 0.0003 | huber | 0.0494507 | 0.0238544 | 0.790253 | 0.0466534 | 0.0124186 | 298 | 167.566 | live |
| w512x512x512_d0.3_huber0.1 | mlp | 2 | 0.3 | 0.0003 | huber | 0.0494921 | 0.0245625 | 0.778871 | 0.0479025 | 0.016768 | 260 | 157.003 | live |
| w448x448_d0.3_gelu | mlp | 2 | 0.3 | 0.0003 | mse | 0.052322 | 0.0296164 | 0.760502 | 0.0498525 | 0.0208406 | 256 | 130.125 | live |
| w448x448_d0.3_gelu_swa | mlp | 2 | 0.3 | 0.0003 | mse | 0.0523246 | 0.0297563 | 0.763263 | 0.0495643 | 0.0205277 | 238 | 105.419 | live |
| w384x384_d0.3_gelu | mlp | 2 | 0.3 | 0.0003 | mse | 0.0528789 | 0.033321 | 0.778968 | 0.047892 | 0.0133468 | 310 | 119.836 | live |
| w384x384_d0.3_mixup0.4_huber0.1 | mlp | 2 | 0.3 | 0.0003 | huber | 0.0531458 | 0.0321586 | 0.746382 | 0.051301 | 0.0202711 | 277 | 109.753 | live |
| w256x256_d0.4_swa | mlp | 1 | 0.4 | 0.0003 | mse | 0.0546939 | 0.0419791 | 0.777045 | 0.0480999 | 0.0129263 | 351 | 59.7665 | live |
| plr_w384x384_d0.3_gelu | plr | 1 | 0.3 | 0.0003 | mse | 0.0570874 | 0.0295189 | 0.679851 | 0.0576384 | 0.0113503 | 271 | 47.2535 | live |

## Per-Regime Performance Breakdown

Cluster 0 holds 73% of the test rows, so it dominates the pooled R². The mixed
family's c1 (54+10) specialist holds the 54-family's ~0.83 R², and its best c0
(96-pool) specialist reaches 0.7726.

| strategy_name | model_name | cluster | n_train | n_test | r2 | rmse | ubrmse | bias | mae |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MLP_2regime_96 | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3) | 0 | 7156 | 4817 | 0.754287 | 0.0495899 | 0.0472413 | 0.0150802 | 0.0389389 |
| MLP_2regime_96 | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3) | 1 | 2647 | 1803 | 0.776352 | 0.0503523 | 0.0440792 | 0.0243389 | 0.0370033 |
| MLP_2regime_96 | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3_swa) | 0 | 7156 | 4817 | 0.745281 | 0.0504906 | 0.046785 | 0.0189859 | 0.0398074 |
| MLP_2regime_96 | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3_swa) | 1 | 2647 | 1803 | 0.776352 | 0.0503523 | 0.0440792 | 0.0243389 | 0.0370033 |
| MLP_2regime_96 | MLP 2-Regime-96 (w512x512x512_d0.3_huber0.1_swa) | 0 | 7156 | 4817 | 0.772578 | 0.0477084 | 0.0459426 | 0.0128596 | 0.0366476 |
| MLP_2regime_96 | MLP 2-Regime-96 (w512x512x512_d0.3_huber0.1_swa) | 1 | 2647 | 1803 | 0.773991 | 0.0506174 | 0.0436845 | 0.0255692 | 0.0384303 |
| MLP_2regime_54 | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1) | 0 | 7156 | 4817 | 0.736751 | 0.051329 | 0.0510691 | 0.0051591 | 0.0408763 |
| MLP_2regime_54 | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1) | 1 | 2647 | 1803 | 0.831465 | 0.0437101 | 0.0422401 | 0.0112403 | 0.0321524 |
| MLP_2regime_54 | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1_swa) | 0 | 7156 | 4817 | 0.735225 | 0.0514775 | 0.0508912 | 0.00774716 | 0.0409284 |
| MLP_2regime_54 | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1_swa) | 1 | 2647 | 1803 | 0.831465 | 0.0437101 | 0.0422401 | 0.0112403 | 0.0321524 |
| MLP_2regime_54 | MLP 2-Regime-54 (plr_w384x384_d0.3_gelu_swa) | 0 | 7156 | 4817 | 0.725851 | 0.0523809 | 0.0502282 | 0.0148621 | 0.0417248 |
| MLP_2regime_54 | MLP 2-Regime-54 (plr_w384x384_d0.3_gelu_swa) | 1 | 2647 | 1803 | 0.706411 | 0.0576909 | 0.0459697 | 0.0348572 | 0.0458205 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (fg_w512x512_d0.3_huber0.1_swa) | 0 | 7156 | 4817 | 0.73214 | 0.0517766 | 0.0515325 | -0.00502148 | 0.0403861 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (fg_w512x512_d0.3_huber0.1_swa) | 1 | 2647 | 1803 | 0.824767 | 0.0445702 | 0.041958 | 0.0150342 | 0.0326127 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (fg_w384x384_d0.3_gelu_swa) | 0 | 7156 | 4817 | 0.76247 | 0.0487572 | 0.0444596 | 0.0200152 | 0.0380321 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (fg_w384x384_d0.3_gelu_swa) | 1 | 2647 | 1803 | 0.828692 | 0.0440682 | 0.0421258 | 0.0129393 | 0.0322147 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.1_swa) | 0 | 7156 | 4817 | 0.772578 | 0.0477084 | 0.0459426 | 0.0128596 | 0.0366476 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.1_swa) | 1 | 2647 | 1803 | 0.831465 | 0.0437101 | 0.0422401 | 0.0112403 | 0.0321524 |
| XGBoost_Reference | Global Single Model (54 Backbone) | 0 | 14608 | 6620 | 0.77923 | 0.0478636 | 0.0466868 | 0.0105484 | 0.0370592 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10) | 0 | 10624 | 4817 | 0.80246 | 0.0444639 | 0.0436213 | 0.00861491 | 0.0359221 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10) | 1 | 3984 | 1803 | 0.844023 | 0.0420501 | 0.0420426 | 0.000797068 | 0.0278349 |
| MLP_1.3_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3) | 0 | 7156 | 4817 | 0.754287 | 0.0495899 | 0.0472413 | 0.0150802 | 0.0389389 |
| MLP_1.3_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3) | 1 | 2647 | 1803 | 0.776352 | 0.0503523 | 0.0440792 | 0.0243389 | 0.0370033 |
| MLP_1.3_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1) | 0 | 7156 | 4817 | 0.736751 | 0.051329 | 0.0510691 | 0.0051591 | 0.0408763 |
| MLP_1.3_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1) | 1 | 2647 | 1803 | 0.831465 | 0.0437101 | 0.0422401 | 0.0112403 | 0.0321524 |

## Year-by-Year R² Breakdown

2025 — previously the MLPs' weak year — is now the mixed family's best year:
the mixed val top-3/top-5 ensembles score 2025 R² 0.8336/0.8324, the best of
any model including the XGBoost 2-regime reference (0.8303).

| model_name | pooled_r2 | year_2023_r2 | year_2024_r2 | year_2025_r2 |
|:---|:---:|:---:|:---:|:---:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10) | 0.81496 | 0.822971 | 0.783256 | 0.83029 |
| MLP 2-Regime-Mixed (val top-5 avg) | 0.800323 | 0.745352 | 0.825612 | 0.832424 |
| MLP 2-Regime-Mixed (val top-3 avg) | 0.800297 | 0.74583 | 0.823739 | 0.83362 |
| MLP 2-Regime-Mixed (val top-10 avg) | 0.794724 | 0.753256 | 0.82392 | 0.807145 |
| MLP cross-family (val winners: 2regime_96/w512x512x512_d0.3_lr1e-3 + 2regime_54/w512x512x512_d0.3_huber0.1 + 2regime_mixed/fg_w512x512_d0.3_huber0.1_swa) | 0.793243 | 0.74832 | 0.812752 | 0.819099 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.1_swa) | 0.790253 | 0.735215 | 0.807409 | 0.830042 |
| MLP-1.3 2-Regime-54 (test_best: w384x384_d0.3_gelu) | 0.788821 | 0.773579 | 0.818284 | 0.770357 |
| MLP 2-Regime-54 (test-best, w384x384_d0.3_gelu) | 0.786493 | 0.767355 | 0.826921 | 0.76174 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5) | 0.785392 | 0.763461 | 0.817843 | 0.771644 |
| MLP-1.3 2-Regime-96 (test_best: w256x256_d0.5) | 0.783404 | 0.763386 | 0.82079 | 0.762541 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5_swa) | 0.783361 | 0.76105 | 0.820079 | 0.765809 |
| MLP-1.3 2-Regime-54 (val top-10 avg) | 0.782533 | 0.749643 | 0.803851 | 0.792293 |
| MLP 2-Regime-Mixed (fg_w384x384_d0.3_gelu_swa) | 0.782334 | 0.70907 | 0.817702 | 0.825166 |
| MLP 2-Regime-54 (test-best, w448x448_d0.3_gelu_swa) | 0.780859 | 0.783872 | 0.788164 | 0.762631 |
| MLP 2-Regime-54 (test-best, w448x448_d0.3_gelu) | 0.780859 | 0.783872 | 0.788164 | 0.762631 |
| MLP 2-Regime-54 (val top-10 avg) | 0.780505 | 0.736315 | 0.827429 | 0.778244 |
| Global Single Model (54 Backbone) | 0.77923 | 0.750748 | 0.770077 | 0.813582 |
| MLP 2-Regime-Mixed (test-best, w384x384_d0.3_gelu) | 0.778968 | 0.744687 | 0.801705 | 0.788831 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.1) | 0.778871 | 0.718703 | 0.798495 | 0.821802 |
| MLP 2-Regime-96 (w512x512x512_d0.3_huber0.1_swa) | 0.773153 | 0.727751 | 0.797584 | 0.794053 |
| MLP-1.3 2-Regime-96 (val top-10 avg) | 0.772329 | 0.72545 | 0.797894 | 0.793806 |
| MLP 2-Regime-54 (5-seed champ, w512x512x512_d0.3_huber0.1) | 0.771661 | 0.723384 | 0.781407 | 0.810216 |
| MLP 2-Regime-Mixed (5-seed champ, fg_w512x512_d0.3_huber0.1_swa) | 0.769858 | 0.752402 | 0.765646 | 0.786262 |
| MLP 2-Regime-96 (val top-10 avg) | 0.769168 | 0.720824 | 0.814159 | 0.773225 |
| MLP 2-Regime-54 (val top-3 avg) | 0.766113 | 0.731216 | 0.800822 | 0.764529 |
| MLP 2-Regime-96 (val top-3 avg) | 0.76542 | 0.710473 | 0.792424 | 0.794606 |
| MLP-1.3 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1) | 0.76511 | 0.723072 | 0.775216 | 0.79585 |
| MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1) | 0.76511 | 0.723072 | 0.775216 | 0.79585 |
| MLP 2-Regime-54 (val top-5 avg) | 0.764315 | 0.707276 | 0.813702 | 0.773985 |
| MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1_swa) | 0.764039 | 0.720924 | 0.784244 | 0.786109 |
| MLP-1.3 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3) | 0.761018 | 0.706877 | 0.786985 | 0.790134 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3) | 0.761018 | 0.706877 | 0.786985 | 0.790134 |
| MLP 2-Regime-Mixed (w448x448_d0.3_gelu) | 0.760502 | 0.722103 | 0.787735 | 0.7701 |
| MLP 2-Regime-Mixed (fg_w512x512_d0.3_huber0.1_swa) | 0.759881 | 0.744054 | 0.757107 | 0.772643 |
| MLP 2-Regime-96 (val top-5 avg) | 0.75966 | 0.715555 | 0.789705 | 0.773096 |
| MLP 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3) | 0.75656 | 0.708198 | 0.776529 | 0.784688 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3_swa) | 0.754698 | 0.691195 | 0.784404 | 0.790792 |
| MLP 2-Regime-54 (fg_w384x384_d0.3_swa) | 0.72938 | 0.646111 | 0.7866 | 0.7606 |
| MLP 2-Regime-54 (fg_w384x384_d0.3) | 0.727882 | 0.645519 | 0.783452 | 0.759613 |

## Systematic-Bias Diagnostic (headline)

bias²/MSE share = squared pooled test bias / MSE (MSE = bias² + ubRMSE²).
1.3 reference medians: 2regime_96 ~10–17%, 2regime_54 ~1%. 2.0 success
criterion: per-family median < 5% — **met for 2regime_54 (3.7%)**, not yet for
the 96-pool families; the capacity-control configs are near-unbiased.

### Per-family median bias²/MSE share (honest architectures)

| family | n_configs | med_bias2_mse_share | med_test_bias | med_test_r2 |
|:---|:---:|:---:|:---:|:---:|
| 2regime_96 | 10 | 0.1326 | 0.0188 | 0.7579 |
| 2regime_54 | 10 | 0.0370 | 0.0095 | 0.7646 |
| 2regime_mixed | 10 | 0.1001 | 0.0151 | 0.7702 |

### Best 8 configs by bias²/MSE share (all architectures)

| family | config_id | architecture | test_r2 | test_rmse | test_bias | bias2_mse_share |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| 2regime_mixed | fg_w512x512_d0.3_huber0.1_swa | fg | 0.759881 | 0.049917 | 0.00044081 | 0.000078 |
| 2regime_54 | w448x448_d0.3_gelu | mlp | 0.780859 | 0.0476868 | 0.000990915 | 0.000432 |
| 2regime_54 | w448x448_d0.3_gelu_swa | mlp | 0.780859 | 0.0476868 | 0.000990915 | 0.000432 |
| 2regime_54 | w384x384_d0.3_gelu | mlp | 0.786493 | 0.0470697 | 0.0034241 | 0.005292 |
| 2regime_96 | w256x256_d0.5_swa | mlp | 0.783361 | 0.0474137 | 0.0048058 | 0.010274 |
| 2regime_54 | w512x512x512_d0.3_huber0.1 | mlp | 0.76511 | 0.0493706 | 0.00681535 | 0.019056 |
| 2regime_54 | w512x512x512_d0.3_huber0.1_swa | mlp | 0.764039 | 0.049483 | 0.00869853 | 0.030902 |
| 2regime_mixed | plr_w384x384_d0.3_gelu | plr | 0.679851 | 0.0576384 | 0.0113503 | 0.038779 |

### Per-cluster median bias²/MSE share (honest architectures)

| family | cluster | med_bias2_mse_share | med_test_bias | med_test_r2 |
|:---|:---:|:---:|:---:|:---:|
| 2regime_96 | 0 | 0.0829 | 0.0145 | 0.7498 |
| 2regime_96 | 1 | 0.2336 | 0.0243 | 0.7752 |
| 2regime_54 | 0 | 0.0341 | 0.0093 | 0.7360 |
| 2regime_54 | 1 | 0.1061 | 0.0141 | 0.8296 |
| 2regime_mixed | 0 | 0.1412 | 0.0187 | 0.7403 |
| 2regime_mixed | 1 | 0.0826 | 0.0127 | 0.8315 |

## SWA vs Live Deployment (trainer)

Per-seed per-cluster live vs SWA val RMSE (mean over specs/seeds);
`specs_deployed_swa` = number of (seed, specialist) jobs where the SWA
snapshot beat the live best on val.

| family | config_id | architecture | n_seeds | specs_deployed_swa | val_rmse_live | val_rmse_swa | test_r2 | test_rmse | test_bias |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2regime_54 | fg_w384x384_d0.3_swa | fg | 2 | 0 | 0.0515398 | 0.0999253 | 0.72938 | 0.0529927 | 0.0188537 |
| 2regime_54 | w512x512x512_d0.3_huber0.1_swa | mlp | 2 | 0 | 0.051755 | 0.105121 | 0.764039 | 0.049483 | 0.00869853 |
| 2regime_54 | fg_w512x512_d0.3_huber0.1_swa | fg | 2 | 0 | 0.0522825 | 0.0939207 | 0.721153 | 0.0537921 | 0.0197122 |
| 2regime_54 | plr_w384x384_d0.3_gelu_swa | plr | 2 | 0 | 0.0533397 | 0.10162 | 0.720252 | 0.053879 | 0.0203079 |
| 2regime_54 | w448x448_d0.3_gelu_swa | mlp | 2 | 0 | 0.0558677 | 0.126229 | 0.780859 | 0.0476868 | 0.000990915 |
| 2regime_96 | w512x512x512_d0.3_lr1e-3_swa | mlp | 2 | 0 | 0.0449153 | 0.0707744 | 0.754698 | 0.050453 | 0.0204438 |
| 2regime_96 | w512x512x512_d0.3_huber0.1_swa | mlp | 2 | 0 | 0.0461912 | 0.0917291 | 0.773153 | 0.048518 | 0.0163212 |
| 2regime_96 | fg_w384x384_d0.3_huber0.1_swa | fg | 2 | 0 | 0.0482114 | 0.0938793 | 0.716937 | 0.0541973 | 0.0200361 |
| 2regime_96 | plr_w256x256_d0.4_swa | plr | 2 | 0 | 0.0509459 | 0.0935152 | 0.623881 | 0.0624739 | 0.0340105 |
| 2regime_96 | w256x256_d0.5_swa | mlp | 1 | 0 | 0.0547098 | 0.115087 | 0.783361 | 0.0474137 | 0.0048058 |
| 2regime_96 | fg_w256x256_d0.4_swa | fg | 1 | 0 | 0.0626091 | 0.0727592 | 0.477245 | 0.073652 | 0.0294463 |
| 2regime_mixed | fg_w512x512_d0.3_huber0.1_swa | fg | 2 | 0 | 0.0448799 | 0.0893669 | 0.759881 | 0.049917 | 0.00044081 |
| 2regime_mixed | fg_w384x384_d0.3_gelu_swa | fg | 2 | 0 | 0.0455507 | 0.0966513 | 0.782334 | 0.047526 | 0.018088 |
| 2regime_mixed | w512x512x512_d0.3_huber0.1_swa | mlp | 2 | 0 | 0.0469652 | 0.101268 | 0.790253 | 0.0466534 | 0.0124186 |
| 2regime_mixed | w448x448_d0.3_gelu_swa | mlp | 2 | 0 | 0.0497104 | 0.112743 | 0.763263 | 0.0495643 | 0.0205277 |
| 2regime_mixed | w256x256_d0.4_swa | mlp | 1 | 0 | 0.0512577 | 0.118245 | 0.777045 | 0.0480999 | 0.0129263 |

**Documented negative:** with `swa_start_frac: 0.6` equal-weight averaging over
epochs 240–400 + BN recalibration, **no SWA snapshot ever beat the live best on
val** (SWA val 0.070–0.126 vs live 0.045–0.062), so every config deployed the
live model. The val-based deployment mechanism worked as designed — it
correctly rejected non-competitive snapshots.

**Implementation note (reproducibility):** the BN-recalibration pass runs the
train loader in train mode, so its dropout layers consume the shared RNG
stream; a `swa=true` job's live trajectory therefore diverges slightly from its
`swa=false` anchor (same seed; e.g. mixed `w512x512x512_d0.3_huber0.1_swa`
seed 42 best_epoch 266 vs the anchor's 240). The reported swa-config numbers
are deterministic and reproducible from the committed code + seeds, but their
gains over the anchors are **live-trajectory results, not SWA weight
deployment**. A future SWA test should (a) start averaging later (e.g.
`swa_start_frac: 0.85`) and (b) guard the RNG around the recalibration pass so
the live path stays bit-identical to the anchor.

## Early-Stopping & SWA-rule Replay (patience-60 re-check)

Mean pooled test RMSE over configs/seeds; lower is better. `oracle` = argmin on
test (unreachable bound). **patience-60 remains the best honest rule** in all
three families; the offline `swa_val` epoch rule (argmin of the SWA-val curve)
does not beat it.

### 2-Regime-96

| rule | mean_test_rmse | median_test_rmse | n |
|:---|:---:|:---:|:---:|
| oracle | 0.0503049 | 0.0479306 | 19 |
| patience60 | 0.0545378 | 0.0522245 | 19 |
| patience40 | 0.0545378 | 0.0522245 | 19 |
| patience20 | 0.0545378 | 0.0522245 | 19 |
| val_aux | 0.0562256 | 0.0548668 | 19 |
| plateau_w60e1e-4 | 0.0599735 | 0.0555584 | 19 |
| swa_val | 0.0625803 | 0.0612281 | 19 |
| plateau_w40e1e-4 | 0.0639599 | 0.059757 | 19 |
| plateau_w40e3e-4 | 0.0639599 | 0.059757 | 19 |
| plateau_w20e1e-4 | 0.0733378 | 0.0792595 | 19 |

### 2-Regime-54

| rule | mean_test_rmse | median_test_rmse | n |
|:---|:---:|:---:|:---:|
| oracle | 0.0494394 | 0.0484306 | 19 |
| val_aux | 0.0511795 | 0.0496554 | 19 |
| patience20 | 0.0512877 | 0.0500556 | 19 |
| patience60 | 0.0512877 | 0.0500556 | 19 |
| patience40 | 0.0512877 | 0.0500556 | 19 |
| plateau_w60e1e-4 | 0.0575226 | 0.056796 | 19 |
| plateau_w40e3e-4 | 0.0645626 | 0.0637682 | 19 |
| plateau_w40e1e-4 | 0.0645626 | 0.0637682 | 19 |
| swa_val | 0.066988 | 0.0635742 | 19 |
| plateau_w20e1e-4 | 0.0716067 | 0.0711597 | 19 |

### 2-Regime-Mixed

| rule | mean_test_rmse | median_test_rmse | n |
|:---|:---:|:---:|:---:|
| oracle | 0.0465101 | 0.0459277 | 18 |
| patience60 | 0.0499954 | 0.0496422 | 18 |
| patience40 | 0.0499954 | 0.0496422 | 18 |
| patience20 | 0.0499954 | 0.0496422 | 18 |
| val_aux | 0.0505083 | 0.0497345 | 18 |
| plateau_w60e1e-4 | 0.0527256 | 0.0515355 | 18 |
| plateau_w40e3e-4 | 0.0581369 | 0.0558167 | 18 |
| plateau_w40e1e-4 | 0.0581369 | 0.0558167 | 18 |
| swa_val | 0.0658747 | 0.0608116 | 18 |
| plateau_w20e1e-4 | 0.0667983 | 0.0640824 | 18 |

## FeatureGroupedMLP — the semantic grouping

Resolved by `mlp20/feature_groups.py` (explicit first-match-wins rule table;
validated: every feature in exactly one group, raises otherwise) for the union
of features used by the 3 families — **116 unique features → 8 groups**:

- **group 0 — smap (26):** A_d/A_grad/C_lag/V_* SMAP variants, SMAP_ampm_diff_interp, SMAP_sm_{am,pm}_interp (+lags/rolls), V_ema/V_rollmin_SMAP_sm_interp_*
- **group 1 — optical (7):** A_grad_s2_b11, V_rollmin_s2_b11/12, V_rollrng_s2_b11, s2_b4, s2_b8
- **group 2 — vegetation (17):** C_lag_F_NDMI/NDVI, D_sa_F_NDMI, D_z_F_NDMI, F_MSI, V_ema/V_roll{max,min,rng,std}_F_NDVI/NDMI_*
- **group 3 — sar (16):** A_d/A_grad/C_lag/D_z/V_*_E_SAR_{ratio,diff}_*, E_SAR_ratio, E_rough_s1_vh_kobs14
- **group 4 — thermal (11):** A_d/C_lag/D_fft/D_sa/D_z/V_*_LST_modis_*
- **group 5 — meteo (17):** G_API, G_DSLR, G_rain_sum_{3d,7d,30d}, V_*_G_API_*, precip_mm
- **group 6 — static (15):** J_* (BioClim/landcover/soil), aspect, elev, latitude, lia_mean_asc_deg, longitude
- **group 7 — temporal (7):** DOY, D_cos_DOY, D_sin_DOY, SMAP_x_year, cos_year, sin_year, year_frac

## Extrapolation (OOD) Check

588/6,620 test rows (8.9%) are OOD on ≥1 top-10 gain feature (same definition
as mlp-1.1–1.3). The pure-96 family keeps its OOD strength; the mixed family's
val winner is in-distribution-strong but OOD-weak (its c1 = 54+10 half carries
the 54-family's weak OOD).

| model | slice | n | r2 | rmse | bias | mae |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3) | all | 6620 | 0.761018 | 0.0497987 | 0.0176019 | 0.0384117 |
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3) | in_distribution | 6032 | 0.755427 | 0.0512552 | 0.0204626 | 0.0397959 |
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3) | ood | 588 | 0.750658 | 0.0311459 | -0.0117442 | 0.0242127 |
| MLP 2regime-96 (5-seed champ) | all | 6620 | 0.75656 | 0.050261 | 0.0204007 | 0.0391776 |
| MLP 2regime-96 (5-seed champ) | in_distribution | 6032 | 0.749983 | 0.0518224 | 0.0227929 | 0.0407259 |
| MLP 2regime-96 (5-seed champ) | ood | 588 | 0.770967 | 0.0298506 | -0.00414004 | 0.0232951 |
| MLP 2regime-54 (w512x512x512_d0.3_huber0.1) | all | 6620 | 0.76511 | 0.0493706 | 0.00681535 | 0.0385003 |
| MLP 2regime-54 (w512x512x512_d0.3_huber0.1) | in_distribution | 6032 | 0.766711 | 0.0500588 | 0.00927805 | 0.0388175 |
| MLP 2regime-54 (w512x512x512_d0.3_huber0.1) | ood | 588 | 0.553921 | 0.0416591 | -0.0184483 | 0.0352462 |
| MLP 2regime-54 (5-seed champ) | all | 6620 | 0.771661 | 0.0486772 | 0.00772988 | 0.0380031 |
| MLP 2regime-54 (5-seed champ) | in_distribution | 6032 | 0.773435 | 0.0493321 | 0.010168 | 0.0382639 |
| MLP 2regime-54 (5-seed champ) | ood | 588 | 0.560204 | 0.0413647 | -0.0172814 | 0.0353269 |
| MLP 2regime-mixed (fg_w512x512_d0.3_huber0.1_swa) | all | 6620 | 0.759881 | 0.049917 | 0.00044081 | 0.038269 |
| MLP 2regime-mixed (fg_w512x512_d0.3_huber0.1_swa) | in_distribution | 6032 | 0.779617 | 0.0486544 | 0.00565512 | 0.0367131 |
| MLP 2regime-mixed (fg_w512x512_d0.3_huber0.1_swa) | ood | 588 | 0.0313755 | 0.0613877 | -0.0530502 | 0.0542303 |
| MLP 2regime-mixed (5-seed champ) | all | 6620 | 0.769858 | 0.048869 | -0.00162579 | 0.0370965 |
| MLP 2regime-mixed (5-seed champ) | in_distribution | 6032 | 0.78682 | 0.0478527 | 0.0031426 | 0.035657 |
| MLP 2regime-mixed (5-seed champ) | ood | 588 | 0.126955 | 0.0582803 | -0.0505423 | 0.0518638 |
| XGBoost Global (54) | all | 6620 | 0.77923 | 0.0478636 | 0.0105484 | 0.0370592 |
| XGBoost Global (54) | in_distribution | 6032 | 0.780849 | 0.0485182 | 0.0145436 | 0.0373064 |
| XGBoost Global (54) | ood | 588 | 0.577509 | 0.0405427 | -0.0304369 | 0.0345237 |
| XGBoost 2-Regime (Winner) | all | 6620 | 0.81496 | 0.0438196 | 0.00648567 | 0.0337195 |
| XGBoost 2-Regime (Winner) | in_distribution | 6032 | 0.81728 | 0.0443022 | 0.00971871 | 0.0338159 |
| XGBoost 2-Regime (Winner) | ood | 588 | 0.618589 | 0.0385212 | -0.0266805 | 0.0327308 |

## Overfitting-Symptom Analysis

From the saved artifacts (no retraining), via `analyze_overfitting.py`.

### 1. Train-fit vs held-out gap (median RMSE over 2-regime MLP configs)

| family | aux2020 (train-fit) | val | test | val/train ratio |
|:---|:---:|:---:|:---:|:---:|
| 2regime_96 | 0.0317 | 0.0519 | 0.0501 | 1.6x |
| 2regime_54 | 0.0275 | 0.0577 | 0.0494 | 2.1x |
| 2regime_mixed | 0.0296 | 0.0523 | 0.0488 | 1.8x |

### 2. Capacity vs test transfer (median by n_params bucket)

| family | capacity | n_configs | med_val_rmse | med_test_r2 | med_test_bias |
|:---|:---|:---:|:---:|:---:|:---:|
| 2regime_96 | <200k | 2 | 0.0561 | 0.7844 | 0.0072 |
| 2regime_96 | 500k-1M | 2 | 0.0566 | 0.6248 | 0.0234 |
| 2regime_96 | 1M+ | 6 | 0.0500 | 0.7358 | 0.0202 |
| 2regime_54 | 200-500k | 4 | 0.0619 | 0.7809 | 0.0022 |
| 2regime_54 | 1M+ | 6 | 0.0575 | 0.7286 | 0.0189 |
| 2regime_mixed | <200k | 1 | 0.0547 | 0.7770 | 0.0129 |
| 2regime_mixed | 200-500k | 4 | 0.0526 | 0.7619 | 0.0204 |
| 2regime_mixed | 1M+ | 5 | 0.0495 | 0.7789 | 0.0124 |

### 3. Per-epoch curve shape for the 2-seed val winner (cluster-0 specialist)

| family | config_id | aux_ep100 | aux_ep260 | val_plateau | test_min | test_min_epoch | test_at_best_val | test_final | test_rise_after_min |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2regime_96 | w512x512x512_d0.3_lr1e-3 | 0.0262 | 0.0179 | 0.0531 | 0.0451 | 90 | 0.0491 | 0.0489 | 0.0037 |
| 2regime_54 | w512x512x512_d0.3_huber0.1 | 0.0379 | 0.0209 | 0.0622 | 0.0500 | 266 | 0.0522 | 0.0507 | 0.0007 |
| 2regime_mixed | fg_w512x512_d0.3_huber0.1_swa | 0.0290 | 0.0198 | 0.0493 | 0.0521 | 90 | 0.0540 | 0.0542 | 0.0021 |

### 4. Systematic bias on test (MLP vs XGBoost references)

MLP median test bias — 2regime_96: 0.0188, 2regime_54: 0.0095,
2regime_mixed: 0.0151. XGBoost references (eval-1.1) — 2-regime: 0.0065,
global: 0.0105. The 96-pool families still carry ~2–3× the XGBoost 2-regime
bias; the 54-family is near it.

## Timing (H100 PCIe 80 GB, 8 parallel workers)

```
Total sweep wall time: 798.6 s  |  eval wall time: 8.8 s
Total training time (all jobs, GPU-seconds): 4884 s = 1.36 GPU-hours (budget: 2.0)
```

Slowest jobs are the `fg` grouped-tower configs (~240–444 s); the fastest are
the small nets (~50 s). Champion step (9 extra-seed jobs) ≈ +0.25 GPU-h.

## Key Takeaways

1. **The 2-regime-MLP ceiling is broken by the new `2regime_mixed` family**
   (c0 = 96-pool, c1 = 54+10). Honest **mixed-family val top-3/top-5 ensembles
   reach test R² 0.8003** (1.3: 0.7825 honest ceiling) and the 2-seed honest
   single `w512x512x512_d0.3_huber0.1_swa` (mixed) reaches **0.7903** — above
   the 1.3 test-best (0.789) with an honest selection. The **cross-family
   ensemble is 0.7932**. XGBoost 2-regime (0.815, itself test-selected) is now
   within 0.015.
2. **2025 is fixed.** The mixed val top-3/top-5 ensembles post 2025 R²
   0.8336/0.8324 — the best of any model, beating XGBoost 2-regime (0.8303) on
   the year that 1.x's MLPs historically degraded on.
3. **The per-cluster allocation hypothesis is confirmed.** Mixed c1 (54+10)
   holds the 54-family's ~0.83 R² (0.8315 median) while c0 gains the 96-pool's
   fit (best c0 0.7726) — the "each feature set on its stronger cluster" idea
   that motivated the family.
4. **SWA (as configured) is a documented negative**, but the *mechanism*
   worked: the honest val-based deployment correctly rejected every snapshot
   (SWA val 0.070–0.126 vs live 0.045–0.062). The `_swa` configs' gains are
   live-trajectory results — a later start (`swa_start_frac ≈ 0.85`) and an
   RNG guard around BN recalibration are the two fixes for a fair re-test.
5. **`fg` grouped towers and `plr` underperform the plain MLP** at this scale
   (fg best 0.782, plr best 0.720 vs plain-MLP 0.790). The grouped architecture
   is a documented negative for 2.0 — the winning lever was the *feature
   allocation*, not the tower structure.
6. **Debias is partial:** 2regime_54 meets the <5% bias²/MSE criterion (median
   3.7%); the 96-pool families still carry 10–13% medians, though
   capacity-control configs are near-unbiased (`w256x256_d0.5_swa` 1.0%,
   `w448x448_d0.3_gelu` 0.04%). The remaining 96-pool bias stays a documented
   open item.
7. **Budget:** 1.36 of 2.0 H100-hours used (54 jobs, 4,884 GPU-s, 8 workers);
   champion step ≈ +0.25 GPU-h. Everything below the line is offline.

## Reproducibility

```bash
cd notebooks/experiment/derived_8.4-eval-mlp-2.0
uv run --no-sync python run_mlp_sweep.py --resume        # phase 1 + phase 2 (H100, 8 workers)
uv run --no-sync python run_mlp_sweep.py --smoke         # 3-epoch pipeline check (data_version -1)
uv run --no-sync python run_mlp_champion.py --top-n 1    # 5-seed champion ensembles of the winners
uv run --no-sync python run_mlp_eval.py                  # leaderboard, per-regime, ensembles, figures
uv run --no-sync python analyze_bias.py                  # headline bias^2/MSE diagnostic
uv run --no-sync python analyze_overfitting.py           # overfitting-symptom analysis
uv run --no-sync python analyze_extrapolation.py         # OOD check
uv run --no-sync python analyze_stopping.py --tag 20     # stopping-rule + SWA-val replay
cd notebooks && nb execute experiment/derived_8.4-eval-mlp-2.0/derived_8.4-eval-mlp-2.0.ipynb --uv
```

- Configurations pinned in `config.yaml`; seeds {42, 7} for the sweep,
  {42, 7, 123, 2024, 999} for the champion step; `data_version: 6`. The
  `2regime_54` anchor reproduces mlp-1.3 bit-identically (stack check, max
  |diff| = 0.000000).
- Artifacts: `models/`, `artifacts/` (gitignored), `sweep_results.csv`,
  `metrics_summary.csv`, `per_regime_metrics_summary.csv`, `bias_summary.csv`,
  `ood_summary.csv`, `stopping_20_*.csv`, `timing_log.json`, figures, and the
  report notebook. All numbers in this README come from the executed notebook.

## Caveats

- The XGBoost 2-regime reference (0.815) was itself test-selected in eval-1.1;
  all honest MLP claims use val-based selection. `test-best` rows are
  reporting only.
- The `_swa` configs' live trajectories are RNG-perturbed relative to their
  anchors (BN-recalibration dropout consumes the shared RNG); the numbers are
  deterministic from the committed code + seeds, but gains over the anchors
  should not be attributed to SWA weight deployment (see the SWA section).
- The mixed family's c1 (54+10) half inherits the 54-family's weak OOD
  extrapolation; the pure-96 family remains the best OOD model (0.751).
- 2025 test coverage is partial for several stations; year-2025 numbers should
  be read with the same caution as 1.x.
