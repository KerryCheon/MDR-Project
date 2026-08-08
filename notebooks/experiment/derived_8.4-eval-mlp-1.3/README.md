# Experiment: `derived_8.4-eval-mlp-1.3` — one more shot at the 2-regime MLP (1.75 H100-hours)

## Objective

Follow-up to `derived_8.4-eval-mlp-1.2` (2-seed val-selected winners 2regime_96 0.761, 2regime_54 0.765; test-best 0.789; XGBoost 2-regime 0.815). 1.2 documented the remaining gap as (1) **systematic positive test bias** (bias² ≈ 10–17% of MSE for the 96-family) and (2) **test error bottoming out early** (~ep 90) while val stays flat. 1.3 gives the current architecture (plain 2-regime MLP) one final shot within a **1.75 H100-hour budget**, attacking those two inefficiencies plus completing 2-seeding of 1.2's strong 1-seed configs and testing train-time regularizers (EMA, mixup, target centering, wd 1e-2, batch 256, huber δ 0.02, and the 54-family's good-capacity region). Two offline analyses of 1.2's saved artifacts (per-cluster calibration, early-stopping-rule replay) were run **first** (zero GPU) and both are documented negatives.

Changes vs 1.2: phase-2 2nd-seed metric switched from the failed aux2020-based robust score to plain **val RMSE**; trainer knobs added behind config keys (defaults = 1.2 behavior); config list = 1.2 anchors + 1-seed completion + new gap-targeting configs.

All numbers below are the stdout of the executed report notebook (`derived_8.4-eval-mlp-1.3.ipynb`). Trained weights, checkpoints, test predictions, and loss curves are archived under `models/`; preprocessed tensors and per-job logs under `artifacts/`; figures under the experiment root.

## Verdict (TL;DR)

- **The plain-2-regime-MLP ceiling is confirmed.** The val-selected winners are **bit-identical to 1.2** (deterministic re-run): 2regime_96 `w512x512x512_d0.3_lr1e-3` test R² 0.761, 2regime_54 `w512x512x512_d0.3_huber0.1` 0.765. The gap to the XGBoost 2-regime winner (0.815) is unchanged (~0.03).
- **Both pre-registered offline fixes were refuted:** val-fit per-cluster calibration does not transfer to test (54-family 0/12 configs helped; medians 0.774→0.746), and no honest early-stopping rule beats patience-60 (plateau rules stop too early).
- **Modest real gains from the 54-family's good-capacity region:** `w448x448_d0.3_gelu` (2-seed test 0.7809, near-zero bias +0.001) and the 54-family **val top-10 avg 0.7825** — both edge past the XGBoost global baseline (0.779), still short of the 2-regime winner (0.815).
- **EMA is a documented trainer-level failure** (inherited decay-0.999-per-step EMA lags the fast-moving head; excluded from selection).
- **Budget:** 0.87 of 1.75 H100-hours used (3,144 GPU-seconds; 8 parallel workers).

## Selection Protocol v4 Diagnostic

1.2 found the aux2020 holdout (2020 ⊂ train) measures *train fit*, not generalization — so it was **dropped as a selection signal**. 1.3 selects by **2-seed mean val RMSE** (phase-2 2nd seed on the val top-10 MLP configs per family, expanded to top-15) and keeps aux2020 as a diagnostic only. Both rankings and their Spearman correlations vs test are reported for auditability.

### 2-Regime-96 — top-10 by val RMSE

| config_id                   | architecture   |   n_seeds |   val_rmse |   aux_rmse |   robust_score |   test_r2 |   test_rmse |   robust_rank |
|:----------------------------|:---------------|----------:|-----------:|-----------:|---------------:|----------:|------------:|--------------:|
| w512x512x512_d0.3_lr1e-3    | mlp            |         2 |  0.0482834 |  0.0249457 |      0.0366146 |  0.761018 |   0.0497987 |             3 |
| w512x512x512_d0.3_huber0.05 | mlp            |         2 |  0.0484297 |  0.0248293 |      0.0366295 |  0.770174 |   0.0488354 |             4 |
| w512x512x512_d0.3_huber0.02 | mlp            |         2 |  0.0486934 |  0.0226742 |      0.0356838 |  0.762261 |   0.049669  |             1 |
| w512x512x512_d0.3_huber0.1  | mlp            |         2 |  0.0491336 |  0.02649   |      0.0378118 |  0.761771 |   0.0497203 |             7 |
| w512x512x512_d0.3_centered  | mlp            |         2 |  0.049351  |  0.0304607 |      0.0399059 |  0.732288 |   0.0527072 |            12 |
| w1024x512x256_d0.3_gelu     | mlp            |         2 |  0.0494225 |  0.0240356 |      0.0367291 |  0.770037 |   0.04885   |             5 |
| w512x512x256_d0.3_huber0.1  | mlp            |         2 |  0.0496386 |  0.0222265 |      0.0359325 |  0.754131 |   0.0505112 |             2 |
| w1024x1024x512_d0.3_gelu    | mlp            |         2 |  0.0502418 |  0.0236676 |      0.0369547 |  0.774762 |   0.0483455 |             6 |
| w512x512x512_d0.3           | mlp            |         2 |  0.0502962 |  0.0302872 |      0.0402917 |  0.761995 |   0.0496969 |            14 |
| w512x512x512_d0.3_wd1e-2    | mlp            |         2 |  0.0503022 |  0.0302793 |      0.0402907 |  0.761677 |   0.0497301 |            13 |

- Spearman(val_rmse, test_r2) = −0.206 (p=0.357, n=22) · Spearman(robust_score, test_r2) = −0.312 (p=0.157, n=22) · Spearman(aux_rmse, test_r2) = −0.354 (p=0.106, n=22)
- val winner (MLP): `w512x512x512_d0.3_lr1e-3` (test_r2=0.7610) | test best (ref): `w256x256_d0.5` (test_r2=0.7834)

### 2-Regime-54 — top-10 by val RMSE

| config_id                  | architecture   |   n_seeds |   val_rmse |   aux_rmse |   robust_score |   test_r2 |   test_rmse |   robust_rank |
|:---------------------------|:---------------|----------:|-----------:|-----------:|---------------:|----------:|------------:|--------------:|
| w512x512x512_d0.3_huber0.1 | mlp            |         2 |  0.0564972 |  0.0232794 |      0.0398883 |  0.76511  |   0.0493706 |             2 |
| w1024x512x256_d0.3_gelu    | mlp            |         2 |  0.0566017 |  0.0221734 |      0.0393875 |  0.773898 |   0.0484382 |             1 |
| w384x384x256_d0.3_gelu     | mlp            |         2 |  0.0568884 |  0.0253214 |      0.0411049 |  0.769538 |   0.048903  |             4 |
| w512x512x256_d0.3_gelu     | mlp            |         2 |  0.057377  |  0.0242083 |      0.0407927 |  0.766228 |   0.0492529 |             3 |
| w384x384_d0.3_huber0.02    | mlp            |         2 |  0.0574505 |  0.0247853 |      0.0411179 |  0.766799 |   0.0491928 |             5 |
| w512x512x512_d0.3          | mlp            |         2 |  0.0582977 |  0.0261998 |      0.0422487 |  0.764522 |   0.0494324 |             6 |
| w512x512x512_d0.3_gelu     | mlp            |         2 |  0.0584996 |  0.0265102 |      0.0425049 |  0.763688 |   0.0495198 |             7 |
| w384x384_d0.3_huber0.1     | mlp            |         2 |  0.0592377 |  0.0298896 |      0.0445637 |  0.774155 |   0.0484107 |             8 |
| w256x256_d0.3_huber0.1     | mlp            |         2 |  0.0604845 |  0.0287274 |      0.044606  |  0.774056 |   0.0484213 |             9 |
| w384x384_d0.3_gelu         | mlp            |         2 |  0.061132  |  0.0310517 |      0.0460919 |  0.788821 |   0.0468125 |            11 |

- Spearman(val_rmse, test_r2) = +0.150 (p=0.579, n=16)
- val winner (MLP): `w512x512x512_d0.3_huber0.1` (test_r2=0.7651) | test best (ref): `w384x384_d0.3_gelu` (test_r2=0.7888)

## Overall Leaderboard (2023–2025 Test Set)

`test-best` rows are reported **for reference only** (selection on test would be leakage); the XGBoost 2-regime reference itself was test-selected in eval-1.1. `(val top-k avg)` rows are offline seed-averaged ensembles of the top-k val-selected MLP configs (no extra training). `MLP-1.2` rows are the 1.2 references.

| model_name                                                | strategy_name          |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:----------------------------------------------------------|:-----------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)                | XGBoost_Reference      |    0.81496  |     0.0438196 |       0.043337  |   0.00648567  |    0.0337195 |         0.905594 |
| MLP 2-Regime-54 (test-best, w384x384_d0.3_gelu)           | MLP_testbest_reference |    0.788821 |     0.0468125 |       0.0467953 |   0.00126699  |    0.0362252 |         0.888558 |
| MLP-1.2 2-Regime-54 (test_best: w384x384_d0.3_gelu)       | MLP_1.2_Reference      |    0.788821 |     0.0468125 |       0.0467953 |   0.00126699  |    0.0362252 |         0.888558 |
| MLP 2-Regime-54 (test-best, w384x384_d0.3)                | MLP_testbest_reference |    0.78409  |     0.0473339 |       0.0469986 |   0.00562395  |    0.0371044 |         0.88722  |
| MLP-1.2 2-Regime-96 (test_best: w512x512x512_d0.4)        | MLP_1.2_Reference      |    0.783883 |     0.0473565 |       0.0451069 |   0.0144227   |    0.0367854 |         0.897946 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)                | MLP_testbest_reference |    0.783404 |     0.047409  |       0.0471517 |   0.00493258  |    0.0360624 |         0.887208 |
| MLP 2-Regime-54 (val top-10 avg)                          | MLP_2regime_54         |    0.782533 |     0.0475043 |       0.0470162 |   0.0067925   |    0.0369583 |         0.888139 |
| MLP 2-Regime-54 (test-best, w448x448_d0.3_gelu)           | MLP_testbest_reference |    0.780859 |     0.0476868 |       0.0476765 |   0.000990915 |    0.0366186 |         0.883737 |
| Global Single Model (54 Backbone)                         | XGBoost_Reference      |    0.77923  |     0.0478636 |       0.0466868 |   0.0105484   |    0.0370592 |         0.889432 |
| MLP 2-Regime-96 (test-best, w512x512x512_d0.3_gelu)       | MLP_testbest_reference |    0.777923 |     0.0480052 |       0.045269  |   0.0159755   |    0.0369576 |         0.896341 |
| MLP 2-Regime-54 (val top-5 avg)                           | MLP_2regime_54         |    0.777435 |     0.0480578 |       0.0472973 |   0.00851601  |    0.0372733 |         0.887438 |
| MLP 2-Regime-96 (test-best, w1024x1024x512_d0.3_gelu)     | MLP_testbest_reference |    0.774762 |     0.0483455 |       0.0461497 |   0.0144047   |    0.036901  |         0.893628 |
| MLP 2-Regime-54 (val top-3 avg)                           | MLP_2regime_54         |    0.774269 |     0.0483985 |       0.0478742 |   0.00710417  |    0.0374487 |         0.885478 |
| MLP 2-Regime-54 (w1024x512x256_d0.3_gelu)                 | MLP_2regime_54         |    0.773898 |     0.0484382 |       0.0478423 |   0.00757439  |    0.0372673 |         0.884533 |
| MLP 2-Regime-96 (val top-10 avg)                          | MLP_2regime_96         |    0.772329 |     0.048606  |       0.0446198 |   0.0192772   |    0.0375105 |         0.899578 |
| MLP 2-Regime-96 (w512x512x512_d0.3_huber0.05)             | MLP_2regime_96         |    0.770174 |     0.0488354 |       0.0457126 |   0.0171832   |    0.0374011 |         0.895721 |
| MLP 2-Regime-54 (w384x384x256_d0.3_gelu)                  | MLP_2regime_54         |    0.769538 |     0.048903  |       0.0484105 |   0.00692277  |    0.0379482 |         0.882997 |
| MLP 2-Regime-96 (val top-5 avg)                           | MLP_2regime_96         |    0.769487 |     0.0489084 |       0.0446924 |   0.019865    |    0.0379116 |         0.898928 |
| MLP 2-Regime-96 (val top-3 avg)                           | MLP_2regime_96         |    0.767136 |     0.0491571 |       0.0459086 |   0.0175734   |    0.0376345 |         0.894247 |
| MLP 2-Regime-54 (w384x384_d0.3_huber0.02)                 | MLP_2regime_54         |    0.766799 |     0.0491928 |       0.0489655 |   0.0153431   |    0.0388109 |         0.88515  |
| MLP 2-Regime-54 (w512x512x256_d0.3_gelu)                  | MLP_2regime_54         |    0.766228 |     0.0492529 |       0.0488953 |   0.00592441  |    0.0381331 |         0.880968 |
| MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)              | MLP_2regime_54         |    0.76511  |     0.0493706 |       0.0488979 |   0.00681535  |    0.0385003 |         0.882441 |
| MLP-1.2 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1) | MLP_1.2_Reference      |    0.76511  |     0.0493706 |       0.0488979 |   0.00681535  |    0.0385003 |         0.882441 |
| MLP 2-Regime-96 (w512x512x512_d0.3_huber0.02)             | MLP_2regime_96         |    0.762261 |     0.049669  |       0.0453999 |   0.0179351   |    0.0379238 |         0.897273 |
| MLP 2-Regime-96 (w512x512x512_d0.3_huber0.1)              | MLP_2regime_96         |    0.761771 |     0.0497203 |       0.0452198 |   0.0206706   |    0.0386133 |         0.89771  |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                | MLP_2regime_96         |    0.761018 |     0.0497987 |       0.0465842 |   0.0176019   |    0.0384117 |         0.890751 |
| MLP-1.2 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)   | MLP_1.2_Reference      |    0.761018 |     0.0497987 |       0.0465842 |   0.0176019   |    0.0384117 |         0.890751 |
| MLP 2-Regime-96 (w512x512x512_d0.3_centered)              | MLP_2regime_96         |    0.732288 |     0.0527072 |       0.0459118 |   0.0259344   |    0.042467  |         0.889593 |

## Hyperparameter Sweep Summary (val RMSE ranking)

34 curated configs in the 2-regime families with 8 parallel H100 workers; 31/38 configs are 2-seeded (phase 2 = val top-15/family). Residual/FT are NOT re-run (documented failures in 1.2).

### Sweep Top-10 — 2-Regime-96

| config_id                   |   n_seeds |   dropout |     lr | loss   |   val_rmse |   aux_rmse |   robust_score |   test_r2 |   test_rmse |   best_epoch |   train_time_s |
|:----------------------------|----------:|----------:|-------:|:-------|-----------:|-----------:|---------------:|----------:|------------:|-------------:|---------------:|
| w512x512x512_d0.3_lr1e-3    |         2 |       0.3 | 0.001  | mse    |  0.0482834 |  0.0249457 |      0.0366146 |  0.761018 |   0.0497987 |          263 |        95.435  |
| w512x512x512_d0.3_huber0.05 |         2 |       0.3 | 0.0003 | huber  |  0.0484297 |  0.0248293 |      0.0366295 |  0.770174 |   0.0488354 |          260 |        97.8043 |
| w512x512x512_d0.3_huber0.02 |         2 |       0.3 | 0.0003 | huber  |  0.0486934 |  0.0226742 |      0.0356838 |  0.762261 |   0.049669  |          220 |        86.4382 |
| w512x512x512_d0.3_huber0.1  |         2 |       0.3 | 0.0003 | huber  |  0.0491336 |  0.02649   |      0.0378118 |  0.761771 |   0.0497203 |          260 |        99.2696 |
| w512x512x512_d0.3_centered  |         2 |       0.3 | 0.0003 | mse    |  0.049351  |  0.0304607 |      0.0399059 |  0.732288 |   0.0527072 |           60 |        37.6738 |
| w1024x512x256_d0.3_gelu     |         2 |       0.3 | 0.0003 | mse    |  0.0494225 |  0.0240356 |      0.0367291 |  0.770037 |   0.04885   |          331 |       103.29   |
| w512x512x256_d0.3_huber0.1  |         2 |       0.3 | 0.0003 | huber  |  0.0496386 |  0.0222265 |      0.0359325 |  0.754131 |   0.0505112 |          290 |       101.193  |
| w1024x1024x512_d0.3_gelu    |         2 |       0.3 | 0.0003 | mse    |  0.0502418 |  0.0236676 |      0.0369547 |  0.774762 |   0.0483455 |          386 |       110.499  |
| w512x512x512_d0.3           |         2 |       0.3 | 0.0003 | mse    |  0.0502962 |  0.0302872 |      0.0402917 |  0.761995 |   0.0496969 |          301 |       107.783  |
| w512x512x512_d0.3_wd1e-2    |         2 |       0.3 | 0.0003 | mse    |  0.0503022 |  0.0302793 |      0.0402907 |  0.761677 |   0.0497301 |          301 |       106.364  |

### Sweep Top-10 — 2-Regime-54

| config_id                  |   n_seeds |   dropout |     lr | loss   |   val_rmse |   aux_rmse |   robust_score |   test_r2 |   test_rmse |   best_epoch |   train_time_s |
|:---------------------------|----------:|----------:|-------:|:-------|-----------:|-----------:|---------------:|----------:|------------:|-------------:|---------------:|
| w512x512x512_d0.3_huber0.1 |         2 |       0.3 | 0.0003 | huber  |  0.0564972 |  0.0232794 |      0.0398883 |  0.76511  |   0.0493706 |          345 |       118.477  |
| w1024x512x256_d0.3_gelu    |         2 |       0.3 | 0.0003 | mse    |  0.0566017 |  0.0221734 |      0.0393875 |  0.773898 |   0.0484382 |          320 |       117.648  |
| w384x384x256_d0.3_gelu     |         2 |       0.3 | 0.0003 | mse    |  0.0568884 |  0.0253214 |      0.0411049 |  0.769538 |   0.048903  |          393 |       106.041  |
| w512x512x256_d0.3_gelu     |         2 |       0.3 | 0.0003 | mse    |  0.057377  |  0.0242083 |      0.0407927 |  0.766228 |   0.0492529 |          310 |       117.362  |
| w384x384_d0.3_huber0.02    |         2 |       0.3 | 0.0003 | huber  |  0.0574505 |  0.0247853 |      0.0411179 |  0.766799 |   0.0491928 |          240 |        80.8929 |
| w512x512x512_d0.3          |         2 |       0.3 | 0.0003 | mse    |  0.0582977 |  0.0261998 |      0.0422487 |  0.764522 |   0.0494324 |          345 |       126.221  |
| w512x512x512_d0.3_gelu     |         2 |       0.3 | 0.0003 | mse    |  0.0584996 |  0.0265102 |      0.0425049 |  0.763688 |   0.0495198 |          306 |       106.308  |
| w384x384_d0.3_huber0.1     |         2 |       0.3 | 0.0003 | huber  |  0.0592377 |  0.0298896 |      0.0445637 |  0.774155 |   0.0484107 |          240 |        77.2921 |
| w256x256_d0.3_huber0.1     |         2 |       0.3 | 0.0003 | huber  |  0.0604845 |  0.0287274 |      0.044606  |  0.774056 |   0.0484213 |          357 |        88.1292 |
| w384x384_d0.3_gelu         |         2 |       0.3 | 0.0003 | mse    |  0.061132  |  0.0310517 |      0.0460919 |  0.788821 |   0.0468125 |          256 |       103.821  |

## Per-Regime Performance Breakdown

Cluster 0 holds 73% of the test rows, so it dominates the pooled R².

| strategy_name     | model_name                                    |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |        bias |       mae |
|:------------------|:----------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|------------:|----------:|
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)    |         0 |      7156 |     4817 | 0.754287 | 0.0495899 | 0.0472413 | 0.0150802   | 0.0389389 |
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)    |         1 |      2647 |     1803 | 0.776352 | 0.0503523 | 0.0440792 | 0.0243389   | 0.0370033 |
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_huber0.05) |         0 |      7156 |     4817 | 0.770502 | 0.0479257 | 0.0459775 | 0.0135256   | 0.0370535 |
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_huber0.05) |         1 |      2647 |     1803 | 0.768879 | 0.0511867 | 0.0435144 | 0.026955    | 0.0383297 |
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_huber0.02) |         0 |      7156 |     4817 | 0.771203 | 0.0478525 | 0.0460305 | 0.0130785   | 0.0366999 |
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_huber0.02) |         1 |      2647 |     1803 | 0.74063  | 0.0542247 | 0.044552  | 0.0309101   | 0.0411947 |
| MLP_2regime_54    | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)  |         0 |      7156 |     4817 | 0.736751 | 0.051329  | 0.0510691 | 0.0051591   | 0.0408763 |
| MLP_2regime_54    | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)  |         1 |      2647 |     1803 | 0.831465 | 0.0437101 | 0.0422401 | 0.0112403   | 0.0321524 |
| MLP_2regime_54    | MLP 2-Regime-54 (w1024x512x256_d0.3_gelu)     |         0 |      7156 |     4817 | 0.748627 | 0.0501578 | 0.0497549 | 0.00634486  | 0.0393056 |
| MLP_2regime_54    | MLP 2-Regime-54 (w1024x512x256_d0.3_gelu)     |         1 |      2647 |     1803 | 0.832991 | 0.0435118 | 0.0421349 | 0.0108593   | 0.0318214 |
| MLP_2regime_54    | MLP 2-Regime-54 (w384x384x256_d0.3_gelu)      |         0 |      7156 |     4817 | 0.742484 | 0.050767  | 0.0503142 | 0.00676529  | 0.040319  |
| MLP_2regime_54    | MLP 2-Regime-54 (w384x384x256_d0.3_gelu)      |         1 |      2647 |     1803 | 0.832826 | 0.0435333 | 0.0429095 | 0.00734352  | 0.0316141 |
| XGBoost_Reference | Global Single Model (54 Backbone)             |         0 |     14608 |     6620 | 0.77923  | 0.0478636 | 0.0466868 | 0.0105484   | 0.0370592 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10)    |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 | 0.00861491  | 0.0359221 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10)    |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 | 0.000797068 | 0.0278349 |
| MLP_1.2_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)    |         0 |      7156 |     4817 | 0.754287 | 0.0495899 | 0.0472413 | 0.0150802   | 0.0389389 |
| MLP_1.2_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)    |         1 |      2647 |     1803 | 0.776352 | 0.0503523 | 0.0440792 | 0.0243389   | 0.0370033 |
| MLP_1.2_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)  |         0 |      7156 |     4817 | 0.736751 | 0.051329  | 0.0510691 | 0.0051591   | 0.0408763 |
| MLP_1.2_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)  |         1 |      2647 |     1803 | 0.831465 | 0.0437101 | 0.0422401 | 0.0112403   | 0.0321524 |

## Year-by-Year R² Breakdown

| model_name                                                |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:----------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)                |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| MLP 2-Regime-54 (test-best, w384x384_d0.3_gelu)           |    0.788821 |       0.773579 |       0.818284 |       0.770357 |
| MLP-1.2 2-Regime-54 (test_best: w384x384_d0.3_gelu)       |    0.788821 |       0.773579 |       0.818284 |       0.770357 |
| MLP 2-Regime-54 (test-best, w384x384_d0.3)                |    0.78409  |       0.755911 |       0.818958 |       0.775181 |
| MLP-1.2 2-Regime-96 (test_best: w512x512x512_d0.4)        |    0.783883 |       0.747432 |       0.826116 |       0.777354 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)                |    0.783404 |       0.763386 |       0.82079  |       0.762541 |
| MLP 2-Regime-54 (val top-10 avg)                          |    0.782533 |       0.749643 |       0.803851 |       0.792293 |
| MLP 2-Regime-54 (test-best, w448x448_d0.3_gelu)           |    0.780859 |       0.783872 |       0.788164 |       0.762631 |
| Global Single Model (54 Backbone)                         |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| MLP 2-Regime-96 (test-best, w512x512x512_d0.3_gelu)       |    0.777923 |       0.741591 |       0.798701 |       0.792043 |
| MLP 2-Regime-54 (val top-5 avg)                           |    0.777435 |       0.736147 |       0.799199 |       0.796323 |
| MLP 2-Regime-96 (test-best, w1024x1024x512_d0.3_gelu)     |    0.774762 |       0.730862 |       0.784069 |       0.808778 |
| MLP 2-Regime-54 (val top-3 avg)                           |    0.774269 |       0.736777 |       0.788373 |       0.796139 |
| MLP 2-Regime-54 (w1024x512x256_d0.3_gelu)                 |    0.773898 |       0.738942 |       0.792343 |       0.788568 |
| MLP 2-Regime-96 (val top-10 avg)                          |    0.772329 |       0.72545  |       0.797894 |       0.793806 |
| MLP 2-Regime-96 (w512x512x512_d0.3_huber0.05)             |    0.770174 |       0.722288 |       0.791691 |       0.796706 |
| MLP 2-Regime-54 (w384x384x256_d0.3_gelu)                  |    0.769538 |       0.734862 |       0.780887 |       0.790675 |
| MLP 2-Regime-96 (val top-5 avg)                           |    0.769487 |       0.721999 |       0.7949   |       0.791721 |
| MLP 2-Regime-96 (val top-3 avg)                           |    0.767136 |       0.715613 |       0.790294 |       0.796175 |
| MLP 2-Regime-54 (w384x384_d0.3_huber0.02)                 |    0.766799 |       0.693432 |       0.823182 |       0.788648 |
| MLP 2-Regime-54 (w512x512x256_d0.3_gelu)                  |    0.766228 |       0.744572 |       0.776491 |       0.773213 |
| MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)              |    0.76511  |       0.723072 |       0.775216 |       0.79585  |
| MLP-1.2 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1) |    0.76511  |       0.723072 |       0.775216 |       0.79585  |
| MLP 2-Regime-96 (w512x512x512_d0.3_huber0.02)             |    0.762261 |       0.712124 |       0.782611 |       0.792272 |
| MLP 2-Regime-96 (w512x512x512_d0.3_huber0.1)              |    0.761771 |       0.711239 |       0.788669 |       0.785812 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |    0.761018 |       0.706877 |       0.786985 |       0.790134 |
| MLP-1.2 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)   |    0.761018 |       0.706877 |       0.786985 |       0.790134 |
| MLP 2-Regime-96 (w512x512x512_d0.3_centered)              |    0.732288 |       0.701949 |       0.763594 |       0.727575 |

## Offline Analysis of 1.2 — Per-Cluster Calibration (documented negative)

1.2 found the 2-regime MLPs carry a systematic positive test bias (bias² ≈ 10–17% of MSE). The obvious fix — fit a per-cluster (and global) affine map `y' = a·y + b` on the **val** predictions of every saved 1.2 model and apply it to test — **does not transfer**: calibrated test R² is worse for nearly every config and the test bias grows. Raw predictions stand; `run_mlp_eval.py` only reports calibrated rows when they beat raw (in practice none do).

### 2-Regime-96 — top-8 by raw R²

| config_id                   |   n_seeds |   raw_r2 |   cal_pc_r2 |   cal_g_r2 |   raw_rmse |   cal_pc_rmse |   raw_bias |   cal_pc_bias |
|:----------------------------|----------:|---------:|------------:|-----------:|-----------:|--------------:|-----------:|--------------:|
| w512x512x512_d0.4           |         1 | 0.783883 |    0.758638 |   0.75857  |  0.0473565 |     0.0500461 | 0.0144227  |     0.0222451 |
| w256x256_d0.5               |         1 | 0.783404 |    0.727816 |   0.718496 |  0.047409  |     0.0531456 | 0.00493258 |     0.0264161 |
| w512x512x512_d0.3_gelu      |         2 | 0.777923 |    0.757836 |   0.758278 |  0.0480052 |     0.0501292 | 0.0159755  |     0.0217491 |
| w1024x1024x512_d0.3_gelu    |         2 | 0.774762 |    0.761033 |   0.76234  |  0.0483455 |     0.0497972 | 0.0144047  |     0.0194923 |
| w512x512_d0.35_gelu         |         1 | 0.772616 |    0.729467 |   0.729677 |  0.0485754 |     0.0529842 | 0.0184685  |     0.0282447 |
| w512x512x512_d0.3_huber0.05 |         2 | 0.770174 |    0.763554 |   0.762832 |  0.0488354 |     0.0495338 | 0.0171832  |     0.0204169 |
| w1024x512x256_d0.3_gelu     |         2 | 0.770037 |    0.758969 |   0.758461 |  0.04885   |     0.0500118 | 0.0150089  |     0.0191207 |
| w256x256x256_d0.3_gelu      |         2 | 0.769784 |    0.754578 |   0.754589 |  0.0488769 |     0.0504653 | 0.0145621  |     0.0192221 |

configs where per-cluster calibration HELPED on test: **5/43** · median raw_r2 0.7541 → cal_pc_r2 0.7351

### 2-Regime-54 — top-8 by raw R²

| config_id               |   n_seeds |   raw_r2 |   cal_pc_r2 |   cal_g_r2 |   raw_rmse |   cal_pc_rmse |    raw_bias |   cal_pc_bias |
|:------------------------|----------:|---------:|------------:|-----------:|-----------:|--------------:|------------:|--------------:|
| w384x384_d0.3_gelu      |         2 | 0.788821 |    0.75117  |   0.750309 |  0.0468125 |     0.0508144 |  0.00126699 |     0.0201202 |
| w384x384_d0.3           |         2 | 0.78409  |    0.732802 |   0.730947 |  0.0473339 |     0.0526565 |  0.00562395 |     0.0240768 |
| w512x512_d0.3_gelu      |         2 | 0.778515 |    0.742887 |   0.738369 |  0.0479411 |     0.0516533 | -0.00103458 |     0.0203462 |
| w384x384_d0.3_huber0.1  |         2 | 0.774155 |    0.732557 |   0.729407 |  0.0484107 |     0.0526806 |  0.0125825  |     0.0248351 |
| w256x256_d0.3_huber0.1  |         2 | 0.774056 |    0.741537 |   0.735597 |  0.0484213 |     0.0517887 |  0.0060214  |     0.0210429 |
| w1024x512x256_d0.3_gelu |         2 | 0.773898 |    0.758144 |   0.75791  |  0.0484382 |     0.0500973 |  0.00757439 |     0.0157877 |
| w256x256_d0.3_gelu      |         2 | 0.773816 |    0.753044 |   0.74825  |  0.048447  |     0.0506227 | -0.00596485 |     0.017245  |
| w512x512x256_d0.3_gelu  |         2 | 0.766228 |    0.75193  |   0.751192 |  0.0492529 |     0.0507368 |  0.00592441 |     0.0155435 |

configs where per-cluster calibration HELPED on test: **0/12** · median raw_r2 0.7739 → cal_pc_r2 0.7456

The same negative was confirmed on the 1.3 sweep artifacts (`calibration_13_summary.csv`): e.g., 2regime_54 `w448x448_d0.3_gelu` raw 0.7809 → calibrated 0.7535.

## Offline Analysis of 1.2 — Early-Stopping-Rule Replay (patience-60 confirmed)

1.2's winner's test error bottomed out at ~ep 90 while val stayed flat to ep 260. We replayed alternative **honest** epoch-selection rules on every saved 1.2 per-epoch curve (val/aux/test): patience-60 (baseline), val+aux joint minimum, and first-sustained-plateau variants. **Finding: no honest rule beats patience-60** — the plateau rules stop too early (undertrained), and val+aux helps the 54-family slightly but hurts the 96-family. The oracle (argmin test) bound shows ~0.004 RMSE of headroom that is not honestly reachable. The 1.3 trainer therefore keeps patience-60.

### 2-Regime-96

| rule             |   mean_test_rmse |   median_test_rmse |   n |
|:-----------------|-----------------:|-------------------:|----:|
| oracle           |        0.0476222 |          0.0477377 |  69 |
| patience60       |        0.0520159 |          0.0519324 |  69 |
| patience40       |        0.0520159 |          0.0519324 |  69 |
| patience20       |        0.0520159 |          0.0519324 |  69 |
| val_aux          |        0.0537652 |          0.0528206 |  69 |
| plateau_w60e1e-4 |        0.0555998 |          0.0547605 |  69 |
| plateau_w40e3e-4 |        0.060673  |          0.0598781 |  69 |
| plateau_w40e1e-4 |        0.060673  |          0.0598781 |  69 |
| plateau_w20e1e-4 |        0.0674415 |          0.0677908 |  69 |

### 2-Regime-54

| rule             |   mean_test_rmse |   median_test_rmse |   n |
|:-----------------|-----------------:|-------------------:|----:|
| oracle           |        0.0483177 |          0.0482512 |  24 |
| val_aux          |        0.0492256 |          0.0491575 |  24 |
| patience20       |        0.0493439 |          0.0496086 |  24 |
| patience60       |        0.0493439 |          0.0496086 |  24 |
| patience40       |        0.0493439 |          0.0496086 |  24 |
| plateau_w60e1e-4 |        0.0571079 |          0.0571608 |  24 |
| plateau_w40e3e-4 |        0.0638651 |          0.0639195 |  24 |
| plateau_w40e1e-4 |        0.0638651 |          0.0639195 |  24 |
| plateau_w20e1e-4 |        0.071378  |          0.0716148 |  24 |

The replay on the 1.3 curves (`stopping_13_aggregate.csv`) confirms: patience-60 median test RMSE 0.0508 (96) / 0.0494 (54); every plateau rule is worse.

## New Trainer Knobs (1.3) — EMA / mixup / target-centering / wd / bs

| family     | config_id                       |   n_seeds |   val_rmse |   aux_rmse |     test_r2 |   test_rmse |   test_bias |   best_epoch |
|:-----------|:--------------------------------|----------:|-----------:|-----------:|------------:|------------:|------------:|-------------:|
| 2regime_54 | w384x384_d0.3_huber0.02         |         2 |  0.0574505 |  0.0247853 |    0.766799 |   0.0491928 |  0.0153431  |          240 |
| 2regime_54 | w384x384_d0.3_mixup0.2          |         2 |  0.06208   |  0.035957  |    0.778403 |   0.0479532 |  0.00454811 |          301 |
| 2regime_54 | w384x384_d0.3_gelu_ema          |         1 |  0.184839  |  0.240567  | -163.444    |   1.3063    |  1.27094    |            2 |
| 2regime_54 | w512x512x512_d0.3_huber0.1_ema  |         2 |  0.197799  |  0.190174  |  -45.173    |   0.692197  |  0.664229   |           20 |
| 2regime_96 | w512x512x512_d0.3_huber0.02     |         2 |  0.0486934 |  0.0226742 |    0.762261 |   0.049669  |  0.0179351  |          220 |
| 2regime_96 | w512x512x512_d0.3_centered      |         2 |  0.049351  |  0.0304607 |    0.732288 |   0.0527072 |  0.0259344  |           60 |
| 2regime_96 | w512x512x512_d0.3_wd1e-2        |         2 |  0.0503022 |  0.0302793 |    0.761677 |   0.0497301 |  0.0210488  |          301 |
| 2regime_96 | w512x512_d0.3_gelu_bs256        |         2 |  0.0505512 |  0.0279411 |    0.692174 |   0.0565182 |  0.0319373  |          218 |
| 2regime_96 | w512x512x512_d0.3_mixup0.2      |         2 |  0.0505851 |  0.0309629 |    0.761063 |   0.0497941 |  0.0191002  |          231 |
| 2regime_96 | w512x512x512_d0.3_huber0.05_ema |         1 |  0.150789  |  0.140795  |  -28.6682   |   0.554857  |  0.528373   |           49 |
| 2regime_96 | w512x512x512_d0.3_lr1e-3_ema    |         1 |  0.160122  |  0.148066  |  -23.1512   |   0.500616  |  0.461101   |           21 |
| 2regime_96 | w1024x512x256_d0.3_gelu_ema     |         1 |  0.168236  |  0.175942  |  -55.8832   |   0.768295  |  0.73744    |            3 |

**EMA diagnosis (documented trainer-level failure):** the inherited EMA (decay 0.999 per optimizer step, ~14 steps/epoch) lags ~70 epochs behind the fast-moving head layer; with early stopping at ep 21–60 the EMA never converges. Isolated empirically: swapping only the head layer (`net.12`) for its EMA values turns val RMSE 0.057 → 0.558, i.e. the EMA head norm is ~6.6× the live head norm. EMA configs are excluded from the winner pool by the val ranking (val_rmse ≈ 0.15–0.20).

## Extrapolation (OOD) Check

Test rows whose top-10 gain features fall outside the trainval [min, max] range are flagged OOD (same definition as mlp-1.1/1.2) — 588/6,620 rows (8.9%).

| model                                       | slice           |    n |       r2 |      rmse |        bias |       mae |
|:--------------------------------------------|:----------------|-----:|---------:|----------:|------------:|----------:|
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3)   | all             | 6620 | 0.761018 | 0.0497987 |  0.0176019  | 0.0384117 |
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3)   | in_distribution | 6032 | 0.755427 | 0.0512552 |  0.0204626  | 0.0397959 |
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3)   | ood             |  588 | 0.750658 | 0.0311459 | -0.0117442  | 0.0242127 |
| MLP 2regime-54 (w512x512x512_d0.3_huber0.1) | all             | 6620 | 0.76511  | 0.0493706 |  0.00681535 | 0.0385003 |
| MLP 2regime-54 (w512x512x512_d0.3_huber0.1) | in_distribution | 6032 | 0.766711 | 0.0500588 |  0.00927805 | 0.0388175 |
| MLP 2regime-54 (w512x512x512_d0.3_huber0.1) | ood             |  588 | 0.553921 | 0.0416591 | -0.0184483  | 0.0352462 |
| XGBoost Global (54)                         | all             | 6620 | 0.77923  | 0.0478636 |  0.0105484  | 0.0370592 |
| XGBoost Global (54)                         | in_distribution | 6032 | 0.780849 | 0.0485182 |  0.0145436  | 0.0373064 |
| XGBoost Global (54)                         | ood             |  588 | 0.577509 | 0.0405427 | -0.0304369  | 0.0345237 |
| XGBoost 2-Regime (Winner)                   | all             | 6620 | 0.81496  | 0.0438196 |  0.00648567 | 0.0337195 |
| XGBoost 2-Regime (Winner)                   | in_distribution | 6032 | 0.81728  | 0.0443022 |  0.00971871 | 0.0338159 |
| XGBoost 2-Regime (Winner)                   | ood             |  588 | 0.618589 | 0.0385212 | -0.0266805  | 0.0327308 |

## Overfitting-Symptom Analysis

Quantifies the generalization failure modes of the 1.3 sweep from the saved artifacts (no retraining): (1) train-fit vs held-out gap; (2) capacity vs transfer by n_params bucket; (3) per-epoch curve shape of each family's 2-seed val winner; (4) systematic bias on test vs XGBoost. Backed by `analyze_overfitting.py` (see also `overfitting_analysis.md` from 1.2 for the full mechanism).

### 1. Train-fit vs held-out gap (median RMSE over 2-regime MLP configs)

| family     |   aux2020 (train-fit) |   val |   test |   val/train ratio |
|:-----------|----------------------:|------:|-------:|------------------:|
| 2regime_96 | 0.0295 | 0.0505 | 0.0497 | 1.7x |
| 2regime_54 | 0.0293 | 0.0599 | 0.0487 | 2.0x |

### 2. Capacity vs test transfer (median by n_params bucket)

| family     | capacity   |   n_configs |   med_val_rmse |   med_test_r2 |   med_test_bias |
|:-----------|:-----------|------------:|---------------:|--------------:|----------------:|
| 2regime_96 | <200k | 1 | 0.0569 | 0.7834 | 0.0049 |
| 2regime_96 | 200-500k | 1 | 0.0509 | 0.7698 | 0.0146 |
| 2regime_96 | 500k-1M | 5 | 0.0506 | 0.7541 | 0.0185 |
| 2regime_96 | 1M+ | 14 | 0.0503 | 0.7617 | 0.0203 |
| 2regime_54 | <200k | 1 | 0.0605 | 0.7741 | 0.0060 |
| 2regime_54 | 200-500k | 7 | 0.0614 | 0.7784 | 0.0056 |
| 2regime_54 | 500k-1M | 3 | 0.0574 | 0.7695 | 0.0059 |
| 2regime_54 | 1M+ | 5 | 0.0583 | 0.7645 | 0.0068 |

### 3. Per-epoch curve shape for the 2-seed val winner (cluster-0 specialist)

| family     | config_id |   aux_ep100 |   aux_ep260 |   val_plateau |   test_min |   test_min_epoch |   test_at_best_val |   test_final |   test_rise_after_min |
|:-----------|:----------|------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|-------------:|----------------------:|
| 2regime_96 | w512x512x512_d0.3_lr1e-3 | 0.0262 | 0.0179 | 0.0531 | 0.0451 | 90 | 0.0491 | 0.0489 | 0.0037 |
| 2regime_54 | w512x512x512_d0.3_huber0.1 | 0.0379 | 0.0209 | 0.0622 | 0.0500 | 266 | 0.0522 | 0.0507 | 0.0007 |

### 4. Systematic bias on test (MLP vs XGBoost references)

MLP median test bias — 2regime_96: 0.0188, 2regime_54: 0.0060. XGBoost references (eval-1.1) — 2-regime: 0.0065, global: 0.0105. The 54-family (matching the XGBoost winner's feature structure) stays near-unbiased; the 96-family's ~3× bias (bias² ≈ 10–17% of MSE) persists and is **not** removable by val-fit calibration (see above).

## Timing (H100 PCIe 80 GB, 8 parallel workers)

```
Total sweep wall time (final invocation): 162.3 s  |  eval wall time: 5.8 s
Total training time (all jobs, GPU-seconds): 3144 s = 0.87 GPU-hours (budget: 1.75)
```

Fastest jobs are the broken-EMA configs (early-stopped at ep 2–21) and small nets; the slowest are the 2-seed big 2-regime MLPs (~110–126 s).

## Key Takeaways

1. **The val-selected winners are bit-identical to 1.2** (deterministic seeds): 2regime_96 `w512x512x512_d0.3_lr1e-3` 0.7610, 2regime_54 `w512x512x512_d0.3_huber0.1` 0.7651 — the anchors reproduce 1.2 exactly under the v5 protocol. The gap to the XGBoost 2-regime winner (0.815) is unchanged (~0.03), so the **plain-2-regime-MLP ceiling is confirmed**.
2. **The two offline fixes were refuted before spending GPU.** Per-cluster affine calibration fit on val does not transfer to test (calibrated R² is worse for 54-family 0/12 configs and 43/48 of 96-family; medians 0.774→0.746 / 0.754→0.735). No honest early-stopping rule beats patience-60 (plateau rules stop too early; val+aux helps 54 slightly but hurts 96). Both are documented negatives.
3. **EMA is a documented trainer-level failure**: the inherited decay-0.999-per-step EMA lags ~70 epochs behind the fast-moving head layer given ~14 steps/epoch, so EMA-evaluated val never falls below ~0.15 and test R² is catastrophic (−20 to −160). Excluded from selection; root cause isolated (the EMA head alone causes the blowup).
4. **What did move the needle (modestly):** the 54-family's good-capacity region — `w448x448_d0.3_gelu` (2-seed test R² 0.7809, val rank 11, near-zero bias +0.001) and `w384x384x256_d0.3_gelu` (0.7695, val rank 3) beat the 54 val winner on test; the 54-family **val top-10 avg 0.7825** edges past the XGBoost global baseline (0.779). mixup helped the 54 family (`w384x384_d0.3_mixup0.2` 0.7784, 2-seed); target centering (0.732) and batch 256 (0.692) hurt.
5. **Extrapolation advantage holds:** the 96-family winner remains the best OOD model (OOD R² 0.751 vs XGBoost 2-regime 0.619).
6. **Budget:** 0.87 of the 1.75 H100-hours used (3,144 GPU-seconds; sweep wall 162 s at 8 workers).

## Reproducibility Notes

- **Protocol (data_version 5):** train on train (2017–2020, n=9,803), early-stop / select configs on the official val split (2021–2022, n=4,805), evaluate on the untouched test set (2023–2025, n=6,620). Final winners selected by **2-seed mean val RMSE** among plain MLPs (phase-2 2nd seed on the val top-10, expanded to top-15). No trainval retrain (documented negative in 1.2).
- **Preprocessing:** median imputation + standardization fit on train only, clip to [−5, 5]; target in original units.
- **Training:** AdamW + warmup (5%) + cosine LR, grad clip 1.0, patience 60; 2-seed sweep (seeds {42, 7}); new knobs (ema, mixup_alpha, center_target, stop_rule) default to 1.2 behavior. `torch.backends.cudnn.deterministic = True` — the anchors reproduce 1.2 bit-for-bit.
- **Offline analyses:** `analyze_calibration.py` (per-cluster affine fit on val) and `analyze_stopping.py` (rule replay on saved curves) run on the 1.2 artifacts (tags `12`) and again on 1.3's (tags `13`) — zero GPU, fully reproducible from saved checkpoints/curves.
- **Reproduce:** `uv run --no-sync python run_mlp_sweep.py --resume` (phase 1 + phase 2) → `uv run --no-sync python run_mlp_eval.py` → `uv run --no-sync python analyze_calibration.py --exp-dir ../derived_8.4-eval-mlp-1.2 --tag 12` (and `--tag 13`) → `uv run --no-sync python analyze_stopping.py` (tags 12/13) → `uv run --no-sync python analyze_overfitting.py` → `uv run --no-sync python analyze_extrapolation.py` → `nb execute derived_8.4-eval-mlp-1.3.ipynb` (from `notebooks/`, with `JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime` if the default runtime dir is not writable).
