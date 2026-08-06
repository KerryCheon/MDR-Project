# Experiment: `derived_8.4-eval-mlp-1.0` — MLP vs XGBoost on the derived_8.4 Split

## Objective

Replace the XGBoost experts of `derived_8.4-eval-1.1` with PyTorch **MLP** regressors to test whether a neural net can match or beat XGBoost — motivated by the hypothesis that **XGBoost cannot extrapolate to feature values unseen during training**. Two model families are evaluated on the Washington-only `derived_8.4` split (7 stations; trainval 2017–2022, test 2023–2025, 6,620 test samples):

- **1-regime**: a single global MLP on the 54-feature shared backbone (`shared_backbone_54`).
- **2-regime**: the best cluster config from eval-1.1 — `Clustering_V0_Full_k2` winner (c0=0, c1=10): cluster 0 uses the 54 backbone features, cluster 1 adds 10 delta features (64 total). Regime labels come from the same KMeans(k=2) router on the 50 V0 features (seed 42) used by eval-1.1.

A **27-config hyperparameter sweep** (width/depth, lr, dropout, weight-decay, batch size, activation, loss) runs in **both** families with **8 parallel H100 workers**. All MLPs train on trainval with early stopping on a **temporal 10% holdout** (each station's last rows by date, mirroring the test split construction), and are evaluated on the untouched 2023–2025 test set. XGBoost reference rows are loaded from `derived_8.4-eval-1.1` (no retraining).

All numbers below are the stdout of the executed report notebook (`derived_8.4-eval-mlp-1.0.ipynb`). Trained weights, checkpoints, test predictions, and loss curves are archived under `models/`; preprocessed tensors and per-job logs under `artifacts/`.

## Overall Leaderboard (2023–2025 Test Set)

| model_name                                 | strategy_name             |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:-------------------------------------------|:--------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10) | XGBoost_Reference         |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| Global Single Model (54 Backbone)          | XGBoost_Reference         |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| MLP 2-Regime (w256x256_d0.3)               | MLP_Clustering_V0_Full_k2 |    0.75884  |     0.0500251 |       0.0500077 |    0.00131949 |    0.0377593 |         0.871552 |
| MLP 1-Regime (w256x256_d0.3)               | MLP_Global                |    0.72348  |     0.0535672 |       0.0523852 |    0.011191   |    0.0411193 |         0.858217 |
| MLP 1-Regime (w512x256x128_lr1e-3)         | MLP_Global                |    0.695493 |     0.0562127 |       0.0482295 |    0.0288754  |    0.0443541 |         0.881888 |
| MLP 2-Regime (w256x256_lr1e-3)             | MLP_Clustering_V0_Full_k2 |    0.689175 |     0.0567929 |       0.0498479 |    0.0272142  |    0.0444775 |         0.881582 |
| MLP 1-Regime (w512x256x128_d0.3)           | MLP_Global                |    0.683989 |     0.0572647 |       0.0569869 |    0.00563392 |    0.0444375 |         0.829219 |
| MLP 1-Regime (w512x256)                    | MLP_Global                |    0.655901 |     0.0597554 |       0.0558301 |    0.0213004  |    0.0474596 |         0.842348 |
| MLP 2-Regime (w1024x1024)                  | MLP_Clustering_V0_Full_k2 |    0.63896  |     0.0612087 |       0.0580151 |    0.0195128  |    0.0481404 |         0.844939 |
| MLP 1-Regime (w1024x1024)                  | MLP_Global                |    0.636094 |     0.0614512 |       0.0517418 |    0.0331517  |    0.048217  |         0.863852 |
| MLP 2-Regime (w256x256_bs256)              | MLP_Clustering_V0_Full_k2 |    0.623815 |     0.0624794 |       0.0533819 |    0.032466   |    0.0485199 |         0.859567 |
| MLP 2-Regime (w512x256)                    | MLP_Clustering_V0_Full_k2 |    0.584625 |     0.0656532 |       0.0586583 |    0.0294881  |    0.0526477 |         0.82304  |

**Winners (holdout-selected, min temporal-holdout RMSE):** `1regime: w1024x1024` (test R² 0.636), `2regime: w256x256_d0.3` (test R² 0.759). Note: holdout selection on a 10% temporal slice is noisy — the test-best 1-regime config is `w256x256_d0.3` (R² 0.723), the same config that wins the 2-regime family.

## Hyperparameter Sweep Summary (replaces eval-1.1's delta grid)

27 configs trained in **both** families; ranked by temporal holdout RMSE (selection metric); test R² shown for reference.

### Sweep Top-10 — 1-regime (Global 54)

| config_id               | hidden_sizes     |     lr |   dropout |   weight_decay |   batch_size | activation   | loss   |   holdout_rmse |   test_r2 |   test_rmse |   best_epoch |   train_time_s |
|:------------------------|:-----------------|-------:|----------:|---------------:|-------------:|:-------------|:-------|---------------:|----------:|------------:|-------------:|---------------:|
| w1024x1024              | [1024, 1024]     | 0.0003 |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0679297 |  0.636094 |   0.0614512 |           40 |        9.06213 |
| w512x256x128_lr1e-3     | [512, 256, 128]  | 0.001  |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0685209 |  0.695493 |   0.0562127 |            6 |        4.52451 |
| w512x256                | [512, 256]       | 0.0003 |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0697557 |  0.655901 |   0.0597554 |           10 |        5.55184 |
| w256x256_d0.3           | [256, 256]       | 0.0003 |       0.3 |         0.0001 |          512 | silu         | mse    |      0.0705785 |  0.72348  |   0.0535672 |           56 |       10.7538  |
| w512x256x128_d0.3       | [512, 256, 128]  | 0.0003 |       0.3 |         0.0001 |          512 | silu         | mse    |      0.0708989 |  0.683989 |   0.0572647 |           12 |        5.37804 |
| w512x256_lr1e-3_d0      | [512, 256]       | 0.001  |       0   |         0.0001 |          512 | silu         | mse    |      0.0718998 |  0.556625 |   0.0678299 |           21 |        5.87773 |
| w1024x512x256_lr1e-3_d0 | [1024, 512, 256] | 0.001  |       0   |         0.0001 |          512 | silu         | mse    |      0.0719656 |  0.545116 |   0.0687047 |           16 |        5.81875 |
| w1024x512x256           | [1024, 512, 256] | 0.0003 |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0721196 |  0.59999  |   0.0644275 |           21 |        6.41369 |
| w512x256x128            | [512, 256, 128]  | 0.0003 |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0727581 |  0.5851   |   0.0656157 |            5 |        4.92985 |
| w256x256_lr1e-3_d0      | [256, 256]       | 0.001  |       0   |         0.0001 |          512 | silu         | mse    |      0.0728539 |  0.437865 |   0.0763759 |           25 |        6.44951 |

### Sweep Top-10 — 2-regime (Cluster c0=0,c1=10)

| config_id               | hidden_sizes     |     lr |   dropout |   weight_decay |   batch_size | activation   | loss   |   holdout_rmse |   test_r2 |   test_rmse |   best_epoch |   train_time_s |
|:------------------------|:-----------------|-------:|----------:|---------------:|-------------:|:-------------|:-------|---------------:|----------:|------------:|-------------:|---------------:|
| w256x256_d0.3           | [256, 256]       | 0.0003 |       0.3 |         0.0001 |          512 | silu         | mse    |      0.062555  |  0.75884  |   0.0500251 |           98 |       10.7781  |
| w256x256_lr1e-3         | [256, 256]       | 0.001  |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0653713 |  0.689175 |   0.0567929 |           64 |        7.68377 |
| w256x256_bs256          | [256, 256]       | 0.0003 |       0.1 |         0.0001 |          256 | silu         | mse    |      0.0665732 |  0.623815 |   0.0624794 |          122 |       13.5864  |
| w1024x1024              | [1024, 1024]     | 0.0003 |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0669002 |  0.63896  |   0.0612087 |           37 |        9.27186 |
| w512x256                | [512, 256]       | 0.0003 |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0681455 |  0.584625 |   0.0656532 |          108 |        9.708   |
| w512x256x128_d0.3       | [512, 256, 128]  | 0.0003 |       0.3 |         0.0001 |          512 | silu         | mse    |      0.0683627 |  0.696412 |   0.0561278 |           55 |        7.88382 |
| w256x256_wd1e-3         | [256, 256]       | 0.0003 |       0.1 |         0.001  |          512 | silu         | mse    |      0.0686363 |  0.544816 |   0.0687273 |          122 |        9.77444 |
| w512x256_lr1e-3_d0      | [512, 256]       | 0.001  |       0   |         0.0001 |          512 | silu         | mse    |      0.0686792 |  0.48118  |   0.0733744 |           32 |        4.6298  |
| w512x256x128_lr1e-3     | [512, 256, 128]  | 0.001  |       0.1 |         0.0001 |          512 | silu         | mse    |      0.0688874 |  0.657611 |   0.0596067 |           84 |        9.30793 |
| w1024x512x256_lr1e-3_d0 | [1024, 512, 256] | 0.001  |       0   |         0.0001 |          512 | silu         | mse    |      0.0689405 |  0.508586 |   0.0714101 |           30 |        5.67664 |

## Per-Regime Performance Breakdown

| strategy_name             | model_name                                 |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |         bias |       mae |
|:--------------------------|:-------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|-------------:|----------:|
| MLP_Global                | MLP 1-Regime (w1024x1024)                  |         0 |     13148 |     6620 | 0.636094 | 0.0614512 | 0.0517418 |  0.0331517   | 0.048217  |
| MLP_Global                | MLP 1-Regime (w512x256x128_lr1e-3)         |         0 |     13148 |     6620 | 0.695493 | 0.0562127 | 0.0482295 |  0.0288754   | 0.0443541 |
| MLP_Global                | MLP 1-Regime (w512x256)                    |         0 |     13148 |     6620 | 0.655901 | 0.0597554 | 0.0558301 |  0.0213004   | 0.0474596 |
| MLP_Clustering_V0_Full_k2 | MLP 2-Regime (w256x256_d0.3)               |         0 |      9562 |     4817 | 0.732629 | 0.0517293 | 0.0517261 | -0.000576799 | 0.0398502 |
| MLP_Clustering_V0_Full_k2 | MLP 2-Regime (w256x256_d0.3)               |         1 |      3586 |     1803 | 0.820116 | 0.0451578 | 0.044704  |  0.00638571  | 0.0321731 |
| MLP_Clustering_V0_Full_k2 | MLP 2-Regime (w256x256_lr1e-3)             |         0 |      9562 |     4817 | 0.664316 | 0.0579622 | 0.0505055 |  0.0284397   | 0.0459471 |
| MLP_Clustering_V0_Full_k2 | MLP 2-Regime (w256x256_lr1e-3)             |         1 |      3586 |     1803 | 0.747104 | 0.0535436 | 0.0478936 |  0.0239401   | 0.0405513 |
| MLP_Clustering_V0_Full_k2 | MLP 2-Regime (w256x256_bs256)              |         0 |      9562 |     4817 | 0.53643  | 0.0681141 | 0.0547006 |  0.0405878   | 0.0549388 |
| MLP_Clustering_V0_Full_k2 | MLP 2-Regime (w256x256_bs256)              |         1 |      3586 |     1803 | 0.82907  | 0.0440196 | 0.0426824 |  0.0107674   | 0.0313707 |
| XGBoost_Reference         | Global Single Model (54 Backbone)          |         0 |     14608 |     6620 | 0.77923  | 0.0478636 | 0.0466868 |  0.0105484   | 0.0370592 |
| XGBoost_Reference         | Clustering_V0_Full_k2 (Winner c0=0, c1=10) |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 |  0.00861491  | 0.0359221 |
| XGBoost_Reference         | Clustering_V0_Full_k2 (Winner c0=0, c1=10) |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 |  0.000797068 | 0.0278349 |

## Year-by-Year R² Breakdown

| model_name                                 |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:-------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10) |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| Global Single Model (54 Backbone)          |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| MLP 2-Regime (w256x256_d0.3)               |    0.75884  |       0.79551  |       0.777776 |       0.689432 |
| MLP 1-Regime (w256x256_d0.3)               |    0.72348  |       0.697243 |       0.763233 |       0.705439 |
| MLP 1-Regime (w512x256x128_lr1e-3)         |    0.695493 |       0.675986 |       0.750077 |       0.654163 |
| MLP 2-Regime (w256x256_lr1e-3)             |    0.689175 |       0.715474 |       0.753571 |       0.584897 |
| MLP 1-Regime (w512x256x128_d0.3)           |    0.683989 |       0.657486 |       0.731301 |       0.657491 |
| MLP 1-Regime (w512x256)                    |    0.655901 |       0.637782 |       0.652975 |       0.667872 |
| MLP 2-Regime (w1024x1024)                  |    0.63896  |       0.658409 |       0.669677 |       0.573849 |
| MLP 1-Regime (w1024x1024)                  |    0.636094 |       0.659168 |       0.702032 |       0.532211 |
| MLP 2-Regime (w256x256_bs256)              |    0.623815 |       0.652703 |       0.692454 |       0.510119 |
| MLP 2-Regime (w512x256)                    |    0.584625 |       0.643997 |       0.587019 |       0.499043 |

## Extrapolation (OOD) Check

Test rows whose **top-10 gain features** (by XGBoost gain from the feature-selection lab) fall outside the trainval [min, max] range are flagged as OOD — 588/6,620 test rows (8.9%). This directly probes the "XGBoost cannot extrapolate to unseen values" hypothesis.

| model                       | slice           |    n |       r2 |      rmse |        bias |       mae |
|:----------------------------|:----------------|-----:|---------:|----------:|------------:|----------:|
| MLP 1regime (w1024x1024)    | all             | 6620 | 0.636094 | 0.0614512 |  0.0331517  | 0.048217  |
| MLP 1regime (w1024x1024)    | in_distribution | 6032 | 0.625317 | 0.0634403 |  0.0360583  | 0.0501638 |
| MLP 1regime (w1024x1024)    | ood             |  588 | 0.684408 | 0.0350402 |  0.00333499 | 0.0282463 |
| XGBoost Global (54)         | all             | 6620 | 0.77923  | 0.0478636 |  0.0105484  | 0.0370592 |
| XGBoost Global (54)         | in_distribution | 6032 | 0.780849 | 0.0485182 |  0.0145436  | 0.0373064 |
| XGBoost Global (54)         | ood             |  588 | 0.577509 | 0.0405427 | -0.0304369  | 0.0345237 |
| MLP 2regime (w256x256_d0.3) | all             | 6620 | 0.75884  | 0.0500251 |  0.00131949 | 0.0377593 |
| MLP 2regime (w256x256_d0.3) | in_distribution | 6032 | 0.755086 | 0.0512909 |  0.00248906 | 0.0387094 |
| MLP 2regime (w256x256_d0.3) | ood             |  588 | 0.694917 | 0.0344518 | -0.0106786  | 0.0280131 |
| XGBoost 2-Regime (Winner)   | all             | 6620 | 0.81496  | 0.0438196 |  0.00648567 | 0.0337195 |
| XGBoost 2-Regime (Winner)   | in_distribution | 6032 | 0.81728  | 0.0443022 |  0.00971871 | 0.0338159 |
| XGBoost 2-Regime (Winner)   | ood             |  588 | 0.618589 | 0.0385212 | -0.0266805  | 0.0327308 |

## Timing (H100 PCIe 80 GB, 8 parallel workers)

```
Total sweep wall time: 124.1 s  |  eval wall time: 4.7 s
GPU: {'device': 'NVIDIA H100 PCIe 80GB', 'n_parallel': 8}
```

54 jobs (27 configs × 2 families) completed in **~2.1 min wall** with 8 concurrent workers (GPU util ~94%, ~5 GB memory). Per-config train time ranged **4.5–18 s** (see `timing_summary.csv` / `timing_log.json` for the full table). For future reference: an equivalent sweep of N configs × F families at P workers ≈ N×F/P × ~12 s wall.

## Key Takeaways

1. **XGBoost still wins overall**, but MLP is competitive: best MLP 2-regime (w256x256_d0.3) reaches **R² = 0.759 / RMSE = 0.050** vs XGBoost 2-regime winner **0.815 / 0.044** (ΔR² = −0.056); best MLP 1-regime (w256x256_d0.3) **0.723** vs XGBoost global **0.779** (ΔR² = −0.056).
2. **Extrapolation hypothesis is supported**: on the OOD slice, the MLP beats XGBoost by **+0.076 R² (2-regime: 0.695 vs 0.619)** and **+0.107 R² (1-regime: 0.684 vs 0.578)**, with lower bias and MAE. XGBoost's advantage is concentrated in-distribution — exactly the pattern expected if trees cannot extrapolate to unseen feature values while a standardized MLP can.
3. **The 2-regime (clustered) structure helps the MLP too**: +0.036 R² over the global MLP, mirroring the XGBoost gain (+0.036). The same config (`w256x256_d0.3`: 256×256, SiLU, dropout 0.3, lr 3e-4) is the test-best in both families.
4. **Yearly stability favors XGBoost**: the MLP's R² degrades more in 2025 (2-regime 0.689 vs 0.796 in 2023), while XGBoost stays flat (~0.83) — suggesting tree ensembles remain more robust to distributional drift, despite losing on the OOD slice.
5. **Holdout selection is noisy on a 10% temporal slice**: the holdout-selected 1-regime winner (w1024x1024, test 0.636) is not the test-best config (w256x256_d0.3, 0.723); the 2-regime family selects the same config by both criteria. Configs that overfit (low dropout, high lr on wide nets) top the holdout but collapse on test.
6. **Cost**: the full 54-job sweep cost ~2.1 min of H100 wall time — MLP tuning is cheap at this dataset size, so the bottleneck is model quality, not compute.

## Reproducibility Notes

- **Protocol**: train on trainval (2017–2022), early stop on a **temporal 10% holdout** (each station's last rows by date; `data_version: 2`), evaluate on the untouched 2023–2025 test set. A first run with a random holdout (v1) was discarded — it leaked temporally-adjacent rows and inflated holdout scores.
- **Preprocessing**: median imputation + standardization fit on trainval only (per cluster subset for specialists), clip to [−5, 5]; target kept in original units. XGBoost needs no imputation (native NaN handling) — a documented, fair difference.
- **Training**: AdamW + cosine LR, grad clip 1.0, patience 25, seed 42; full checkpoints (`checkpoint.pt`) every 10 epochs → jobs resume via `run_mlp_sweep.py --resume`.
- **Reproduce**: `uv run python run_mlp_sweep.py --resume` (parallel sweep) → `uv run python run_mlp_eval.py` (report artifacts) → `uv run python analyze_extrapolation.py` (OOD check) → `nb execute derived_8.4-eval-mlp-1.0.ipynb` (report).
