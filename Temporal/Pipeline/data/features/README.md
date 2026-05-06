# Temporal Soil Moisture Model

## Results Log and Experimental History

**Author:** Jakob Balkovec

---

## Table of Contents

1. [Best Model](#best-model)
2. [Overview](#overview)
3. [v1.x Series: Early Baselines and Invalid Experiments](#v1x-series--early-baselines-and-invalid-experiments)
   - [v1.0.0: Valid Baseline (No Derived Features)](#v100--valid-baseline-no-derived-features)
   - [v1.1.0: Invalid (Feature Leakage)](#v110--invalid-feature-leakage)
   - [v1.2.0: Invalid (Feature Leakage)](#v120--invalid-feature-leakage)
4. [v2.x Series: Expanded Data and Corrected Features](#v2x-series--expanded-data-and-corrected-features)
   - [v2.1.0: Invalid (Expanded Split, Leakage)](#v210--invalid-expanded-split-leakage)
   - [v2.2.0: Valid Baseline (Derived Features Fixed)](#v220--valid-baseline-derived-features-fixed)
   - [v2.3.0: Valid Baseline (Handpicked Derived Features)](#v230--valid-baseline-handpicked-derived-features)
5. [v3.x Series: Full Derived Feature Models](#v3x-series--full-derived-feature-models)
   - [v3.1.0: Valid Baseline (All Derived Features)](#v310--valid-baseline-all-derived-features)
   - [v3.2.0: Valid (Pruned Feature Set)](#v320--valid-pruned-feature-set)
   - [v3.3.0: Valid (Hyperparameter-Tuned, Leakage-Checked)](#v330--valid-hyperparameter-tuned-leakage-checked)
6. [v4.x Series: Architectural Experiments](#v4x-series--architectural-experiments)
   - [v4.1.0: Valid (Two-Stage Residual Model)](#v410--valid-two-stage-residual-model)
7. [v5.x Series: Linear Diagnostic Models](#v5x-series--linear-diagnostic-models)
   - [v5.1.0: Valid (ElasticNet, Diagnostic)](#v510--valid-elasticnet-diagnostic)
8. [v7.x Series: New Features](#v7x-series-new-features)
   - [v7.1.0: Valid Baseline](#v710-valid-baseline)
   - [v7.2.0: Valid](#v720-valid)
   - [v7.3.0: Valid](#v730-valid)
   - [v7.4.0: Valid](#v740-valid)
   - [v7.5.0: Valid](#v750-valid)
   - [v7.6.0: In Progress](#v760-in-progress)
9. [v8.x Series: Rain Features](#v8x-series-rain-features)
   - [v8.1.0: Valid (Rain Feature Additions)](#v810-valid-rain-feature-additions)
   - [v8.2.0: Valid (Rain Variance Study)](#v820-valid-rain-variance-study)
10. [v9.x Series: Gated Models](#v9x-series-gated-models)
    - [v9.1.0: Valid (Gated Mixture of Experts)](#v910-valid-gated-mixture-of-experts)
    - [v9.2.0: Valid (Wet Expert Rain Impulses)](#v920-valid-wet-expert-rain-impulses)
    - [v9.3.0: Valid (Winter/Non-Winter Gate)](#v930-valid-winternon-winter-gate)
    - [v9.4.0: Valid (Improved Winter Expert)](#v940-valid-improved-winter-expert)
11. [v11.x Series: Rain Modeling and Residual Correction](#v11x-series-rain-modeling-and-residual-correction)
    - [v11.1.0: Valid (Rain-Only XGB Diagnostic)](#v1110-valid-rain-only-xgb-diagnostic)
    - [v11.2.0: Valid (Residual-Correction Prototype)](#v1120-valid-residual-correction-prototype)
12. [v12.x Series: Shallow NN Baselines](#v12x-series-shallow-nn-baselines)
    - [v12.1.0: Valid (Shallow NN Baseline)](#v1210-valid-shallow-nn-baseline)
    - [v12.2.0: Valid (Shallow NN Retune)](#v1220-valid-shallow-nn-retune)
    - [v12.3.0: Valid (Shallow NN Variant)](#v1230-valid-shallow-nn-variant)
    - [v12.4.0: Valid (Stacked v12.1 + v12.3)](#v1240-valid-stacked-v121--v123)
    - [v12.5.0: Valid (Shallow NN Variant)](#v1250-valid-shallow-nn-variant)
    - [v12.6.0: In Progress (High-Val / Mid-Test Regime)](#v1260-in-progress-high-val--mid-test-regime)
    - [v12.7.0: In Progress (Metric Inconsistency)](#v1270-in-progress-metric-inconsistency)
13. [v13.x Series: Temporal Backbone and AR Rollout](#v13x-series-temporal-backbone-and-ar-rollout)
    - [v13.1.0: Valid (Temporal Backbone + HMM State Features)](#v1310-valid-temporal-backbone--hmm-state-features)
    - [v13.2.0: Invalid (AR Rollout Feature Mismatch)](#v1320-invalid-ar-rollout-feature-mismatch)
    - [v13.2.1: Valid (AR Rollout Fixed)](#v1321-valid-ar-rollout-fixed)
14. [v14.x Series: Spatial Feature Integration](#v14x-series-spatial-feature-integration)
    - [v14.1.0: Valid (Temporal + Static Spatial Features)](#v1410-valid-temporal--static-spatial-features)
15. [v15.x Series: derived_6.0 Feature Set](#v15x-series-derived_60-feature-set)
    - [v15.1.0: Valid (108-Feature Stack Baseline)](#v1510-valid-108-feature-stack-baseline)
    - [v15.2.0: Valid (Config Re-run)](#v1520-valid-config-re-run)
    - [v15.3.0: Valid (Early-Stopped XGB + Ridge Stack)](#v1530-valid-early-stopped-xgb--ridge-stack)
16. [v16.x Series: Drift-Aware Tuning on derived_6.0](#v16x-series-drift-aware-tuning-on-derived_60)
    - [v16.1.0: Valid (Year-Weighted Stack Baseline)](#v1610-valid-year-weighted-stack-baseline)
    - [v16.2.0: Valid (Drift Features + Calibration)](#v1620-valid-drift-features--calibration)
    - [v16.3.0: Valid (Drift Expansion + Weighting Study)](#v1630-valid-drift-expansion--weighting-study)
    - [v16.4.0: Valid (Aggressive Drift Tuning)](#v1640-valid-aggressive-drift-tuning)
    - [v16.5.0: Valid (Add `J_` Feature Family)](#v1650-valid-add-j_-feature-family)
17. [v17.x Series: Expanded Test Split](#v17x-series-expanded-test-split)
    - [v17.1.0: Valid (Expanded Test Years)](#v1710-valid-expanded-test-years)
18. [v18.x Series: Calibration Pass](#v18x-series-calibration-pass)
    - [v18.3.0: Valid (Calibrated Drift Model)](#v1830-valid-calibrated-drift-model)
19. [v19.x Series: derived_8.0 Feature Set](#v19x-series-derived_80-feature-set)
    - [v19.2.0: Valid (Feature Set Refresh)](#v1920-valid-feature-set-refresh)
20. [v20.x Series: Variance Checks](#v20x-series-variance-checks)
    - [v20.1.0: Valid (Variance and Heteroskedasticity Pass)](#v2010-valid-variance-and-heteroskedasticity-pass)
    - [v20.3.0: Valid (Dry Regime Specialist Pass)](#v2030-valid-dry-regime-specialist-pass)

> **Note**: Splits have been re-named

```yaml
split_renaming:
  base_1.0: base
  base_2.0: base_no_met

  derived_1.0: derived
  derived_2.0: derived_all
  derived_3.0: derived_new
  derived_4.0: derived_updated
  derived_5.0: derived_new_updated
  derived_6.0: null
```

---

## Best Model

![Best model dashboard](Figures/best_model.png)

Updated Feb 27, 2026

> Current pick is `v20.3` (hard-gated mixture-of-experts). It achieves the best logged test $R^2$ so far (≈ 0.8586).

## Test Performance (Full Test Set, N = 4,016)

| Model                          | MAE         | RMSE        |     UB-RMSE | $R^2$         | Bias     | MedAE   | P90 AE  |
|--------------------------------|-------------|-------------|-------------|---------------|----------|---------|---------|
| `v20.3` hard gated (oracle)    | **0.02550** | **0.03541** | **0.03423** | **0.85860**   | -0.00904 | 0.01837 | 0.05776 |

---

## Overview

This document tracks the evolution of the **temporal soil moisture model**, including all major experimental versions, validation status, and known issues.

Each model version is explicitly labeled as **VALID** or **INVALID** based on data leakage, split correctness, and feature integrity.
Only **VALID** models should be used as baselines for ablation studies or future comparisons.

Metrics reported:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **$R^2$**: Coefficient of Determination

---

## v1.x Series: Early Baselines and Invalid Experiments

### v1.0.0: **VALID Baseline (No Derived Features)**

**Description:**
Initial baseline model using only base satellite bands and metadata. No derived temporal features included.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.024815 | 0.035413 | 0.879066 |
| Val   | 0.050936 | 0.066775 | 0.613661 |
| Test  | 0.049829 | 0.064362 | 0.520847 |

**Comments:**

- Serves as a clean reference point for all future experiments
- No derived features
- No known data leakage

---

### v1.1.0: **INVALID (Feature Leakage)**

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.029109 | 0.040223 | 0.874408 |
| Val   | 0.045965 | 0.059679 | 0.679553 |
| Test  | 0.064540 | 0.075874 | 0.017134 |

**Comments:**

- Removed `DOY` following ablation in v1.0.0
- **INVALID:** Feature leakage introduced from sensor-derived data
- Model had access to information unavailable at training time

---

### v1.2.0: **INVALID (Feature Leakage)**

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.031976 | 0.044604 | 0.845555 |
| Val   | 0.046963 | 0.060642 | 0.669136 |
| Test  | 0.057298 | 0.068399 | 0.201255 |

**Comments:**

- Continued from v1.1.0 with modified feature set
- **INVALID:** Same leakage issue persisted
- Results not comparable to valid baselines

---

## v2.x Series: Expanded Data and Corrected Features

### v2.1.0: **INVALID (Expanded Split, Leakage)**

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.019661 | 0.026586 | 0.930787 |
| Val   | 0.042565 | 0.057601 | 0.702951 |
| Test  | 0.039524 | 0.052080 | 0.712817 |

**Comments:**

- Expanded dataset by adding additional stations
- **INVALID:** Sensor-derived feature leakage still present
- Apparent performance gain is misleading

---

### v2.2.0: **VALID Baseline (Derived Features Fixed)**

**Description:**
First corrected model with derived features properly constructed and leakage removed.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.017946 | 0.028135 | 0.923666 |
| Val   | 0.052999 | 0.067946 | 0.599990 |
| Test  | 0.050461 | 0.065313 | 0.506571 |

**Comments:**

- Removed all sensor-related leaked features
- **VALID** baseline
- Suitable for controlled ablation studies

---

### v2.3.0: **VALID Baseline (Handpicked Derived Features)**

**Description:**
Derived feature set expanded using a small, curated subset informed by earlier feature importance analysis.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.014148 | 0.019481 | 0.963402 |
| Val   | 0.039717 | 0.052083 | 0.764966 |
| Test  | 0.041054 | 0.053289 | 0.671525 |

**Comments:**

- Added 8 handpicked derived features
- All features verified to be temporally valid
- **VALID** baseline for:
  - family-level ablation
  - contribution analysis
  - future architectural experiments

---

### v3.1.0: **VALID Baseline (All Derived Features)**

**Description:**
Added all derived features except the unstable ones identified in prior analyses.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.011984 | 0.016691 | 0.973135 |
| Val   | 0.039207 | 0.050659 | 0.777642 |
| Test  | 0.043802 | 0.056370 | 0.632453 |

**Comments:**

- Added all features (89 total)
- All features verified to be temporally valid
- **VALID** baseline

---

### v3.2.0: **VALID (All Derived Features)**

**Description:**
Picked 40 features out of 89 by pruning the families with lowest importance from v3.1.0.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.012754 | 0.017935 | 0.968981 |
| Val   | 0.039758 | 0.051282 | 0.772138 |
| Test  | 0.043800 | 0.056701 | 0.628125 |

**Comments:**

- 40 features selected from 89 in v3.1.0
- All features verified to be temporally valid
- **VALID** baseline

---

### v3.3.0: **VALID (All Derived Features)**

**Description:**
Tuned the Hyperparameters further and removed one feature (`I_ts_spike_s1_vv`) due to lingering leakage concerns.

**Comments:**

- 40 features selected from 89 in v3.1.0
- All features verified to be temporally valid
- **VALID** baseline

#### v3.3.1: **Row 177**

##### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.023669 | 0.032485 | 0.898234 |
| Val   | 0.023175 | 0.030704 | 0.918315 |
| Test  | 0.038198 | 0.049690 | 0.714397 |

#### v3.3.2: **Row 405**

##### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.023638 | 0.032479 | 0.898272 |
| Val   | 0.023065 | 0.030626 | 0.918733 |
| Test  | 0.038369 | 0.049903 | 0.711952 |

#### v3.3.3: **Row 267**

##### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.023898 | 0.032851 | 0.895931 |
| Val   | 0.023388 | 0.031064 | 0.916390 |
| Test  | 0.038324 | 0.049914 | 0.711818 |

---

### v4.1.0: **VALID Baseline (2 Model Approach)**

**Description:**
Upgrade of v3.2 where I replaced the single XGB regressor with a two-stage modeling strategy

- **Model A** learns the dominant soil-moisture drivers (rainfall, API, seasonality, static context)
- **Model B** learns the residual signal left behind by **Model A** (radar, optical, LST dynamics)

- Final prediction is the sum of both models:

$$\hat{y}_t = \hat{y}_t^{(A)} + \hat{r}_t^{(B)}$$

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.013086 | 0.018172 | 0.968154 |
| Val   | 0.038476 | 0.048866 | 0.793100 |
| Test  | 0.046930 | 0.060299 | 0.579425 |

**Comments:**

- Didn't bump $$R^2$$ like I expected...actually it made everything worse. I could tune the hyperparameters more, but I think this is most likely a dead end

---

### v5.1.0: **VALID (ElasticNet, Diagnostic!)**

**Description:**
Using a new type of model (ElasticNet) instead of XGB.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.048441 | 0.060676 | 0.644965 |
| Val   | 0.052522 | 0.065754 | 0.625384 |
| Test  | 0.048862 | 0.061532 | 0.562049 |

**Comments:**

- Switched to ElasticNet for better interpretability and feature selection
- Still using the same 40 features from v3.2.0
- Preliminary results look promising, but further tuning and validation needed

---

## v7.x Series: New Features

### v7.1.0: **VALID BASELINE**

**Description:**
Used the 40 features obtained from the pipeline to train a new regressor, fully tuned.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.034470 | 0.043565 | 0.818483 |
| Val   | 0.036769 | 0.047804 | 0.774720 |
| Test  | 0.036493 | 0.046039 | 0.757695 |

**Comments:**

- Very stable and robust model

### v7.2.0: **VALID**

**Description:**
Iterative pruning of features based on importance (remove bottom 10% after every iteration).

#### Results

| Split | $R^2$  |
| ----- | ------ |
| Train | 0.7269 |
| Val   | 0.7265 |
| Test  | 0.7051 |

**Comments:**

- Quick pruning pass to test sensitivity
- Performance dipped vs v7.1.x, so this was more diagnostic than final

### v7.3.0: **VALID**

**Description:**
Fully tuned `XGBRegressor` with post-hoc processing and calibration

#### Results

**Before calibration**

| Split | MAE      | RMSE     | $R^2$    | Bias Mean (True vs. Pred) |
| ----- | -------- | -------- | -------- | ------------------------- |
| Train | 0.034470 | 0.043565 | 0.818483 | 0.000045                  |
| Val   | 0.036769 | 0.047804 | 0.774720 | 0.023734                  |
| Test  | 0.036493 | 0.046039 | 0.757695 | 0.012184                  |

**After calibration**

| Split | MAE      | RMSE     | $R^2$    | Bias Mean (True vs. Pred) |
| ----- | -------- | -------- | -------- | ------------------------- |
| Train | 0.038399 | 0.048168 | 0.778098 | -2.348640e-02             |
| Val   | 0.031648 | 0.040660 | 0.837026 | -1.138516e-08             |
| Test  | 0.034252 | 0.045696 | 0.761296 | -1.228967e-02             |

**Comments:**

- Strong single-model baseline
- Test R^2 around 0.76 after calibration

### v7.4.0: **VALID**

**Description:**
Used `Ridge` to stack a `RandomForest` and an `XGBRegressor`

#### Results

**Stacked (XGB + RF → Ridge) TEST metrics**

| Metric | Value      |
| ------ | ---------- |
| $R^2$  | 0.77681427 |
| MAE    | 0.03312991 |
| RMSE   | 0.04418567 |
| Bias   | 0.00535984 |

Weights: `[0.10768029, 0.88591085]` with intercept `0.00514084`

**Comments:**

- Very stable and robust model

**Best Params for Stacking**

**Best XGB params**:

```JSON
{
  "subsample": 0.9,
  "reg_lambda": 2.0,
  "reg_alpha": 0.05,
  "n_estimators": 4000,
  "min_child_weight": 3,
  "max_depth": 7,
  "learning_rate": 0.05,
  "gamma": 0.0,
  "colsample_bytree": 0.75
}
```

**Best RF params**:

```JSON
{
  "model__n_estimators": 800,
  "model__min_samples_split": 10,
  "model__min_samples_leaf": 5,
  "model__max_features": 0.5,
  "model__max_depth": 16,
}
```

---

### v7.5.0: **VALID**

**Description:**
Re-ran the v7.4 stack idea with a cleaner eval flow: baseline XGB, post-hoc Ridge calibration, and a basic XGB + RF stack.

#### Results

**Baseline XGB**

| Split | MAE      | RMSE     | $R^2$    | Bias Mean (True vs. Pred) |
| ----- | -------- | -------- | -------- | ------------------------- |
| Train | 0.001786 | 0.002553 | 0.999377 | -0.000003                 |
| Val   | 0.034186 | 0.045249 | 0.798166 | 0.022045                  |
| Test  | 0.039043 | 0.050376 | 0.709898 | 0.018762                  |

**After post-hoc Ridge calibration (XGB)**

| Split | MAE      | RMSE     | $R^2$    | Bias Mean (True vs. Pred) |
| ----- | -------- | -------- | -------- | ------------------------- |
| Train | 0.022184 | 0.022668 | 0.950855 | -0.022183                 |
| Val   | 0.030050 | 0.039355 | 0.847319 | -0.000000                 |
| Test  | 0.036139 | 0.046745 | 0.750212 | -0.003312                 |

**Stacked (XGB + RF → Ridge) TEST metrics**

| Metric | Value       |
| ------ | ----------- |
| $R^2$  | 0.75377602  |
| MAE    | 0.03556066  |
| RMSE   | 0.04641019  |
| Bias   | -0.00464616 |

Weights: `[0.45864664, 0.50093636]` with intercept `0.03028375`

**Comments:**

- Calibration helped a lot on val, modest lift on test
- Stacking beat calibrated XGB by a hair, but not a huge jump
- Val split is missing Touchet data, so keep that in mind

### v7.6.0: **IN PROGRESS**

**Description:**
OOF constrained blending (XGB + RF) using time-based folds per station. Same features, no data changes.

#### Results

---

## v8.x Series: Rain Features

### v8.1.0: **VALID (Rain Feature Additions)**

**Description:**
Added rain features from domain analysis to quantify rainfall effects while keeping the standard temporal split.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.012255 | 0.017278 | 0.971450 |
| Val   | 0.036803 | 0.047996 | 0.772912 |
| Test  | 0.041720 | 0.053383 | 0.674228 |

**Comments:**

- Rain-derived features integrated into the main feature set
- Performance remains in line with prior valid baselines

---

### v8.2.0: **VALID (Rain Variance Study)**

**Description:**
Focused experiment to estimate how much soil moisture variance can be explained by rain signals alone and with a rain-backed feature.

#### Results

**Main model (baseline for this run)**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.011103 | 0.015629 | 0.976813 |
| Val   | 0.035866 | 0.046442 | 0.787377 |
| Test  | 0.042880 | 0.055056 | 0.671544 |

**Rain-only ShallowXGB**

| Split | MAE      | RMSE     | $R^2$    | Bias     |
| ----- | -------- | -------- | -------- | -------- |
| Train | 0.037474 | 0.048972 | 0.772345 | 0.000028 |
| Val   | 0.037166 | 0.048253 | 0.770475 | 0.017825 |
| Test  | 0.035145 | 0.047089 | 0.759728 | 0.008382 |

**Main model with rain backbone feature**

| Split | MAE      | RMSE     | $R^2$    | Bias      |
| ----- | -------- | -------- | -------- | --------- |
| Train | 0.003077 | 0.004533 | 0.998050 | -0.000002 |
| Val   | 0.036036 | 0.047154 | 0.780810 | 0.020987  |
| Test  | 0.041493 | 0.053943 | 0.684687 | 0.018086  |

**Comments:**

- Rain-only model captures a sizable share of variance, especially on the held-out station
- Rain-backbone feature slightly lifts test $R^2$ vs the baseline for this run

---

## v9.x Series: Gated Models

### v9.1.0: **VALID (Gated Mixture of Experts)**

**Description:**
Mixture-of-experts setup to separate winter vs non-winter regimes with a simple gate. Expert A targets the learnable (dry/shoulder/summer) regime, Expert B targets winter/wet behavior.

#### Results

**Overall (full splits)**

| Model         | Split | MAE      | RMSE     | $R^2$    |
| ------------- | ----- | -------- | -------- | -------- |
| Expert A only | Val   | 0.035421 | 0.045815 | 0.793084 |
| Expert B only | Val   | 0.060447 | 0.078256 | 0.396304 |
| Soft Mix      | Val   | 0.033962 | 0.044036 | 0.808841 |
| Expert A only | Test  | 0.041679 | 0.053824 | 0.686077 |
| Expert B only | Test  | 0.065581 | 0.085960 | 0.199311 |
| Soft Mix      | Test  | 0.039376 | 0.051813 | 0.709101 |

**Slice checks (soft mix)**

| Slice | Split | MAE      | RMSE     | $R^2$     |
| ----- | ----- | -------- | -------- | --------- |
| Wet   | Val   | 0.033928 | 0.042908 | 0.051728  |
| Wet   | Test  | 0.043370 | 0.052990 | -0.971714 |
| Dry   | Val   | 0.033973 | 0.044422 | 0.814365  |
| Dry   | Test  | 0.037998 | 0.051400 | 0.723531  |

**Comments:**

- Soft mix marginally improves overall test $R^2$ vs Expert A alone
- Wet-only performance remains weak, indicating winter regime is still hard to model

---

### v9.2.0: **VALID (Wet Expert Rain Impulses)**

**Description:**
Builds on v9.1 with rain impulse features added only to the wet expert to boost wet-regime skill while keeping the dry expert fixed.

#### Results

**Overall (full splits)**

| Model         | Split | MAE      | RMSE     | $R^2$    |
| ------------- | ----- | -------- | -------- | -------- |
| Expert A only | Val   | 0.042298 | 0.054899 | 0.702896 |
| Expert B only | Val   | 0.034365 | 0.043735 | 0.811442 |
| Soft Mix      | Val   | 0.033442 | 0.043639 | 0.812268 |
| Expert A only | Test  | 0.047030 | 0.059615 | 0.614892 |
| Expert B only | Test  | 0.043379 | 0.055267 | 0.669013 |
| Soft Mix      | Test  | 0.038446 | 0.050425 | 0.724468 |

**Slice checks (soft mix)**

| Slice | Split | MAE      | RMSE     | $R^2$    |
| ----- | ----- | -------- | -------- | -------- |
| Wet   | Val   | 0.034302 | 0.044611 | 0.370624 |
| Wet   | Test  | 0.042409 | 0.053164 | 0.161911 |
| Dry   | Val   | 0.032737 | 0.042826 | 0.820266 |
| Dry   | Test  | 0.035167 | 0.048041 | 0.771591 |

**Comments:**

- Soft mix improves overall test $R^2$ vs v9.1 and Expert A alone
- Wet-slice $R^2$ remains low, but improves substantially relative to v9.1

---

### v9.3.0: **VALID (Winter/Non-Winter Gate)**

**Description:**
Hard/soft gating between a non-winter expert and a winter expert using season-based splits (winter vs non-winter).

#### Results

**Overall (full splits)**

| Model         | Split | MAE      | RMSE     | $R^2$     |
| ------------- | ----- | -------- | -------- | --------- |
| Expert A only | Val   | 0.033750 | 0.045402 | 0.796798  |
| Expert B only | Val   | 0.075901 | 0.104020 | -0.066633 |
| Soft Mix      | Val   | 0.033246 | 0.043457 | 0.813832  |
| Expert A only | Test  | 0.036568 | 0.049098 | 0.738779  |
| Expert B only | Test  | 0.082181 | 0.110853 | -0.331585 |
| Soft Mix      | Test  | 0.038327 | 0.049736 | 0.731950  |

**Slice checks (soft mix)**

| Slice      | Split | MAE      | RMSE     | $R^2$     |
| ---------- | ----- | -------- | -------- | --------- |
| Winter     | Val   | 0.027886 | 0.036437 | 0.094157  |
| Winter     | Test  | 0.039873 | 0.052290 | -0.311979 |
| Non-winter | Val   | 0.035256 | 0.045813 | 0.797342  |
| Non-winter | Test  | 0.037591 | 0.048472 | 0.725748  |

**Comments:**

- Soft mix slightly improves overall val but dips on test vs v9.2
- Winter-slice performance remains unstable, especially on test

---

### v9.4.0: **VALID (Improved Winter Expert)**

**Description:**
Focused on improving the winter expert while keeping the non-winter expert fixed; evaluates gating on winter/non-winter splits.

#### Results

**Overall (full splits)**

| Model         | Split | MAE      | RMSE     | $R^2$    |
| ------------- | ----- | -------- | -------- | -------- |
| Expert A only | Val   | 0.033750 | 0.045402 | 0.796798 |
| Expert B only | Val   | 0.046793 | 0.058678 | 0.660583 |
| Soft Mix      | Val   | 0.033274 | 0.044243 | 0.807040 |
| Expert A only | Test  | 0.036568 | 0.049098 | 0.738779 |
| Expert B only | Test  | 0.048082 | 0.060434 | 0.604231 |
| Soft Mix      | Test  | 0.036480 | 0.048690 | 0.743106 |

**Slice checks (soft mix)**

| Slice      | Split | MAE      | RMSE     | $R^2$     |
| ---------- | ----- | -------- | -------- | --------- |
| Winter     | Val   | 0.031671 | 0.043478 | -0.289769 |
| Winter     | Test  | 0.038878 | 0.054469 | -0.423605 |
| Non-winter | Val   | 0.033876 | 0.044526 | 0.808566  |
| Non-winter | Test  | 0.035337 | 0.045680 | 0.756427  |

**Comments:**

- Soft mix improves overall test $R^2$ vs v9.3 and v9.2
- Winter-slice performance is still negative on test despite expert changes

---

## v11.x Series: Rain Modeling and Residual Correction

### v11.1.0: **VALID (Rain-Only XGB Diagnostic)**

**Description:**
Rain-only XGBoost study targeting precipitation dynamics (`log1p(precip_mm)`) to quantify how much signal is captured without the full soil feature stack.

#### Results

| Variant    | Val MAE (mm) | Val RMSE (mm) | Val $R^2$ (mm) | Val $R^2$ (log) |
| ---------- | ------------ | ------------- | -------------- | --------------- |
| Baseline   | 1.481858     | 4.466299      | 0.846668       | 0.902927        |
| Aggressive | 1.150342     | 3.844999      | 0.886360       | 0.935285        |

**Comments:**

- Aggressive configuration improves all reported validation metrics vs baseline.
- This notebook reports validation metrics only (no full train/val/test table).

---

### v11.2.0: **VALID (Residual-Correction Prototype)**

**Description:**
Two-stage soil-moisture prototype with a base model and planned rain-based residual correction. Note: notebook title is `v11.3`, but the file path/version folder is `v11.2`.

#### Results

**Base soil model (train loop output)**

| Split | MAE    | RMSE   | $R^2$  |
| ----- | ------ | ------ | ------ |
| Train | 0.0296 | 0.0381 | 0.8619 |
| Val   | 0.0350 | 0.0464 | 0.7880 |
| Test  | 0.0390 | 0.0506 | 0.7231 |

**Final corrected test metrics (Section 8.1 output)**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Test  | 0.036493 | 0.046039 | 0.757695 |

**Comments:**

- Temporal split checks in-notebook report no date overlap leakage.
- Final test metrics improve over the base test metrics shown earlier in the run.

---

## v12.x Series: Shallow NN Baselines

### v12.1.0: **VALID (Shallow NN Baseline)**

**Description:**
Initial shallow MLP baseline on `derived_new` features with early stopping and full split evaluation.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.022029 | 0.028785 | 0.920755 |
| Val   | 0.038880 | 0.053101 | 0.722038 |
| Test  | 0.044309 | 0.057026 | 0.628253 |

**Comments:**

- Best validation checkpoint reported: `val_r2 = 0.722038`.
- Test wet-slice remains weak (`R^2 = -3.772777`) while dry-slice is positive (`R^2 = 0.643049`).

---

### v12.2.0: **VALID (Shallow NN Retune)**

**Description:**
Re-tuned shallow MLP baseline with improved validation fit over v12.1

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.018255 | 0.025646 | 0.937094 |
| Val   | 0.038508 | 0.051509 | 0.738449 |
| Test  | 0.043314 | 0.057329 | 0.624292 |

**Comments:**

- Best validation checkpoint reported: `val_r2 = 0.738449`.
- Overall val improves vs v12.1, but test remains similar and wet-slice is still strongly negative (`R^2 = -4.342402`).

---

### v12.3.0: **VALID (Shallow NN Variant)**

**Description:**
Shallow MLP variant with the strongest validation score in the early v12 sequence. Used Gaussian NLL loss function...hence the low train $R^2$

#### Results

| Split | MAE      | RMSE     | $R^2$     |
| ----- | -------- | -------- | --------- |
| Train | 0.025014 | 0.107761 | -0.110605 |
| Val   | 0.034431 | 0.044782 | 0.802305  |
| Test  | 0.041470 | 0.052819 | 0.681080  |

**Comments:**

- Best validation checkpoint reported: `val_r2 = 0.802305`.
- Train metrics are anomalous (negative train `R^2`) despite stronger val/test performance, keep as experimental

---

### v12.4.0: **VALID (Stacked v12.1 + v12.3)**

**Description:**
Simple ridge stacking of prediction outputs from v12.1 and v12.3 (`preds_mlp_v12_1.csv` + `preds_mlp_v12_3.csv`).

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Val   | 0.032268 | 0.042294 | 0.823660 |
| Test  | 0.041064 | 0.052111 | 0.689568 |

**Comments:**

- Improves overall validation and modestly improves test vs individual v12.1/v12.3 runs.
- Test wet-slice remains unstable (`R^2 = -1.693497`) while dry-slice is positive (`R^2 = 0.684708`).

---

### v12.5.0: **VALID (Shallow NN Variant)**

**Description:**
Follow-up shallow MLP variant with moderate overall performance and persistent wet-regime difficulty

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.030277 | 0.073856 | 0.478320 |
| Val   | 0.035017 | 0.046540 | 0.786477 |
| Test  | 0.042050 | 0.053456 | 0.673340 |

**Comments:**

- Best validation checkpoint reported: `val_r2 = 0.786477`.
- Wet-slice test performance is still negative (`R^2 = -2.710989`).

---

### v12.6.0: **Valid (High-Val / Mid-Test Regime)**

**Description:**
Shallow MLP run with very high validation scores but only moderate overall test transfer

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.013614 | 0.069401 | 0.539350 |
| Val   | 0.011720 | 0.016609 | 0.972805 |
| Test  | 0.013437 | 0.055973 | 0.641860 |

**Comments:**

- Best validation checkpoint reported: `val_r2 = 0.972805`.
- Large validation-to-test gap means ... instability and requires additional verification

---

### v12.7.0: **INVALID (Shallow NN + Lag-1 Update)**

**Description:**
Latest shallow MLP run with lag-1 row handling updates.

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.009564 | 0.018075 | 0.968748 |
| Val   | 0.009161 | 0.016096 | 0.974460 |
| Test  | 0.008695 | 0.015670 | 0.971928 |

**Comments:**

- Test slice check from artifacts: wet `R^2 = 0.572281`, dry `R^2 = 0.973953`
- INVALID because the new lagged features we've constructed from the ground truth (leak)

---

## v13.x Series: Temporal Backbone and AR Rollout

### v13.1.0: **VALID (Temporal Backbone + HMM State Features)**

**Description:**
Adds a sequential wetness bucket and train-only HMM state features on top of the temporal baseline, then uses quantile-style XGB experts with ridge stacking.

#### Results

**Final stacked model (`ridge_stack`)**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.029755 | 0.036814 | 0.870384 |
| Val   | 0.029957 | 0.039034 | 0.849801 |
| Test  | 0.034997 | 0.046343 | 0.754492 |

**Comments:**

- This was the strongest run in the v13 folder
- Temporal checks still show one station warning in notebook logs (`Touchet_WA_824` missing val rows)

---

### v13.2.0: **INVALID (AR Rollout Feature Mismatch)**

**Description:**
First recursive autoregressive rollout attempt with `P_lag1` and `P_lag1_missing`

**Comments:**

- Notebook fails during rollout predict with an XGBoost feature name mismatch
- Error shows train model missing fields `P_lag1` and `P_lag1_missing` at inference
- Marked INVALID since the run does not complete with a consistent feature schema

---

### v13.2.1: **VALID (AR Rollout Fixed)**

**Description:**
Fixed version of the v13.2 AR experiment (`MDR-v13.2-ar-rollout-fixed.ipynb`) with aligned feature columns

#### Results

| Model      | Split | MAE      | RMSE     | $R^2$    |
| ---------- | ----- | -------- | -------- | -------- |
| Baseline   | Val   | 0.034737 | 0.045313 | 0.797590 |
| Baseline   | Test  | 0.039489 | 0.050993 | 0.702751 |
| AR Rollout | Val   | 0.034149 | 0.044088 | 0.808391 |
| AR Rollout | Test  | 0.039898 | 0.051268 | 0.699530 |

**Comments:**

- AR rollout gives a small lift on validation
- Test performance is slightly lower than baseline in this setup

---

## v14.x Series: Spatial Feature Integration

### v14.1.0: **VALID (Temporal + Static Spatial Features)**

**Description:**
Adds static spatial columns to the split (`derived_with_spatial`) and trains a full XGB model

#### Results

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.000980 | 0.001418 | 0.999799 |
| Val   | 0.036482 | 0.046329 | 0.790106 |
| Test  | 0.045071 | 0.056951 | 0.637054 |

**Comments:**

- Train fit is near perfect while val and test lag hard, so this run looks heavily overfit
- Notebook logs include the same split warning for `Touchet_WA_824` with missing validation rows

---

## v15.x Series: derived_6.0 Feature Set

### v15.1.0: **VALID (108-Feature Stack Baseline)**

**Description:**
First v15 run on `derived_6.0` with a 108-feature set, XGB + RF base learners, and ridge stack

#### Results

**Base XGB**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.001182 | 0.001727 | 0.999715 |
| Val   | 0.030309 | 0.040403 | 0.839083 |
| Test  | 0.037712 | 0.048471 | 0.731428 |

**Final stacked model**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.018155 | 0.018412 | 0.967579 |
| Val   | 0.027748 | 0.036115 | 0.871422 |
| Test  | 0.036667 | 0.045751 | 0.760723 |

**Comments:**

- Big val lift from stacking
- Test improves vs base XGB and lands in the same range as stronger v13 results

---

### v15.2.0: **VALID (Config Re-run)**

**Description:**
Re-run of v15.1 style config with the same 108-feature setup.

#### Results

**Final stacked model**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.018155 | 0.018412 | 0.967579 |
| Val   | 0.027748 | 0.036115 | 0.871422 |
| Test  | 0.036667 | 0.045751 | 0.760723 |

**Comments:**

- Metrics match v15.1 in notebook outputs
- Useful mostly as a reproducibility check

---

### v15.3.0: **VALID (Early-Stopped XGB + Ridge Stack)**

**Description:**
v15 follow-up with early-stopped native XGBoost training before RF + ridge stacking.

#### Results

**Base XGB**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.026787 | 0.035228 | 0.881312 |
| Val   | 0.024126 | 0.031645 | 0.901284 |
| Test  | 0.036420 | 0.046557 | 0.752211 |

**Final stacked model**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.027858 | 0.036491 | 0.872646 |
| Val   | 0.023081 | 0.029901 | 0.911862 |
| Test  | 0.035724 | 0.045650 | 0.761775 |

**Comments:**

- Best validation score in v15 so far
- Best v15 test score too, but still below the historical v7.4 reference

---

## v16.x Series: Drift-Aware Tuning on derived_6.0

### v16.1.0: **VALID (Year-Weighted Stack Baseline)**

**Description:**
Introduced year-weighting on `derived_6.0` and compared base XGB, RF, and stacked ridge combinations

#### Results

**Base XGB (train+val fit)**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Train | 0.001208 | 0.001781 | 0.999697 |
| Val   | 0.001032 | 0.001484 | 0.999783 |
| Test  | 0.037152 | 0.047758 | 0.739264 |

**Final stacked model (`XGB + RF -> Ridge`)**

| Split | MAE      | RMSE     | $R^2$    |
| ----- | -------- | -------- | -------- |
| Test  | 0.036008 | 0.046100 | 0.757054 |

**Comments:**

- Stacking improved over base XGB in `v16.1`, but remained below the later drift-aware variants
- A calibrated-XGB stacking variant was effectively identical on test (`R^2 = 0.757042`)

---

### v16.2.0: **VALID (Drift Features + Calibration)**

**Description:**
Added drift-aware temporal interaction features and tested both train+val fits and post-hoc ridge calibration flows.

#### Results

| Model                                  | Test MAE     | Test RMSE    | Test $R^2$ | Test bias (true - pred) |
| -------------------------------------- | ------------ | ------------ | ---------- | ----------------------- |
| Baseline XGB (train+val)               | 0.037150     | 0.047756     | 0.739264   | 0.013023                |
| Drift XGB (train+val)                  | _not logged_ | _not logged_ | 0.790900   | 0.003315                |
| Baseline train-only + ridge calibrator | 0.039309     | 0.048865     | 0.727043   | -0.000641               |
| Drift train-only + ridge calibrator    | _not logged_ | _not logged_ | 0.761099   | -0.000861               |

**Comments:**

- Drift-aware features produced a major test $R^2$ gain vs the baseline branch.
- 2023 slice was still unstable/negative (`R^2 = -0.101033`) in this version

---

### v16.3.0: **VALID (Drift Expansion + Weighting Study)**

**Description:**
Expanded drift feature set and explicitly compared weighted vs unweighted drift-XGB variants.

#### Results

| Model                        | Test MAE | Test RMSE | Test $R^2$ | Test bias (true - pred) |
| ---------------------------- | -------- | --------- | ---------- | ----------------------- |
| Drift (no weights)           | 0.032218 | 0.041547  | 0.802674   | 0.002829                |
| Drift (`beta = 0.2` weights) | 0.031798 | 0.041061  | 0.807260   | 0.002716                |

**Comments:**

- Weighted branch was the winner inside v16.3 (`R^2 = 0.807260`).
- 2023 metrics improved vs v16.2 but remained negative (`R^2 = -0.060867`).

---

### v16.4.0: **VALID (Aggressive Drift Tuning)**

**Description:**
More aggressive drift-model tuning on the expanded feature stack.

#### Results

| Model              | Test $R^2$   | Test bias (true - pred) |
| ------------------ | ------------ | ----------------------- |
| Drift (no weights) | **0.811571** | 0.002041                |
| Drift (weighted)   | 0.799298     | 0.001932                |

**Comments:**

- Highest full-test $R^2$ among all logged valid runs to date!
- 2023 slice crossed into slightly positive territory (`R^2 = 0.002914`, slope `0.861997`).

---

### v16.5.0: **VALID (Add `J_` Feature Family)**

**Description:**
Added `J_` family features and reran drift weighted/unweighted variants.

#### Results

| Model              | Test $R^2$ | Test bias (true - pred) |
| ------------------ | ---------- | ----------------------- |
| Drift (no weights) | 0.804560   | 0.001712                |
| Drift (weighted)   | 0.810545   | 0.002032                |

**Comments:**

- Slightly below v16.4 on full-test $R^2$, but stronger 2023 behavior.
- Best 2023 diagnostic in v16 series (`R^2 = 0.037897`, slope `1.010004`).

---

## v17.x Series: Expanded Test Split

### v17.1.0: **VALID (Expanded Test Years)**

**Description:**
Same drift baseline idea with a tougher split. Test now covers full 2023 to 2025 and train is smaller.

#### Results

| Model              | Test MAE | Test RMSE | Test $R^2$ | Test bias (true - pred) |
| ------------------ | -------- | --------- | ---------- | ----------------------- |
| Drift (no weights) | 0.03005  | 0.04058   | 0.81424    | -0.00400                |
| Drift (weighted)   | 0.02879  | 0.04005   | 0.81910    | -0.00044                |

**Comments:**

- Weighted won again
- 2023 slice looked great here (`R^2 = 0.813318`, slope `0.942258`)

---

## v18.x Series: Calibration Pass

### v18.3.0: **VALID (Calibrated Drift Model)**

**Description:**
Focused calibration pass on `derived_8.0` with weighted and unweighted drift branches.

#### Results

| Model              | Test MAE | Test RMSE | Test $R^2$ | Test bias (true - pred) |
| ------------------ | -------- | --------- | ---------- | ----------------------- |
| Drift (no weights) | 0.02925  | 0.03974   | 0.82186    | -0.00218                |
| Drift (weighted)   | 0.02804  | 0.03939   | 0.82503    | -0.00308                |

**Comments:**

- Best logged test score so far
- Weighted branch is the one to keep as the reference

---

## v19.x Series: derived_8.0 Feature Set

### v19.2.0: **VALID (Feature Set Refresh)**

**Description:**
Moved forward with `derived_8.0` and tested a fresh feature set pass.

#### Results

| Model              | Test MAE | Test RMSE | Test $R^2$ | Test bias |
| ------------------ | -------- | --------- | ---------- | --------- |
| Drift (no weights) | 0.03039  | 0.04059   | 0.81418    | -0.00284  |
| Drift (weighted)   | 0.02832  | 0.03968   | 0.82239    | -0.00320  |

**Comments:**

- Nice run, but still a touch behind `v18.3`
- 2023 slice stayed strong (`R^2 = 0.802718`, slope `0.919413`)

### v19.3.0: **VALID (Base Model Benchmark)**

**Description:**
Forked `v19.2` into a benchmark notebook that compares `XGBoost`, `CatBoost`, `LightGBM`, and `Random Forest` as base models on the same `derived_8.0` weighted train+val to held-out test-station setup. All four models trained on year-weighted train+val, evaluated on the held-out test station (N = 4,016).

#### Results

**Full test set (N = 4,016)**

| Model         | $R^2$   | MAE     | RMSE    | ubRMSE  | Bias     | MedAE   | P90 AE  |
|---------------|---------|---------|---------|---------|----------|---------|---------|
| Random Forest | **0.8309** | 0.02824 | **0.03873** | **0.03871** | **−0.00112** | 0.02138 | 0.05975 |
| CatBoost      | 0.8242  | 0.02946 | 0.03948 | 0.03923 | −0.00442 | 0.02293 | 0.05975 |
| XGBoost       | 0.8224  | **0.02832** | 0.03968 | 0.03956 | −0.00320 | **0.02032** | 0.06152 |
| LightGBM      | 0.8067  | 0.03044 | 0.04140 | 0.04124 | −0.00370 | 0.02305 | 0.06301 |

**2023 slice diagnostics (N = 1,266)**

| Model         | $R^2_{2023}$ | Slope  | Intercept |
|---------------|--------------|--------|-----------|
| Random Forest | **0.8398**   | 0.9428 | 0.01376   |
| CatBoost      | 0.8305       | **0.9567** | **0.00765** |
| LightGBM      | 0.8031       | 0.9278 | 0.01287   |
| XGBoost       | 0.8027       | 0.9194 | 0.01692   |

**Top-5 consensus features (mean normalized importance across all four models)**

| Feature                      | Consensus Importance |
|------------------------------|----------------------|
| `V_rollmin_LST_modis_kobs30` | 0.1270               |
| `K_aspect_cos`               | 0.0714               |
| `C_lag_LST_modis_kobs30`     | 0.0479               |
| `D_sin_DOY`                  | 0.0422               |
| `SMAP_sm_pm_interp_ema02`    | 0.0407               |

**Comments:**

- **Random Forest wins overall** on $R^2$, RMSE, ubRMSE, and bias; XGBoost edges it on MAE and MedAE.
- All four models cluster tightly between $R^2 = 0.807$ and $R^2 = 0.831$ — no single model dominates by a large margin, which is a useful signal for ensembling.
- LightGBM trails the other three on every metric at this configuration and is the weakest base learner in this setup.
- The 2023 slice ranking mirrors the full-test ranking: RF leads, CatBoost second, XGB and LGB tied.
- CatBoost shows the best calibration on the 2023 slice (slope closest to 1.0 at 0.957, smallest intercept).
- Rolling-minimum LST (`V_rollmin_LST_modis_kobs30`) is the dominant signal by a wide margin; terrain aspect (`K_aspect_cos`) is unexpectedly strong as the second-ranked consensus feature.
- Artifacts exported: `base_model_benchmark_metrics.csv`, `base_model_benchmark_predictions.csv`, `base_model_benchmark_2023_slice.csv`, `base_model_feature_importance_top30.csv`.

---

## v20.x Series: Variance Checks

### v20.1.0: **VALID (Variance and Heteroskedasticity Pass)**

**Description:**
Kept the drift model on `derived_8.0`, then checked how much variance and heteroskedasticity were still hanging around.

#### Results

| Model              | Test MAE | Test RMSE | Test $R^2$ | Test bias |
| ------------------ | -------- | --------- | ---------- | --------- |
| Drift (no weights) | 0.03039  | 0.04059   | 0.81418    | -0.00284  |
| Drift (weighted)   | 0.02832  | 0.03968   | 0.82239    | -0.00320  |

**Comments:**

- Same top-line performance as the logged `v19.2` weighted branch
- Formal variance tests still came back strongly heteroskedastic
- Good diagnostic run, not a new best model

---

### v20.3.0: **VALID (Dry Regime Specialist Pass)**

**Description:**
3 regime specialists.

#### Results

| Model                | Test MAE | Test RMSE | Test $R^2$ | Test bias |
| -------------------- | -------- | --------- | ---------- | --------- |
| Dry regime model     | 0.02550  | 0.03541   | 0.85860    | -0.00904  |

**Comments:**

- Best reported score in the v20 block
- Stronger than `v18.3` on the numbers provided here
- Logged as a dry-regime result, so compare it with that context in mind
