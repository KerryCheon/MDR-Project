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

---

## Best Model

**Section Last Updated on:** Sat Feb 14th, 2026
**Selected Model:** **v7.4.0 — Stacked XGB + RF → Ridge**
**Status:** **VALID**

### Configuration Summary

- Base learners: tuned `XGBRegressor` + `RandomForest`
- Meta-learner: `Ridge`
- Features: 40 temporally valid derived features
- Trained under strict temporal split

### Final Performance

| Split | MAE      | RMSE     | $R^2$        |
| ----- | -------- | -------- | ------------ |
| Test  | 0.033130 | 0.044186 | **0.776814** |

### Rationale for Selection

This model is the current **best valid reference** after excluding v12.7.0.

**Pros**

- Highest test-set $R^2$ among stable, historically validated models
- Stacking reduces over-reliance on any single model's bias
- Remains robust under temporal drift

**Cons**

- More moving parts than a single model, so it is harder to interpret
- Needs periodic re-checks when feature distributions drift

This trade-off is intentional: robustness and consistency are prioritized for the reference baseline.

### Notes

- v12.7.0 is marked **INVALID** and is excluded from model selection.
- v3.3.x remains the best single-model XGB baseline.
- This model is the **current reference point** for downstream analysis and reporting.

---

## Overview

This document tracks the evolution of the **temporal soil moisture regression model**, including all major experimental versions, validation status, and known issues.

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

Dump from v10

```
Best validation R2: 0.8237884024403164
Best params:
  learning_rate: 0.031802897015642956
  max_depth: 8
  min_child_weight: 14.153263657924109
  subsample: 0.9720330017585932
  colsample_bytree: 0.8303197403379534
  gamma: 2.3861026002162537e-08
  reg_alpha: 0.02411624638533363
  reg_lambda: 18.137253935049337
```

Results

```
FINAL RESULTS
Test -> R2: 0.77014 | MAE: 0.03470 | RMSE: 0.04606

Best params (final):
  learning_rate: 0.031802897015642956
  max_depth: 8
  min_child_weight: 14.153263657924109
  subsample: 0.9720330017585932
  colsample_bytree: 0.8303197403379534
  gamma: 2.3861026002162537e-08
  reg_alpha: 0.02411624638533363
  reg_lambda: 18.137253935049337
```

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

- Test slice check from artifacts: wet `R^2 = 0.572281`, dry `R^2 = 0.973953`.
- INVALID because the new lagged features we're constructed from the ground truth (leak)
