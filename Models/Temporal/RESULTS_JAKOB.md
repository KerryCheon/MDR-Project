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

---

## Best Model

**Section Last Updated on:** Sun Jan 18th, 2025
**Selected Model:** **v7.4.0 — Stacked XGB + RF → Ridge**
**Status:** **VALID**

### Configuration Summary

- Base learners: tuned `XGBRegressor` + `RandomForest`
- Meta-learner: `Ridge`
- Features: 40 temporally valid derived features
- Trained under strict temporal split

### Final Performance

| Split | MAE      | RMSE     | R²           |
| ----- | -------- | -------- | ------------ |
| Test  | 0.033130 | 0.044186 | **0.776814** |

### Rationale for Selection

This model is the current **best overall** because it improves test performance without sacrificing stability.

**Pros**

- Highest test-set R^2 among all temporally valid models
- Stacking reduces over-reliance on any single model's bias
- Still robust under temporal drift
- Stable performance on the held-out period

**Cons**

- More moving parts than a single model, so it is a little harder to interpret
- Needs periodic re-checks when feature distributions drift

This trade-off is intentional. Under real-world temporal generalization, robustness and stability were prioritized over marginal validation gains.

### Notes

- v7.3.0 (post-hoc calibrated XGB) was strong, but v7.4.0 improved test R^2 and RMSE.
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
- **R²**: Coefficient of Determination

---

## v1.x Series: Early Baselines and Invalid Experiments

### v1.0.0: **VALID Baseline (No Derived Features)**

**Description:**
Initial baseline model using only base satellite bands and metadata. No derived temporal features included.

#### Results

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
| ----- | -------- | -------- | -------- |
| Train | 0.023669 | 0.032485 | 0.898234 |
| Val   | 0.023175 | 0.030704 | 0.918315 |
| Test  | 0.038198 | 0.049690 | 0.714397 |

#### v3.3.2: **Row 405**

##### Results

| Split | MAE      | RMSE     | R²       |
| ----- | -------- | -------- | -------- |
| Train | 0.023638 | 0.032479 | 0.898272 |
| Val   | 0.023065 | 0.030626 | 0.918733 |
| Test  | 0.038369 | 0.049903 | 0.711952 |

#### v3.3.3: **Row 267**

##### Results

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
| ----- | -------- | -------- | -------- |
| Train | 0.013086 | 0.018172 | 0.968154 |
| Val   | 0.038476 | 0.048866 | 0.793100 |
| Test  | 0.046930 | 0.060299 | 0.579425 |

**Comments:**

- Didn't bump \[R^2\] like I expected...actually it made everything worse. I could tune the hyperparameters more, but I think this is most likely a dead end

---

### v5.1.0: **VALID (ElasticNet, Diagnostic!)**

**Description:**
Using a new type of model (ElasticNet) instead of XGB.

#### Results

| Split | MAE      | RMSE     | R²       |
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

| Split | MAE      | RMSE     | R²       |
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

| Split | R²     |
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

| Split | MAE      | RMSE     | R²       | Bias Mean (True vs. Pred) |
| ----- | -------- | -------- | -------- | ------------------------- |
| Train | 0.034470 | 0.043565 | 0.818483 | 0.000045                  |
| Val   | 0.036769 | 0.047804 | 0.774720 | 0.023734                  |
| Test  | 0.036493 | 0.046039 | 0.757695 | 0.012184                  |

**After calibration**

| Split | MAE      | RMSE     | R²       | Bias Mean (True vs. Pred) |
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
| R²     | 0.77681427 |
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

| Split | MAE      | RMSE     | R²       | Bias Mean (True vs. Pred) |
| ----- | -------- | -------- | -------- | ----------------------- |
| Train | 0.001786 | 0.002553 | 0.999377 | -0.000003               |
| Val   | 0.034186 | 0.045249 | 0.798166 | 0.022045                |
| Test  | 0.039043 | 0.050376 | 0.709898 | 0.018762                |

**After post-hoc Ridge calibration (XGB)**

| Split | MAE      | RMSE     | R²       | Bias Mean (True vs. Pred) |
| ----- | -------- | -------- | -------- | ----------------------- |
| Train | 0.022184 | 0.022668 | 0.950855 | -0.022183               |
| Val   | 0.030050 | 0.039355 | 0.847319 | -0.000000               |
| Test  | 0.036139 | 0.046745 | 0.750212 | -0.003312               |

**Stacked (XGB + RF → Ridge) TEST metrics**

| Metric | Value      |
| ------ | ---------- |
| R²     | 0.75377602 |
| MAE    | 0.03556066 |
| RMSE   | 0.04641019 |
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

TBD
