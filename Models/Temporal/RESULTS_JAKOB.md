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

---

## Best Model

**Section Last Updated on:** Fri Jan 9th, 2025
**Selected Model:** **v3.3.1 — Robust XGBoost (Row 177)**
**Status:** **VALID**

### Configuration Summary

- `max_depth = 3`
- `min_child_weight = 100`
- `lambda = 5`
- `subsample = 1.0`
- `colsample_bytree = 0.6`
- 39 temporally valid derived features
- Trained under strict per-station temporal split

### Final Performance

| Split | MAE      | RMSE     | R²       |
| ----- | -------- | -------- | -------- |
| Train | 0.023669 | 0.032485 | 0.898234 |
| Val   | 0.023175 | 0.030704 | 0.918315 |
| Test  | 0.038198 | 0.049690 | 0.714397 |

### Rationale for Selection

This model was selected as the **final best model** due to its superior balance between performance and robustness:

**Pros**

- Highest test-set R^2 among all temporally valid models
- Extremely robust under temporal drift
- Clean inductive bias via shallow trees and high `min_child_weight`
- Low risk of station-level memorization
- Stable performance across train, validation, and test splits

**Cons**

- Sacrifices approximately **0.001–0.0015 R²** compared to more aggressive configurations

This trade-off is intentional. Under real-world temporal generalization, robustness and stability were prioritized over marginal validation gains.

### Notes

- More aggressive configurations (v3.3.2, v3.3.3) achieved comparable validation performance but showed slightly worse generalization to the held-out test period.
- Architectural experiments (v4.x) and linear models (v5.x) did not improve test performance.
- This model serves as the **final reference point** for all future analysis, visualization, and reporting.

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

- **Model A**learns the dominant soil-moisture drivers (rainfall, API, seasonality, static context)
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

INVALID

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
