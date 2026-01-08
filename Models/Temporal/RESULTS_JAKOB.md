# Temporal Soil Moisture Model

## Results Log and Experimental History

**Author:** Jakob Balkovec

---

## Table of Contents

1. [Overview](#overview)
2. [v1.x Series — Early Baselines and Invalid Experiments](#v1x-series--early-baselines-and-invalid-experiments)
   - [v1.0.0 — Valid Baseline (No Derived Features)](#v100--valid-baseline-no-derived-features)
   - [v1.1.0 — Invalid (Feature Leakage)](#v110--invalid-feature-leakage)
   - [v1.2.0 — Invalid (Feature Leakage)](#v120--invalid-feature-leakage)
3. [v2.x Series — Expanded Data and Corrected Features](#v2x-series--expanded-data-and-corrected-features)
   - [v2.1.0 — Invalid (Expanded Split, Leakage)](#v210--invalid-expanded-split-leakage)
   - [v2.2.0 — Valid Baseline (Derived Features Fixed)](#v220--valid-baseline-derived-features-fixed)
   - [v2.3.0 — Valid Baseline (Handpicked Derived Features)](#v230--valid-baseline-handpicked-derived-features)
4. [Template for New Model Versions](#template-for-new-model-versions)

---

## Overview

This document tracks the evolution of the **temporal soil moisture regression model**, including all major experimental versions, validation status, and known issues.

Each model version is explicitly labeled as **VALID** or **INVALID** based on data leakage, split correctness, and feature integrity.
Only **VALID** models should be used as baselines for ablation studies or future comparisons.

Metrics reported:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

---

## v1.x Series — Early Baselines and Invalid Experiments

### v1.0.0 — **VALID Baseline (No Derived Features)**

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

### v1.1.0 — **INVALID (Feature Leakage)**

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

### v1.2.0 — **INVALID (Feature Leakage)**

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

## v2.x Series — Expanded Data and Corrected Features

### v2.1.0 — **INVALID (Expanded Split, Leakage)**

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

### v2.2.0 — **VALID Baseline (Derived Features Fixed)**

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

### v2.3.0 — **VALID Baseline (Handpicked Derived Features)**

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

### v3.1.0 — **VALID Baseline (All Derived Features)**

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

### v3.2.0 — **VALID Baseline (All Derived Features)**

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

## Template for New Model Versions

```markdown
## vX.Y.Z — [VALID / INVALID] Short Description

**Description:**
Brief summary of what changed relative to the previous version.

### Results

| Split | MAE | RMSE | R²  |
| ----- | --- | ---- | --- |
| Train |     |      |     |
| Val   |     |      |     |
| Test  |     |      |     |

**Comments:**

- Key changes
- Feature additions/removals
- Split details
- VALID or INVALID justification
```
