## Feature Cadence, Imputation Strategy, and Required Additions

**Author**: Jakob Balkovec

**Purpose**: Document decisions, constraints, and next steps for handling temporal resolution issues in satellite-derived soil-moisture features.

### 1. Core Issue

Sentinel-1 provides ~6-day sampling. Sentinel-2/MODIS provide even sparser sampling (cloud-dependent).
Many temporal features you originally proposed assume daily measurements, which are not natively available.

We now face two valid paths:

1. Use raw 6-day data and drop or reformulate daily features
2. Impute daily values and compute daily features from the reconstructed time series

My plan is to pursue (2), but with scientifically defensible safeguards!

---

### 2. Strategy: Daily Imputation + Transparent Documentation

**Why?**

- Enables Lag-1, gradients, rolling windows, etc.
- Much stronger ML performance
- Maintains internal temporal coherence
- Still grounded by true Sentinel-1 observations

**"How to stay scientifically safe?"**

- The imputer must be trained only on real data
- Imputed values should be labeled as synthetic
- Validate the imputer using held-out real observations
- Compute all temporal features after imputation
- Provide diagnostics (errors, time-series comparisons, confidence)

This approach apparently aligns with common practices in hydrology and remote sensing ML research...

---

### 3. Required Additions to the Pipeline

#### A. Additional Imputers

I'm currently mostly relying on `LinearRegression` and `XGBoost`. To support more complex features and reduce bias, I have to add:

- a `KNN` Imputer (temporal distance aware)
- Gaussian Process Regression (captures smooth trends + uncertainty)
- Random Forest Regression (nonlinear alternative) `[maybe]`
- Seasonal naive imputer (fallback for long gaps) `[gotta do some research]`
- Spline interpolation models (for optical signals)

> **Note**: Each imputer needs to be selectable per feature...maybe selectable is not the right word, but I need to be able to pick a certain imputer for a feature. Can be hardcoded or added to the `config.yaml`

#### B. Validation System for Imputed Time Series

This jsut crossed my mind. Something that we need is a mechanism to evaluate each imputer before it is used...

Validation tasks [`not_final]:

- Mask out real observations, pretend they are missing
- Predict them with the imputer

- **Compute**:

  - RMSE
  - MAE
  - MAPE
  - Peak error
  - Bias
  - Correlation

- **Visualize**:
  - Predicted vs. actual
  - Error histogram
  - Time-series overlay

> **Note**: This proves that the synthetic timestamps are credible.

#### C. Diagnostics Dashboard Enhancements

Diagnostics has to also include:

- Error metrics per **imputer**
- Error metrics per **feature**
- Error metrics per **gap-length category**

- **Summary statistics**:
  - “Gaps ≤ 3 days ... RMSE = X”
  - “Gaps 4–7 days ... RMSE = Y”
  - “Gaps ≥ 10 days ... unreliable”

This will show the quality of synthetic data

#### D. Exportable Imputation Report

If this ever becomes a paper, we need to track what imputer was used for what feature (unless hardcoded ... don't want to hardcode if we don't have to) and how it performed. This ensures "scientific" transparency.

The report could be a `.json/.csv/.xlsx/.log` file. I'm more leaning towards `.json` or `.log` just because the format is easier to read and we wouldn't need heavy post processing (might do some plots/summaries/dashboards)

The report has to include:

- Which imputer was used for each feature
- Summary of validation errors
- List of gaps filled
- Diagnostic plots [`maybe`...more of a postprocessing task]
- Feature-level confidence assessment
- Warnings for large-gap extrapolations [`mandatory`...keep it transparent]

This will/can essentially become part of the project documentation.

---

### 4. Updated Feature Definitions (Cadence-Aware)

**Features computed from real Sentinel-1 data (every 6 days):**

- `VV/VH` ratio
- Backscatter difference
- Coherence
- Lag-6 (instead of Lag-7)
- Lag-30
- Rolling Range (6-day window stepping)
- Fourier features
- Spectral entropy

**Features computed from imputed daily data:**

- Lag-1
- Lag-3
- Rolling mean
- Rolling std
- Rolling CV
- EMA
- Gradients
- Percent change
- Temporal roughness

Features based on optical data:

- Still use MODIS NDVI and MODIS LST (or use only when Sentinel-2 is missing)
- Impute missing dates only if gaps are short (set ceiling to 10 [`maybe`])
- Otherwise treat as sparse features and `WARN`

> **Note:** Set ceiling to **7 or 10**. The lower the better (the error will be smaller but we lose the ability to impute for larger gaps)

---

### 5. Updated Principles

Things that we have to stick to at all times...

1. Always identify which sensor each feature comes from

`S1, S2, MODIS, or imputed.`

2. Never allow synthetic values to train the imputer (Mostly for `XGBoost` as of Nov 21st but this could change)

> **Note:** Imputer is trained once...only on real timestamps.

3. Clearly separate:

- Original satellite measurements
- Imputed synthetic daily values
- Derived temporal features

> **Note**: These all go into the same dataset (master). Use the cache file (`satellite_cache.json`) for raw satellite measurements/data. Use a separate dataset for derived features...will have to go through some pipe to get computed anyways, and we can just export there. We might not have to separate the imputed synthetic data since we can just left-join and filter.

4. Provide evidence the imputer is accurate enough, before relying on daily temporal features.

5. **NEVER IMPUTE BEYOND THE TEMPORAL SUPPORT OF THE MODEL**

- If the longest gap is `x` we don't want to impute for `x + n`; where `n < 0`...means the model could/would hallucinate.

---

### 6. Action Items (Implementation Checklist)

**Imputer improvements**

- Add 3–5 new imputers
- Add cross-validation for imputers
- Add gap-length–aware evaluation

**Diagnostics**

- Implement RMSE/MAE/etc. reporting
- Implement time-series overlays
- Implement gap-based error summary
- Add per-feature diagnostics

**Reporting**

- Generate imputation report
- Document which imputer was used per feature
- Document validation error for each

**Feature engineering**

- Rewrite definitions to specify imputed/daily/sparse inputs
- Update rolling/lag windows to match cadence
- Add warnings for features that rely on long-gap imputations

No cross-feature dependency injection framework

---

**HARD TODOs:**

- Learn the weights from validation

Right now LM and XGB manually pull from: `["LST", "NDVI", "Rain_sat"]`

But the rules for what cross-features to use should be dynamic.

You need:

`CrossFeatureRegistry`

A small object that:

- looks up permitted predictors for each feature
- enforces no circular dependencies
- applies per-feature masks
- provides derived predictors (like NDVI anomalies, smoothed LST, etc.)

Right now cross-feature usage is too hardcoded..

If a cross-feature is missing during training or inference, LM and XGB quietly ffill/bfill it.

That is a correctness bomb.

You need:

`MissingnessPropagator`

If predictors are unreliable, the target should be considered unreliable too.

This requires:

- per-predictor confidence
- predictor gap lengths
- predictor missingness masks feeding into target

Right now, missing predictors are silently imputed, which contaminates multivariate imputation

No caching layer for repeated `.fit()` calls

Every time you run imputation for NDVI, LST, Rain_sat, you train models from scratch.

A caching layer saves:

- trained models
- fitted climatology
- temporal encodings
- weight estimates

This is required once the dataset grows (10+ years daily sats? It’ll kill you).
