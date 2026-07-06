# Feature Selection Pipeline Review: Verification Report

**Date:** 2026-07-06
**Source document:** `docs/plans/20260706-feature_selection_pipeline_review.md`
**Verifier:** CommandCodeBot
**Overall verdict:** The review is accurate. The critical bug is real and reproducible, and the design concerns are grounded in the current implementation.

---

## 1. [CRITICAL BUG] Silent Feature Alignment Mismatch

**Status:** VERIFIED

**Evidence:**
- `Modeling/Src/soilmoist_fl/Selectors/mi.py` fits `SimpleImputer(strategy="median")` and then calls `_rank_dict_from_scores(feature_cols, scores)`, which zips the original column list with the returned scores.
- `Modeling/Src/soilmoist_fl/Selectors/elasticnet.py` builds a `Pipeline` of `imputer → scaler → ElasticNetCV` and then zips `feature_cols` with `enet.coef_`.
- `sklearn.SimpleImputer` v1.8.0 (the version in the project's `notebooks/` uv environment) drops all-NaN columns by default (`keep_empty_features=False`).

**Reproduction:**
A synthetic 3-column dataframe was constructed where `colA` is all-NaN and `colB`/`colC` are informative.

- `select_mi` returned `ranked: ['colA', 'colB']` and assigned `colB`'s mutual-information score to `colA`. `colC` was silently truncated.
- `select_elasticnet` returned `ranked: ['colA', 'colB']` and selected both, including the useless `colA`.

**Proposed fix:**
Using `get_feature_names_out()` to obtain the kept-feature names and assigning `0.0` to any dropped features was tested and works correctly.

---

## 2. Linear Bias in a Non-Linear Modeling Stack

**Status:** VERIFIED

**Evidence:**
- `Modeling/Configs/default.yaml` configures final models as `linear`, `xgb`, and `rf`.
- Selection stages are `mi → elasticnet → stability`, and `stability_bootstrap_elasticnet` calls `select_elasticnet` repeatedly.
- ElasticNet is a linear regularized model, so it can assign zero coefficients to features whose predictive power is non-linear or interaction-based, filtering them out before the tree-based models (`xgb`, `rf`) can use them.

---

## 3. Collinearity and Coefficient Dilution in Lags

**Status:** VERIFIED as a sound theoretical concern

**Evidence:**
- The pipeline ranks features by absolute ElasticNet coefficient and keeps only the top `k=60` after the elasticnet stage.
- ElasticNet combines L1 and L2 penalties; the L2 component distributes coefficient mass across highly correlated variables such as lagged soil-moisture features.
- The stability stage requires `min_freq=0.6`. Collinear lag features can fail this threshold because their individual selection frequencies are diluted across bootstrap runs.

---

## 4. Computational and Statistical Flaws in Stability Selection

**Status:** VERIFIED

**Evidence:**
- `stability_bootstrap_elasticnet` (`Modeling/Src/soilmoist_fl/Selectors/stability.py`) calls `select_elasticnet` inside every bootstrap iteration.
- `select_elasticnet` instantiates `ElasticNetCV(cv=5)` (`Modeling/Src/soilmoist_fl/Selectors/elasticnet.py`), so each bootstrap performs 5-fold cross-validation.
- The default config does not set `stability_n_boot`, so `Modeling/Src/soilmoist_fl/cli.py` falls back to `n_boot=10`. This results in 10 × 5 = 50 ElasticNet fits.
- The recommendation to fit `ElasticNetCV` once to choose `alpha` and `l1_ratio`, then run a plain `ElasticNet` in the bootstrap loop, matches standard stability-selection methodology.

---

## 5. Entropy Estimation Issues under High Missingness (mi.py)

**Status:** VERIFIED for `derived_9.0`; does **not** hold for `derived_8.2`.

**Data check:** The canonical split `data/splits/derived_9.0/train.csv` was inspected.

- Shape: `29,362 rows × 499 columns`.
- After excluding `station_id` and `date`, there are **497 feature columns**.
- **426 feature columns have >50% missing values** (85.7% of feature columns).
- Maximum per-column missingness ratio is **0.8137** (81.37%), matching the source document.
- **426 feature columns are completely NaN for at least one station** (30 stations total).

**Source of the missingness:**
The high missingness is driven primarily by satellite-derived features that are absent for most stations in `derived_9.0`:

| Feature / family | All-NaN stations in derived_9.0 | Derived from |
| --- | --- | --- |
| `F_NDMI` | 25 / 30 | MODIS/Sentinel spectral index |
| `F_NDVI` | 25 / 30 | MODIS/Sentinel spectral index |
| `F_MSI` | 25 / 30 | MODIS/Sentinel spectral index |
| `s1_vv`, `s1_vh` | 25 / 30 | Sentinel-1 SAR |
| `s2_b4`, `s2_b8`, `s2_b11`, `s2_b12` | 25 / 30 | Sentinel-2 bands |
| `LST_modis` | 25 / 30 | MODIS land-surface temperature |
| `SMAP_sm_am_interp` | 26 / 30 | SMAP soil moisture |
| **V family** (168 columns) | up to 25 / 30 | Rolling stats on `F_NDMI` |
| **A family** (80 columns) | up to 25 / 30 | Diffs/gradients on `F_NDMI` |
| **C family** (57 columns) | up to 25 / 30 | Lags on `F_NDMI` and other remote features |

Per-station counts of all-NaN feature columns in `derived_9.0`:

- **19 stations** have exactly 426 all-NaN feature columns.
- **5 stations** have 424 all-NaN feature columns.
- **5 stations** escape the bulk missingness: Touchet, Quinault, Darrington, Spokane, and SourdoughGulch.

**Comparison with `derived_8.2`:**
`data/splits/derived_8.2/train.csv` (`15,704 rows × 499 columns`, 12 stations) shows a completely different pattern:

- **0 feature columns have >50% missing values.**
- Maximum per-column missingness ratio is **14.69%**.
- **0 feature columns are completely NaN for any station.**
- The same satellite features and their derivatives are fully populated (0% missing).

**Impact:**
`mutual_info_regression` uses the Kraskov-Stögbauer-Grassberger (KSG) k-NN estimator. In `derived_9.0`, median-imputing 50–80% of a column's values creates dense ties and biases nearest-neighbor distances, producing noisy mutual-information scores. Because `derived_8.2` has only ~15% missingness at worst, this entropy-estimation issue does not apply there.

**Note:** The source document states "426 out of 499 columns"; the precise figure is **426 out of 497 feature columns** (or 426 out of 499 total columns). The substantive claim is unchanged.

---

## Additional Context

The `Modeling/Configs/default.yaml` still points to a macOS path under `/Users/jbalkovec/Desktop/.../derived_new/`, which does not exist in this repo. The canonical local split is `data/splits/derived_9.0/`. This path issue is separate from the review findings but should be updated when reproducing the pipeline.

---

## What if the data were complete?

If the satellite-derived features were fully populated (i.e., no high missingness), the impact of the five findings would change as follows:

**Still present:**

- **Issue 2 — Linear bias in a non-linear stack:** Using ElasticNet to pre-select features for Random Forest and XGBoost would still suppress non-linear and interaction effects.
- **Issue 3 — Collinearity / coefficient dilution:** Lagged soil-moisture features would still be highly collinear, and ElasticNet's L2 penalty would still split their coefficients.
- **Issue 4 — Stability selection waste:** Running `ElasticNetCV(cv=5)` inside every bootstrap iteration would still be computationally wasteful and add hyperparameter noise, independent of missingness.

**Would disappear or become latent:**

- **Issue 5 — MI entropy bias:** Would disappear. With complete data, there are no median-imputed ties to bias the KSG k-NN estimator.
- **Issue 1 — Silent alignment bug:** Would become latent. With no all-NaN columns, `SimpleImputer` would not drop anything, so the `zip`-based alignment would not trigger. However, the bug remains in the code: if the pipeline is run per-station, on a subset, or in a bootstrap sample where a feature becomes all-NaN, it would silently corrupt rankings again.

In short, fixing the missing data would resolve the immediate MI problem and reduce the chance of hitting the alignment bug, but the design-level issues (linear selector, collinearity handling, stability methodology) would still need to be addressed.
