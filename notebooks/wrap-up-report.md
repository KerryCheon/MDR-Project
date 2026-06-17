# MoE Handoff Report

**Author:** Jakob Balkovec
**Date:** June 8th 2026

---

## 1. Context: Why MoE?

The baseline XGBoost model (v19.3, dataset derived 9.0, internally relabeled as v25 now) achieves strong temporal performance ($R^2$ = 0.822, MAE = 0.0283, RMSE = 0.0397) on held-out test data (Jan 2023 -- Dec 2025). However, spatial generalization is the main open problem.

**Within-WA LOSO results:**

| Station | R² |
|---|---|
| Darrington | 0.730 |
| Spokane | 0.728 |
| SourdoughGulch | 0.545 |
| Quinault | 0.498 |
| Touchet | 0.403 |

**Out-of-state (6 stations, western US):**

| Station | Climate Match | R² |
|---|---|---|
| OR Riley | Similar | 0.614 |
| ID Murphy | Similar | 0.608 |
| CA Redding | Moderate | 0.497 |
| CO Boulder | Moderate | 0.431 |
| WY Lander | Different | 0.358 |
| MT Wolf Point | Different | 0.209 |

A broader 10-test spatial diagnostic suite across 31 stations showed near-zero or negative R² at many unseen stations. Two root causes were confirmed:

1. **Single-regime low-variance stations**: R² misbehaves on flat targets (the denominator of R² collapses when a station has low variance in its target). Not a model failure; it's a metric artifact.

2. **Feature distribution shift**: Test stations operate outside the training feature range. Key examples: `G_rain_sum_3d` dropped ~4x from train to test; `year_frac` shifted substantially. SHAP analysis showed that when a station's top driver is a static descriptor (elevation, soil texture) rather than a seasonal/temporal feature, mean R² drops sharply.

> **Important Note:** Touchet was excluded from the primary model scope due to regime imbalance. It is 68% Dry and semi-arid, making it unrepresentative of the WA training distribution.

The regime-aware (MoE) approach was motivated by the hypothesis that a single XGBoost model handles `Dry`, `Transition`, and `Wet` conditions with a single learned mapping, which may be suboptimal at the tails.

---

## 2. Regime Definitions

Regimes are defined by thresholds on `soil_moisture_5cm`. Thresholds were originally set on derived 8.0 and then recalibrated for derived 9.0 using the **train-set 33rd and 66th percentiles**.

**Derived 9.0 thresholds (current):**

| Regime | Condition |
|---|---|
| Dry | $\text{SM} < 0.0993$ |
| Transition | $0.0993 \leq \text{SM} < 0.2115$ |
| Wet | $\text{SM} \geq 0.2115$ |

> **Warning:** The original thresholds (DRY\_MAX = 0.20, WET\_MIN = 0.313) were calibrated on derived 8.0 and are misaligned with the 9.0 test distribution. **Do not reuse them.**

> **Note:** Might be worth exploring a different approach to obtain these thresholds. The current method is very simple and does not account for the fact that the regimes are not equally sized in feature space. One thing I had in mind then (but never got to it) was a more clustering approach or using domain knowledge to define more meaningful boundaries...

Separability analysis confirmed strong statistical separation between regimes (K2 ratios in the range 167–722), so the boundaries are not the problem. The Transition class is the hard one because it is the most heterogeneous regime by definition.

> **See report** on regime separability analysis for more details.

---

## 3. What I Tried

### 3.1 3-Class Regime Classifier (Dry | Transition | Wet)

Trained a multi-class XGBoost classifier to route samples to one of three expert models.

**Result:**
- Overall accuracy: ~63%
- Transition recall: 41–48% (stuck regardless of tuning)
- Wet recall: ~23%

The Wet recall issue is a **class imbalance artifact**, not a boundary definition problem. Separability analysis confirmed the regimes are statistically distinct; the classifier is just undersupplied with Wet examples.

The Transition recall ceiling is a harder problem. Transition samples by definition span an intermediate moisture range where Dry-like and Wet-like dynamics co-exist. The model cannot reliably distinguish them from their neighbors. This is **poor class separability**, not a tuning problem...further hyperparameter search is unlikely to move the needle meaningfully.

> **Note:** I confirmed this by running separability diagnostics. The K2 ratios are high across all pairs, but Dry–Transition and Transition–Wet pairs are closer together in feature space than Dry–Wet. The classifier struggles most at those boundaries.

**SMOTE was explored and abandoned.** SMOTE oversamples the minority class (Wet) in feature space, but it does not address temporal distribution shift. The unseen station problem is about feature range mismatch across geographic locations, not class imbalance within a station.

### 3.2 Binary Classifier (Dry | Wet+Transition)

Collapsed Wet and Transition into a single "Non-Dry" class.

**Result:**
- Overall accuracy: 80%

This was proposed as the basis for a **2-expert MoE** architecture: one expert for Dry conditions, one for Wet+Transition. The gating model is simpler, better calibrated, and avoids the Transition recall wall.

> **Note:** The final decision between the 3-class and binary path was **LEFT OPEN**.

> **See**: Section 5

### 3.3 Soft Gating (Weighted Blending of Expert Predictions)

Explored soft gating where the router outputs class probabilities, and the final prediction is a weighted blend:

$$\hat{y} = \sum_{k} p_k \cdot \hat{y}_k$$

where $p_k$ is the gating probability for regime $k$ and $\hat{y}_k$ is the prediction from expert $k$.

**Critical implementation bug discovered:** Expert predictions are positionally aligned arrays (not pandas Series with matching indices). When doing weighted blending after subsetting by regime, predictions from sparse regimes are not re-indexed. This causes silent failures:

- `0.0 * NaN = NaN` propagates without warning
- The final blended prediction has NaN in rows where a regime is absent

**Fix:** Reindex each expert's prediction array back to the full index before blending, or initialize the output array to zeros and fill by regime.

### 3.4 Gate Leakage Bug (v21.3)

v21.3 reported a val $R^2$ of **0.980**, which is invalid. The routing model was trained on train+val combined, then evaluated on val. This is a data leakage issue...the gating model had already seen the val labels during training.

**v22.3** corrected this with OOF (out-of-fold) routing trained on the train split only. This is the clean baseline for any MoE routing evaluation. Use v22.3 as the reference point.

> **Warning:** Do not cite v21.3 metrics anywhere. They are not reproducible under clean conditions

### 3.5 Interaction Features (Abandoned)

Explored interaction terms between static soil properties ($\text{texture} \times \text{elevation}$) and dynamic weather features ($\text{rain accumulation} \times \text{API}$) to try to give the model regime-specific sensitivity.

**Result:** Near-perfect collinearity (Pearson $r > 0.995$) across all interaction pairs. The features added no independent signal and were dropped entirely

> **Note:** Do not revisit this unless you change the feature set substantially

---

## 4. What Did Not Work and Why

| Approach | Result | Why |
|---|---|---|
| 3-class gating (XGBoost) | ~63% accuracy, Transition recall ~41–48% | Class separability problem at Dry–Transition and Transition–Wet boundaries; not a tuning issue |
| SMOTE for class imbalance | No spatial improvement | Does not address temporal/geographic distribution shift |
| Interaction features (static × dynamic) | Dropped | Near-perfect collinearity ($r > 0.995$), zero marginal signal |
| Soft gating (naive blending) | Silent NaN failures | Index alignment mismatch between expert arrays |
| v21.3 MoE (gated) | R² = 0.980 (invalid) | Gate leakage: router trained on train+val, evaluated on val |

---

## 5. Open Decision: Binary vs. 3-Class

> **Note**: This was NOT resolved before handoff

**Option A: Binary (2-expert MoE)**
- Gating: Dry vs. Non-Dry, 80% accuracy
- Simpler, cleaner, better recall across both classes
- Loses the Transition-specific expert, but Transition was the hardest to classify anyway
- Recommended if you want a working MoE baseline fast

**Option B: 3-class (3-expert MoE)**
- Gating: Dry | Transition | Wet, ~63% accuracy
- Theoretically richer; Transition gets its own expert
- Transition recall ceiling makes this architecturally fragile
- Higher risk; lower expected payoff without a fundamentally different gating approach

> **Note:** My lean was toward the binary path as the pragmatic starting point, with 3-class as a stretch goal only if binary shows a clear ceiling. But this is a judgment call...both are defensible.

---

## 6. What Is Still Pending

- Final MoE architecture decision (binary vs. 3-class)
- Clean end-to-end MoE evaluation using v22.3 routing baseline
- Spatial $R^2$ improvement: the MoE framing alone is unlikely to fix feature distribution shift. The more impactful path is probably **more geographically diverse training data** (add stations from OR, ID, CA to training), which directly attacks the root cause

---

## 8. Relevant Notebook Versions

| Version | Status | Notes |
|---|---|---|
| v19.3 (v25) | Baseline | Current best. Derived 9.0, temporal $R^2$ = 0.822 |
| v21.3 | Invalid | Gate leakage. Do not use for evaluation |
| v22.3 | Clean MoE baseline | OOF routing on train only. Use this as MoE reference |
| v23.1+ | Active | Where continued MoE/spatial work lives |

---

*Jakob Balkovec*
