## Summary of Observations (Diagnostics)

### 1) Generalization Gap (XGB vs RF)

XGB:

- Train $R^2 = 0.9997$
- Val $R^2 = 0.836$
- Test $R^2 = 0.692$

RF:

- Val $R^2 = 0.845$
- Test $R^2 = 0.761$

**Interpretation:** XGB generalizes worse than RF under temporal shift. This looks like temporal nonstationarity + high model capacity (boosting sensitivity), not feature selection failure.

---

### 2) Systematic Bias in XGB (Temporal Drift Signature)

Base XGB test bias:

$$
\mathbb{E}[y - \hat{y}] \approx +0.017
$$

So XGB underpredicts on average.

Bias decays over test years:

- 2023: +0.0375
- 2024: +0.0201
- 2025: +0.0120

Post-hoc calibration:

$$
\hat{y}_{\text{cal}} = a \hat{y}_{\text{XGB}} + b
$$

improves test $R^2$ from $0.692 \rightarrow 0.727$, indicating a smooth global misalignment (scale/offset + mild curvature), not structural failure.

---

### 3) Stacking Behavior (Why It Helps)

Stacked (XGB + RF → Ridge):

- Test $R^2 \approx 0.768$

Weights:

$$
w_{\text{XGB}} \approx 0.46,\quad w_{\text{RF}} \approx 0.55
$$

Residual similarity is high:

$$
\text{Corr}(e_{\text{XGB}}, e_{\text{RF}}) \approx 0.91,\quad
\text{Corr}(|e_{\text{XGB}}|, |e_{\text{RF}}|) \approx 0.82
$$

So stacking is not “two totally different regime experts.” It helps mainly via stability under drift and smoothing bias/variance. The benefit concentrates where XGB struggles (especially 2023 + mid-range).

---

### 4) Prediction Compression (Confirmed, Not Just Suspected)

Global test calibration fit:

- slope $\approx 0.944$
- std ratio $\sigma_{\text{pred}}/\sigma_y \approx 0.904$

Year-wise compression (major drift signal):

- 2023: slope $\approx 0.385$, std ratio $\approx 0.660$ (severe compression)
- 2024: slope $\approx 0.893$, std ratio $\approx 0.965$
- 2025: slope $\approx 0.996$, std ratio $\approx 0.854$

So compression is primarily a 2023 phenomenon.

---

## Why 2023 Is So Bad (Hard Evidence)

Target distribution shift:

- Train mean $y \approx 0.195$, 2023 mean $y \approx 0.288$
- Train std $y \approx 0.102$, 2023 std $y \approx 0.046$
- 2023 $y$ range is compressed: min/max $\approx 0.146/0.378$

Feature distribution shifts (joint shift, not just one variable):

- $G\_API$ mean: train $\approx 43$ → 2023 $\approx 51$ (wetter conditions)
- SMAP mean: train $\approx 0.395$ → 2023 $\approx 0.298$ (lower satellite signal)
- LST mean: train $\approx 287.5$ → 2023 $\approx 276.4$ (cooler period)

Missingness is _not_ the cause:

- avg missing train $\approx 4.6\%$
- avg missing 2023 $\approx 0\%$

---

## Core Problem

**Temporal transition / nonstationarity** entering 2023:

- higher baseline moisture
- lower variance
- shifted joint feature geometry

Boosting is more sensitive to this abrupt drift than RF. Stacking helps largely because RF is more robust in the drift-heavy region (2023) and in mid-range predictions where XGB compresses.

---

## Solution Directions (Pool)

- Downweight older years during training (preserve sample size, bias toward current regime).
- Add drift-aware features (year index, anomaly features, trend features) to relax stationarity.
- Regularize XGB / reduce capacity to limit over-specialization to old regimes.
- Use calibration and/or stacking as the robust baseline while experimenting with drift-aware training.
