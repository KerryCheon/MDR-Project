# Regime Gating Analysis Report (v19 to v25)

This report details how regime-based gating (Dry, Transition, Wet) has been explored, implemented, and modified across model versions from **v19** to the latest **v25** in the MDR soil moisture modeling project.

---

## 1. Context & Motivation

Soil moisture dynamics exhibit highly non-linear, regime-dependent behavior:
- **Dry Regime**: Characterized by low-variance, slow-drying, soil-texture-dominated behavior.
- **Wet Regime**: Characterized by rapid gravitational drainage, saturation, and rainfall-driven peaks.
- **Transition Regime**: Characterized by high heterogeneity where dry-like and wet-like dynamics co-exist.

A single global model (like the XGBoost baseline in v19.3) handles these regimes with a single learned mapping, which can be suboptimal at tails and struggles with **spatial generalization** when transferred to unseen stations in Washington state (or out-of-state). 

To specialize the learning process, a **Mixture of Experts (MoE)** framework was proposed:
1. Divide the soil moisture space into three regimes.
2. Train specialist models (experts) for each regime.
3. Use a gating model (router) to dynamically route samples or blend expert predictions.

---

## 2. Regime Boundaries & Threshold Calibration

Regimes are defined by thresholds on `soil_moisture_5cm` (and its proxy, the base model prediction). The thresholds evolved across dataset versions:

| Dataset / Context | Dry Threshold ($T_1$) | Wet Threshold ($T_2$) | Calibration Strategy |
| :--- | :--- | :--- | :--- |
| **Derived 8.0** | $0.2000$ | $0.3130$ | Historical baseline. |
| **Derived 8.1** | $0.1910$ | $0.2910$ | Recalibrated on 8.1 train set. |
| **Derived 9.0 (Recalibrated)** | $0.0993$ | $0.2115$ | 33rd and 66th percentiles of the 9.0 train set. |
| **v24 / v25 (Quantile baseline)** | $0.0912$ | $0.2130$ | Latest train set 33rd/66th percentiles. |

> [!WARNING]
> **Threshold Misalignment / Lockout**: Several later notebooks (e.g., `MDR-v21.5.ipynb` and `MDR-v22.3-portable.ipynb`) locked thresholds to $T_1 = 0.200$ and $T_2 = 0.313$, which are misaligned with the `derived_9.0` test distribution. In contrast, `MDR-v24-main.ipynb` uses the updated $0.0912$ / $0.2130$ boundaries.

---

## 3. Timeline of Gating Architectures (v19 to v25)

The table below outlines the evolution of gating approaches in the notebooks:

| Version | Status | Gating/Routing Mechanism | Test $R^2$ | Key Findings & Notes |
| :--- | :--- | :--- | :--- | :--- |
| **v19.3** | Valid | **None (Baseline)** | `0.822` | Single global XGBoost regression model on `derived_9.0`. Set as the baseline anchor. |
| **v20.2** | Valid | **Hard vs. Soft Gating** | `0.822` (Soft)<br>`0.791` (Hard) | Soft gating (weighted blending using sigmoid weights around $T_1$/$T_2$) outperformed hard routing by smoothing predictions at boundaries. |
| **v20.3** | Valid | **Oracle Hard Gating** | `0.859` | Upper-bound diagnostic using true labels to route samples. Showed maximum MoE potential. |
| **v20.5** | Valid | **Transition Classifier Gating** | — | A binary classifier routes transition samples to base (if stable) or transition specialist (if unstable). |
| **v20.6** | Valid | **Double Pass Self-Gating & Residual Experts** | `0.801` (Resid) | Gating based on base predictions. Residual expert (`base_pred + specialist_residual`) beat direct expert routing (`0.759`) but lagged the baseline. |
| **v21.1 / 2**| Valid | **Predicted Soft Gating** | `0.822` / `0.810` | Explored "Gating Damage"—misrouting errors propagate and compound quickly. |
| **v21.3** | **Invalid** | **Soft Gating + Base Fallback** | `0.980` (Val) | **Gate Leakage**: Router was trained on train+val combined, causing an invalid $R^2$ spike on validation data. |
| **v21.4 / 5**| Valid | **Base-Dominant Ensemble** | `0.826` | Introduced confidence-based blending. Blends expert predictions up to a maximum weight $\lambda_{max} = 0.25$, falling back to base at boundaries. |
| **v21.4 / 5**| Valid | **SMAP Satellite Gating** | `0.321` (Gating Acc) | Replaced the 100+ feature router with SMAP satellite data. Failed due to high scaling mismatch. |
| **v22.3** | Valid | **OOF Routing & Blending Bug Fixes** | `-0.621` (Buggy)<br>`0.822` (Fixed) | Corrected gate leakage using Out-of-Fold (OOF) base predictions. Resolved a critical silent NaN propagation bug. |
| **v23.1** | Valid | **Survey (No Gating)** | `0.826` | Shifted back to baseline survey comparing RF, XGB, and KNN. |
| **v24 / v25** | Valid | **Weighted Single-Model (No Gating)** | `0.832` | Abandoned MoE in favor of **temporal recency** and **regime-balance** sample weighting. |

---

## 4. Key Gating Methodologies

### 4.1 Hard Gating (Deterministic Routing)
Samples are assigned a single discrete regime based on the gating signal:
$$R = \begin{cases} 
0 \text{ (Dry)} & \hat{y}_{base} \le T_1 \\ 
1 \text{ (Transition)} & T_1 < \hat{y}_{base} \le T_2 \\ 
2 \text{ (Wet)} & \hat{y}_{base} > T_2 
\end{cases}$$
The prediction is routed entirely to a single specialist: $\hat{y} = \hat{y}_{R}$. Hard boundaries suffer from **regime misclassification propagation**—if the base model predicts $0.199$ instead of $0.201$, the sample is routed to the wrong specialist, introducing step-function errors.

### 4.2 Soft Gating (Weighted Blending)
Introduced in `MDR-v20.2.ipynb` to smooth transitions:
- Compute sigmoid weights centered on the thresholds:
  $$s_1 = \sigma\left(\frac{\hat{y}_{base} - T_1}{w_1}\right), \quad s_2 = \sigma\left(\frac{\hat{y}_{base} - T_2}{w_2}\right)$$
- Derive regime weights:
  $$w_{\text{dry}} = 1 - s_1, \quad w_{\text{wet}} = s_2, \quad w_{\text{trans}} = s_1 - s_2$$
- Normalize weights to sum to 1 and blend:
  $$\hat{y}_{\text{blended}} = w_{\text{dry}} \hat{y}_{\text{dry}} + w_{\text{trans}} \hat{y}_{\text{trans}} + w_{\text{wet}} \hat{y}_{\text{wet}}$$

### 4.3 Base-Dominant Ensemble (Confidence Blending)
Introduced in `MDR-v21.4.ipynb` to limit the damage of incorrect routing. Instead of picking an expert, the ensemble behaves as a regularized correction to the base model:
$$\hat{y} = \hat{y}_{base} + \alpha \cdot \left(\hat{y}_{expert\_mix} - \hat{y}_{base}\right)$$
where the blending weight $\alpha$ is dynamically scaled by two confidence factors:
1. **Distance Confidence** ($\text{conf}_{dist}$): Blending weight goes to 0 near thresholds $T_1$ and $T_2$ (falling back entirely to the base model).
2. **Agreement Confidence** ($\text{conf}_{agree}$): Blending weight goes to 0 if the experts disagree wildly (high standard deviation among expert predictions).
3. **Scale Factor** ($\alpha = \lambda_{max} \cdot \text{conf}_{dist} \cdot \text{conf}_{agree}$): The maximum expert influence is restricted by capping $\lambda_{max} = 0.25$ (meaning the model remains at least 75% dominated by the base model).

### 4.4 SMAP Satellite Gating
A 1-D satellite-derived gating signal using `SMAP_sm_interp_lag1` was evaluated in `MDR-v21.4-portable.ipynb`. Gating thresholds for SMAP were calibrated by matching the train percentiles:
- Dry: $\text{SMAP} \le t_{1\_smap}$ (where $t_{1\_smap}$ is the 85th percentile of SMAP in dry regime rows).
- Wet: $\text{SMAP} \ge t_{2\_smap}$ (where $t_{2\_smap}$ is the 15th percentile of SMAP in wet regime rows).
- **Result**: Gating accuracy collapsed to **32%** on test data. Due to the spatial resolution mismatch (9 km SMAP vs. point in-situ sensor), SMAP values are too coarse and noisy to serve as direct gating signals for local station models.

---

## 5. Key Implementation Bugs & Troubleshooting

### 5.1 Gate Leakage (v21.3)
- **Problem**: In `v21.3`, the gating model was trained on `train + val` combined and then evaluated on `val`. This led to an artificially inflated validation $R^2 = 0.980$.
- **Fix (v22.3)**: Implement **Out-of-Fold (OOF) cross-validation** on the train set only:
  1. Train the base model on `train_df` using 5-fold cross-validation to generate out-of-fold predictions (`pred_train_base_oof`).
  2. Use these OOF predictions to construct the regime masks and features used to train the experts on the training set.
  3. Validate the combined MoE pipeline on the validation set using clean, unseen base predictions from a model trained strictly on the train split.

### 5.2 Silent NaN Propagation Bug
- **Problem**: In early versions of v22.3, the test $R^2$ collapsed to `-0.621`. This was caused by an array alignment mismatch. Specialists were trained only on their respective subset of samples. When returning predictions, the subsets were not re-aligned to the full dataset index.
- Since `0.0 * NaN = NaN`, multiplying the soft gating weights (which are defined for all rows) with the unaligned sparse expert prediction arrays propagated `NaN` across the entire validation/test series.
- **Fix (v22.3)**: 
  1. Initialize full-sized pandas Series for each expert using `np.nan` indexed to the full dataset.
  2. Populate the subsets using `.loc[mask]` or `.update()`.
  3. Fill the remaining NaNs in the expert prediction series with `0` before multiplying by weights:
     ```python
     pred_combined = (
         w_dry * pred_dry_full.fillna(0).values +
         w_trans * pred_trans_full.fillna(0).values +
         w_wet * pred_wet_full.fillna(0).values
     )
     ```

---

## 6. The Transition to Sample Weighting (v24/v25)

By **v24** and **v25**, Mixture of Experts (MoE) gating was largely abandoned in favor of a single-model approach trained with **custom sample weights**. This shift was driven by two key insights:
1. **Transition Recall Ceiling**: The transition regime is too heterogeneous to classify or route reliably. The K2 separability metrics between Dry-Transition and Transition-Wet are close, making routing classification errors unavoidable.
2. **Gating Error Propagation**: Gating errors degrade overall performance so severely that even highly optimized specialist experts fail to outperform a well-regularized single model.

### Current Gating-Alternative (v24/v25 Weighting System):
Instead of routing samples to specialists, a single XGBoost model is trained with combined sample weights:
$$w_{\text{combined}} = w_{\text{temporal}} \times w_{\text{regime}}$$

1. **Temporal Recency Weighting**: 
   $$w_{\text{temporal}} = e^{\beta(Y - Y_{\text{max}})}$$
   (where $\beta = 0.2$ and $Y$ is the year). This prioritizes recent years to prevent temporal drift.
2. **Regime Balance Weighting**:
   $$w_{\text{regime}} = \frac{1}{\text{frequency of regime } k}$$
   This weights the samples inversely by their regime size (Dry, Transition, Wet) to prevent the model from ignoring underrepresented regimes.

This weighted single-model architecture achieved a test $R^2$ of **0.832**, outperforming all prior MoE variants under clean, leak-free validation.

---

## 7. Conclusions & Recommendations

- **Soft Gating vs. Hard Gating**: If a future MoE architecture is pursued, **always prefer soft blending** (Section 4.2) or **base-dominant confidence blending** (Section 4.3). Hard gating introduces step-function errors near thresholds that degrade performance.
- **Data Leakage Risk**: When constructing gating features, ensure that the base model's predictions on the training set are strictly **Out-of-Fold (OOF)** to prevent gate leakage.
- **Satellite Gating Limitations**: Do not use raw satellite measurements (like SMAP) as a direct gating signal due to spatial scaling mismatch.
- **Current Best Practice**: Prioritize sample weights (temporal + regime balance) over MoE specialist routing. It provides equivalent or superior performance with a fraction of the architectural complexity.
