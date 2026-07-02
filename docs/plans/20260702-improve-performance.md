Viewed sweep_beta.py:1-47

To squeeze more performance out of the **2-regime model** ($T = 0.159$), we can leverage several techniques that were successful in the 3-regime experiments but have not yet been fully adapted or optimized for the 2-regime layout.

Here are the four most promising strategies to explore:

---

### 1. Soft Gating (Sigmoid Blending) around $T = 0.159$
Currently, Model 6 uses a **hard boundary** (discrete step routing). If the base model predicts $0.158$, it routes 100% to the Dry Specialist; if it predicts $0.160$, it routes 100% to the Wet Specialist. This introduces a step-discontinuity (boundary step error) at the transition.

Following the success of **v20.2**, you can implement soft gating for the 2-regime model:
$$\hat{y} = (1 - s) \hat{y}_{\text{dry}} + s \hat{y}_{\text{wet}}$$
where the blending weight $s$ is computed using a sigmoid centered on the threshold $T = 0.159$:
$$s = \sigma\left(\frac{\hat{y}_{\text{base}} - 0.159}{w}\right)$$
Here, $w$ represents the transition window size (e.g., $w = 0.02$ or $0.03$). Blending predictions near the threshold will smooth out predictions and significantly reduce boundary errors.

---

### 2. Residual Specialists (MoE Residual Correction)
In **v20.6**, training specialist models to predict the **residuals** of the global baseline model rather than the raw soil moisture target yielded much higher stability:
1. Generate predictions from the global model: $\hat{y}_{\text{global}}$.
2. Compute training targets for the specialists as residuals: $\text{target}_{\text{specialist}} = y_{\text{true}} - \hat{y}_{\text{global}}$.
3. Train the Dry and Wet specialists to fit these residuals.
4. Predict: $\hat{y} = \hat{y}_{\text{global}} + \hat{y}_{\text{specialist\_residual}}$.

This is much easier to train because the global model handles the broad physical mapping, and the specialists only have to learn the local regime corrections.

---

### 3. Hyperparameter Tuning for the 2-Regime Wet Specialist
Currently, `XGB_PARAMS_WET_2R` uses:
- `max_depth = 10`
- `min_child_weight = 1`
- `n_estimators = 6000`

These parameters were tuned for the **3-regime Wet Specialist** (which only trains on $\text{SM} \ge 0.248$). 
In the 2-regime model, the "Wet Specialist" is trained on everything $\ge 0.159$, which is a much larger and more heterogeneous slice containing both transition drying and saturated wet peaks.
A deep tree (`max_depth=10`) trained on this broad set is highly prone to overfitting the high-density transition points at the expense of wet peaks. 

**Recommendation**: 
- Reduce the maximum depth to `max_depth = 7` or `8`.
- Increase `min_child_weight = 5` to regularize the leaf nodes.
- Use a safer loss objective like `objective="reg:pseudohubererror"` or `reg:absoluteerror` to handle transient rain peaks without warping the model.

---

### 4. Regime Balance Weighting within Slices
Within the 2-regime wet slice ($\text{SM} \ge 0.159$), the distribution is highly skewed towards the threshold (many transition points, very few extreme wet saturation peaks). 
Following the **v24/v25** concept, you can apply **Regime Balance Weighting** inside the specialist's training subset to penalize the loss of rarer points:
$$w_{\text{sample}} = w_{\text{temporal}} \times \frac{1}{\text{density}(y)}$$
This forces the Wet Specialist to learn the true upper-tail physics instead of ignoring them to minimize the loss on transition samples.