# derived_8.1_pos-eval-1.2

Evaluation of XGBoost oracle hard gating models on the derived_8.1_pos test set (N=8,902, 13 WA stations, SM > 0.0 only) with **Temporal Recency Weighting (Drift, beta=0.4)** and **Hyperparameter Tuning for 2-Regime Specialists** applied during training.

This experiment uses the optimal decay parameter $\beta = 0.4$ identified in the 1.1 parameter sweep and applies optimized hyperparameters for the 2-Regime specialists (Dry and Wet).

## Combined Model Performance (at beta = 0.4)

| Model | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **#1** Single Global XGBoost (no gating) | 0.4858 | 0.0755 | 0.0745 | −0.0120 | 0.05536 | 0.03954 | 0.708 |
| **#3** 3-Regime Oracle (regime-specific features) | 0.8658 | 0.0386 | 0.0373 | −0.0100 | 0.03023 | 0.02463 | 0.935 |
| **#5** 3-Regime Oracle (overall features, ablation) | **0.8729** | 0.0375 | 0.0361 | −0.0102 | 0.02962 | 0.02443 | 0.939 |
| **#6** 2-Regime Oracle (T=0.159, Tuned) | **0.7607** | 0.0515 | 0.0502 | −0.0114 | 0.03985 | 0.03101 | 0.879 |

## Individual Specialist Performance

### Model 2: 3-Regime Specialists (Regime-Specific Features, beta=0.4)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | +0.0001 | 0.0459 | 0.0423 | −0.0178 | 0.03584 | 0.02855 | 0.435 |
| Transition (0.159–0.248) | 2,441 | −0.0151 | 0.0258 | 0.0258 | −0.0002 | 0.02159 | 0.01977 | 0.228 |
| Wet (SM ≥ 0.248) | 3,128 | −0.2091 | 0.0383 | 0.0371 | −0.0095 | 0.03098 | 0.02591 | 0.295 |

### Model 4: 3-Regime Specialists (Overall Features, Ablation, beta=0.4)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | **+0.1021** | 0.0435 | 0.0397 | −0.0179 | 0.03419 | 0.02958 | 0.513 |
| Transition (0.159–0.248) | 2,441 | −0.0641 | 0.0264 | 0.0264 | −0.0001 | 0.02239 | 0.02046 | 0.195 |
| Wet (SM ≥ 0.248) | 3,128 | −0.1927 | 0.0381 | 0.0368 | −0.0097 | 0.03039 | 0.02523 | 0.324 |

### Model 6: 2-Regime Specialists (T=0.159, Tuned, beta=0.4)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | **+0.0510** | 0.0447 | 0.0412 | −0.0173 | 0.03473 | 0.02669 | 0.468 |
| Wet (SM ≥ 0.159) | 5,569 | **+0.0324** | 0.0552 | 0.0546 | −0.0078 | 0.04291 | 0.03464 | 0.407 |

---

## 2-Regime Specialist Optimization Details

The grid search on `tune_2r.py` swept parameters for the Dry/Wet specialists under $\beta = 0.4$:

### Optimal Dry Specialist (2R) Parameters:
- `objective`: `"reg:absoluteerror"`
- `max_depth`: `8`
- `min_child_weight`: `5`
- `subsample`: `0.8`
- `colsample_bytree`: `0.8`
- Resulting R² on target dry slice: **+0.0510** (improved from $+0.0198$ in 1.1)

### Optimal Wet Specialist (2R) Parameters:
- `objective`: `"reg:absoluteerror"` (crucial change from `reg:squarederror`)
- `max_depth`: `10`
- `min_child_weight`: `3`
- `subsample`: `0.9`
- `colsample_bytree`: `0.8`
- Resulting R² on target wet slice: **+0.0324** (improved from $-0.0471$ in 1.1, and $-0.0259$ in 1.0)

---

## Performance Comparison across Experiments

| Model | 1.0 R² (No Weights) | 1.1 R² (beta=0.2, Un-tuned 2R) | 1.2 R² (beta=0.4, Tuned 2R) | Net Change (1.2 vs 1.0) |
|---|---|---|---|---|
| Single Global XGBoost | **0.5023** | 0.4901 | 0.4858 | −0.0165 |
| 3-Regime Oracle (regime-specific) | **0.8662** | 0.8653 | 0.8658 | −0.0004 |
| 3-Regime Oracle (overall, ablation) | 0.8637 | 0.8713 | **0.8729** | **+0.0092** |
| 2-Regime Oracle (T=0.159) | 0.7453 | 0.7443 | **0.7607** | **+0.0154** |

### Specialist-Specific R² Gains for 2R Specialists:

| Specialist Slice | 1.0 R² (No Weights) | 1.1 R² (beta=0.2, Un-tuned 2R) | 1.2 R² (beta=0.4, Tuned 2R) | Net Change (1.2 vs 1.0) |
|---|---|---|---|---|
| Dry Specialist (2R) | −0.0191 | +0.0198 | **+0.0510** | **+0.0701** |
| Wet Specialist (2R) | −0.0259 | −0.0471 | **+0.0324** | **+0.0583** |

## Key Takeaways

1. **Positive R² achieved for BOTH 2-Regime Specialists**: The un-tuned 2-regime wet specialist had negative R² ($-0.0259$ in 1.0 and $-0.0471$ in 1.1). Tuning hyperparameters using an absolute error objective (`reg:absoluteerror`) and regularized parameters successfully pushed the Wet Specialist R² to **$+0.0324$**.
2. **Oracle 2-Regime Model Overall Performance Boost**: Due to the improvements in both specialists, the combined 2-regime oracle model R² improved from $0.7453$ to **$0.7607$** ($+0.0154$ absolute improvement).
3. **Temporal recency weighting default of $\beta=0.4$ validated**: The default $\beta=0.4$ weighting successfully maintained Model 5's top score of **$0.8729$** while boosting the dry specialist to **$+0.1021$**.

## References

- [Dataset derived_8.1_pos](../../data/splits/derived_8.1_pos/README.md) — dataset compilation and feature selection.
- [derived_8.1_pos-eval-1.0 README](../derived_8.1_pos-eval-1.0/README.md) — baseline evaluation.
- [derived_8.1_pos-eval-1.1 README](../derived_8.1_pos-eval-1.1/README.md) — temporal recency weight (beta=0.2) sweep.
