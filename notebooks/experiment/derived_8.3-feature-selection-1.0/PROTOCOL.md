# Locked Eval Protocol: derived_8.3-feature-selection-1.0

This document defines the evaluation protocol for feature selection on `derived_8.3`.

## Data Splits & Target
- Dataset: `data/splits/derived_8.3/` (`train.csv`, `val.csv`, `test.csv`)
- Target column: `soil_moisture_5cm`
- Train setup: `train.csv` + `val.csv` combined for training, `test.csv` for evaluation.

## Evaluator (1.3-Lite XGBoost)
```python
XGB_PARAMS_LITE = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "min_child_weight": 10,
    "reg_lambda": 1.5,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 1500,
    "learning_rate": 0.01,
    "tree_method": "hist",
    "device": "cuda",  # fallback to cpu if probe fails
    "n_jobs": -1,
    "random_state": 42,
}
```

## Weighting Schemes
1. **No drift (unweighted)**: Uniform sample weights. (Primary metric for selecting `OVERALL_SELECTED_FEATURES_V1`).
2. **With drift**: Exponential sample weights $\beta = 0.2$, mean-normalized:
   $$w_i = \frac{\exp(\beta (t_i - t_{\max}))}{\text{mean}(\exp(\beta (t - t_{\max}))}$$

## Key Metrics
- $R^2$, RMSE, ubRMSE, Bias, MAE, Med|Err|, Pearson correlation.
