# Locked eval protocol (2.1)

Aligned with `MDR-v25.ipynb` and `derived_8.0-optimization-1.0`. Use this checklist for any future XGBoost global eval so metrics stay comparable.

## Checklist

| Item | Required value |
|------|----------------|
| Seed | `SEED = 42` (`random`, `numpy`, `PYTHONHASHSEED`, XGB `random_state`) |
| Train split | `train + val` concatenated; score on `test` |
| Hparams | 1.3-lite: `reg:squarederror`, `max_depth=8`, `min_child_weight=10`, `n_estimators=1500`, `learning_rate=0.01`, `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=1.5`, `reg_alpha=0.03` |
| Device | **`device="cuda"`** when XGBoost CUDA probe succeeds (same as opt-1.0); else `cpu` |
| Tree method | `hist` |
| Temporal weights (drift) | `w = exp(β · (year − t_max)) / mean(w)` with **β = 0.2** (mean-normalized) |
| No drift | `sample_weight=None` (uniform) |
| Missing values | **No median impute** — missing feature columns raise; XGBoost handles NaNs in-feature if present |
| Residual / Bias | `err = y_true − y_pred`; `Bias = mean(err)` |
| ubRMSE | `std(err)` (population std as in opt-1.0 / sklearn-style on residual) |
| RMSE | `√mean(err²)` |
| MAE / Med\|Err\| | mean / median of `\|err\|` |
| Pearson | `corrcoef(y_true, y_pred)[0, 1]` |

## Do not reintroduce (bugs from 2.0)

1. Unnormalized temporal weights (`exp(-β·Δyear)` without `/ mean(w)`).
2. Median imputation on features before fit.
3. Bias as `mean(pred − true)` (sign flip).
4. ubRMSE as `√(RMSE² − bias²)` only if Bias uses a different residual definition.

## Feature lists

This directory is self-contained. Load V6 lists from **local** artifacts (seeded from 2.0; re-run `run_selection.py` to regenerate):

- V6 variants: `artifacts/<dataset>/<variant>/selected_features.json`
- opt-1.0 pipeline set: `../derived_8.0-optimization-1.0/selected_features.json` (shared experiment; not copied)
- hand MDR-v25 (38 features, hardcoded in `eval.ipynb` / `run_eval.py`)
- 8.2 V3/V5: `data/splits/derived_8.2/dataset_metadata.py`

## Sanity targets (this machine / GPU)

| Config | Reference |
|--------|-----------|
| 8.0 hand + drift + 1.3-lite | opt-1.0 Model 5 ≈ **R² 0.8253** (GPU) |
| Env noise band | ΔR² ≲ **0.003** vs that reference is acceptable |
