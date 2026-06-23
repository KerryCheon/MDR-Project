# notebooks/

Jupyter notebooks for model training and evaluation.

## training/

Active notebooks representing the final/ongoing model versions. Top-level files are the current best models; versioned subdirectories hold newer experiment lines that bypass `archive/`:

- `MDR-TemporalSpatial-v2.1.ipynb` — temporal-spatial transfer notebook
- `MDR-v25.ipynb` — latest model version
- `TemporalDelta-v0/MDR-TD-v0.ipynb` — temporal-delta modeling, v0
- `TemporalDelta-v1/MDR-TD-v1.0.ipynb`, `MDR-TD-v1.1.ipynb` — temporal-delta modeling, v1
- `Temporal-v20/MDR-v20.4.1-portable.ipynb`, `MDR-v20.4.2.ipynb`, `MDR-v20.5-portable.ipynb` — v20 portable/new variants
- `Temporal-v21/MDR-v21.4-portable.ipynb`, `MDR-v21.5.ipynb` — v21 portable/new variants
- `Temporal-v22/MDR-v22.3-portable.ipynb` — v22 portable variant

### training/archive/

Full versioned notebook history (v1–v24, all sub-versions). Organized by version number. Notable archived entries:

- `archive/v20/MDR-v20.8.ipynb` — three-regime XGBoost with spatial evaluation
- `archive/v23/MDR-v23.1.ipynb` — model survey (XGBoost, Gradient Boosting baseline comparison)
- `archive/v24/MDR-v24.ipynb` — spatial generalization experiments on the 5-station split
- `archive/v24/MDR-v24-main.ipynb` — spatial generalization main analysis

Do not modify archived notebooks — create new versions for new experiments to preserve history.

## Notebook environment (uv)

The notebook environment is managed with [uv](https://docs.astral.sh/uv/) and lives in this directory (`notebooks/`):

- `pyproject.toml` — project deps (torch, scikit-learn, xgboost, pandas, matplotlib, statsmodels) + jupyter dev deps + uv torch index (cpu/cu128 extras)
- `uv.lock` — locked dependencies
- `.python-version` — Python 3.12

To start Jupyter Lab with the notebook environment loaded:

```bash
cd notebooks
uv run --with jupyter jupyter lab
```

## evaluation/

Notebooks focused on model diagnostics and result analysis:

- `eval.ipynb` — primary evaluation notebook
- `main_eval.ipynb` — main evaluation pipeline
- `regime_separability.ipynb` — dry/transition/wet regime analysis
- `best_model_analysis.ipynb` — analysis of the best checkpoint
- `regime_feature_importance_top30.ipynb` — SHAP feature importance by regime
