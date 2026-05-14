# notebooks/

Jupyter notebooks for model training and evaluation.

## training/

Curated notebooks representing the final model versions used in the paper:

- `MDR-v20.8.ipynb` — three-regime XGBoost with spatial evaluation
- `MDR-v23.1.ipynb` — model survey (XGBoost, Gradient Boosting baseline comparison)
- `MDR-v24.ipynb` — spatial generalization experiments on the 5-station split
- `MDR-v24-main.ipynb` — spatial generalization main analysis
- `MDR-TemporalSpatial-v2.1.ipynb` — temporal-spatial transfer notebook

### training/archive/

Full versioned notebook history (v0–v22, v21, v22, all sub-versions). Organized by version number, mirroring the `Models/Temporal/` directory structure.

## evaluation/

Notebooks focused on model diagnostics and result analysis:

- `eval.ipynb` — primary evaluation notebook
- `main_eval.ipynb` — main evaluation pipeline
- `regime_separability.ipynb` — dry/transition/wet regime analysis
- `best_model_analysis.ipynb` — analysis of the best checkpoint
- `regime_feature_importance_top30.ipynb` — SHAP feature importance by regime
