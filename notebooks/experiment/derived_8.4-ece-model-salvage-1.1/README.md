# Experiment: `derived_8.4-ece-model-salvage-1.1`

This experiment is a reproducible descendant of `derived_8.4-ece-model-salvage-1.0`. It locally reruns the MI → ElasticNet → stability → wrapper feature-selection pipeline after removing every case-insensitive `SMAP` feature, produces nested 40/50/60/69 feature manifests, and evaluates six routing families at 60 features plus the global model at all four sizes with learner seeds `[42, 7, 13]`. Router seed is fixed at 42. All selector, router, and model fitting uses WA trainval only; ECE targets are evaluation-only.

The executed notebook and this README are regenerated after the dependent Slurm stages complete:

```bash
cd notebooks/experiment/derived_8.4-ece-model-salvage-1.1
uv run --no-sync python run_feature_selection.py --stage all
uv run --no-sync python run_model_salvage.py --resume
uv run --no-sync python build_notebook.py
nb execute derived_8.4-ece-model-salvage-1.1.ipynb --uv --timeout 1800
uv run --no-sync python update_readme.py
```

The final notebook-generated report will include the full three-seed metrics, selector/router provenance, original-model and 1.0 comparisons, no-SMAP invariance checks, and five seven-line ECE station charts comparing the original global model, 1.0, the four 1.1 global sizes, and ground truth.

## Comparison of 1.1 selected features with 1.0 no-SMAP models

Pending completion of the 1.0 five-seed reference artifacts and the 1.1 three-seed GPU stage.

## Global model version comparison

Pending completion of the common-seed prediction artifacts. The final notebook-generated section will embed one seven-line chart for each of the five ECE stations.
