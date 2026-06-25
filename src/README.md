# src/

Source code for the data ingestion and feature engineering pipeline.

## pipeline/

End-to-end pipeline for processing raw station and satellite data into model-ready feature matrices.

- `main.py` — entry point; runs all configured stations
- `config.yaml` — station list and pipeline settings
- `pipes/` — modular processing stages (parse, clean, merge, feature engineering, satellite retrieval, weather, save)
- `imputers/` — missing value imputation strategies (KNN, spline, climatology, XGBoost, voting ensemble, etc.)
- `records/` — daily observation record handling
- `utils/` — shared utilities (config loading, logging, math helpers)
- `smoothing/` — Whittaker smoother and Fourier transform smoothing
- `validation/` — input data validation
- `logs/` — log filtering utilities

## Usage

The preferred way to run the pipeline is using `uv`:

```bash
# Set Python path and run the pipeline
export PYTHONPATH="src"   # On Windows: $env:PYTHONPATH="src"
uv run -m pipeline.main
uv run -m pipeline.main --station spokane_17_ssw
```

Processed outputs are written to `src/pipeline/data/processed/<station>/final.csv`.
