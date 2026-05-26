# MDR Project

Spatio-temporal soil moisture modeling research project.

## Repo Layout
| Directory | Purpose |
|-----------|---------|
| `Temporal/Pipeline/` | Ingestion → cleaning → imputation → feature engineering data pipeline (modular "pipes") |
| `Temporal/Pipeline/data/splits/` | Train/validation/test split CSVs (station keys) |
| `Modeling/` | Feature selection lab (`soilmoist-fl`): MI → ElasticNet → Stability → Model eval |
| `Models/Temporal/` | Temporal model experiment notebooks (v20, v21, etc.) |
| `Models/TemporalSpatial/` | Combined temporal-spatial experiment notebooks (v1.x) |
| `Spatial/` | Spatial modeling work in progress (satellite retrieval, data prep scripts) |
| `WriteUp/` | LaTeX paper draft |

## Key Entrypoints
- **Temporal Pipeline:** `Temporal/Pipeline/main.py` — CLI with `--station <key>`, `--config <path>`, `--list-stations`
- **Root runner:** `run_pipeline.py` — adds project root to `sys.path`, imports `Temporal.Pipeline.main`
- **Feature selection lab:** `Modeling/main.py` (or `python -m Modeling.Src.soilmoist_fl.cli run`)
- **Jupyter kernels:** Run `uv run --with jupyter jupyter lab` from `Models/Temporal/` to start notebooks with project environment loaded.

## Config Notes
- **Temporal Pipeline config:** `Temporal/Pipeline/config.yaml` — station blocks + global impute/satellite/logging settings
- **Modeling config:** `Modeling/Configs/default.yaml` — data splits, selection stages, models, scoring
- **Paths in Modeling config are hardcoded macOS paths** (e.g., `/Users/jbalkovec/Desktop/MDR/...`) — update for your environment

## Data Flow (Temporal Pipeline)
```
RequestPipe → ParsePipe → CleanPipe → MergePipe → SatellitePipe → WeatherPipe → TemporalFillPipe → WhittakerPipe → FeaturePipe → SavePipe
```
Output per station: `Temporal/Pipeline/data/processed/<station>/final.csv`

### Dataset Splits
- Train/validation/test splits defined in `Temporal/Pipeline/data/splits/`; they combined multiple stations into one file.
- Currently it versioned till v9.

## Notebooks
Experiment notebooks at `Models/Temporal/v*/` and `Models/TemporalSpatial/v*/`. If user mentions something like "v20" or "v22", check those directories for the relevant notebook.
There're are many models and new models are being added, so please list all the versions in the `Models/Temporal/` directory to know what versions are available.
There might be nested directories like `v20/v20.1/....ipynb`, or in `v24/....ipynb` directory, so check those as well.

## Important Notes
- Do not modify existing versioned notebooks (e.g., `v9`, `v10`) — create new versions for new experiments to preserve history.
