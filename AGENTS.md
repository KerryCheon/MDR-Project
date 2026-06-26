# MDR Project

Spatio-temporal soil moisture modeling research project (IEEE AIIoT 2026).

## Repo Layout

| Directory | Purpose |
|-----------|---------|
| `src/pipeline/` | Ingestion → cleaning → imputation → feature engineering data pipeline (modular "pipes") |
| `data/splits/` | Train/validation/test split CSVs; `derived_9.0/` is the canonical split, `archive/` holds older versions |
| `notebooks/training/` | Active model training notebooks (final versions + newer experiment lines) |
| `notebooks/training/archive/` | Full versioned notebook history (v1–v24, all sub-versions) |
| `notebooks/evaluation/` | Model diagnostics: eval, regime separability, SHAP feature importance, best-model analysis |
| `experiments/` | EDA and one-off investigations (domain analysis, correlation, interpolation, missing values) |
| `Modeling/` | Feature selection lab (`soilmoist-fl`): MI → ElasticNet → Stability → Model eval |
| `d_models/` | Deep-learning model history (LSTM, GRU-transformer, TCN, transformer) |
| `results/figures/` | Saved figures organized by type (temporal, loso, spatial, shap, writeup) |
| `writeup/` | LaTeX paper draft (compile with `cd writeup && bash compile.sh`) |
| `paper/` | Submitted IEEE PDF |
| `tests/` | pytest suite |
| `docs/` | Reserved (empty) |

### Restructuring
The repo was restructured not long ago, so some paths may be outdated in the docs. The current structure is as above.
The last commit of the old structure was `6a9b9a044a5d5ddac7818b78544abaa7815e1fdd`, look up the commit history if you need to find old paths.

## Key Entrypoints

- **Data pipeline:** `PYTHONPATH=. python src/pipeline/main.py` (all stations) or `--station <key>` for one station. Config at `src/pipeline/config.yaml`.
- **Feature selection lab:** `Modeling/main.py` (or `python -m Modeling.Src.soilmoist_fl.cli run`). Config at `Modeling/Configs/default.yaml`. Note: paths in Modeling config may be hardcoded macOS paths — update for your environment.
- **Tests:** `make test` or `pytest tests/`.
- **Lint:** `make lint`.

## Notebooks

Active experiment notebooks live in `notebooks/training/`. Top-level files are the current best models; versioned subdirectories hold newer experiment lines that bypass `archive/`:

- `MDR-TemporalSpatial-v2.1.ipynb` — temporal-spatial transfer
- `MDR-v25.ipynb` — latest model version
- `TemporalDelta-v0/MDR-TD-v0.ipynb` — temporal-delta modeling, v0
- `TemporalDelta-v1/MDR-TD-v1.0.ipynb`, `MDR-TD-v1.1.ipynb` — temporal-delta modeling, v1
- `Temporal-v20/MDR-v20.4.1-portable.ipynb`, `MDR-v20.4.2.ipynb`, `MDR-v20.5-portable.ipynb` — v20 portable/new variants
- `Temporal-v21/MDR-v21.4-portable.ipynb`, `MDR-v21.5.ipynb` — v21 portable/new variants
- `Temporal-v22/MDR-v22.3-portable.ipynb` — v22 portable variant

The full version history (v1–v24) is in `notebooks/training/archive/`, organized by version number. If a user mentions a version like "v20" or "v22", check both `notebooks/training/Temporal-vXX/` (newer active lines) and `notebooks/training/archive/vXX/` (history). List the directory contents to see what's available; there are nested subdirectories (e.g. `archive/v20/v20.1/MDR-v20.1.ipynb`).

### Interacting with Notebooks

Use `jupyter-mcp-server` for reading, running and modifying notebooks rather than manipulating the string content directly, to ensure proper formatting and metadata handling.

The MCP interface is automatically configured to connect to a local Jupyter Lab. Reuse existing kernel sessions if possible (use `list_kernels` command), avoid starting multiple Jupyter kernels to prevent data out of sync issues.

#### Steps when using MCP to interact with notebooks:

1. `list_kernels` to see if there's an existing kernel session, remember the `id` of the kernel you want to use.
2. `use_notebook` with the notebook path and kernel id to connect to the notebook. This is essentially opening the notebook you want to use.

#### Notes on MCP Commands

- `list_files`: it starts from the `notebooks/` directory, not the project root, so the path should be relative to `notebooks/` (e.g. `training/TemporalDelta-v1/MDR-TD-v1.0.ipynb`). Since there are many notebooks, increase the `limit` parameter to 100 when listing from `/` to ensure you see all notebooks.
- `read_notebook` only works for notebooks you have `use_notebook`d. If it doesn't work, remind the user to check if the jupyter-collaboration extension is enabled.
- Never run `connect_to_jupyter` because it requires a token and can cause confusion.

## Notebook Environment (uv)

The notebook environment is managed with [uv](https://docs.astral.sh/uv/) and lives in `notebooks/`:

- `notebooks/pyproject.toml` — deps (torch, scikit-learn, xgboost, pandas, matplotlib, statsmodels) + jupyter dev deps + uv torch index (cpu/cu128 extras)
- `notebooks/uv.lock` — locked dependencies
- `notebooks/.python-version` — Python 3.12

To start Jupyter Lab with the notebook environment loaded:

```bash
cd notebooks
uv run --with jupyter jupyter lab
```

## Data Flow (Pipeline)

```
RequestPipe → ParsePipe → CleanPipe → MergePipe → SatellitePipe → WeatherPipe → TemporalFillPipe → WhittakerPipe → FeaturePipe → SavePipe
```

Output per station: processed final CSV under `src/pipeline/data/processed/`.

## Important Notes

- This is a team project with many contributors, so ensure configs are well-documented and never hard-code paths.
- This is a research project, so make sure to show all the works and ensure reproducibility.
- This is a large project with many moving parts, so only make changes you are confident about. If unsure, ask the team first.
- Do not modify existing versioned notebooks (e.g. in `archive/v9/`, `archive/v10/`) — create new versions for new experiments to preserve history.
- Must use the MCP interface for notebook interactions to maintain formatting and metadata integrity. Never read the notebook file directly.
