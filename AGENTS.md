# MDR Project

Spatio-temporal soil moisture modeling research project (IEEE AIIoT 2026).

## Status
There're two models in development:
1. Spatial model: `notebooks/training`.
2. LSTM-based Temporal models: `Models/Temporal/lstm`.

Currently we're focusing on stations within Washington state as part of the collaboration with the ECE team that develop and deploy new in-situ soil moisture sensors, for which will be used to evaluate the model's performance. Thus, out-of-region spatial generalization is not a concern for now (but we still need some sptial generalization within the state).

## Repo Layout

| Directory | Purpose |
|-----------|---------|
| `src/pipeline/` | Ingestion → cleaning → imputation → feature engineering data pipeline (modular "pipes") |
| `data/splits/` | Train/validation/test split CSVs; `derived_9.0/` is the canonical split, `archive/` holds older versions |
| `notebooks/experiment/` | New experiments goes here |
| `notebooks/training/` | Active model training notebooks (final versions + newer experiment lines) |
| `notebooks/training/archive/` | Full versioned notebook history (v1–v24, all sub-versions) |
| `notebooks/evaluation/` | Model diagnostics: eval, regime separability, SHAP feature importance, best-model analysis |
| `experiments/` | Legacy EDA and one-off investigations (domain analysis, correlation, interpolation, missing values) |
| `Modeling/` | Feature selection lab (`soilmoist-fl`): MI → ElasticNet → Stability → Model eval |
| `d_models/` | Deep-learning model history (LSTM, GRU-transformer, TCN, transformer) |
| `results/figures/` | Saved figures organized by type (temporal, loso, spatial, shap, writeup) |
| `writeup/` | LaTeX paper draft (compile with `cd writeup && bash compile.sh`) |
| `paper/` | Submitted IEEE PDF |
| `tests/` | pytest suite |
| `docs/` | Knowledge base (WIP) and random notes |

### Restructuring
The repo was restructured not long ago, so some paths may be outdated in the docs. The current structure is as above.
The last commit of the old structure was `6a9b9a044a5d5ddac7818b78544abaa7815e1fdd`, look up the commit history if you need to find old paths.

## Key Entrypoints

- **Data pipeline:** `PYTHONPATH=. python src/pipeline/main.py` (all stations) or `--station <key>` for one station, use `--config` to specify the config file; default config at `src/pipeline/config.yaml` but new datasets should version their own config files.
- **Feature selection lab:** `Modeling/main.py` (or `python -m Modeling.Src.soilmoist_fl.cli run`). Config at `Modeling/Configs/default.yaml`. Note: paths in Modeling config may be hardcoded macOS paths — update for your environment.
- **Tests:** `make test` or `pytest tests/`.
- **Lint:** `make lint`.

## Runtime Environment (uv)

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
- This is a research project, so make sure to show all the works and ensure full reproducibility. Don't just randomly define random constants, include the code used to generate/select these constants. Things in `/scratch` are tracked, so they don't considered reproducible. The rule of thumb is: if you can't provide the code to generate it, it's not reproducible.
- This is a large project with many moving parts, so only make changes you are confident about. If unsure, ask the team first.
- Do not modify existing versioned notebooks (e.g. in `archive/v9/`, `archive/v10/`) — create new versions for new experiments to preserve history.
- When the user asks to read, edit, execute, or work with .ipynb files, use the notebook-cli skill, which provides the `nb` command-line tool. Do not use the built-in Read/Write tools for `.ipynb` files. Never read or write the notebook file directly.

## General Coding Rules
- Prefer import existing constants/functions instead of redefining them.
- Use double quotes for string literals, unless there's nested quotes where using single quotes outside will save the expense of escaping.