![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

# MDR — Temporal Pipeline

This repository contains the Temporal Pipeline which fetches, parses, cleans, merges,
and enriches environmental station data (USCRN/SNOTEL) and satellite-derived features.
It performs robust temporal gap-filling (ensemble imputation), smoothing, and
feature engineering to produce cleaned timeseries ready for downstream modeling or analysis.

Key features:

- Modular "pipes" architecture: Request → Parse → Clean → Merge → Satellite → Temporal Fill → Smooth → Feature → Save
- Pluggable imputer ensemble with voting strategy and per-imputer confidence
- Modular configuration via `config.yaml` (station-specific blocks)
- Docker-supported and runnable locally via Python venv
- Tests using pytest

## Quickstart — Docker

The recommended way to run the pipeline in a self-contained environment is via Docker.

From the repository root:

```bash
docker build -t temporal-pipeline -f src/pipeline/Dockerfile .

# Run the pipeline with data path mounted
docker run --rm \
  -v $(pwd)/src/pipeline/data:/app/src/pipeline/data \
  temporal-pipeline
```

## Local Development (Python)

The preferred method to manage the local environment and run the pipeline is using [uv](https://docs.astral.sh/uv/).

### Option A: Using uv (Preferred)

From the repository root:

```bash
# Sync and set up the virtual environment with all required dependencies
uv sync --all-packages

# Run the pipeline for all stations (module runner mode)
$env:PYTHONPATH="src"
uv run -m pipeline.main

# Run a single station from the CLI
uv run -m pipeline.main --station spokane_17_ssw

# Provide a custom config file
uv run -m pipeline.main --config src/pipeline/config.yaml

# List configured stations
uv run -m pipeline.main --list-stations
```

*Note: On Linux/macOS, use `export PYTHONPATH="src"` instead of `$env:PYTHONPATH="src"`.*

### Option B: Traditional pip venv (Fallback)

If you do not have `uv` installed, you can use standard Python virtual environments:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r src/pipeline/requirements.txt

# Set PYTHONPATH and run as module
export PYTHONPATH=src     # On Windows: $env:PYTHONPATH="src"
python -m pipeline.main
```

Notes:

- Use the `config.yaml` in `src/pipeline/config.yaml` to modify station parameters, years, and IMPUTER/SATELLITE settings.
- To run only a subset of stations, edit `config.yaml` (comment-out or remove the unwanted station blocks), or add a shorter year range.

## Configuration

- `config.yaml` controls pipeline execution. Top-level keys include `stations`, `temporal_fill`, `whittaker`, `satellite`, `logging`, and `imputer`.
- Each station block provides `request`, `parse`, `clean`, `merge`, and `save` values.
- The `temporal_fill` section configures which bands/features are interpolated. The `imputer` section configures which algorithms are enabled and their hyperparameters.
- For non-network/custom sensors, set `parse.manual_mode: true` with `parse.latitude`, `parse.longitude`, `parse.start_date`, and `parse.end_date`. In this mode, the pipeline creates a date/coordinate scaffold and skips `RequestPipe`.

## Pipeline structure

- `main.py` — orchestration script that sequences the pipeline for each station in `config.yaml`.
- `pipes/` — folder containing pipeline steps (each named `*Pipe`). Major pipes:

  - `request_pipe.py` — download data (USCRN) by year or read local SNOTEL files.
  - `parse_pipe.py` — parse raw station text files or build a manual scaffold DataFrame for custom stations.
  - `clean_pipe.py` — remove or convert invalid values and perform column selection.
  - `merge_pipe.py` — combine DataFrames from multiple inputs.
  - `satellite_pipe.py` — fetch/cache satellite data per station and add satellite features.
  - `temporal_fill_pipe.py` — run the ensemble imputation on configured features.
  - `whittaker_pipe.py` — optional smoothing of features with Whittaker smoother.
  - `feature_pipe.py` — derive features like `NDVI`, `NDMI`, `Rain_3d`, `DOY` and set mask flags.
  - `save_pipe.py` — persist final output as CSV/Parquet/JSON/etc.

- `imputers/` — collection of imputation implementations (linear, rolling mean, GP, KNN, XGBoost, climatology, etc.) and the ensemble voting logic in `api.py`.
- `smoothing/` — Whittaker smoothing functionality.
- `utils/` — helpers (configuration loader, logger, math utils, imputer helpers, etc.).
- `data/` — default local data directories: `data/raw`, `data/processed`, and `data/cache`.
- `tests/` — pytest-based tests for data and pipeline checks.

## Imputer ensemble — short summary

The imputation subsystem provides a flexible, weighted voting approach. Each imputer produces a filled value and a confidence score. The ensemble computes an effective weight (base_weight \* confidence) and produces the final fill by a weighted average with outlier suppression. Diagnostics and validation can be generated per-run (`imputers.api.transform_with_ensemble`).

See `imputers/IMPUTER_INFO.md` for implementation details and design rationale.

## Logging

- Default logging settings are defined in `config.yaml` under `logging`.
- Logs are written to `logs/pipeline.log` and imputer logs to `logs/imputer.log` by default.

## Running tests

Install test dependencies (in the same venv), then run:

```bash
cd Temporal/Pipeline
pytest -q
````

The tests analyze the final processed CSV and perform sanity checks for schema, ranges, and derived features.

## Contribution & Development

- Use Python 3.10 (recommended) or matching environment from `requirements.txt`.
- Follow project style and maintain small API changes.
- Tests use `pytest` and should be run before merging changes.

### Makefile

There is a small `Makefile` located in `Temporal/Pipeline/` with a few common targets to make local development and docker tasks easier:

- `make build` — Build Docker image from `Temporal/` root using the top-level Dockerfile.
- `make build-pipeline` — Build the pipeline-specific Docker image using `Pipeline/Dockerfile`.
- `make docker-run` — Run the image with data volume mounted and the working dir set to `Pipeline`.
- `make run-station STATION=<station_key>` — Run the pipeline locally for a single station.

## License

This project is MIT licensed. See the top-level LICENSE file for details.
