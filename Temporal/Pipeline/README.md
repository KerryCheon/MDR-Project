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

From the `Temporal` root (recommended):

```bash
cd Temporal
docker build -t temporal-pipeline .

# Run the pipeline with data path mounted and current workdir set to Pipeline
docker run --rm \
  --workdir /app/Temporal/Pipeline \
  -v $(pwd)/Pipeline/data:/app/Temporal/Pipeline/data \
	temporal-pipeline
```

Notes:

- The repository includes a `Dockerfile` in `Temporal/` that installs dependencies and copies the project into the image.
  -- The `--workdir /app/Temporal/Pipeline` ensures `main.py` is executed from the `Pipeline/` directory inside the container.
  If you prefer to build a Pipeline-only image, there is `Pipeline/Dockerfile` which can be built from the `Temporal/` root:

```bash
cd Temporal
docker build -t temporal-pipeline -f Pipeline/Dockerfile .
docker run --rm --workdir /app/Temporal/Pipeline -v $(pwd)/Pipeline/data:/app/Temporal/Pipeline/data temporal-pipeline
```

## Local development (Python)

To run the pipeline locally without Docker:

````bash
cd Temporal/Pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/..   # make sure absolute imports `Temporal.Pipeline.*` resolve
python main.py

# Run a single station from the CLI
```bash
python main.py --station spokane_17_ssw
````

# Provide a custom config file

```bash
python main.py --config /path/to/config.yaml
```

````

Notes:

- Use the `config.yaml` in the same folder to modify station parameters, years, and IMPUTER/SATELLITE settings.
- To run only a subset of stations, edit `config.yaml` (comment-out or remove the unwanted station blocks), or add a shorter year range.

## Configuration

- `config.yaml` controls pipeline execution. Top-level keys include `stations`, `temporal_fill`, `whittaker`, `satellite`, `logging`, and `imputer`.
- Each station block provides `request`, `parse`, `clean`, `merge`, and `save` values.
- The `temporal_fill` section configures which bands/features are interpolated. The `imputer` section configures which algorithms are enabled and their hyperparameters.

## Pipeline structure

- `main.py` — orchestration script that sequences the pipeline for each station in `config.yaml`.
- `pipes/` — folder containing pipeline steps (each named `*Pipe`). Major pipes:

  - `request_pipe.py` — download data (USCRN) by year or read local SNOTEL files.
  - `parse_pipe.py` — parse raw station text files to unified DataFrame.
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
