# MDR Spatio-Temporal Soil Moisture Model

This repo is for a spatio-temporal soil moisture modeling project.

Current status:
- Temporal pipeline is implemented and runnable.
- Spatial pipeline is a placeholder for now (teammate work not pushed yet).

## Project Structure

- `Temporal/Pipeline/`: Main temporal data pipeline (ingestion, cleaning, feature building, saving).
- `Models/Temporal/`: Temporal model experiments and results.
- `Spatial/`: Spatial work area (in progress).
- `Models/Spatial/placeholder.txt`: Placeholder for upcoming spatial model outputs.

## Quick Start (Temporal)

```bash
cd Temporal/Pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all configured stations:

```bash
PYTHONPATH=$(pwd)/.. python main.py
```

Run one station:

```bash
PYTHONPATH=$(pwd)/.. python main.py --station spokane_17_ssw
```

Outputs are written to station-specific files under:
- `Temporal/Pipeline/data/processed/<station>/final.csv`

## Spatial Placeholder

Spatial modeling integration will be added here once the spatial branch is pushed.
