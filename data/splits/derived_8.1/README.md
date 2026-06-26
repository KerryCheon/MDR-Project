# derived_8.1

## 🚀 How to Run the Pipeline and Compile the Dataset

Follow these steps to generate the `derived_8.1` splits locally:

### Step 1: Place SNOTEL Raw Files
Place your downloaded SNOTEL `.stm` files for the 8 new SNOTEL stations in their raw directories:
- `src/pipeline/data/raw/BeaverPass/`
- `src/pipeline/data/raw/BurntMountain/`
- `src/pipeline/data/raw/CayusePass/`
- `src/pipeline/data/raw/HartsPass/`
- `src/pipeline/data/raw/MFNooksack/`
- `src/pipeline/data/raw/MartenRidge/`
- `src/pipeline/data/raw/Paradise/`
- `src/pipeline/data/raw/RainyPass/`

### Step 2: Run the Pipeline
Authenticate your Google Earth Engine account (if not already authenticated) and run the pipeline to parse raw data, query GEE, run Weather API inputs, and perform ensemble temporal gap-filling:
```bash
# Run the pipeline for all 13 Washington stations
PYTHONPATH=. uv run -m pipeline.main --config data/splits/derived_8.1/config.yaml
```
*(The GEE cache will be saved incrementally to `src/pipeline/data/cache/*_satellite_cache.json` after every 20 successful fetches, guarding against quota limit crashes).*

### Step 3: Fetch LIA Angle Features
Run the GEE LIA fetcher script using the coordinates defined in your `derived_8.1` station list:
```bash
PYTHONPATH=. uv run data/splits/derived_8.0/LIA/fetch_lia.py \
  --stations-csv data/splits/derived_8.1/LIA/stations.csv \
  --out-csv data/splits/derived_8.1/LIA/stations_lia.csv
```

### Step 4: Compile and Split the Dataset
Execute the compilation script to derive the 350+ training features, merge LIA values, calculate year-fraction drift factors, and split into final CSVs:
```bash
PYTHONPATH=. uv run data/splits/derived_8.1/make_derived_8.1.py
```

Upon completion, you will find the generated splits under:
- `data/splits/derived_8.1/train.csv`
- `data/splits/derived_8.1/val.csv`
- `data/splits/derived_8.1/test.csv`
- `data/splits/derived_8.1/split_meta.json`

## Threshold and Selected Features
See [dataset_metadata.py](dataset_metadata.py) for the bimodal valley-based regime thresholds (T1=0.16, T2=0.25) and the list of selected features used fro each regime.