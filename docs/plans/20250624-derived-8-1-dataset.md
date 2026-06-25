# Creating the fully-featured derived_8.1 Washington dataset

Create a new split dataset called `derived_8.1` containing exactly the 13 Washington-only stations (5 original Washington stations + 8 new Washington SNOTEL stations). The dataset will be fully featured like `derived_8.0`, meaning it will contain all satellite bands, SMAP soil moisture, Whittaker-smoothed features, Local Incident Angle (LIA) features, and downstream rolling/lag features.

The 13 Washington stations are:
* **Original 5 WA Stations:**
  1. `spokane_17_ssw` (USCRN WA)
  2. `quinault_4_ne` (USCRN WA)
  3. `darrington_21_nne` (USCRN WA)
  4. `sourdough_gulch` (SNOTEL WA)
  5. `touchnet` (SNOTEL WA)
* **New 8 WA SNOTEL Stations:**
  6. `cayuse_pass_wa`
  7. `paradise_wa`
  8. `burnt_mountain_wa`
  9. `beaver_pass_wa`
  10. `harts_pass_wa`
  11. `marten_ridge_wa`
  12. `mf_nooksack_wa`
  13. `rainy_pass_wa`

---

## User Review Required

> [!IMPORTANT]
> To compile this dataset, the user must provide the raw SNOTEL `.stm` files for the 8 new SNOTEL stations since these files are too large to be committed to Git. They must be placed in:
> `src/pipeline/data/raw/<StationName>/` (e.g. `src/pipeline/data/raw/CayusePass/` for `cayuse_pass_wa`).

> [!WARNING]
> Running the pipeline fetches satellite, SMAP, and terrain data from Google Earth Engine (GEE). The user must have a registered GEE account and have authenticated `ee` on their local machine (by running `ee.Authenticate()`) before execution.

---

## Proposed Changes

### Configuration Modifications

Modify the pipeline configuration to comment out all non-Washington stations. This restricts pipeline execution to the 13 WA stations.

#### [MODIFY] [config.yaml](file:///c:/Users/pan/Documents/GitHub/MDR-Project/src/pipeline/config.yaml)
- Comment out all `device_*`, non-WA USCRN (`uscrn_arco_17_sw`, etc.), and non-WA SCAN (`scan_conrad_agrc`, etc.) stations.
- Retain only the 13 Washington stations.

---

### Dataset Merging and Splitting Scripts

Create a new split folder `data/splits/derived_8.1/` containing a script to compile and split the processed station data.

#### [NEW] [stations.csv](file:///c:/Users/pan/Documents/GitHub/MDR-Project/data/splits/derived_8.1/LIA/stations.csv)
- List the coordinates and IDs of the 13 Washington stations for LIA extraction.

#### [NEW] [make_derived_8.1.py](file:///c:/Users/pan/Documents/GitHub/MDR-Project/data/splits/derived_8.1/make_derived_8.1.py)
- Script that aggregates the processed `final.csv` files of the 13 stations, runs the math functions to compute all 350+ lag, rolling mean, and FFT features, merges the LIA data, computes scaled drift features, and splits the data by year (Train: 2017-2020, Val: 2021-2022, Test: 2023-2025).

---

## Verification Plan

### Automated Steps (After User Approval)
1. **Comment out stations**: Comment out unneeded stations in `config.yaml`.
2. **Setup directories & scripts**: Write `make_derived_8.1.py` and `stations.csv`.
3. **LIA Generation**: Verify we can run `fetch_lia.py` for the 13 WA stations to produce `stations_lia.csv`.
4. **Data Generation Walkthrough**: Provide a step-by-step description to the user of how they should run the pipeline and resplit script locally once they have the raw SNOTEL `.stm` files.
