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

## Robust GEE Fetching Enhancements

### 1. Robust Initialization & Fallback
- Initializing GEE will try the project ID `mdr-project-475522` first.
- If that project is not accessible, it will fall back to calling `ee.Initialize()` without a project parameter (which utilizes the default cloud project of the authenticated Google account).
- Fix the bug in `fetch_lia.py` where a duplicate `ee.Initialize()` call at the end of the script overrides the initial project authentication.

### 2. Incremental Caching & Validity Guards
- Modify `satellite_pipe.py` to write the GEE response cache to disk incrementally (every 20 successful fetches).
- Only save a GEE response to the cache dictionary if the data is **valid** (i.e. at least one retrieved satellite, SMAP, or terrain feature is not `None`).
- If a query fails or returns entirely `None` fields, do not save it to the cache dictionary so that it is retried in future runs.

---

## Proposed Changes

### Configuration Modifications

Modify the pipeline configuration to comment out all non-Washington stations. This restricts pipeline execution to the 13 WA stations.

#### [MODIFY] [config.yaml](file:///c:/Users/pan/Documents/GitHub/MDR-Project/src/pipeline/config.yaml)
- Comment out all `device_*`, non-WA USCRN (`uscrn_arco_17_sw`, etc.), and non-WA SCAN (`scan_conrad_agrc`, etc.) stations.
- Retain only the 13 Washington stations.

---

### Earth Engine Setup & Script Bug Fixes

#### [MODIFY] [satellite_pipe.py](file:///c:/Users/pan/Documents/GitHub/MDR-Project/src/pipeline/pipes/satellite_pipe.py)
- Update `ee.Initialize()` with a fallback block.
- Update the GEE processing loop in the `run()` method to perform validation checks (`any(v is not None for v in data.values())`) and save the cache to disk incrementally every 20 successful fetches.

#### [MODIFY] [fetch_lia.py](file:///c:/Users/pan/Documents/GitHub/MDR-Project/data/splits/derived_8.0/LIA/fetch_lia.py)
- Update `ee.Initialize()` with the fallback block.
- Remove the redundant/buggy `ee.Initialize()` call on line 129.

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
1. **Apply configurations**: Comment out unneeded stations in `config.yaml`.
2. **Apply Earth Engine fixes**: Modify `satellite_pipe.py` and `fetch_lia.py`.
3. **Setup directories & scripts**: Write `make_derived_8.1.py` and `stations.csv`.
4. **LIA Generation**: Verify we can run `fetch_lia.py` for the 13 WA stations to produce `stations_lia.csv`.
5. **Data Generation Walkthrough**: Provide a step-by-step description to the user of how they should run the pipeline and resplit script locally once they have the raw SNOTEL `.stm` files.
