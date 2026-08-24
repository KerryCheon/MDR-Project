# Satellite Pipeline Optimization & Regression Monitoring Plan

## Goal Description
The objective is to optimize the Google Earth Engine (GEE) satellite data extraction pipeline (`src/pipeline/pipes/satellite_pipe.py`) to drastically reduce GEE compute credit consumption (EECU-hours) and API call overhead (latency), while establishing a robust regression monitoring framework to guarantee that optimizations do not alter data semantics, produce invalid values, or break downstream pipes (`temporal_fill_pipe`, `whittaker_pipe`, `feature_pipe`, `save_pipe`).

---

## Current Architecture & Bottleneck Analysis

### 1. Repeated Static DEM / Terrain Queries
- **Current Behavior**: For each weekly time window across a multi-year station (e.g. 52 weeks/year × 5–10 years = 260–520 weeks), `SatellitePipe` executes:
  - `ee.Image("USGS/SRTMGL1_003").reduceRegion(...)` for elevation.
  - `ee.Terrain.products(...)` and `reduceRegion(...)` for slope and aspect.
- **Problem**: Elevation, slope, and terrain aspect are static and do not change over time. Querying them 520 times generates over 1,000 redundant raster reductions per station.
- **Credit & Performance Impact**: Wastes compute credits on redundant raster sampling and accounts for ~1,560 unnecessary HTTP roundtrips per station.

### 2. Redundant `.size().getInfo()` Pre-checks
- **Current Behavior**: For every weekly window, before reducing each dataset (MODIS LST, MODIS NDVI, Sentinel-1, Sentinel-2, SMAP), `size().getInfo() > 0` is called.
- **Problem**: In Earth Engine, reducing an empty `ImageCollection` evaluates safely and returns null values without error. Calling `.size().getInfo()` sends 5 synchronous blocking HTTP requests per week purely to check collection size.
- **Impact**: For 520 weeks, this adds 2,600 redundant roundtrips per station.

### 3. Client-Side Python Band Loops with Individual `.getInfo()` Calls
- **Current Behavior**: For Sentinel-2, `stats` is computed once, but Python iterates over `["B2", "B3", "B4", "B8", "B11", "B12"]` calling `stats.get(band).getInfo()` for each band. Similarly, Sentinel-1 and SMAP make separate `.getInfo()` calls for each property.
- **Problem**: Each `.getInfo()` is an individual network RPC to GEE.
- **Impact**: 12+ separate RPCs per week instead of evaluating the complete dictionary once.

### 4. Unbatched Weekly Processing (Thread Pool with High Network Latency)
- **Current Behavior**: Python iterates over every week and submits individual weekly tasks via `ThreadPoolExecutor(max_workers=4)`.
- **Problem**: With 20–22 RPCs per week, a single station triggers ~10,400 HTTP calls. High request frequency risks GEE interactive request throttling (HTTP 429) and high connection latency.

---

## Technical Design of Optimizations

```mermaid
flowchart TD
    subgraph Current_Pipeline ["Current Pipeline (High Overhead)"]
        A1[Merged Station DataFrame] --> B1[Group by Week: 260-520 Weeks]
        B1 --> C1[520 x ThreadPool Tasks]
        C1 --> D1[5x size.getInfo per week]
        C1 --> E1[5x Sensor reduceRegions]
        C1 --> F1[12x Band getInfo calls]
        C1 --> G1[520x Static DEM Queries]
        D1 & E1 & F1 & G1 --> H1[Total ~10,400 RPCs per station]
    end

    subgraph Optimized_Pipeline ["Optimized Pipeline (SatellitePipeV2)"]
        A2[Merged Station DataFrame] --> B2[1. Static Terrain Extraction: 1 RPC]
        B2 --> C2[2. Identify Uncached Weeks]
        C2 --> D2[3. Server-Side Batching via FeatureCollection: 1-5 RPCs]
        D2 --> E2[Single Server Reduction per Week Composite]
        E2 --> F2[4. Return Full Dictionary in 1 Response]
        F2 --> G2[5. Merge into DataFrame & Update JSON Cache]
        G2 --> H2[Total: 2-6 RPCs per station (>99% reduction)]
    end
```

### Proposed Key Optimizations:
1. **Static Feature Extraction (One-Shot per Station)**:
   - Extract `elev`, `slope`, and `aspect` exactly **once** per station coordinate buffer.
   - Broadcast the static values across all weekly records, eliminating 99.8% of DEM compute credit usage.

2. **Unified Multi-Sensor Dictionary Reduction**:
   - Construct a single server-side `ee.Dictionary` encompassing all dynamic sensors (MODIS LST, MODIS NDVI, Sentinel-1 VV/VH, Sentinel-2 B2-B12, SMAP SM & Qual).
   - Eliminate all `.size().getInfo()` calls and Python band loops.
   - Evaluate in **1 single `.getInfo()`** call per time window.

3. **Temporal Batching via `ee.FeatureCollection` (Station/Chunk Batching)**:
   - Convert missing/uncached date ranges into an `ee.FeatureCollection` where each feature represents a weekly time window.
   - Map a server-side reduction function across the feature collection.
   - GEE executes the batch in parallel across Google's cloud infrastructure and returns all weeks in **1 single network roundtrip** per chunk (e.g. 52 weeks / 1 year per batch or full uncached station range).
   - For an entire 10-year station dataset, total network requests drop from ~10,400 to **under 10 requests**.

4. **100% Cache Compatibility & Fallback Mechanism**:
   - The JSON cache schema `{date_key: {features...}}` remains 100% identical and backward-compatible.
   - If a multi-week server-side batch fails (e.g. timeout on huge date range), the pipeline automatically falls back to single-week unified dictionary extraction.

---

## User Review Required

> [!IMPORTANT]
> **Zero Breaking Changes / Isolated Development**:
> As requested, we will **NOT modify the existing `satellite_pipe.py` directly**. Instead:
> 1. We create `src/pipeline/pipes/optimized_satellite_pipe.py` (`SatellitePipeV2`).
> 2. We build regression monitoring fixtures and comparison scripts.
> 3. We provide a configuration switch (`satellite.version: "v2"` or `use_optimized: true`) with default fallback to ensure uninterrupted workflow.

> [!NOTE]
> **Numeric Precision & Floating-Point Parity**:
> Server-side composite reductions in Earth Engine evaluate identical pixel buffers and scale factors (`0.02` for MODIS LST, `0.0001` for MODIS NDVI, `1/10000` for S2, `10^(dB/10)` for S1). Small differences at floating-point precision ($< 10^{-5}$) can occur due to server-side reduction order. Our regression tests will assert floating-point tolerance `rtol=1e-5, atol=1e-5` for continuous features and exact equality for integer flags/status masks.

---

## Proposed Changes

### 1. Pipeline Component (`src/pipeline/pipes/`)

#### [NEW] `src/pipeline/pipes/optimized_satellite_pipe.py`
A high-performance, credit-efficient replacement for `SatellitePipe` (`SatellitePipeV2`):
- Implements `fetch_static_terrain(lat, lon)` (runs once per station).
- Implements `build_weekly_expression(buffer, start_date, end_date)` server-side.
- Implements `fetch_satellite_chunked(lat, lon, uncached_ranges, chunk_size=52)` using `ee.FeatureCollection`.
- Implements fallback to `fetch_single_week_unified(lat, lon, start, end)` for resilient execution.
- Integrates with standard cache path and maintains exact schema and data types.

#### [MODIFY] `src/pipeline/pipes/__init__.py`
Export both `SatellitePipe` and `OptimizedSatellitePipe` (or `SatellitePipeV2`).

#### [MODIFY] `src/pipeline/main.py`
Add optional configuration check:
```python
sat_version = global_cfg.get("satellite", {}).get("version", "v1")
if sat_version == "v2" or global_cfg.get("satellite", {}).get("use_optimized", False):
    from .pipes.optimized_satellite_pipe import OptimizedSatellitePipe as SatPipe
else:
    from .pipes.satellite_pipe import SatellitePipe as SatPipe

with_sat = SatPipe(config=global_cfg, station_name=station_name).run(merged)
```

---

### 2. Regression Monitoring & Validation Framework

#### [NEW] `src/pipeline/validation/sample_data_generator.py`
- Utility script to extract and store reference baseline satellite samples (coordinates, dates, and expected satellite dictionary outputs) from existing processed station datasets (e.g. `src/pipeline/data/processed/quinault/final.csv` and Spokane/Quinault sample periods).
- Generates JSON fixtures in `tests/fixtures/satellite_regression_samples.json`.

#### [NEW] `src/pipeline/validation/compare_satellite_pipes.py`
- Standalone CLI comparison tool:
  - Runs both `SatellitePipe` (baseline) and `OptimizedSatellitePipe` (v2) on identical inputs / station configurations.
  - Measures wall-clock runtime, network RPC count, and memory footprint.
  - Performs column-by-column schema, nullability, and numeric parity checks.
  - Outputs a detailed parity report table.

#### [NEW] `tests/test_satellite_pipeline.py`
- Automated pytest regression suite:
  - `test_satellite_output_schema`: Asserts all 19 satellite columns (`LST_modis`, `NDVI_modis`, `s1_vv`, `s1_vh`, `s1_vv_dB`, `s1_vh_dB`, `s2_b2`..`s2_b12`, `elev`, `slope`, `aspect`, `SMAP_sm_am`..`SMAP_qual_pm`) exist with expected dtypes.
  - `test_satellite_regression_parity`: Compares outputs of `OptimizedSatellitePipe` against reference fixtures.
  - `test_static_dem_reuse`: Verifies DEM is queried once and reused correctly across all rows.
  - `test_cache_roundtrip_compatibility`: Ensures cache written by v2 can be read by v1 and vice versa.
  - `test_empty_and_edge_cases`: Tests handling of missing dates, leap years, boundary weeks, and empty dataframes.

---

### 3. Environment & Workspace Config

#### [MODIFY] `pyproject.toml` and `src/pipeline/pyproject.toml`
- Fix `requires-python` from `>=3.14` to `>=3.12` to match `notebooks/.python-version` and cluster Python 3.12 installation, enabling clean `uv` runs and test execution without quota errors.

---

## Verification Plan

### Automated Tests
1. **Pytest Suite**:
   ```bash
   UV_CACHE_DIR=/scratch/user/u.rp352032/.cache/uv uv run pytest tests/test_satellite_pipeline.py -v
   ```
2. **Full Test Suite**:
   ```bash
   UV_CACHE_DIR=/scratch/user/u.rp352032/.cache/uv uv run pytest tests/
   ```

### Regression & Performance Benchmarks
1. **Side-by-Side Parity & Benchmark Execution**:
   ```bash
   UV_CACHE_DIR=/scratch/user/u.rp352032/.cache/uv uv run python -m src.pipeline.validation.compare_satellite_pipes --station quinault_4_ne --samples 10
   ```
2. **Schema & Null Integrity Check**:
   Verify that all output columns, missing masks, and downstream pipelines (`TemporalFillPipe` → `WhittakerPipe` → `FeaturePipe`) process the output identically without warnings or errors.
