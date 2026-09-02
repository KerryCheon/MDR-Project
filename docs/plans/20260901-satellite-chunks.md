# Implementation Plan: ECE Farm Satellite Base Map & Upstream Grid Chunk Analysis (Enumclaw, WA)

## Goal Description

The ECE research team is preparing to deploy several new in-situ soil moisture sensor nodes on a commercial farm in **Enumclaw, King County, Washington** (King County Parcel PIN **`3420069035`**, Section 34, Township 20N, Range 06E, covering $\sim 69.4\text{ acres}$ / $3.02\text{ million sq ft}$).

### The Problem
During earlier deployments (e.g. Bellevue Botanical Garden, Renton garden), sensor nodes placed in close proximity ($\sim 50\text{ m}$ apart) fell into identical satellite pixel footprints and buffer extraction zones. As a result:
1. The pipeline extracted identical satellite feature vectors (identical MODIS, Sentinel-1/2, SMAP, and SRTM DEM values).
2. The ML models predicted identical soil moisture values despite ground-truth variations.

### The Objective & Major Revisions
We will update the mapping, chunking, and feature validation pipeline with the following critical enhancements:
1. **Accurate Farm Boundary (King County Parcel `3420069035`)**: Retrieve the official 32-vertex parcel geometry directly from the King County GIS Feature Service and render the exact property boundary with a prominent gold/yellow highlight on all base maps.
2. **Remove Artificial "Farm Center" Box**: Eliminate the obstructive "farm center" callout text box and crosshairs from all maps.
3. **Upstream-Aligned Grid Chunking**: Rather than partitioning an arbitrary grid centered on a single point, align the grid lines directly to the **native upstream source coordinate systems**:
   - **Sentinel-2 & Sentinel-1 (UTM Zone 10N / MGRS)**: $250\text{ m} \times 250\text{ m}$ standard integer grid cells.
   - **MODIS Thermal LST & Global Products**: $1000\text{ m} \times 1000\text{ m}$ macro-scale grids.
   - **SRTM DEM / 3DEP**: 1 arc-second ($\approx 30\text{m}$) topographical cells.
4. **High-Visibility 1000m Macro Chunks**: Make the $1000\text{ m}$ macro chunk boundaries prominently visible using bold, high-contrast lines and dedicated macro-chunk labels, showing exactly how the 1km boundary slices through the farm parcel.
5. **Dedicated Satellite Source Overlay Figures**: Create individual, publication-quality figures for each key satellite channel:
   - **Figure 1**: Base map with King County Parcel `3420069035` and Upstream Grid Overlays ($1000\text{m}$ Macro + $250\text{m}$ Sub-chunks).
   - **Figure 2**: Static Soil Feature Grid Overlay (USDA SSURGO series + SoilGrids $250\text{m}$ sand/clay/OM).
   - **Figure 3**: Optical Vegetation & Surface Reflectance Grid Overlay (Sentinel-2 / High-Res Greenness, `GRVI`, `VARI`).
   - **Figure 4**: Thermal LST & Macro Chunk Overlay (MODIS Land Surface Temperature variations across macro boundaries).
   - **Figure 5**: Topography & SRTM/USGS Digital Elevation Model Overlay (elevation contours, slope, aspect).
   - **Figure 6**: Multivariate Feature Dissimilarity Matrix & Cross-Feature Correlation Heatmap.

---

## User Review Required

> [!IMPORTANT]
> **Parcel Verification**: King County Parcel PIN **`3420069035`** in **Enumclaw, King County, WA**:
> - Centroid: $47.181208^\circ\text{ N}, -122.029209^\circ\text{ W}$ ($47^\circ 10' 52.3''\text{ N}, 122^\circ 01' 45.2''\text{ W}$).
> - Bounding Box: $[47.17746^\circ\text{ N}, 47.18464^\circ\text{ N}]$, $[-122.03737^\circ\text{ W}, -122.02688^\circ\text{ W}]$.
> - Width: $\approx 1168\text{ m}$, Height: $\approx 1153\text{ m}$, Area: $69.4\text{ acres}$.
>
> **Upstream Grid Alignment**:
> The $1000\text{m}$ macro grid and $250\text{m}$ sub-grid are anchored to absolute integer coordinates in standard UTM Zone 10N / Web Mercator coordinate space, matching the upstream tiling of Sentinel-2, MODIS, and DEM products.

---

## Proposed Changes

### Component: ECE Farm Satellite & Soil Analysis (`notebooks/experiment/ece_farm_satellite_chunks/`)

```
notebooks/experiment/ece_farm_satellite_chunks/
├── README.md                                     # Full experiment documentation with stdout tables & figures
├── farm_grid_generator.py                        # Updated Python module / CLI with parcel fetching & upstream alignment
├── farm_satellite_chunks.ipynb                   # Executable Jupyter notebook with all 6 figures & validation cells
└── figures/
    ├── farm_basemap_upstream_grid.png            # Figure 1: Basemap + Parcel 3420069035 + Upstream 1000m/250m grid
    ├── farm_basemap_soil_grid.png                # Figure 2: Static soil properties (SSURGO & SoilGrids) per chunk
    ├── farm_basemap_optical_ndvi_grid.png        # Figure 3: Sentinel-2 / Optical greenness & reflectance per chunk
    ├── farm_basemap_thermal_lst_grid.png         # Figure 4: MODIS Land Surface Temperature (LST) across macro chunks
    ├── farm_basemap_terrain_dem_grid.png         # Figure 5: Topographical elevation contours & slope per chunk
    └── farm_feature_heterogeneity_heatmap.png     # Figure 6: Inter-chunk feature dissimilarity & correlation matrix
```

---

#### [MODIFY] [`notebooks/experiment/ece_farm_satellite_chunks/farm_grid_generator.py`](file:///scratch/user/u.rp352032/MDR-Project/notebooks/experiment/ece_farm_satellite_chunks/farm_grid_generator.py)

1. **Parcel Retrieval**:
   - Query King County Parcel Feature Service (`https://gismaps.kingcounty.gov/arcgis/rest/services/Property/KingCo_Parcels/MapServer/0/query`) for `PIN = '3420069035'` (with local caching).
   - Extract the 32-vertex parcel polygon and convert to Web Mercator & WGS-84.
2. **Upstream Grid Partitioner**:
   - Generate standard $250\text{ m} \times 250\text{ m}$ sub-chunks and $1000\text{ m} \times 1000\text{ m}$ macro-chunks snapped to global integer grid coordinates.
   - Tag each chunk with whether it intersects or is inside Parcel `3420069035` (`in_farm_parcel: bool`).
3. **Multi-Channel Satellite Feature Extractors**:
   - **Soil**: USDA SSURGO map units (`mukey: 300971` Buckley gravelly loam, `300985` Wilkeson silt loam) & SoilGrids texture fractions.
   - **Optical**: Mean RGB reflectance, `GRVI`, `VARI`, surface texture.
   - **Thermal LST**: Simulated/interpolated MODIS daytime LST showing macro-grid thermal steps across 1km chunk lines.
   - **DEM**: USGS 3DEP / SRTM elevation, slope percentage, slope degrees, and aspect.
4. **Enhanced Plotting Functions**:
   - `plot_upstream_grid_basemap`: Satellite imagery with Parcel `3420069035` boundary highlighted in solid gold/yellow, bold orange $1000\text{m}$ macro-chunk lines, cyan dashed $250\text{m}$ sub-chunk lines, and chunk IDs *(no farm center text box)*.
   - `plot_soil_grid_basemap`: Satellite imagery with parcel boundary + static soil texture / series per chunk.
   - `plot_optical_ndvi_basemap`: Satellite imagery with parcel boundary + optical greenness / vegetation index per chunk.
   - `plot_thermal_lst_basemap`: Satellite imagery with parcel boundary + MODIS thermal LST per chunk.
   - `plot_terrain_dem_basemap`: Satellite imagery with parcel boundary + elevation contours and slope vectors.
   - `plot_feature_heterogeneity_heatmap`: Multivariate dissimilarity matrix and cross-feature correlation.

---

#### [MODIFY] [`notebooks/experiment/ece_farm_satellite_chunks/farm_satellite_chunks.ipynb`](file:///scratch/user/u.rp352032/MDR-Project/notebooks/experiment/ece_farm_satellite_chunks/farm_satellite_chunks.ipynb)

Update the notebook to execute all 6 analysis sections:
- **Cell 1**: Title, location description (Enumclaw, King County, WA), and Parcel `3420069035` background.
- **Cell 2**: Imports and configuration.
- **Cell 3**: Fetching King County Parcel `3420069035` geometry and computing upstream-aligned grid coordinates.
- **Cell 4 & 5**: Figure 1 — Upstream Satellite Grid & Farm Parcel Map.
- **Cell 6 & 7**: Figure 2 — Static Soil Property Chunks Map.
- **Cell 8 & 9**: Figure 3 — Optical Greenness & Reflectance Chunks Map.
- **Cell 10 & 11**: Figure 4 — MODIS Thermal LST & 1000m Macro Chunks Map.
- **Cell 12 & 13**: Figure 5 — Topographical Elevation & Slope Contours Map.
- **Cell 14 & 15**: Figure 6 — Multivariate Feature Dissimilarity Matrix & Validation Report.
- **Cell 16 & 17**: Field deployment coordinate table filtered to chunks intersecting the farm parcel.

---

#### [MODIFY] [`notebooks/experiment/ece_farm_satellite_chunks/README.md`](file:///scratch/user/u.rp352032/MDR-Project/notebooks/experiment/ece_farm_satellite_chunks/README.md)

Update the documentation to include:
- Enumclaw, King County, WA geographic profile.
- Parcel `3420069035` dimensions ($69.4\text{ acres}$, $1168\text{m} \times 1153\text{m}$).
- Embedded Figures 1 through 6.
- Complete feature variance and chunk validation table.
- Parcel-intersecting sensor placement coordinate reference table.

---

## Verification Plan

### Automated Execution & Parity Checks
1. **Notebook Execution**:
   ```bash
   cd notebooks
   uv run nb execute experiment/ece_farm_satellite_chunks/farm_satellite_chunks.ipynb
   ```
   Verify all 17 cells execute with exit code 0.
2. **Artifact Verification**:
   Verify that all 6 figures are generated under `figures/`:
   - `farm_basemap_upstream_grid.png`
   - `farm_basemap_soil_grid.png`
   - `farm_basemap_optical_ndvi_grid.png`
   - `farm_basemap_thermal_lst_grid.png`
   - `farm_basemap_terrain_dem_grid.png`
   - `farm_feature_heterogeneity_heatmap.png`
3. **Statistical Feature Variance Checks**:
   - Confirm $CV > 0$ across all satellite channels (optical, thermal, soil, elevation).
   - Confirm parcel `3420069035` intersects at least 9 distinct $250\text{m}$ chunks and spans across the $1000\text{m}$ macro-grid boundary.

### Manual Verification
1. Visually check that the King County Parcel `3420069035` boundary perfectly aligns with the real physical field/property lines in the satellite imagery.
2. Verify that the 1000m macro-chunk lines and labels are crisp, prominent, and easily visible.
3. Verify that the "farm center" text box has been completely removed.
