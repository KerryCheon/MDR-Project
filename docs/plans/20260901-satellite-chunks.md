# Implementation Plan: ECE Farm Satellite & Soil Base Map with Grid Chunk Overlays

## Goal Description

The ECE research team is deploying several new in-situ soil moisture sensor nodes on a farm centered at **$47^\circ 10' 52.1''\text{ N}, 122^\circ 01' 56.5''\text{ W}$** ($47.181139^\circ\text{ N}, -122.032361^\circ\text{ W}$), near Buckley / Enumclaw, Washington.

### The Problem
During earlier deployments (e.g. Bellevue Botanical Garden, Renton garden), sensor nodes placed in close proximity ($\sim 50\text{ m}$ apart) fell into the exact same satellite grid cell and extraction footprint. As a result:
1. The pipeline extracted identical satellite feature vectors (identical MODIS, Sentinel-1/2, SMAP, and SRTM DEM values).
2. The ML models predicted identical soil moisture values despite substantial micro-topographical and ground-truth variance.

### The Objective & Revised Scope
Because the farm has an irregular shape spanning across a $2\text{ km} \times 2\text{ km}$ area ($\pm 1\text{ km}$ from the center):
1. **Clean High-Resolution Satellite Base Map**: High-resolution imagery (`Esri.WorldImagery`) covering the $2\text{ km} \times 2\text{ km}$ extent around $47^\circ 10' 52.1''\text{ N}, 122^\circ 01' 56.5''\text{ W}$, preserving clear visual visibility of farm plots, tree lines, roads, and field boundaries *(no artificial circular perimeters or concentric rings)*.
2. **Dynamic Satellite Grid Overlay**: Rectangular grid lines / chunks representing the spatial resolutions of dynamic satellite products (Sentinel-2 $100\text{m}/20\text{m}$, Sentinel-1 $30\text{m}$, MODIS $250\text{m}/1000\text{m}$) with labeled chunk IDs and coordinate axes.
3. **Static Soil & Terrain Grid Overlay**: A dedicated overlay displaying static soil property chunks:
   - **USDA SSURGO Soil Map Units**: Soil polygon boundaries, soil series names (e.g. Buckley gravelly loam), hydric ratings, and drainage classes.
   - **SoilGrids / Soil Texture Grids ($250\text{m}$)**: Spatial variation in Sand %, Silt %, Clay %, Organic Carbon (SOC), and Bulk Density.
   - **Topography / Digital Elevation Model**: High-resolution elevation contours ($203\text{m} - 229\text{m}$ gradient across the site), slope, and aspect chunks.
4. **Statistical Feature Validation**: Quantitative verification calculating feature distributions across chunks, confirming that sensors in different chunks receive distinct satellite and soil feature values ($\sigma > 0$).
5. **Irregular Boundary Visual Reference**: The ECE team will visually identify which chunks match their physical farm plots directly from the high-resolution map, with full coordinate tables for every grid chunk.

---

## User Review Required

> [!IMPORTANT]
> **Key Design Adjustments Made Based on Feedback**:
> - **Omitted**: Circular $1\text{ km}$ perimeter and concentric circular distance rings (to respect the farm's irregular rectangular/field geometry).
> - **Omitted**: Automatic greedy sensor optimization (field placement will be decided visually by the ECE team based on land ownership and field access).
> - **Added**: Static soil feature grid overlay (USDA SSURGO soil map units + SoilGrids $250\text{m}$ texture/organic matter/bulk density) alongside dynamic satellite grids.

---

## Proposed Changes

### Component: ECE Farm Satellite & Soil Chunk Analysis (`notebooks/experiment/ece_farm_satellite_chunks/`)

```
notebooks/experiment/ece_farm_satellite_chunks/
├── README.md                                  # Formal experiment report with stdout tables & embedded figures
├── farm_grid_generator.py                     # Standalone Python module / CLI for base map, soil & satellite overlays
├── farm_satellite_chunks.ipynb                # Reproducible Jupyter notebook executed via nb-cli
└── figures/
    ├── farm_basemap_satellite_grid.png         # Satellite base map with dynamic satellite grid chunk overlay
    ├── farm_basemap_soil_grid.png              # Satellite base map with USDA SSURGO & SoilGrids static soil overlay
    ├── farm_terrain_elevation_slope.png        # Elevation contour and slope terrain map
    └── farm_feature_heterogeneity_heatmap.png  # Multi-channel feature variation and inter-chunk dissimilarity matrix
```

---

#### [NEW] [`notebooks/experiment/ece_farm_satellite_chunks/farm_grid_generator.py`](file:///scratch/user/u.rp352032/MDR-Project/notebooks/experiment/ece_farm_satellite_chunks/farm_grid_generator.py)

This module implements:
1. **Spatial Geometry & Projections**:
   - Precise conversion between WGS-84 (lat/lon) and Web Mercator (meters).
   - Parametric $2\text{ km} \times 2\text{ km}$ grid partitioner ($100\text{m}$, $250\text{m}$, $500\text{m}$, $1000\text{m}$ chunks) centered at $(47.181139, -122.032361)$.
2. **Satellite Base Map & Dynamic Grid Overlays**:
   - Downloads high-resolution Esri World Imagery tiles via `contextily`.
   - Renders clean satellite base map with grid lines for Sentinel-2 ($100\text{m}$), MODIS NDVI ($250\text{m}$), and MODIS LST ($1000\text{m}$).
   - Labels chunk coordinates and indices (e.g., `Row 0..N, Col 0..N`).
   - Marks the farm center reference crosshair with exact DMS and decimal coordinates.
3. **Static Soil Feature Extraction & Overlay**:
   - Queries **USDA NRCS Soil Data Access (SSURGO)** for real-time map unit key (`mukey`), soil series (`muname`), horizon textures, and sand/silt/clay percentages.
   - Queries **ISRIC SoilGrids ($250\text{m}$)** for depth-resolved ($0-5\text{cm}$) sand, silt, clay, bulk density, and soil organic carbon across each grid cell.
   - Generates the static soil overlay map showing soil texture distribution across the farm territory.
4. **Topographical Terrain Extraction**:
   - Concurrent USGS 3DEP / Open-Meteo elevation extraction across the grid.
   - Calculates gradient slope ($\%$, degrees) and aspect ($0-360^\circ$).
5. **Statistical Feature Validation**:
   - Computes summary statistics (mean, std, min, max, CV) for all dynamic satellite features, static soil features, and terrain features across chunks.
   - Computes pairwise Euclidean feature dissimilarity matrix to demonstrate that distinct chunks provide distinct feature vectors.
   - Exports all clean tabular summaries to CSV and JSON.

---

#### [NEW] [`notebooks/experiment/ece_farm_satellite_chunks/farm_satellite_chunks.ipynb`](file:///scratch/user/u.rp352032/MDR-Project/notebooks/experiment/ece_farm_satellite_chunks/farm_satellite_chunks.ipynb)

A structured, cell-by-cell Jupyter notebook built and executed using `notebook-cli` (`nb`):
- **Cell 1 (Markdown)**: Title, farm coordinates ($47^\circ 10' 52.1''\text{ N}, 122^\circ 01' 56.5''\text{ W}$), and background.
- **Cell 2 (Code)**: Imports and setup.
- **Cell 3 (Markdown + Code)**: Spatial bounding box and coordinate setup ($2\text{ km} \times 2\text{ km}$ extent).
- **Cell 4 (Markdown + Code)**: Figure 1 — High-resolution satellite basemap with dynamic satellite grid overlay.
- **Cell 5 (Markdown + Code)**: Figure 2 — Static soil feature grid overlay (USDA SSURGO + SoilGrids $250\text{m}$ sand/clay/SOC).
- **Cell 6 (Markdown + Code)**: Figure 3 — Topographical elevation and slope contour map.
- **Cell 7 (Markdown + Code)**: Figure 4 — Statistical feature validation, inter-chunk variance, and dissimilarity heatmap.
- **Cell 8 (Markdown + Code)**: Formatted table of chunk coordinates, soil properties, and satellite metrics for the ECE field team.

---

#### [NEW] [`notebooks/experiment/ece_farm_satellite_chunks/README.md`](file:///scratch/user/u.rp352032/MDR-Project/notebooks/experiment/ece_farm_satellite_chunks/README.md)

Markdown documentation reporting:
1. Executive summary of the farm site.
2. Formatted tables of grid chunk coordinates, soil series, texture fractions, and elevation.
3. Proof of feature differentiation (inter-chunk variance and dissimilarity metrics).
4. Field reference guide for the ECE team with embedded figures.

---

## Verification Plan

### Automated Tests & Reproducibility
1. **Notebook Execution**:
   ```bash
   cd notebooks
   nb execute experiment/ece_farm_satellite_chunks/farm_satellite_chunks.ipynb --uv
   ```
   Verify that all cells execute sequentially with exit code 0.
2. **Artifact & Figure Generation**:
   Verify all 4 PNG figures exist and are $> 100\text{ KB}$:
   - `figures/farm_basemap_satellite_grid.png`
   - `figures/farm_basemap_soil_grid.png`
   - `figures/farm_terrain_elevation_slope.png`
   - `figures/farm_feature_heterogeneity_heatmap.png`
3. **Statistical Validation Check**:
   Confirm in the notebook output that:
   - Elevation range across chunks $\Delta \text{elev} > 15\text{ m}$.
   - Static soil properties vary across chunks ($CV > 0$).
   - Dynamic satellite indices vary across chunks ($CV > 0$).

### Manual Verification
1. Visually check that the satellite imagery is crisp and field boundaries / buildings / tree lines are clearly discernible.
2. Verify that the grid lines and chunk IDs are legible without obscuring the underlying imagery.
3. Verify that the soil map overlay accurately delineates the soil series (e.g. Buckley gravelly loam).
