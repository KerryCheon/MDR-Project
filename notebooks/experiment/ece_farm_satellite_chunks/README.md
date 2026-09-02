# ECE Farm Upstream Satellite, Soil & Rainfall Grid Chunk Validation
**Enumclaw, King County, Washington | Parcel PIN: `3420069035`**

## 1. Executive Summary

This experiment generates high-resolution satellite basemaps with upstream-aligned dynamic satellite, static soil, and gridded rainfall chunk overlays for the ECE soil moisture sensor deployment on a commercial farm in **Enumclaw, King County, Washington**.

- **Farm Location**: Enumclaw, King County, Washington (Section 34, Township 20N, Range 06E)
- **Parcel PIN**: `3420069035` (King County GIS official parcel boundary)
- **Parcel Area**: $69.4\text{ acres}$ ($3,023,914.9\text{ sq ft}$, $32\text{ polygon vertices}$)
- **Parcel Extent**: Lat $[47.17746^\circ\text{ N}, 47.18464^\circ\text{ N}]$, Lon $[-122.03737^\circ\text{ W}, -122.02688^\circ\text{ W}]$
- **Upstream Grid Alignment**:
  - **1000m Macro Grid**: Aligned to integer 1000m coordinates in Web Mercator / UTM Zone 10N (MODIS LST thermal scale)
  - **250m Sub-Grid**: Aligned to integer 250m coordinates (MODIS NDVI / Sentinel-2 pixel grouping zone)
  - **Precipitation**: PRISM & GridMET gridded rainfall (annual, 30-day, 7-day storm totals)
  - **Topography**: USGS 3DEP / SRTM 1-arc-second (~30m) digital elevation model

### Problem Addressed
In previous sensor deployments (e.g. Bellevue Botanical Garden, Renton garden), sensor nodes placed in close proximity ($\sim 50\text{ m}$ apart) fell into identical satellite pixel footprints and buffer extraction zones. As a result, the spatial machine learning models predicted identical soil moisture values despite ground-truth variations.

Because Parcel `3420069035` is over $1.1\text{ km}$ across, it intersects **23 distinct $250\text{m}$ sub-chunks** and crosses the **1000m MODIS Macro Thermal boundary**. This tool visualizes the exact parcel boundaries, upstream grids, and proves statistically that placing sensors in different chunks provides distinct satellite, soil, and meteorological feature inputs.

---

## 2. Upstream Satellite Base Map & Grid Reference (Figure 1)

Sub-meter **Esri World Imagery** overlaid with:
- **Farm Parcel Boundary**: Solid gold/yellow line showing official 32-vertex King County parcel boundary.
- **1000m Macro Grid**: Bold orange solid lines (`#FF3D00`) aligned to standard integer 1000m coordinates.
- **250m Sub-Grid**: Cyan dashed lines (`#00E5FF`) with gold badges for parcel-intersecting chunks.
- **Opaque Legend**: Positioned strictly on top of all grid lines and basemap layers (`zorder=100`).

![Figure 1: Farm Upstream Satellite Grid Reference Map](figures/farm_basemap_upstream_grid.png)

---

## 3. Static Soil Properties & Texture Grid Overlay (Figure 2)

Integration of **USDA NRCS SSURGO** soil survey map units and **ISRIC SoilGrids 250m** depth profiles:
- **Buckley series** (`mukey: 300971`): Rich alluvial lowland flat ($10.0\%$ OM, bulk density $1.05\text{ g/cm}^3$, poorly drained)
- **Wilkeson series** (`mukey: 300985`): Silt loam terrace soil ($58.0\%$ silt, $7.5\%$ OM, bulk density $1.16\text{ g/cm}^3$, moderately well drained)
- **Kapowsin series** (`mukey: 300962`): Upland glacial till ($14.5\%$ clay, bulk density $1.24\text{ g/cm}^3$)

![Figure 2: Farm Static Soil Features & Texture Grid Overlay](figures/farm_basemap_soil_grid.png)

---

## 4. Optical Vegetation & Surface Reflectance Grid (Figure 3)

Chunk-level extraction of multi-band optical surface reflectance and vegetation indices (Green-Red Vegetation Index `GRVI`, Visible Atmospherically Resistant Index `VARI`, and RGB reflectance).

![Figure 3: Farm Optical Vegetation & Surface Reflectance Grid](figures/farm_basemap_optical_ndvi_grid.png)

---

## 5. MODIS Thermal Land Surface Temperature (LST) Map (Figure 4)

Thermal land surface temperature variations derived from MODIS MOD11A1 products across the 1000m macro thermal chunk boundary, demonstrating temperature discontinuities that occur when sensors cross 1km tiles.

![Figure 4: Farm MODIS Thermal Land Surface Temperature Map](figures/farm_basemap_thermal_lst_grid.png)

---

## 6. Topographical Elevation & Slope Contours Map (Figure 5)

Digital elevation model contours and slope gradients derived from USGS 3DEP and SRTM 1-arc-second data, illustrating hydrological drainage regimes across the farm parcel.

![Figure 5: Farm Topographical Elevation Profile & Contours](figures/farm_basemap_terrain_dem_grid.png)

---

## 7. Gridded Precipitation & Rainfall Map (Figure 6)

High-resolution gridded rainfall derived from **PRISM** and **GridMET** datasets across Enumclaw, WA:
- **Normal Annual Precipitation**: $1436\text{ mm} - 1492\text{ mm}$ (Cascade foothill orographic gradient).
- **30-Day Wet Season Accumulation**: $227\text{ mm} - 242\text{ mm}$.
- **7-Day Storm Total**: $78\text{ mm} - 83\text{ mm}$.

![Figure 6: Farm Gridded Precipitation & Rainfall Map](figures/farm_basemap_rainfall_grid.png)

---

## 8. Statistical Feature Validation & Inter-Chunk Separability (Figure 7)

All features were evaluated across the domain to verify non-zero variance ($\sigma > 0$):

| Feature | Mean | Std | Min | Max | CV (%) | Distinct Values Confirmed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `elevation_m` | 213.30 | 4.82 | 190.00 | 219.00 | 2.26% | **True** |
| `slope_deg` | 0.63 | 0.93 | 0.00 | 5.80 | 149.54% | **True** |
| `sand_pct` | 40.43 | 12.54 | 26.50 | 56.20 | 31.01% | **True** |
| `clay_pct` | 13.25 | 0.93 | 11.60 | 15.00 | 7.04% | **True** |
| `silt_pct` | 46.30 | 11.94 | 30.80 | 60.30 | 25.80% | **True** |
| `organic_matter_pct` | 8.41 | 1.28 | 5.93 | 10.19 | 15.24% | **True** |
| `bulk_density_g_cm3` | 1.12 | 0.07 | 1.03 | 1.25 | 6.07% | **True** |
| `sand_clay_ratio` | 3.11 | 1.13 | 1.84 | 4.84 | 36.20% | **True** |
| `opt_red_mean` | 79.83 | 14.12 | 46.50 | 134.30 | 17.68% | **True** |
| `opt_green_mean` | 104.40 | 10.93 | 72.00 | 146.50 | 10.47% | **True** |
| `opt_blue_mean` | 68.39 | 11.78 | 51.00 | 125.90 | 17.23% | **True** |
| `opt_grvi` | 0.14 | 0.06 | 0.04 | 0.30 | 42.51% | **True** |
| `opt_vari` | 0.22 | 0.09 | 0.07 | 0.49 | 41.18% | **True** |
| `modis_lst_celsius` | 24.65 | 0.95 | 23.26 | 26.82 | 3.83% | **True** |
| `annual_precip_mm` | 1471.19 | 12.87 | 1436.30 | 1492.30 | 0.87% | **True** |
| `precip_30d_mm` | 236.43 | 3.68 | 226.70 | 242.40 | 1.56% | **True** |
| `precip_7d_mm` | 80.60 | 0.99 | 78.10 | 82.60 | 1.23% | **True** |

![Figure 7: Inter-Chunk Feature Dissimilarity Matrix & Cross-Feature Correlation](figures/farm_feature_heterogeneity_heatmap.png)

---

## 9. Parcel-Intersecting Sensor Deployment Coordinates

The table below lists all **23 chunks intersecting King County Parcel `3420069035`**:

| Chunk ID | Row | Col | Macro Chunk ID | Center Lat (°N) | Center Lon (°W) | Elevation (m) | Soil Series | Sand (%) | Clay (%) | OM (%) | Greenness (GRVI) | MODIS LST (°C) | Annual Precip (mm) | 30d Precip (mm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `R02_C06` | 2 | 6 | Macro_M585_N972 | 47.1853 | -122.031 | 215.0 | Wilkeson | 26.5 | 13.8 | 7.63% | +0.108 | 25.60°C | 1466 mm | 231 mm |
| `R02_C07` | 2 | 7 | Macro_M585_N972 | 47.1853 | -122.028 | 217.0 | Wilkeson | 26.5 | 13.3 | 7.63% | +0.102 | 25.52°C | 1468 mm | 238 mm |
| `R02_C08` | 2 | 8 | Macro_M584_N972 | 47.1853 | -122.026 | 218.0 | Wilkeson | 26.5 | 14.4 | 7.63% | +0.067 | 23.82°C | 1469 mm | 237 mm |
| `R03_C06` | 3 | 6 | Macro_M585_N972 | 47.1838 | -122.031 | 216.0 | Wilkeson | 28.1 | 13.8 | 7.29% | +0.138 | 25.42°C | 1483 mm | 234 mm |
| `R03_C07` | 3 | 7 | Macro_M585_N972 | 47.1838 | -122.028 | 217.0 | Wilkeson | 28.1 | 13.3 | 7.29% | +0.158 | 25.29°C | 1484 mm | 241 mm |
| `R03_C08` | 3 | 8 | Macro_M584_N972 | 47.1838 | -122.026 | 218.0 | Wilkeson | 28.1 | 14.4 | 7.29% | +0.107 | 23.65°C | 1486 mm | 240 mm |
| `R04_C06` | 4 | 6 | Macro_M585_N971 | 47.1823 | -122.031 | 217.0 | Wilkeson | 30.4 | 13.8 | 7.23% | +0.047 | 23.95°C | 1490 mm | 235 mm |
| `R04_C07` | 4 | 7 | Macro_M585_N971 | 47.1823 | -122.028 | 219.0 | Kapowsin | 45.7 | 14.2 | 6.32% | +0.044 | 23.87°C | 1492 mm | 242 mm |
| `R04_C08` | 4 | 8 | Macro_M584_N971 | 47.1823 | -122.026 | 219.0 | Kapowsin | 45.7 | 14.4 | 6.32% | +0.178 | 25.10°C | 1492 mm | 241 mm |
| `R05_C06` | 5 | 6 | Macro_M585_N971 | 47.1807 | -122.031 | 216.0 | Wilkeson | 29.1 | 13.8 | 7.52% | +0.165 | 23.51°C | 1469 mm | 232 mm |
| `R05_C07` | 5 | 7 | Macro_M585_N971 | 47.1807 | -122.028 | 217.0 | Wilkeson | 29.1 | 13.3 | 7.52% | +0.212 | 23.26°C | 1471 mm | 238 mm |
| `R05_C08` | 5 | 8 | Macro_M584_N971 | 47.1807 | -122.026 | 217.0 | Wilkeson | 29.1 | 14.4 | 7.52% | +0.163 | 25.27°C | 1471 mm | 238 mm |
| `R06_C03` | 6 | 3 | Macro_M586_N971 | 47.1792 | -122.037 | 214.0 | Buckley | 51.4 | 12.8 | 10.03% | +0.116 | 25.61°C | 1471 mm | 235 mm |
| `R06_C04` | 6 | 4 | Macro_M585_N971 | 47.1792 | -122.035 | 214.0 | Wilkeson | 26.6 | 13.2 | 7.79% | +0.076 | 23.98°C | 1471 mm | 240 mm |
| `R06_C05` | 6 | 5 | Macro_M585_N971 | 47.1792 | -122.033 | 215.0 | Wilkeson | 26.6 | 14.2 | 7.79% | +0.109 | 23.79°C | 1473 mm | 234 mm |
| `R06_C06` | 6 | 6 | Macro_M585_N971 | 47.1792 | -122.031 | 216.0 | Wilkeson | 26.6 | 13.8 | 7.79% | +0.137 | 23.62°C | 1474 mm | 233 mm |
| `R06_C07` | 6 | 7 | Macro_M585_N971 | 47.1792 | -122.028 | 217.0 | Wilkeson | 26.6 | 13.3 | 7.79% | +0.137 | 23.57°C | 1475 mm | 239 mm |
| `R06_C08` | 6 | 8 | Macro_M584_N971 | 47.1792 | -122.026 | 217.0 | Wilkeson | 26.6 | 14.4 | 7.79% | +0.080 | 25.61°C | 1475 mm | 238 mm |
| `R07_C03` | 7 | 3 | Macro_M586_N971 | 47.1777 | -122.037 | 212.0 | Buckley | 52.9 | 12.8 | 10.19% | +0.149 | 25.57°C | 1486 mm | 238 mm |
| `R07_C04` | 7 | 4 | Macro_M585_N971 | 47.1777 | -122.035 | 213.0 | Wilkeson | 27.7 | 13.2 | 7.68% | +0.212 | 23.46°C | 1487 mm | 242 mm |
| `R07_C05` | 7 | 5 | Macro_M585_N971 | 47.1777 | -122.033 | 214.0 | Wilkeson | 27.7 | 14.2 | 7.68% | +0.187 | 23.51°C | 1488 mm | 237 mm |
| `R07_C06` | 7 | 6 | Macro_M585_N971 | 47.1777 | -122.031 | 215.0 | Wilkeson | 27.7 | 13.8 | 7.68% | +0.216 | 23.34°C | 1489 mm | 235 mm |
| `R07_C07` | 7 | 7 | Macro_M585_N971 | 47.1777 | -122.028 | 216.0 | Wilkeson | 27.7 | 13.3 | 7.68% | +0.070 | 23.90°C | 1491 mm | 242 mm |

---

## 10. Recommendations for ECE Field Placement

1. **Cross-Macro Placement**: Deploy at least one node in **Macro Chunk M586_N971** (e.g. `R06_C03` or `R07_C03`) and at least one node in **Macro Chunk M585_N971** (e.g. `R05_C07` or `R06_C06`) to ensure the sensors capture distinct MODIS 1km LST thermal steps.
2. **Soil Series Contrast**: Place one node in the alluvial lowland **Buckley series** (`R07_C03`, $10.19\%$ OM, $52.9\%$ sand) and another in the terrace **Wilkeson series** (`R05_C07`, $7.52\%$ OM, $29.1\%$ sand) to give the ML models rich soil feature variation.
3. **Vegetation Density Spacing**: Space sensors between pasture areas with high greenness (`R07_C06` GRVI $= +0.216$) and tilled/crop areas with moderate greenness (`R04_C06` GRVI $= +0.047$).
