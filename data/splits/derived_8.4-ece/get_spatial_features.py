# get_spatial_features.py
# Static geospatial features extractor (DEM, WorldCover, WorldClim, OpenLandMap)

import os
import argparse
import pandas as pd
import ee
from src.pipeline.utils.gee import initialize_ee

SRTM_DEM = "USGS/SRTMGL1_003"
WORLDCOVER = "ESA/WorldCover/v200"
WORLDCLIM_BIO = "WORLDCLIM/V1/BIO"

OLM_CLAY = "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
OLM_SAND = "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
OLM_TEXTURE = "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02"

OLM_DEPTH_BANDS = ["b0", "b10", "b30", "b60", "b100", "b200"]

def _rename_bands(img: ee.Image, prefix: str, bands: list[str]) -> ee.Image:
    return img.select(bands).rename([f"{prefix}_{b}" for b in bands])

def build_feature_image() -> ee.Image:
    # --- DEM / Terrain ---
    dem = ee.Image(SRTM_DEM).select("elevation")
    terrain = ee.Terrain.products(dem)

    elev = dem.rename("elev_m")
    slope = terrain.select("slope").rename("slope_deg")
    aspect = terrain.select("aspect").rename("aspect_deg")

    deg2rad = ee.Number(3.141592653589793).divide(180.0)
    slope_rad = slope.multiply(deg2rad)
    aspect_rad = aspect.multiply(deg2rad)

    slope_sin = slope_rad.sin().rename("slope_sin")
    slope_cos = slope_rad.cos().rename("slope_cos")
    aspect_sin = aspect_rad.sin().rename("aspect_sin")
    aspect_cos = aspect_rad.cos().rename("aspect_cos")

    clay = ee.Image(OLM_CLAY)
    sand = ee.Image(OLM_SAND)
    texc = ee.Image(OLM_TEXTURE)

    clay_b = _rename_bands(clay, "clay_wfrac", OLM_DEPTH_BANDS)
    sand_b = _rename_bands(sand, "sand_wfrac", OLM_DEPTH_BANDS)

    clay0 = clay.select("b0")
    sand0 = sand.select("b0")
    sand_clay_ratio = sand0.divide(clay0.add(1e-6)).rename("sand_clay_ratio_b0")
    clay_sand_sum = clay0.add(sand0).rename("clay_plus_sand_b0")

    tex_b = _rename_bands(texc, "soil_texture_usda", OLM_DEPTH_BANDS)

    # --- Landcover (categorical code) ---
    lc = ee.ImageCollection(WORLDCOVER).first().select("Map").rename("lc_code")

    # --- Climate normals (BIO1..BIO19) ---
    bio = ee.Image(WORLDCLIM_BIO)
    bio_names = bio.bandNames()
    bio = bio.rename(bio_names.map(lambda b: ee.String("bio_").cat(ee.String(b))))

    feat = ee.Image.cat([
        elev, slope, aspect,
        slope_sin, slope_cos, aspect_sin, aspect_cos,
        clay_b, sand_b, sand_clay_ratio, clay_sand_sum,
        tex_b,
        lc,
        bio
    ]).toFloat()

    return feat

def sample_points(feature_image: ee.Image, stations_df: pd.DataFrame, scale_m: int) -> pd.DataFrame:
    feats = []
    for _, r in stations_df.iterrows():
        sid = str(r["station_id"])
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        geom = ee.Geometry.Point([lon, lat])
        feats.append(ee.Feature(geom, {"station_id": sid, "latitude": lat, "longitude": lon}))

    fc = ee.FeatureCollection(feats)

    sampled = feature_image.sampleRegions(
        collection=fc,
        scale=scale_m,
        geometries=False
    )

    rows = sampled.getInfo()["features"]
    props = [f["properties"] for f in rows]
    out = pd.DataFrame(props)

    front = ["station_id", "latitude", "longitude"]
    cols = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
    out = out[cols].copy()
    return out

def add_landcover_labels(df: pd.DataFrame) -> pd.DataFrame:
    wc_map = {
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen",
    }
    if "lc_code" in df.columns:
        df["landcover_label"] = df["lc_code"].map(wc_map).fillna("Unknown")
    return df

def apply_family_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    meta = {"station_id", "latitude", "longitude", "landcover_label"}
    renamed = {}
    for c in df.columns:
        if c in meta:
            continue
        renamed[c] = f"J_{c}"
    return df.rename(columns=renamed)

def add_derived_geo_family_k(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6

    if "J_sand_wfrac_b0" in df.columns and "J_clay_wfrac_b0" in df.columns:
        df["K_sand_clay_ratio_b0"] = df["J_sand_wfrac_b0"] / (df["J_clay_wfrac_b0"] + eps)
        df["K_clay_plus_sand_b0"] = df["J_clay_wfrac_b0"] + df["J_sand_wfrac_b0"]

    for base in ["slope_sin", "slope_cos", "aspect_sin", "aspect_cos"]:
        jcol = f"J_{base}"
        if jcol in df.columns:
            df[f"K_{base}"] = df[jcol]
            df = df.drop(columns=[jcol])

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords_csv", default="data/splits/derived_8.4-ece/station_coords.csv")
    ap.add_argument("--out_csv", default="data/splits/derived_8.4-ece/station_static_features.csv")
    ap.add_argument("--scale_m", type=int, default=250)
    ap.add_argument("--include_k", action="store_true", default=True)
    args = ap.parse_args()

    initialize_ee()

    stations = pd.read_csv(args.coords_csv)
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude"]).copy()

    feat_img = build_feature_image()
    out = sample_points(feat_img, stations, scale_m=args.scale_m)
    out = add_landcover_labels(out)
    out = apply_family_prefixes(out)

    if args.include_k:
        out = add_derived_geo_family_k(out)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("Wrote:", args.out_csv)
    print("Columns:", len(out.columns))

if __name__ == "__main__":
    main()
