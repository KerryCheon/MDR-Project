# fetch_lia.py
# Jakob Balkovec
# fetched the station-level LIA summary

import argparse
import pandas as pd
import ee

def load_stations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["station_id", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.copy()
    df["station_id"] = df["station_id"].astype(str)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    return df


def stations_to_fc(df: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for _, r in df.iterrows():
        pt = ee.Geometry.Point([float(r["longitude"]), float(r["latitude"])])
        feats.append(ee.Feature(pt, {"station_id": str(r["station_id"])}))
    return ee.FeatureCollection(feats)


def add_lia_band(img: ee.Image, dem: ee.Image) -> ee.Image:
    # desc: 'lia_deg' band to a Sentinel-1 image using a common GEE approximation for look direction.

    # note:
    #   True LIA needs local look azimuth. In GEE, a practical approximation is to estimate
    #   radar azimuth direction from the incidence-angle surface and combine it with terrain
    #   slope/aspect from DEM

    # incidence angle from ellipsoid (degrees)
    theta_i = img.select("angle")

    # terrain slope/aspect (degrees)
    alpha_s = ee.Terrain.slope(dem).rename("slope")
    phi_s = ee.Terrain.aspect(dem).rename("aspect")

    # approximate look azimuth (degrees) by taking aspect of the incidence angle surface
    # and averaging over the image footprint
    phi_i = ee.Terrain.aspect(theta_i).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=img.geometry(),
        scale=1000,
        bestEffort=True,
        maxPixels=100000000
    ).get("aspect")

    # relative aspect between look direction and slope aspect
    phi_r = ee.Image.constant(phi_i).subtract(phi_s)

    deg2rad = ee.Number(3.141592653589793).divide(180.0)

    phi_r_rad = phi_r.multiply(deg2rad)
    alpha_s_rad = alpha_s.multiply(deg2rad)
    theta_i_rad = theta_i.multiply(deg2rad)
    alpha_r = (alpha_s_rad.tan().multiply(phi_r_rad.cos())).atan()
    alpha_az = (alpha_s_rad.tan().multiply(phi_r_rad.sin())).atan()

    # local incidence angle (radians), then convert to degrees
    theta_lia = (alpha_az.cos().multiply((theta_i_rad.subtract(alpha_r)).cos())).acos()
    lia_deg = theta_lia.divide(deg2rad).rename("lia_deg")

    return img.addBands(lia_deg)


def per_station_lia_stats(stations_fc: ee.FeatureCollection, start: str,
                          end: str, pass_filter: str | None, dem: ee.Image, scale_m: int) -> ee.FeatureCollection:
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filterDate(start, end)
        .select(["angle"])
    )
    if pass_filter in ("ASCENDING", "DESCENDING"):
        s1 = s1.filter(ee.Filter.eq("orbitProperties_pass", pass_filter))

    s1_lia = s1.map(lambda img: add_lia_band(img, dem))

    lia_mean = s1_lia.select("lia_deg").mean().rename("lia_mean_deg")
    lia_std = s1_lia.select("lia_deg").reduce(ee.Reducer.stdDev()).rename("lia_std_deg")

    lia_stack = lia_mean.addBands(lia_std)

    sampled = lia_stack.sampleRegions(
        collection=stations_fc,
        scale=scale_m,
        geometries=False
    )
    return sampled


def fc_to_df(fc: ee.FeatureCollection) -> pd.DataFrame:
    info = fc.getInfo()
    rows = []
    features = info.get("features", []) if info else []
    for f in features:
        props = f.get("properties", {})
        if "station_id" in props:
            rows.append(props)
    return pd.DataFrame(rows)


def main():

    try:
        ee.Initialize(project="mdr-project-475522")
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project="mdr-project-475522")
        except Exception:
            try:
                # Fallback to default user project
                ee.Initialize()
            except Exception as e:
                print(f"Failed to initialize Earth Engine: {e}")
                raise

    ap = argparse.ArgumentParser()
    ap.add_argument("--stations-csv", default="stations.csv")
    ap.add_argument("--out-csv", default="stations_lia.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--dem", default="USGS/SRTMGL1_003")
    ap.add_argument("--scale", type=int, default=30, help="Sampling scale at stations (meters)")
    args = ap.parse_args()

    stations = load_stations(args.stations_csv)
    stations_fc = stations_to_fc(stations)

    dem = ee.Image(args.dem)

    fc_all = per_station_lia_stats(stations_fc, args.start, args.end, None, dem, args.scale)
    df_all = fc_to_df(fc_all).rename(
        columns={"lia_mean_deg": "lia_mean_all_deg", "lia_std_deg": "lia_std_all_deg"}
    )[["station_id", "lia_mean_all_deg", "lia_std_all_deg"]]

    fc_asc = per_station_lia_stats(stations_fc, args.start, args.end, "ASCENDING", dem, args.scale)
    df_asc = fc_to_df(fc_asc).rename(
        columns={"lia_mean_deg": "lia_mean_asc_deg", "lia_std_deg": "lia_std_asc_deg"}
    )[["station_id", "lia_mean_asc_deg", "lia_std_asc_deg"]]

    fc_desc = per_station_lia_stats(stations_fc, args.start, args.end, "DESCENDING", dem, args.scale)
    df_desc = fc_to_df(fc_desc).rename(
        columns={"lia_mean_deg": "lia_mean_desc_deg", "lia_std_deg": "lia_std_desc_deg"}
    )[["station_id", "lia_mean_desc_deg", "lia_std_desc_deg"]]

    out = stations.merge(df_all, on="station_id", how="left").merge(df_asc, on="station_id", how="left").merge(df_desc, on="station_id", how="left")

    miss = out["lia_mean_all_deg"].isna().mean()
    print(f"Missing LIA(all) fraction: {miss:.4f}")

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
