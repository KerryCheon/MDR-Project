import pandas as pd
import ee
import argparse
from datetime import datetime

# ---------------------------
# Initialize Earth Engine
# ---------------------------
def init_ee(project_id='mdr-research'):
    ee.Initialize(project=project_id)
    print(f"✅ Earth Engine initialized with project '{project_id}'")

# ---------------------------
# Load and clean ground data
# ---------------------------
def load_ground_data(csv_path):
    # Load CSV, parse 'Date' column instead of 'datetime'
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    
    # Rename and set index
    df = df.rename(columns={'Date': 'datetime'})
    df.set_index('datetime', inplace=True)
    
    print("✅ Ground data loaded")
    print(df.head())
    
    if 'soil_moisture_2in' not in df.columns:
        raise ValueError("Target column 'soil_moisture_2in' not found in ground data.")
    
    print("Missing values per column:")
    print(df.isna().sum())
    
    return df


# ---------------------------
# Retrieve Sentinel data
# ---------------------------
def get_sentinel_features(lat, lon, start_date, end_date):
    point = ee.Geometry.Point([lon, lat])
    
    # Sentinel-1 VV/VH
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(point)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .select(['VV', 'VH']))
    
    # Sentinel-2 B4,B3,B2,B8
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR")
          .filterBounds(point)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .select(['B4','B3','B2','B8']))
    
    # NDVI calculation
    def add_ndvi(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return img.addBands(ndvi)
    
    s2 = s2.map(add_ndvi)
    
    # Sample point
    def sample_image(img):
        sample = img.sample(point, scale=10).first()
        return sample
    
    def ee_to_pd(ee_ic):
        dates = ee_ic.aggregate_array('system:time_start').getInfo()
        features = []
        for d in dates:
            img_dict = ee_ic.filter(ee.Filter.eq('system:time_start', d)).first().toDictionary().getInfo()
            img_dict['datetime'] = pd.to_datetime(d, unit='ms')
            features.append(img_dict)
        df_feat = pd.DataFrame(features)
        df_feat.set_index('datetime', inplace=True)
        return df_feat
    
    s1_df = ee_to_pd(s1)
    s2_df = ee_to_pd(s2)
    
    # Sanity check
    print(f"✅ Sentinel-1 shape: {s1_df.shape}, Sentinel-2 shape: {s2_df.shape}")
    return s1_df, s2_df

# ---------------------------
# Merge ground + satellite features
# ---------------------------
def merge_features(ground_df, s1_df, s2_df):
    full_df = ground_df.join([s1_df, s2_df], how='outer')
    
    # Interpolate missing values
    full_df = full_df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    
    print("✅ Full dataset prepared")
    print(full_df.head())
    print("Missing values after merge:")
    print(full_df.isna().sum())
    
    return full_df

# ---------------------------
# Main script
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_csv', type=str, help='Path to cleaned ground CSV')
    parser.add_argument('--lat', type=float, default=47.0, help='Station latitude')
    parser.add_argument('--lon', type=float, default=-118.57, help='Station longitude')
    parser.add_argument('--out_dir', type=str, default='./processed', help='Output directory')
    
    args = parser.parse_args()
    
    init_ee()
    
    ground_df = load_ground_data(args.ground_csv)
    start_date = ground_df.index.min().strftime('%Y-%m-%d')
    end_date = ground_df.index.max().strftime('%Y-%m-%d')
    
    s1_df, s2_df = get_sentinel_features(args.lat, args.lon, start_date, end_date)
    
    full_df = merge_features(ground_df, s1_df, s2_df)
    
    # Save merged dataset
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/soil_moisture_dataset.csv"
    full_df.to_csv(out_path)
    print(f"✅ Merged dataset saved to {out_path}")
