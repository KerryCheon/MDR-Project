import sys
import os

# Define the features lists from dataset_metadata.py
OVERALL_SELECTED_FEATURES_V1 = [
    'DOY', 'D_cos_DOY', 'D_sa_E_SAR_ratio', 'D_sa_F_NDMI', 'D_sa_LST_modis', 'D_sin_DOY', 
    'D_z_E_SAR_ratio', 'D_z_F_NDMI', 'D_z_LST_modis', 'E_SAR_ratio', 'F_MSI', 'F_NDMI', 
    'G_API', 'J_aspect_deg', 'J_bio_bio15', 'J_soil_texture_usda_b0', 'K_aspect_cos', 
    'LST_modis', 'V_ema_E_SAR_diff_kobs30', 'V_ema_E_SAR_ratio_kobs30', 'V_ema_F_NDVI_kobs30', 
    'V_rollmax_E_SAR_ratio_kobs30', 'V_rollmax_F_NDMI_kobs7', 'V_rollmean_E_SAR_diff_kobs30', 
    'V_rollmean_E_SAR_ratio_kobs30', 'V_rollmean_F_NDMI_kobs30', 'V_rollmin_F_NDVI_kobs30', 
    'V_rollmin_s2_b12_kobs30', 'latitude', 'lia_std_asc_deg', 's2_b4', 's2_b8', 'sin_year', 
    'slope', 'J_bio_bio03', 'V_rollmax_F_NDMI_kobs30', 'V_rollmax_s2_b11_kobs30', 
    'V_rollmean_LST_modis_kobs30', 'C_smm_E_SAR_diff_alpha0.85_n5', 'J_soil_texture_usda_b200'
]

OVERALL_SELECTED_FEATURES_V2 = [
    'A_d_E_SAR_diff_kobs30', 'A_d_E_SAR_ratio_kobs30', 'A_grad_E_SAR_diff_kobs30', 'A_grad_E_SAR_ratio_kobs30', 
    'C_lag_F_NDVI_kobs30', 'DOY', 'D_cos_DOY', 'D_sin_DOY', 'J_aspect_deg', 'J_bio_bio15', 
    'J_bio_bio16', 'J_bio_bio19', 'J_lc_code', 'J_soil_texture_usda_b0', 'J_soil_texture_usda_b10', 
    'J_soil_texture_usda_b200', 'K_aspect_cos', 'SMAP_sm_pm_interp_rollrange30', 'V_ema_LST_modis_kobs30', 
    'V_rollmean_LST_modis_kobs30', 'V_rollmin_LST_modis_kobs30', 'V_rollmin_s2_b11_kobs30', 'latitude', 
    'lia_std_asc_deg', 's2_b8', 'sin_year', 'slope', 'G_API', 'J_bio_bio13', 'SMAP_x_year', 
    'V_rollmax_LST_modis_kobs14', 'V_rollrng_G_API_kobs7', 'G_DSLR', 'V_rollrng_E_SAR_ratio_kobs30', 
    'V_rollmax_F_NDVI_kobs14', 'C_lag_LST_modis_kobs30', 'V_rollmax_G_API_kobs30', 'A_d_SMAP_sm_interp_kobs5', 
    'V_rollrng_F_NDVI_kobs30', 'J_bio_bio12'
]

OVERALL_SELECTED_FEATURES_V3 = [
    'A_d_E_SAR_diff_kobs30', 'A_grad_E_SAR_diff_kobs30', 'A_grad_E_SAR_ratio_kobs30', 'C_lag_F_NDVI_kobs30', 
    'DOY', 'D_cos_DOY', 'D_sin_DOY', 'J_aspect_deg', 'J_bio_bio15', 'J_lc_code', 
    'J_soil_texture_usda_b10', 'J_soil_texture_usda_b200', 'K_aspect_cos', 'SMAP_sm_pm_interp_rollrange30', 
    'V_rollmean_LST_modis_kobs30', 'V_rollmin_LST_modis_kobs30', 'V_rollmin_s2_b11_kobs30', 'latitude', 
    'lia_std_asc_deg', 's2_b8', 'sin_year', 'A_d_E_SAR_ratio_kobs30', 'G_API', 'J_bio_bio19', 
    'SMAP_x_year', 'V_rollmax_LST_modis_kobs14', 'V_rollrng_G_API_kobs7', 'G_DSLR', 'V_rollrng_E_SAR_ratio_kobs30', 
    'slope', 'J_soil_texture_usda_b0', 'V_rollrng_F_NDVI_kobs30', 'V_rollmax_F_NDVI_kobs14', 'A_d_SMAP_sm_interp_kobs5', 
    'V_rollmax_G_API_kobs30', 'V_rollrng_s2_b11_kobs30', 'C_lag_LST_modis_kobs30', 'E_rough_s1_vh_kobs14', 
    's2_b4', 'A_grad_SMAP_sm_interp_kobs30', 'V_ema_LST_modis_kobs30', 'D_fft_dom_LST_modis_kobs30', 
    'E_rough_s1_vh_kobs7', 'D_fft_ent_LST_modis_kobs30', 'V_rollmin_F_NDMI_kobs14', 'cos_year', 
    'J_bio_bio16'
]

OVERALL_SELECTED_FEATURES_V4 = [
    'A_d_E_SAR_diff_kobs30', 'A_d_E_SAR_ratio_kobs30', 'A_d_F_NDMI_kobs30', 'A_d_LST_modis_kobs30', 
    'A_d_SMAP_sm_interp_kobs30', 'A_grad_E_SAR_diff_kobs30', 'A_grad_E_SAR_ratio_kobs30', 'A_grad_F_NDMI_kobs30', 
    'A_grad_LST_modis_kobs30', 'A_grad_SMAP_sm_interp_kobs30', 'C_lag_E_SAR_ratio_kobs30', 'C_lag_F_NDVI_kobs30', 
    'C_lag_G_API_kobs1', 'C_lag_LST_modis_kobs30', 'I_ts_spike_s1_vv', 'SMAP_ampm_diff_interp', 
    'SMAP_sm_pm_interp_rollrange30', 'V_rollcv_E_SAR_diff_kobs30', 'V_rollcv_G_API_kobs30', 'V_rollmax_E_SAR_ratio_kobs30', 
    'V_rollmin_LST_modis_kobs30', 'V_rollrng_F_NDMI_kobs30', 'V_rollrng_G_API_kobs30', 'V_rollrng_G_API_kobs7', 
    'V_rollstd_G_API_kobs30', 'lia_mean_asc_deg', 'lia_std_asc_deg', 'V_rollrng_E_SAR_diff_kobs30', 
    'V_rollstd_G_API_kobs7', 'SMAP_sm_pm_interp_rollstd30', 'E_rough_s1_vv_kobs14', 'V_rollstd_G_API_kobs14', 
    'V_rollrng_G_API_kobs14', 'V_ema_LST_modis_kobs30', 'A_d_E_SAR_ratio_kobs14', 'A_grad_E_SAR_ratio_kobs14', 
    'V_rollcv_G_API_kobs14', 'V_rollmean_F_NDVI_kobs30', 'V_rollcv_G_API_kobs7', 'A_d_LST_modis_kobs14', 
    'A_grad_LST_modis_kobs14', 'SMAP_sm_pm_interp_rollmean7', 'V_rollrng_E_SAR_ratio_kobs30', 'V_rollcv_E_SAR_diff_kobs14', 
    'V_rollmax_G_API_kobs7', 'E_rough_s1_vv_kobs7', 'C_lag_F_NDVI_kobs12', 'SMAP_sm_pm_interp_lag1', 
    'V_rollmax_G_API_kobs30', 'V_rollrng_E_SAR_diff_kobs14'
]

OVERALL_SELECTED_FEATURES_V5 = [
    'A_d_E_SAR_diff_kobs30', 'A_d_E_SAR_ratio_kobs30', 'A_d_F_NDMI_kobs30', 'A_d_LST_modis_kobs30', 
    'A_d_SMAP_sm_interp_kobs30', 'A_grad_E_SAR_diff_kobs30', 'A_grad_E_SAR_ratio_kobs30', 'A_grad_F_NDMI_kobs30', 
    'A_grad_LST_modis_kobs30', 'A_grad_SMAP_sm_interp_kobs30', 'C_lag_E_SAR_ratio_kobs30', 'C_lag_F_NDVI_kobs30', 
    'C_lag_G_API_kobs1', 'C_lag_LST_modis_kobs30', 'I_ts_spike_s1_vv', 'SMAP_ampm_diff_interp', 
    'SMAP_sm_pm_interp_rollrange30', 'V_rollcv_E_SAR_diff_kobs30', 'V_rollcv_G_API_kobs30', 'V_rollmax_E_SAR_ratio_kobs30', 
    'V_rollmin_LST_modis_kobs30', 'V_rollrng_F_NDMI_kobs30', 'V_rollrng_G_API_kobs30', 'V_rollrng_G_API_kobs7', 
    'V_rollstd_G_API_kobs30', 'lia_mean_asc_deg', 'lia_std_asc_deg', 'V_rollrng_E_SAR_diff_kobs30', 
    'V_rollstd_G_API_kobs7', 'SMAP_sm_pm_interp_rollstd30', 'E_rough_s1_vv_kobs14', 'V_rollstd_G_API_kobs14'
]

def categorize(feature):
    if feature in ['DOY', 's2_b8', 'latitude', 'slope', 's2_b4', 'LST_modis']:
        return "Raw Input (RAW)"
    if feature.startswith("A_d_"):
        return "Temporal Difference (A_d_)"
    if feature.startswith("A_grad_"):
        return "Temporal Gradient (A_grad_)"
    if feature.startswith("C_lag_"):
        return "Autoregressive Lag (C_lag_)"
    if feature.startswith("D_fft_"):
        return "Spectral Fourier (D_fft_)"
    if feature.startswith("D_sa_") or feature.startswith("D_z_") or feature in ['D_cos_DOY', 'D_sin_DOY', 'sin_year', 'cos_year']:
        return "Seasonal / Calendar (D_)"
    if feature.startswith("E_rough_"):
        return "Surface Roughness Proxy (E_rough_)"
    if feature.startswith("I_"):
        return "Temporal Anomalous Spike (I_)"
    if feature.startswith("lia_"):
        return "Local Incidence Angle (lia_)"
    if feature.startswith("J_bio_"):
        return "Bioclimatic Variable (J_bio_)"
    if feature.startswith("J_") or feature.startswith("K_"):
        return "Static GIS / Soil (J_ / K_)"
    if feature.startswith("SMAP_"):
        return "SMAP Interpolations (SMAP_)"
    if feature.startswith("V_") or feature.startswith("C_smm_"):
        return "Rolling / Moving Average (V_)"
    if feature.startswith("G_"):
        return "Hydrologic / Precipitation (G_)"
    # specific V1 raw satellite features
    if feature in ['E_SAR_ratio', 'F_MSI', 'F_NDMI']:
        return "Raw Input (RAW)"
    return "Other"

def analyze():
    v1 = set(OVERALL_SELECTED_FEATURES_V1)
    v2 = set(OVERALL_SELECTED_FEATURES_V2)
    v3 = set(OVERALL_SELECTED_FEATURES_V3)
    v4 = set(OVERALL_SELECTED_FEATURES_V4)
    v5 = set(OVERALL_SELECTED_FEATURES_V5)
    
    print("=== Set Sizes ===")
    print(f"V1: {len(v1)}")
    print(f"V2: {len(v2)}")
    print(f"V3: {len(v3)}")
    print(f"V4: {len(v4)}")
    print(f"V5: {len(v5)}")
    
    print("\n=== Intersections ===")
    print(f"V1 and V2 (both): {len(v1.intersection(v2))}")
    print(f"V1 and V3 (both): {len(v1.intersection(v3))}")
    
    print("\n=== Unique features: V1 vs V2 ===")
    print(f"In V1 but not in V2: {len(v1 - v2)}")
    print(f"In V2 but not in V1: {len(v2 - v1)}")
    
    print("\n=== Features in V1 but NOT in V2 ===")
    for f in sorted(v1 - v2):
        print(f"  {f} ({categorize(f)})")
        
    print("\n=== Features in V2 but NOT in V1 ===")
    for f in sorted(v2 - v1):
        print(f"  {f} ({categorize(f)})")
        
    print("\n=== Category Breakdowns ===")
    all_categories = sorted(list(set(categorize(f) for f in v1 | v2 | v3 | v4 | v5)))
    
    # Let's print table of category counts for each model
    print(f"{'Category':<35} | {'V1':<4} | {'V2':<4} | {'V3':<4} | {'V4':<4} | {'V5':<4} | {'Union':<5}")
    print("-" * 75)
    for cat in all_categories:
        count_v1 = sum(1 for f in v1 if categorize(f) == cat)
        count_v2 = sum(1 for f in v2 if categorize(f) == cat)
        count_v3 = sum(1 for f in v3 if categorize(f) == cat)
        count_v4 = sum(1 for f in v4 if categorize(f) == cat)
        count_v5 = sum(1 for f in v5 if categorize(f) == cat)
        count_union = sum(1 for f in (v1 | v2 | v3 | v4 | v5) if categorize(f) == cat)
        print(f"{cat:<35} | {count_v1:<4} | {count_v2:<4} | {count_v3:<4} | {count_v4:<4} | {count_v5:<4} | {count_union:<5}")

if __name__ == '__main__':
    analyze()
