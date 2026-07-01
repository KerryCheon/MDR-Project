# Metadata and configuration constants for derived_8.1_pos splits

# Calibrated regime thresholds (valleys-based calibration)
# Dry: SM < T1
# Transition: T1 <= SM < T2
# Wet: SM >= T2
T1 = 0.159
T2 = 0.248

# Semantic aliases
DRY_THRESHOLD = T1
WET_THRESHOLD = T2

# Selected features for each regime and overall
OVERALL_SELECTED_FEATURES = [
    "C_lag_E_SAR_diff_kobs30", "C_lag_E_SAR_ratio_kobs30", "C_lag_F_NDMI_kobs12", "C_lag_F_NDMI_kobs30", "C_lag_LST_modis_kobs30",
    "C_lag_s2_b11_kobs30", "D_sa_LST_modis", "D_z_E_SAR_ratio", "E_SAR_ratio", "LST_modis",
    "SMAP_sm_am_interp_lag30", "SMAP_sm_pm_interp_lag1", "SMAP_sm_pm_interp_lag30", "V_rollmax_E_SAR_diff_kobs14", "V_rollmax_E_SAR_ratio_kobs30",
    "V_rollmax_F_NDMI_kobs30", "V_rollmax_F_NDVI_kobs30", "V_rollmax_G_API_kobs14", "V_rollmax_G_API_kobs30", "V_rollmax_LST_modis_kobs30",
    "V_rollmax_SMAP_sm_interp_kobs14", "V_rollmax_s2_b11_kobs7", "V_rollmin_E_SAR_diff_kobs30", "V_rollmin_E_SAR_ratio_kobs30", "V_rollmin_F_NDVI_kobs30",
    "V_rollmin_G_API_kobs14", "V_rollmin_G_API_kobs30", "V_rollmin_LST_modis_kobs30", "V_rollmin_s2_b11_kobs30", "V_rollmin_s2_b12_kobs30",
    "C_lag_E_SAR_diff_kobs12", "C_lag_E_SAR_ratio_kobs5", "C_lag_F_NDMI_kobs6", "C_lag_SMAP_sm_interp_kobs2", "C_lag_SMAP_sm_interp_kobs6",
    "SMAP_sm_am_interp_rollrange30", "V_rollmax_F_NDMI_kobs7", "V_rollmax_F_NDVI_kobs7", "V_rollmax_LST_modis_kobs7", "V_rollmax_SMAP_sm_interp_kobs7"
]

DRY_SELECTED_FEATURES = [
    "C_lag_E_SAR_diff_kobs30", "C_lag_F_NDMI_kobs12", "C_lag_F_NDMI_kobs30", "C_lag_F_NDVI_kobs6", "C_lag_LST_modis_kobs2",
    "C_lag_SMAP_sm_interp_kobs12", "C_lag_s2_b11_kobs6", "DOY", "D_sa_E_SAR_ratio", "E_SAR_diff",
    "F_MSI", "F_NDMI", "SMAP_sm_pm_interp_lag7", "V_ema_LST_modis_kobs30", "V_rollmax_F_NDMI_kobs14",
    "V_rollmax_F_NDMI_kobs30", "V_rollmax_F_NDVI_kobs14", "V_rollmax_G_API_kobs30", "V_rollmax_LST_modis_kobs30", "V_rollmax_SMAP_sm_interp_kobs14",
    "V_rollmax_SMAP_sm_interp_kobs30", "V_rollmax_s2_b11_kobs7", "V_rollmax_s2_b12_kobs14", "V_rollmin_E_SAR_diff_kobs30", "V_rollmin_E_SAR_ratio_kobs30",
    "V_rollmin_F_NDMI_kobs30", "V_rollmin_F_NDVI_kobs30", "V_rollmin_G_API_kobs14", "V_rollmin_LST_modis_kobs30", "V_rollmin_SMAP_sm_interp_kobs30",
    "s1_vh", "C_lag_E_SAR_ratio_kobs30", "C_lag_F_NDVI_kobs30", "C_lag_LST_modis_kobs12", "C_lag_s2_b11_kobs30",
    "SMAP_sm_am_interp_lag30", "V_rollmax_s2_b11_kobs30", "V_rollmax_s2_b12_kobs30", "V_rollmin_E_SAR_ratio_kobs7", "V_rollmin_s2_b11_kobs30"
]

TRANSITION_SELECTED_FEATURES = [
    "C_lag_E_SAR_diff_kobs12", "C_lag_LST_modis_kobs12", "C_lag_LST_modis_kobs30", "C_lag_s2_b12_kobs30", "E_SAR_ratio",
    "F_MSI", "F_NDMI", "V_rollmax_F_NDMI_kobs30", "V_rollmax_F_NDVI_kobs30", "V_rollmax_G_API_kobs14",
    "V_rollmax_LST_modis_kobs30", "V_rollmax_SMAP_sm_interp_kobs14", "V_rollmax_SMAP_sm_interp_kobs30", "V_rollmax_s2_b11_kobs14", "V_rollmin_F_NDMI_kobs14",
    "V_rollmin_F_NDMI_kobs30", "V_rollmin_F_NDVI_kobs30", "V_rollmin_G_API_kobs30", "V_rollmin_LST_modis_kobs30", "V_rollmin_LST_modis_kobs7",
    "V_rollmin_SMAP_sm_interp_kobs30", "V_rollmin_s2_b11_kobs14", "V_rollmin_s2_b12_kobs14", "s2_b8", "C_lag_E_SAR_diff_kobs2",
    "C_lag_F_NDMI_kobs6", "C_lag_LST_modis_kobs1", "C_lag_LST_modis_kobs5", "C_lag_s2_b12_kobs12", "D_sa_LST_modis",
    "V_rollmax_E_SAR_diff_kobs14", "V_rollmax_E_SAR_diff_kobs30", "V_rollmax_F_NDMI_kobs14", "V_rollmax_LST_modis_kobs7", "V_rollmin_E_SAR_diff_kobs14",
    "V_rollmin_F_NDMI_kobs7", "V_rollmin_s2_b12_kobs30", "V_rollrng_E_SAR_ratio_kobs30", "s2_b4", "C_lag_E_SAR_diff_kobs30"
]

WET_SELECTED_FEATURES = [
    "A_d_E_SAR_ratio_kobs14", "C_lag_F_NDVI_kobs30", "D_sa_LST_modis", "E_SAR_diff", "E_SAR_ratio",
    "F_NDVI", "LST_modis", "SMAP_sm_pm_interp_lag30", "SMAP_sm_pm_interp_lag7", "V_rollmax_E_SAR_ratio_kobs30",
    "V_rollmax_F_NDVI_kobs30", "V_rollmax_G_API_kobs30", "V_rollmax_SMAP_sm_interp_kobs30", "V_rollmin_E_SAR_diff_kobs14", "V_rollmin_E_SAR_diff_kobs30",
    "V_rollmin_E_SAR_ratio_kobs14", "V_rollmin_F_NDMI_kobs30", "V_rollmin_G_API_kobs30", "V_rollmin_s2_b12_kobs14", "s1_vh",
    "s2_b4", "C_lag_E_SAR_diff_kobs12", "C_lag_E_SAR_diff_kobs30", "C_lag_E_SAR_diff_kobs6", "C_lag_LST_modis_kobs30",
    "C_lag_SMAP_sm_interp_kobs5", "D_sa_E_SAR_ratio", "F_NDMI", "SMAP_sm_interp_lag30", "V_rollmax_E_SAR_diff_kobs7",
    "V_rollmax_E_SAR_ratio_kobs7", "V_rollmax_F_NDMI_kobs30", "V_rollmin_F_NDVI_kobs7", "V_rollmin_LST_modis_kobs14", "V_rollmin_SMAP_sm_interp_kobs14",
    "V_rollmin_SMAP_sm_interp_kobs30", "V_rollmin_s2_b11_kobs14", "V_rollrng_s2_b12_kobs30", "s2_b8", "C_lag_E_SAR_ratio_kobs30"
]
