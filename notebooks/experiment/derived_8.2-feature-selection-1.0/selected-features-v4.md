# Walkthrough - Creating OVERALL_SELECTED_FEATURES_V4

We have successfully generated `OVERALL_SELECTED_FEATURES_V4` for the `derived_8.2` split by skipping the Mutual Information (MI) stage and increasing `k` in the ElasticNet and stability selection stages to ensure 50 final features are selected.

## Summary of Changes

1. **Created Config**: [config_v4.yaml](./config_v4.yaml)
   - Omitted the `mi` stage entirely.
   - Set the `elasticnet` stage `k` to `80`.
   - Set the `stability` stage `k` to `80`.
   - Set `min_freq: 0.01` under stability selector, using `top_k: 50` as the primary threshold to select the 50 most stable features across the 100 bootstrap rounds.
2. **Created Run Script**: [run_all_selections_v4.py](./run_all_selections_v4.py)
   - Runs the pipeline using the new config and updates `OVERALL_SELECTED_FEATURES_V4` in `data/splits/derived_8.2/dataset_metadata.py`.
3. **Execution**:
   - Ran `uv run notebooks/experiment/derived_8.2-feature-selection/run_all_selections_v4.py` successfully.
   - Output logs and files are saved under the run directory: `C:\Users\pan\Documents\GitHub\MDR-Project\Modeling\Runs\2026-07-10_193906`.
4. **Target File Updated**: [dataset_metadata.py](../../../data/splits/derived_8.2/dataset_metadata.py)
   - Appended `OVERALL_SELECTED_FEATURES_V4` containing exactly 50 selected features.

---

## Model Evaluation Metrics with V4 Features

After selection, standard models (Linear, XGBoost, Random Forest) were trained and evaluated on the 50 selected features:

| Model | Split | R² Score | RMSE | MAE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear** | Train | 0.4488 | 0.0805 | 0.0659 |
| | Val | 0.4064 | 0.0884 | 0.0733 |
| | Test | 0.2598 | 0.0906 | 0.0730 |
| **XGBoost** | Train | 0.9629 | 0.0209 | 0.0149 |
| | Val | 0.6863 | 0.0643 | 0.0439 |
| | Test | 0.5236 | 0.0727 | 0.0552 |
| **Random Forest** | Train | 0.9899 | 0.0109 | 0.0064 |
| | Val | 0.6928 | 0.0636 | 0.0428 |
| | Test | 0.5481 | 0.0708 | 0.0534 |

---

## Selected 50 Features (`OVERALL_SELECTED_FEATURES_V4`)

The selected features are listed below, ordered by their stability frequency (from most stable to least stable):

1. `A_d_E_SAR_diff_kobs30` (Frequency: 1.0)
2. `A_d_E_SAR_ratio_kobs30` (Frequency: 1.0)
3. `A_d_F_NDMI_kobs30` (Frequency: 1.0)
4. `A_d_LST_modis_kobs30` (Frequency: 1.0)
5. `A_d_SMAP_sm_interp_kobs30` (Frequency: 1.0)
6. `A_grad_E_SAR_diff_kobs30` (Frequency: 1.0)
7. `A_grad_E_SAR_ratio_kobs30` (Frequency: 1.0)
8. `A_grad_F_NDMI_kobs30` (Frequency: 1.0)
9. `A_grad_LST_modis_kobs30` (Frequency: 1.0)
10. `A_grad_SMAP_sm_interp_kobs30` (Frequency: 1.0)
11. `C_lag_E_SAR_ratio_kobs30` (Frequency: 1.0)
12. `C_lag_F_NDVI_kobs30` (Frequency: 1.0)
13. `C_lag_G_API_kobs1` (Frequency: 1.0)
14. `C_lag_LST_modis_kobs30` (Frequency: 1.0)
15. `I_ts_spike_s1_vv` (Frequency: 1.0)
16. `SMAP_ampm_diff_interp` (Frequency: 1.0)
17. `SMAP_sm_pm_interp_rollrange30` (Frequency: 1.0)
18. `V_rollcv_E_SAR_diff_kobs30` (Frequency: 1.0)
19. `V_rollcv_G_API_kobs30` (Frequency: 1.0)
20. `V_rollmax_E_SAR_ratio_kobs30` (Frequency: 1.0)
21. `V_rollmin_LST_modis_kobs30` (Frequency: 1.0)
22. `V_rollrng_F_NDMI_kobs30` (Frequency: 1.0)
23. `V_rollrng_G_API_kobs30` (Frequency: 1.0)
24. `V_rollrng_G_API_kobs7` (Frequency: 1.0)
25. `V_rollstd_G_API_kobs30` (Frequency: 1.0)
26. `lia_mean_asc_deg` (Frequency: 1.0)
27. `lia_std_asc_deg` (Frequency: 1.0)
28. `V_rollrng_E_SAR_diff_kobs30` (Frequency: 0.93)
29. `V_rollstd_G_API_kobs7` (Frequency: 0.81)
30. `SMAP_sm_pm_interp_rollstd30` (Frequency: 0.74)
31. `E_rough_s1_vv_kobs14` (Frequency: 0.71)
32. `V_rollstd_G_API_kobs14` (Frequency: 0.68)
33. `V_rollrng_G_API_kobs14` (Frequency: 0.48)
34. `V_ema_LST_modis_kobs30` (Frequency: 0.46)
35. `A_d_E_SAR_ratio_kobs14` (Frequency: 0.43)
36. `A_grad_E_SAR_ratio_kobs14` (Frequency: 0.43)
37. `V_rollcv_G_API_kobs14` (Frequency: 0.38)
38. `V_rollmean_F_NDVI_kobs30` (Frequency: 0.36)
39. `V_rollcv_G_API_kobs7` (Frequency: 0.26)
40. `A_d_LST_modis_kobs14` (Frequency: 0.24)
41. `A_grad_LST_modis_kobs14` (Frequency: 0.24)
42. `SMAP_sm_pm_interp_rollmean7` (Frequency: 0.22)
43. `V_rollrng_E_SAR_ratio_kobs30` (Frequency: 0.19)
44. `V_rollcv_E_SAR_diff_kobs14` (Frequency: 0.18)
45. `V_rollmax_G_API_kobs7` (Frequency: 0.12)
46. `E_rough_s1_vv_kobs7` (Frequency: 0.11)
47. `C_lag_F_NDVI_kobs12` (Frequency: 0.08)
48. `SMAP_sm_pm_interp_lag1` (Frequency: 0.08)
49. `V_rollmax_G_API_kobs30` (Frequency: 0.08)
50. `V_rollrng_E_SAR_diff_kobs14` (Frequency: 0.05)
