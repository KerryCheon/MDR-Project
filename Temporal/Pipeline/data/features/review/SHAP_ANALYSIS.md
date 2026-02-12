# Global SHAP Feature Importance + Correlation Report

**Author**: Jakob Balkovec
**Date:** Feb 11th 2026

## Quick Summary

- **Dataset**: `MDR/Temporal/Pipeline/data/splits/derived_new_updated/test_derived_updated.csv`
  - I had to make another split since some of the features were dropped in the original `test_derived.csv`. This new split is based on the same original data but retains all features for the test set.
- Selected features (after mapping): 15
- Missing after mapping: `None`
- Feature set 10 handling: `xgb`
- MI estimator: `scikit-learn` (symmetrized)
- Dataset selection: `max_coverage`

### Method

For each feature, I defined the following quantities:

- **Frequency**

  $$
  f = \frac{\#\{\text{feature sets where the feature appears in the top-10}\}}{11}
  $$

- **Average SHAP magnitude**

  $$
  \overline{|\mathrm{SHAP}|} = \frac{1}{n_f} \sum_{i=1}^{n_f} |\mathrm{SHAP}_i|
  $$

> Note: Where $n_f$ is the number of feature sets in which the feature appears in the top-10

- **Global feature importance**

  $$
  G = f \times \overline{|\mathrm{SHAP}|}
  $$

> Note: This metric prioritizes features that are both consistently important across feature sets and strongly influential within each set

Feature selection for correlation uses `top 15 by global_importance`. Correlations are computed on `MDR/Temporal/Pipeline/data/splits/derived_new_updated/test_derived_updated.csv`

**Stats:**

```
(rows: 2829, dropped NaNs: 0).
```

> Note: Output column names retain `avg_shap`/`shap_values` for backwards compatibility.

## Top Features by Global Importance

| Feature                  | Freq Count | Frequency | Avg Importance | Global Importance |
| ------------------------ | ---------- | --------- | -------------- | ----------------- |
| DOY                      | 8          | 0.727     | 0.141017       | 0.102558          |
| V_ema_LST_modis_kobs30   | 2          | 0.182     | 0.201660       | 0.036665          |
| NDMI                     | 5          | 0.455     | 0.074629       | 0.033922          |
| C_smm_G_API_alpha0.85_n5 | 2          | 0.182     | 0.181003       | 0.032910          |
| MSI                      | 5          | 0.455     | 0.071109       | 0.032322          |
| s1_vv                    | 5          | 0.455     | 0.065524       | 0.029784          |
| precip_mm                | 5          | 0.455     | 0.065421       | 0.029737          |
| SAR_ratio                | 6          | 0.545     | 0.051482       | 0.028081          |
| API                      | 1          | 0.091     | 0.308204       | 0.028019          |
| V_ema_G_API_kobs7        | 1          | 0.091     | 0.299229       | 0.027203          |
| s1_vh                    | 5          | 0.455     | 0.055946       | 0.025430          |
| V_rollmin_G_API_kobs7    | 2          | 0.182     | 0.124321       | 0.022604          |
| G_API                    | 3          | 0.273     | 0.079060       | 0.021562          |
| s2_b8                    | 4          | 0.364     | 0.056122       | 0.020408          |
| G_rain_sum_30d           | 3          | 0.273     | 0.068193       | 0.018598          |

![Top 15](figures/global_importance_top15.png)

## Consistent Features

> Note: The following features appear in the top-10 for at least 6 out of the 11 feature sets. This indicates they are consistently important across different model configurations and data subsets

| Feature   | Freq Count | Frequency | Avg Importance | Global Importance |
| --------- | ---------- | --------- | -------------- | ----------------- |
| DOY       | 8          | 0.727     | 0.141017       | 0.102558          |
| SAR_ratio | 6          | 0.545     | 0.051482       | 0.028081          |
| NDMI      | 5          | 0.455     | 0.074629       | 0.033922          |
| MSI       | 5          | 0.455     | 0.071109       | 0.032322          |
| s1_vv     | 5          | 0.455     | 0.065524       | 0.029784          |
| precip_mm | 5          | 0.455     | 0.065421       | 0.029737          |
| s1_vh     | 5          | 0.455     | 0.055946       | 0.025430          |

**Condensed:**

> Note: The threshold was dropped from 6 to 5 to include `precip_mm` and `SAR_ratio`

| Feature (>=5/11) |
| ---------------- |
| DOY              |
| NDMI             |
| MSI              |
| s1_vv            |
| precip_mm        |
| SAR_ratio        |
| s1_vh            |

## Correlation Analysis and Thresholding

After identifying features with high global importance, I moved onto examining pairwise relationships between these features using three complementary correlation measures

> Note: The formulas have been truncated for readability + the fact that these are standard definitions that can be easily looked up. The key point is that I used a combination of linear, monotonic, and nonlinear metrics to capture a wide range of potential dependencies between features.

For any feature pair \( (x, y) \), I computed:

- **Pearson correlation**
  \[
  r(x, y) = \frac{\mathrm{cov}(x, y)}{\sigma_x \sigma_y}
  \]

  > Note: This measures the strength of **linear** relationships

- **Spearman rank correlation**
  \[
  \rho(x, y) = r(\mathrm{rank}(x), \mathrm{rank}(y))
  \]

  > Note: This captures **monotonic** relationships, including nonlinear but consistently increasing or decreasing trends

- **Mutual information**
  \[
  \mathrm{MI}(x, y)
  \]
  > Note: This measures which **general nonlinear dependence** and can detect relationships missed by correlation coefficients

**Yoinked from**:

- [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [Spearman’s rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
- [Mutual information](https://en.wikipedia.org/wiki/Mutual_information)

Feature pairs were flagged as **highly correlated** if **any** of the following criteria were satisfied:

- **Pearson correlation**:
  \( |r| \ge 0.6 \), indicating a strong linear relationship

- **Spearman correlation**:
  \( |\rho| \ge 0.6 \), indicating a strong monotonic relationship

- **Mutual information**:
  \( \mathrm{MI} \ge 0.3 \), indicating a meaningful nonlinear dependency

Using all three metrics together helps avoid missing important interactions that may not be strictly linear, while still keeping the set of candidate feature pairs focused and interpretable

> Note: Pearson measures linear relationships, Spearman measures monotonic (rank-based) relationships, and Mutual Information measures general statistical dependence (including nonlinear relationships)

## Heatmaps

These are for the selected features (after mapping)
![](figures/correlation_pearson.png)
![](figures/correlation_spearman.png)
![](figures/correlation_mutual_info.png)

## Top Correlated Pairs

> Note: I broke down the table into three sections based on the type of relationship (linear, monotonic, nonlinear) and the strength of the correlation. See below

> Note: Mutual information values are unnormalized and should be interpreted comparatively within this dataset rather than as absolute magnitudes

| Feature A             | Feature B                | Pearson | Spearman | Mutual Info | Relationship Type         |
| --------------------- | ------------------------ | ------- | -------- | ----------- | ------------------------- |
| F_NDMI                | F_MSI                    | -0.969  | -1.000   | 1.000       | Strong negative linear    |
| V_ema_G_API_kobs7     | C_smm_G_API_alpha0.85_n5 | 0.996   | 0.996    | 0.449       | Strong positive linear    |
| s1_vv                 | s1_vh                    | 0.945   | 0.901    | 0.984       | Strong positive linear    |
| s1_vv                 | E_SAR_ratio              | -0.586  | -0.588   | 0.984       | Nonlinear dependency      |
| s1_vh                 | E_SAR_ratio              | -0.788  | -0.835   | 0.983       | Strong negative linear    |
| G_API                 | V_ema_G_API_kobs7        | 0.972   | 0.980    | 0.307       | Strong positive linear    |
| V_rollmin_G_API_kobs7 | C_smm_G_API_alpha0.85_n5 | 0.979   | 0.979    | 0.405       | Strong positive linear    |
| V_rollmin_G_API_kobs7 | V_ema_G_API_kobs7        | 0.978   | 0.978    | 0.368       | Strong positive linear    |
| G_API                 | C_smm_G_API_alpha0.85_n5 | 0.958   | 0.968    | 0.335       | Strong positive linear    |
| G_API                 | V_rollmin_G_API_kobs7    | 0.926   | 0.940    | 0.454       | Strong positive linear    |
| G_rain_sum_30d        | V_ema_G_API_kobs7        | 0.922   | 0.931    | 0.230       | Strong positive linear    |
| G_rain_sum_30d        | C_smm_G_API_alpha0.85_n5 | 0.907   | 0.918    | 0.202       | Strong positive linear    |
| G_rain_sum_30d        | V_rollmin_G_API_kobs7    | 0.896   | 0.918    | 0.243       | Strong positive linear    |
| G_API                 | G_rain_sum_30d           | 0.882   | 0.902    | 0.161       | Strong positive linear    |
| F_MSI                 | E_SAR_ratio              | 0.424   | 0.440    | 0.783       | Nonlinear dependency      |
| F_NDMI                | E_SAR_ratio              | -0.310  | -0.440   | 0.782       | Nonlinear dependency      |
| s1_vh                 | F_MSI                    | -0.658  | -0.603   | 0.781       | Strong negative linear    |
| s1_vh                 | F_NDMI                   | 0.551   | 0.603    | 0.780       | Strong positive monotonic |
| s2_b8                 | F_NDMI                   | 0.571   | 0.629    | 0.780       | Strong positive monotonic |
| s2_b8                 | F_MSI                    | -0.598  | -0.629   | 0.778       | Strong negative monotonic |
| s1_vv                 | F_MSI                    | -0.720  | -0.709   | 0.775       | Strong negative linear    |
| s1_vv                 | F_NDMI                   | 0.649   | 0.709    | 0.774       | Strong positive linear    |
| s2_b8                 | E_SAR_ratio              | -0.364  | -0.282   | 0.766       | Nonlinear dependency      |
| s1_vv                 | s2_b8                    | 0.386   | 0.409    | 0.758       | Nonlinear dependency      |
| s1_vh                 | s2_b8                    | 0.373   | 0.350    | 0.755       | Nonlinear dependency      |
| s1_vh                 | V_ema_G_API_kobs7        | 0.623   | 0.690    | 0.172       | Strong positive linear    |
| s1_vh                 | C_smm_G_API_alpha0.85_n5 | 0.610   | 0.680    | 0.144       | Strong positive linear    |
| s1_vh                 | G_rain_sum_30d           | 0.623   | 0.680    | 0.320       | Strong positive linear    |
| s1_vh                 | G_API                    | 0.606   | 0.676    | 0.123       | Strong positive linear    |
| E_SAR_ratio           | V_ema_G_API_kobs7        | -0.497  | -0.664   | 0.203       | Strong negative monotonic |

### Strong Linear Relationships

**Positive**:

| Feature A             | Feature B                | Pearson | Spearman | Mutual Info |
| --------------------- | ------------------------ | ------- | -------- | ----------- |
| V_ema_G_API_kobs7     | C_smm_G_API_alpha0.85_n5 | 0.996   | 0.996    | 0.449       |
| s1_vv                 | s1_vh                    | 0.945   | 0.901    | 0.984       |
| G_API                 | V_ema_G_API_kobs7        | 0.972   | 0.980    | 0.307       |
| V_rollmin_G_API_kobs7 | C_smm_G_API_alpha0.85_n5 | 0.979   | 0.979    | 0.405       |
| V_rollmin_G_API_kobs7 | V_ema_G_API_kobs7        | 0.978   | 0.978    | 0.368       |
| G_API                 | C_smm_G_API_alpha0.85_n5 | 0.958   | 0.968    | 0.335       |
| G_API                 | V_rollmin_G_API_kobs7    | 0.926   | 0.940    | 0.454       |
| G_rain_sum_30d        | V_ema_G_API_kobs7        | 0.922   | 0.931    | 0.230       |
| G_rain_sum_30d        | C_smm_G_API_alpha0.85_n5 | 0.907   | 0.918    | 0.202       |
| G_rain_sum_30d        | V_rollmin_G_API_kobs7    | 0.896   | 0.918    | 0.243       |
| G_API                 | G_rain_sum_30d           | 0.882   | 0.902    | 0.161       |
| s1_vh                 | V_ema_G_API_kobs7        | 0.623   | 0.690    | 0.172       |
| s1_vh                 | C_smm_G_API_alpha0.85_n5 | 0.610   | 0.680    | 0.144       |
| s1_vh                 | G_rain_sum_30d           | 0.623   | 0.680    | 0.320       |
| s1_vh                 | G_API                    | 0.606   | 0.676    | 0.123       |
| s1_vv                 | F_NDMI                   | 0.649   | 0.709    | 0.774       |

**Negative**:
| Feature A | Feature B | Pearson | Spearman | Mutual Info |
|-------------|--------------|---------|----------|-------------|
| F_NDMI | F_MSI | -0.969 | -1.000 | 1.000 |
| s1_vh | E_SAR_ratio | -0.788 | -0.835 | 0.983 |
| s1_vh | F_MSI | -0.658 | -0.603 | 0.781 |
| s1_vv | F_MSI | -0.720 | -0.709 | 0.775 |

### Strong Monotonic Relationships

**Positive**:
| Feature A | Feature B | Pearson | Spearman | Mutual Info |
|-----------|-----------|---------|----------|-------------|
| s1_vh | F_NDMI | 0.551 | 0.603 | 0.780 |
| s2_b8 | F_NDMI | 0.571 | 0.629 | 0.780 |

**Negative**:
| Feature A | Feature B | Pearson | Spearman | Mutual Info |
|-------------|-------------------|---------|----------|-------------|
| E_SAR_ratio | V_ema_G_API_kobs7 | -0.497 | -0.664 | 0.203 |
| s2_b8 | F_MSI | -0.598 | -0.629 | 0.778 |s

### Nonlinear Dependencies

| Feature A | Feature B   | Pearson | Spearman | Mutual Info |
| --------- | ----------- | ------- | -------- | ----------- |
| s1_vv     | E_SAR_ratio | -0.586  | -0.588   | 0.984       |
| F_MSI     | E_SAR_ratio | 0.424   | 0.440    | 0.783       |
| F_NDMI    | E_SAR_ratio | -0.310  | -0.440   | 0.782       |
| s2_b8     | E_SAR_ratio | -0.364  | -0.282   | 0.766       |
| s1_vv     | s2_b8       | 0.386   | 0.409    | 0.758       |
| s1_vh     | s2_b8       | 0.373   | 0.350    | 0.755       |

---

_Jakob Balkovec_
