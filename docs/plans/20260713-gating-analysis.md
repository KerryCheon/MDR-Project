# Unsupervised Grouping and Gating Quality Evaluation for derived_8.2

This plan outlines the approach to explore alternative ways of grouping the rows in `derived_8.2` using only features available at inference time (without relying on the target `soil_moisture_5cm` or training a router). We will evaluate both binary ($K=2$) and 3-class ($K=3$) groupings to find regimes where different feature sets are most useful.

---

## Proposed Grouping Strategies

We will compare several unsupervised and proxy-based grouping methods:
1. **Univariate Proxy Binning**:
   - Split the dataset into 2 and 3 groups using quantile thresholds on key physical indicators:
     - **SMAP**: `SMAP_sm_pm_interp_lag1` (direct satellite estimate of moisture)
     - **API**: `G_API` (antecedent precipitation)
     - **LST**: `LST_modis` (surface temperature/seasonality proxy)
     - **NDMI**: `F_NDMI` (spectral index of moisture)
2. **Seasonal / Month-based Grouping**:
   - Grouping by **Month** or **Season** (Pacific Northwest meteorological seasons):
     - *Dry Season*: July, August, September (typically dry, low rain)
     - *Wet Season*: November, December, January, February, March (wet, high precipitation)
     - *Transition Season*: April, May, June, October
3. **Static Features & Location Grouping**:
   - Grouping based on physical station static features (elevation, slope, HWSD clay/sand fractions):
     - Binning based on soil texture/clay fraction (e.g., High Clay vs. Low Clay) or elevation.
     - Hierarchical clustering of stations using static parameters.
4. **Multivariate Clustering (K-Means / Gaussian Mixture Models)**:
   - Run clustering algorithms with $K=2$ and $K=3$ on combinations of non-target features:
     - *Clustering Set A (Dynamic Physical)*: `[SMAP_sm_pm_interp_lag1, G_API, LST_modis]`
     - *Clustering Set B (Hybrid Dynamic + Static)*: `[SMAP_sm_pm_interp_lag1, G_API, LST_modis, elevation, clay_fraction]`
     - *Clustering Set C (Full V3)*: The full V3 feature set (excluding any target-derived or leakage features).

---

## Gating Quality Metrics Framework

To measure whether a grouping separates the data in a "meaningful way," we propose the following quantitative metrics:

1. **Feature Relationship Divergence (Core Metric)**:
   - *Rationale*: A gating strategy is successful if the relationship between features and the target variable differs across groups, justifying separate specialist feature sets.
   - *Metric*:
     - Compute the vector of correlations (Pearson and Spearman) between all features and the target $y$ within each group $k$: $\mathbf{C}_k$.
     - For each feature $f$, compute its correlation standard deviation across groups: $\sigma_f = \text{std}(\{C_{1,f}, C_{2,f}, \dots, C_{K,f}\})$.
     - **Divergence Index ($D$)**: The mean of $\sigma_f$ across the top 20 globally correlated features (or all features). A higher $D$ indicates that feature-to-target relationships change significantly across the groups.
     - **Pairwise Distance**: The Euclidean/Manhattan distance between the correlation vectors of different groups (e.g. $dist(\mathbf{C}_1, \mathbf{C}_2)$).

2. **Target Separability**:
   - *Rationale*: Do the groups correspond to physically distinct moisture levels?
   - *Metric*:
     - **KS Statistic**: Kolmogorov-Smirnov test statistic on $y$ between pairs of groups.
     - **Variance Reduction (VR)**: The percentage of target variance explained by the grouping:
       $$VR = 1 - \frac{\sum_k N_k \text{var}(y_k)}{N \text{var}(y_{global})}$$

3. **Group Balance (Entropy)**:
   - *Rationale*: Prevent degenerate groupings (e.g., a cluster containing 99% of samples).
   - *Metric*: Shannon entropy of group sizes normalized to $[0, 1]$.

---

## Proposed Changes

### Experiment Component

We will create a new experiment folder `notebooks/experiment/derived_8.2-gating-analysis/` to contain the analysis.

#### [NEW] [derived_8.2_gating_analysis.ipynb](./notebooks/experiment/derived_8.2-gating-analysis/derived_8.2_gating_analysis.ipynb)
A Jupyter Notebook that:
- Loads `derived_8.2/train.csv`, `val.csv`, and `station_static_features.csv`.
- Merges static features and extracts month/season features.
- Implements the univariate, seasonal, static, and hybrid K-Means clustering configurations.
- Calculates the Target Separability, Feature Relationship Divergence, and Group Balance metrics for each configuration (both $K=2$ and $K=3$).
- Identifies features that are highly correlated with the target in one group but uncorrelated/differently correlated in others.
- Generates plots:
  - Density/violin plots of target soil moisture across the groups.
  - Heatmap of feature correlations with the target for each group to visually compare relationship divergence.
  - Correlation profile comparison curves for top features.

#### [NEW] [README.md](./notebooks/experiment/derived_8.2-gating-analysis/README.md)
A detailed report presenting tables of the quality metrics for each grouping method, highlighting the best binary and 3-class configurations, and listing the recommended specialist features for each group.

---

## Verification Plan

### Automated Tests
- Create and execute the notebook `derived_8.2_gating_analysis.ipynb` inside the `notebooks` environment using the `nb` tool:
  ```bash
  cd notebooks
  nb execute experiment/derived_8.2-gating-analysis/derived_8.2_gating_analysis.ipynb --uv
  ```
- Confirm the notebook executes without errors and that all cells successfully compute and display metrics tables and plots.
