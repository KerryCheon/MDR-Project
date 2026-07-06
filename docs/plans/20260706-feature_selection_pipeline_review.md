# Feature Selection Pipeline Review: Modeling/

A thorough review was conducted on the feature selection pipeline (`Modeling/Src/soilmoist_fl/`). This review evaluates the selection process, checking whether it might miss useful features or select bad ones. 

We identified a **critical bug** that causes silent feature misalignment under common data conditions, along with several design limitations that lead to linear bias and instability in feature selection.

---

## 1. [CRITICAL BUG] Silent Feature Alignment Mismatch (mi.py & elasticnet.py)

### The Issue
Both `mi.py` and `elasticnet.py` use `SimpleImputer(strategy="median")` to handle missing values. By default, `SimpleImputer` drops columns that contain only `NaN` values in the fit sample.
- In `select_mi`, `SimpleImputer` is fit on `X`. If any column is all-NaN, it is skipped.
- In `select_elasticnet`, `SimpleImputer` is wrapped in a `Pipeline` fit on `X`.

When a column is dropped, the output matrix `X_imp` (or the downstream coefficients `enet.coef_`) has a **shorter length** than the original list of columns (`feature_cols`). The code then aligns scores/coefficients using a simple `zip`:
```python
# mi.py:
ranked, score_map = _rank_dict_from_scores(feature_cols, scores)

# base.py:
def _rank_dict_from_scores(feature_cols, scores):
    pairs = list(zip(feature_cols, scores))
    ...
```
```python
# elasticnet.py:
coefs = enet.coef_
abs_coef = np.abs(coefs)
pairs = list(zip(feature_cols, abs_coef))
```

If `feature_cols` has length 3 (`[colA, colB, colC]`) and `colA` is dropped because it is all-NaN:
1. `scores` / `coefs` will have length 2 (representing the scores for `colB` and `colC`).
2. `zip` matches them sequentially: `colA` gets matched with `colB`'s score, and `colB` gets matched with `colC`'s score.
3. `colC` is completely truncated and dropped.

### Impact
This leads to a silent, massive feature corruption:
- A completely useless (all-NaN) column (`colA`) steals the high score of a good feature (`colB`) and is selected.
- Other features are shifted left, assigning them incorrect scores.
- Some features are completely omitted.

In `train.csv`, **426 out of 499 columns are completely NaN for at least one station** because remote sensing data is "pending upstream generation". In any run where a single station's split is used, or where the bootstrap sampler in `stability_bootstrap_elasticnet` samples a subset where a feature is all-NaN, **feature ranking becomes completely scrambled**.

### Recommended Fix
We must use scikit-learn's `get_feature_names_out` to align coefficients only to features that were actually kept by the preprocessors, and assign a default score of `0.0` to any dropped features.

#### For `mi.py`:
```python
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    
    # Get names of kept features
    kept_features = imp.get_feature_names_out(feature_cols)
    
    # Compute scores for kept features
    scores = mutual_info_regression(
        X_imp,
        y_num,
        random_state=int(random_state),
        n_neighbors=int(n_neighbors),
    )
    
    # Map kept features to their scores
    score_map = {f: float(s) for f, s in zip(kept_features, scores)}
    
    # Assign 0.0 to dropped features
    for f in feature_cols:
        if f not in score_map:
            score_map[f] = 0.0
            
    # Rank features based on scores
    ranked = sorted(feature_cols, key=lambda f: -score_map[f])
```

#### For `elasticnet.py`:
```python
    model.fit(X, y_num)
    enet = model.named_steps["enet"]
    coefs = enet.coef_
    abs_coef = np.abs(coefs)
    
    # Get names of features that reached the ElasticNet step
    kept_features = model[:-1].get_feature_names_out(feature_cols)
    
    # Map kept features to their absolute coefficients
    score_map = {f: float(c) for f, c in zip(kept_features, abs_coef)}
    
    # Assign 0.0 to dropped features
    for f in feature_cols:
        if f not in score_map:
            score_map[f] = 0.0
            
    # Rank features based on scores
    ranked = sorted(feature_cols, key=lambda f: -score_map[f])
```

---

## 2. Linear Bias in a Non-Linear Modeling Stack

### The Issue
The final modeling stack includes **Random Forest** (`rf`) and **XGBoost** (`xgb`), which excel at capturing non-linear relationships and complex feature interactions.
However, the feature selection pipeline uses **ElasticNet** (a linear model) to prune the candidate features from 382 down to 60, and then again in the bootstrap stability loop.

### Impact
If a feature has a strong non-linear relationship (e.g., quadratic, or active only under specific soil wetness thresholds) or is only useful when interacting with another feature (e.g., Temperature × Wind Speed), it will be assigned a coefficient of `0.0` by ElasticNet. 
As a result:
- The pipeline will **miss useful non-linear features**, filtering them out before the XGBoost/RF models ever get to see them.
- The pipeline will select **suboptimal linear features** that perform well in a linear model but have less predictive power than complex features under tree-based architectures.

### Recommendation
Complement the linear selector with a tree-based feature importance step (e.g., Random Forest or XGBoost Feature Importances) to ensure non-linear features are not discarded. We could define a `select_tree_importance` stage or incorporate tree importances into the stability selection loop for non-linear modeling runs.

---

## 3. Collinearity and Coefficient Dilution in Lags

### The Issue
Soil moisture has high temporal autocorrelation. As a result, lag features (e.g., `SMAP_sm_am_interp_lag1` through `SMAP_sm_am_interp_lag30`) are highly collinear.
When multiple collinear features are passed to `ElasticNetCV`, the model splits the coefficient weight among them (due to the Ridge component of the penalty).
For example, if the total lag effect size is `1.0`, three collinear lag features might each receive a coefficient of `0.33`. An independent, less predictive feature might receive a coefficient of `0.5`.

### Impact
When ranking by absolute coefficient value (`abs_coef`), the highly predictive but collinear features are ranked lower (`0.33 < 0.5`) and may be pruned by the `k = 60` limit.
Furthermore, in stability selection (`stability_bootstrap_elasticnet`), minor variations in bootstrap samples will cause the coefficient to shift randomly between the collinear lags. This dilutes their individual selection frequencies across bootstrap runs (e.g., `lag1` selected 40% of the time, `lag2` 30% of the time), causing both to fail the `min_freq = 0.6` threshold and be completely excluded.

### Recommendation
1. **Correlation Filtering**: Perform hierarchical clustering or correlation thresholding on the features before selection to drop redundant collinear columns or group them.
2. **Group-Aware Regularization**: Use Group Lasso or group-based selection if features naturally belong to families (which they do, e.g., the SMAP family, precipitation family, etc.).

---

## 4. Computational and Statistical Flaws in Stability Selection

### The Issue
In `stability_bootstrap_elasticnet`, the bootstrap function runs `select_elasticnet` in a loop:
```python
    def _run_bootstrap(b, idx):
        ...
        out = select_elasticnet(Xb, yb, k=enet_k, random_state=int(random_state) + b, **kwargs)
        return out["selected"]
```
Because `select_elasticnet` runs `ElasticNetCV` with a 5-fold cross-validation internally, the pipeline runs **5-fold CV for every single bootstrap iteration**. 

This leads to:
1. **Massive Computational Waste**: Running `n_boot = 100` means fitting ElasticNet 500 times.
2. **Hyperparameter Fluctuation**: The regularization strength `alpha` and `l1_ratio` fluctuate across bootstrap samples, adding extra noise to the selection frequency.
3. **Low Bootstraps**: To compensate for the slow runtime, `n_boot` is set to `10` in configs. This is statistically too small to compute reliable frequencies, as a feature's selection rate is highly sensitive to random sample fluctuations.

### Recommendation
According to standard stability selection methodology:
1. Fit `ElasticNetCV` **once** on the entire training set to find the optimal `alpha` and `l1_ratio`.
2. Run the bootstrap loop using a simple `ElasticNet` model (no CV) with these fixed parameters.
3. Increase `n_boot` to `100` (which will now run 10x faster than 10 bootstrap iterations with CV).

---

## 5. Entropy Estimation Issues under High Missingness (mi.py)

### The Issue
`select_mi` uses `mutual_info_regression(X_imp, y_num, n_neighbors=3)` to filter features.
Because 426 out of 499 columns in the dataset have missingness ratios greater than 50% (reaching up to 81.37%), replacing missing values with a single constant (the median) creates a massive spike in the probability density function of each feature.
The Kraskov-Stögbauer-Grassberger (KSG) entropy estimator used in scikit-learn's `mutual_info_regression` assumes continuous variables. Identical values (the imputed medians) violate this assumption and bias the nearest-neighbor distances, leading to highly inaccurate and noisy mutual information scores.

### Recommendation
1. Use `discrete_features` parameter in `mutual_info_regression` if columns are highly discrete or constant.
2. Apply a small amount of random noise (jitter) to imputed values before MI calculation, or filter out columns with missingness exceeding a high threshold (e.g., >80%) before running MI.
