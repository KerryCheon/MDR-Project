# Why Per-Regime Features Don't Help: Analysis

The delta grid tells the story starkly:
- **Cluster 0 additions HURT** — 10 additions drops R² from 0.8143 → 0.7891 (−0.025)
- **Cluster 1 additions HELP marginally** — 10 additions improves R² from 0.8143 → 0.8150 (+0.0007)

## Root causes (interacting):

**1. Cluster 0 (larger cluster, 73% of data) has a saturated backbone**

The 54-feature wrapper-selected backbone was optimized for pooled R², so cluster 0 already dominates the objective. The top-ranked specialist additions for cluster 0 are all **static features** (`J_bio_bio04/06/07/14`, `longitude`, `latitude`, `aspect`, `lia_mean_asc_deg`) — station-level constants that don't help with temporal dynamics within-cluster and are collinear with existing backbone features (`J_aspect_deg`, `J_bio_bio02/13` already present). More parameters on a saturated model → overfitting → degradation.

**2. Cluster 1 (smaller cluster, 27%) has more room for improvement**

Its specialist additions are more diverse (`SMAP_sm_am_interp`, SMAP lags, `DOY`, `C_lag_F_NDMI`, `V_rollmin_F_NDMI`) — features that capture different temporal dynamics than what's in the backbone. Since cluster 1 has only 2,647 training rows, the backbone was not optimized for its specific dynamics, leaving marginal signal to exploit.

**3. The ranking mechanism biases toward globally-important features**

Cluster specialist ranking combines (a) XGBoost gain from an all-feature pooled model and (b) Spearman residual correlation on the backbone's test errors. Both are dominated by cluster 0's majority, so cluster 0 gets static features that are globally important but within-cluster redundant, while cluster 1 gets more diverse features.

**4. The structural constraint prevents useful exploration**

The experiment architecture is "shared backbone + add-only deltas." When specialist features help cluster 1 but are also useful for cluster 0 (e.g., `SMAP_sm_am_interp`), they can't be added to cluster 0 without also adding the harmful static features. The delta-grid framework can only add the same ranking's top-N features — it can't selectively add "the useful ones" to each cluster independently.

## Proposed Diagnostic Plan

To rigorously validate these hypotheses, I recommend the following analysis:

| # | Analysis | Method | What It Tests |
|---|----------|--------|---------------|
| 1 | **Cluster-0 specialist feature redundancy** | Compute pairwise correlation / VIF between top-10 cluster-0 additions and the 54 backbone features within cluster-0 training data | Hypothesis 2: collinearity with backbone |
| 2 | **Cluster-0 addition SHAP analysis** | Fit the delta_c0_10_c1_0 model and compute SHAP on cluster-0 test samples; check if specialist features have near-zero or negative contribution | Hypothesis 1 & 3: added noise |
| 3 | **Cluster-0 residual structure** | Compare backbone residuals for cluster-0 vs cluster-1 test samples (residual std, autocorrelation, station-level patterns) | Hypothesis 1: backbone already explains cluster 0 well |
| 4 | **Backbone+specialist interaction** | Fit a model with backbone + top-5 cluster-0 specialists on cluster-0 data, compare gain importance distribution vs backbone-only | Hypothesis 2: specialist features get zero gain |
| 5 | **Alternative ranking sanity check** | Generate cluster-specific specialist rankings using only cluster-specific gain (fit per-cluster XGBoost instead of pooled) and re-evaluate top-10 | Hypothesis 3: ranking bias |
| 6 | **Cluster 0/1 feature overlap analysis** | Check how many of cluster-1's top-10 specialists are in the candidate pool for cluster-0 but ranked lower | Hypothesis 4: structural constraint |
| 7 | **Per-station breakdown** | Break down cluster-0 performance by station — does the degradation come evenly or from specific stations? | Hypothesis 1: overfitting to majority stations |

The core question is: **is the failure due to feature redundancy (fixable by better ranking) or fundamental model saturation (not fixable within this architecture)?** Analysis #5 would answer this most directly.