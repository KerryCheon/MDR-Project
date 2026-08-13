# Audit: clustering setup in `derived_8.4-gating-analysis-1.0` — data leakage check

## Direct answer

**No, the clustering is not fit on the entire dataset before splitting.** The KMeans, `StandardScaler`, and imputation means are fitted **only on trainval (train + val)** and applied to the test split via `predict` only. The test set never enters any fit in the notebook. The splits themselves are clean: temporal and disjoint (train 2017–2020 / val 2021–2022 / test 2023–2025; 9,803 / 4,805 / 6,620 rows; zero (date, station) overlap between any pair of splits). All cluster-quality metrics (silhouette, Calinski–Harabasz, Davies–Bouldin, WSS), t-SNE/PCA, and scaling diagnostics are computed on trainval only. Exported parameters transparently record `fitted_on: "derived_8.4 trainval (train + val)"`.

So the classic leakage pattern the user asked about (fit clustering on all rows, then split) is **not present**.

## Findings (ordered by severity)

### 1. Clustering input features are test-selected (main caveat, inherited)
`Clustering_Backbone54` — the primary strategy — clusters on `shared_backbone_54`, loaded from `derived_8.4-eval-1.1/selected_features.json`. That feature list was produced by `derived_8.4-feature-selection-2.0`, whose artifact records `"selection_goal": "unweighted_pooled_test_r2_2023_2025"` and whose plan explicitly states: *"The 2023–2025 labels intentionally influence candidate discovery, ranking, and final selection … test rows are the direct feature-selection benchmark."* The 54 features were therefore chosen by directly optimizing test-set R². The clustering *algorithm* is fit cleanly, but the feature set it operates on was selected with the test set in the loop. Any test metrics reported through this chain inherit optimism.

`Clustering_V0_Full` (the legacy `OVERALL_SELECTED_FEATURES_V0`, "exact copy of derived_8.2's global c1 feature set") predates the 8.4 split and was selected on 2017–2022-style folds in the older pipelines — not test-selected in the current chain. Its provenance is cleaner.

### 2. The "winning router" the analysis reproduces is a test-selected artifact (inherited)
The README/notebook claims `Clustering_Backbone54` K=2 reproduces "the exact eval-1.1 winning router." That winner (`Clustering_V0_Full_k2`, c0=0, c1=10, test R² 0.815) was chosen in eval-1.1 by:
- ranking specialist feature additions via Spearman correlation of features with **test residuals** (`compute_delta_rankings` in `eval11/evaluator.py`), and
- picking the (c0, c1) grid point with the **highest pooled test R²** (`run_eval.py`: `if grid_res.pooled_r2 > best_grid_r2`).

Additionally, eval-1.1's experts are trained with `eval_set` = the test split (no `early_stopping_rounds`, so the fit is unaffected — the test set is only watched during training). This matters for the overall chain, not for the gating notebook's own fitting.

### 3. K=2 justification cites test numbers (minor, in this notebook)
The README argues "K=2 is the largest K at which every regime corresponds to a whole station group (purity = 1.000 on trainval **and test**)", and §11 computes test station purity for every (strategy, K). The notebook's §12 synthesis table itself uses trainval-only purity, but the K-selection argument as documented uses test purity as evidence — i.e., test data informed the choice of K for the router. Strictly, K should be decided on trainval only, with test purity reported as out-of-sample confirmation.

### 4. `Univariate_G_API` recomputes its threshold on the test split (latent bug, zero current impact)
§4 computes test labels for the two non-clustering strategies via `get_labels_for_strategy(test_df, ...)`, and `quantile_bin` takes `np.nanquantile(val, 0.5)` of the frame passed — i.e., the **test split's own median** becomes the threshold applied to the test split. This is textbook leakage for that strategy. It is currently harmless: those test labels are never consumed by any reported metric (gating metrics are computed on trainval only; per-regime interpretation runs only for clustering strategies). Note eval-1.1's `UnivariateGAPIRouter` does this correctly (threshold fitted on trainval, applied to test), so the eval-1.1 leaderboard entry is clean on this point.

### 5. Router fit protocol inconsistency across experiments (minor)
fs2.0's search fitted the router on **train only** (`V0Router(...).fit(data.train)`), while eval-1.1 and gating-analysis-1.0 fitted on **trainval**. The partitions coincide (both station-pure, counts match), so there is no practical effect, but the protocol differs between the feature-selection stage and the evaluation stage. Fitting an unsupervised transform on trainval is acceptable practice, but it should be consistent and documented.

### 6. SMAP features in clustering (not leakage, interpretational note)
SMAP soil-moisture products are strong proxies of the target (in-situ 5 cm soil moisture) and are legitimate inference-time features. The regimes are partly defined by land-surface wetness state; the analysis acknowledges this explicitly ("dynamic land-surface state (SMAP) matters"). Not a leak, but it explains why regime target distributions differ so sharply.

## Verdict

The clustering fit/apply protocol in `derived_8.4-gating-analysis-1.0` is scientifically sound as far as the notebook itself goes: fit on trainval → apply to test, no test data in any fit, transparent exports. The soundness caveats are **inherited from upstream**: the 54 backbone features were test-selected, and the "winning router" identity was chosen by test performance. Consequently, test-set numbers reported through this chain (e.g., eval-1.1's R² = 0.815) should not be presented as purely out-of-sample without disclosing the test-informed feature/winner selection.

## Recommended follow-ups (if the team wants to harden this)

1. Re-derive clustering features on trainval-only selection (or keep V0_Full as the defensible router feature set) if clean test numbers are required; otherwise disclose the test-informed selection explicitly wherever test metrics are reported.
2. Restrict the K-selection criterion to trainval (drop test purity from the justification, or label it "out-of-sample confirmation only").
3. Fix `quantile_bin` to use a trainval-fitted threshold for test labels, or remove the unused heuristic test-label computation.
4. Unify the router fit protocol (train-only vs trainval) across fs2.0 / eval-1.1 / gating-analysis and document it once.
