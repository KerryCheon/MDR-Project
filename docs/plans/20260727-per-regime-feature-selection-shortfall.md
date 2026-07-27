## History: Independent Per-Regime Selection Already Failed

In `derived_8.3-eval-1.0`, each V0 cluster ran its own `MI→ElasticNet→Stability` pipeline independently. The results:

| Cluster | Train rows | MI candidates | ElasticNet nonzeros | Stability survivors |
|---------|-----------|---------------|-------------------|-------------------|
| Cluster 0 (V0 Full) | 7,156 | 300 | **26** | **26** |
| Cluster 1 (V0 Full) | 2,647 | 300 | **66 → 42** | **42** |
| Cluster 0 (Dynamic) | 4,339 | 300 | **2** | **2** |
| Cluster 1 (Dynamic) | 5,464 | 300 | 25 | 25 |

The independently selected per-cluster sets **scored R² 0.7197**, while routing the same K=2 clusters with the shared 50-feature V0 backbone scored **0.7703**. The difference is entirely feature truncation.

This was the **root cause** that motivated the shift to "shared backbone + add-only deltas" — and it's documented in `docs/plans/20260726-new-feature-selection-2.md`.

---

## Why Fully Independent Per-Regime Selection Collapses

The collapse mechanism is a **multi-stage failure chain**:

### Stage 1: ElasticNetCV with L1-heavy penalty

Per-cluster data has fewer rows (2,647–7,156), so ElasticNetCV selects aggressive regularization (l1_ratio=1.0 or 0.9, high alpha). This picks **one proxy** from each correlated feature group and zeros out the rest. From 300 MI candidates → 2–66 nonzeros.

### Stage 2: Bootstrap stability amplifies sparsity

Stability selection (20 resamples at 80% sample fraction) requires a feature to appear in ≥60% of bootstraps. With only 2–66 ElasticNet nonzeros surviving from Stage 1, stability further prunes to 2–42 features. Features that are marginally useful but not consistently top-ranked in every bootstrap get dropped.

### Stage 3: No fallback pool to repair

The repair mechanism can only fall back to the ranked ElasticNet or MI lists — but those lists are already empty of useful candidates because Stages 1–2 have already aggressively filtered everything. The repair can't find new features to add.

### Stage 4: The feedback loop

Smaller cluster → fewer training rows → more aggressive regularization → fewer survivors → less diverse expert → worse performance → model relies more on the other cluster → the collapsed cluster's expert becomes useless.

---

## The 8.4 Delta Grid Confirms It

The delta grid from the 2.0 experiment validates this indirectly:
- **Cluster 1** (2,647 rows, the smaller cluster that retained 42 features) — adding 10 specialist features gave a marginal +0.0007 R²
- **Cluster 0** (7,156 rows, retained 26 features) — adding 10 specialist features **hurt by −0.025 R²**, because the 26 features it independently selected are a different set than the backbone, and the backbone was globally optimized

The architecture of "independently selected per-cluster features" already lost at the selection stage, before the model was even fitted.

---

## Proposed Diagnostic Plan

To definitively explain **why** the per-cluster pipeline collapses so hard, I'd propose:

| # | Analysis | Method | Rationale |
|---|----------|--------|-----------|
| 1 | **Collapse replicate on 8.4** | Run the independent per-cluster selection pipeline (MI→ElasticNet→Stability) on V0 Full K2 clusters using 8.4 data, capture every intermediate stage count and hyperparameter | Confirm the collapse mechanism reproduces on the current dataset |
| 2 | **Regularization hyperparameter biopsy** | Extract `alpha` and `l1_ratio` selected by ElasticNetCV at each cluster; compare to sample size | Prove smaller clusters → aggressive L1 → collapse |
| 3 | **Feature overlap analysis** | Compare the 26 cluster-0 survivors vs 42 cluster-1 survivors vs the 50-feature V0 backbone — measure overlap and residual correlation | Show that independent selection produces disjoint feature sets that miss the shared backbone signal |
| 4 | **Sample-size ablation** | Artificially downsample cluster 0 to match cluster 1's size (2,647 rows), rerun independent selection, check if it also collapses to similar sparsity | Isolate the cause: small sample or heterogeneity? |
| 5 | **Candidate-pool overlap** | Check how many of the independent per-cluster survivors are even IN the experiment's 96-feature candidate pool | Reveal whether the candidate pool was missing cluster-specific signals from the start |