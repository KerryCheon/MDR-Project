# Locked protocol for `derived_8.3-feature-selection-2.0`

## Methodology boundary

This is a dataset/router rerun of the feature-selection 2.2 methodology, not a
new selector revision. The derived 8.0 control, candidate paths, fold rules,
confidence bounds, temporal weighting, XGBoost parameters, seeds, and
diagnostics are frozen to the 2.2 values.

The only current-data baseline is the exact ordered 50-feature
`OVERALL_SELECTED_FEATURES_V0` list from derived 8.3 metadata. V3 and the 2.1 C1
artifact are not comparison arms.

## Original selection arm

1. Concatenate train and validation (2017–2022) for development only.
2. Construct four deterministic row-balanced station groups.
3. For each of the last four eligible validation years, train on earlier years
   and stations outside the held-out group.
4. Score unweighted and mean-normalized beta 0.2 training protocols.
5. Rank by the one-standard-error lower confidence bound of paired permutation
   delta normalized RMSE, iteratively refitting after every reduction.
6. Choose the feature-count candidate with minimum one-standard-error upper
   confidence bound of grouped-OOF normalized RMSE.
7. For each regime, require a positive paired improvement lower confidence
   bound over the shared global backbone; otherwise select no delta.

Station ID and date are fold metadata only. Feature names, families, and hand
lists never participate in selector scoring. Original column position is the
deterministic tie-break when confidence bounds and mean importance are equal.

## Nested and crossed arms

1. Inner importance and candidate generation use only 2017–2020.
2. Outer feature-count and path selection use only 2021–2022.
3. Outer candidate fits use the locked 1,500-tree evaluation learner.
4. Forward-time and station-time ranking paths are generated independently and
   unioned before outer labels are read.
5. Progressive elimination creates deterministic bridge sizes and never drops
   more features in one refit than the requested retained checkpoint.
6. The progressive canonical arm remains limited to derived 8.0.

## V0-full router

The router uses all 50 ordered V0 features with mean imputation,
`StandardScaler`, K-means `K=2`, `n_init=10`, and seed 42.

- The original arm fits on the combined development pool. Its saved provenance
  is diagnostic and is not reused during evaluation.
- The nested arm fits on inner training data only and saves the frozen router
  used to interpret regime-specific feature sets.
- Every router artifact records the resolved source, source SHA-256, ordered
  columns, preprocessing, fit scope, and whether evaluation may reuse it.

Scopes and evaluation model labels use the `2.0_*` identity and
`clustering_v0_full_k2`; the obsolete dynamic-router scope is invalid here.

## Evaluation and interpretation

Evaluation retains the 2.2 locked learner: seed 42, native XGBoost missing-value
handling, `error = truth - prediction`, and mean-normalized temporal weights for
beta 0.2. The unweighted arm uses no sample weights.

`derived_8.3-eval-1.0` already consumed the 2023–2025 test period. Therefore:

- `--confirm-final` is disabled;
- all 2023–2025 evaluation and diagnostic manifests set
  `unbiased_sota_eligible` to false;
- no historical threshold is a new gate verdict;
- any future unbiased claim requires an untouched time period or new-station
  holdout.

## Artifact and report integrity

Artifact files are atomically replaced and completion markers are written last
with hashes of required outputs. Selection, evaluation, and diagnostic artifacts
record split hashes. Router and report provenance also guard the V0 metadata
source hash.

Every reported value is owned by a saved producer and registered builder in
`generate_results.py`. The generator writes normalized evidence before rendering
Markdown. `generate_results.py --check` rebuilds expected report bytes in memory
and verifies evidence, Markdown, manifest, generator, split, V0 source, and
completion hashes.

Inline scripts and notebook-only calculations are not accepted report sources.
