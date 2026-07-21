# Locked protocol for feature-selection-2.2

## Selection

1. Concatenate train and validation (2017–2022) for development only.
2. Construct four deterministic row-balanced station groups.
3. For each of the last four eligible validation years, train on earlier years and stations outside the held-out group.
4. Score both unweighted and mean-normalized β=0.2 training protocols.
5. Rank by the one-standard-error lower confidence bound of paired permutation Δ normalized RMSE, iteratively refitting after every reduction.
6. Choose the feature-count candidate with minimum one-standard-error upper confidence bound of grouped OOF normalized RMSE.
7. For regimes, require a positive paired lower confidence bound versus the shared global backbone; otherwise select no delta.

Station ID and date are fold metadata only. No feature names, families, or hand lists participate in scoring. 
Original column position is the deterministic tie-break when confidence bounds and mean importance are exactly equal.

## Evaluation

Final evaluation follows feature-selection-2.1: XGBoost 1.3-lite, seed 42,
native missing-value handling, `err = true - prediction`, and mean-normalized temporal weights for β=0.2. The unweighted protocol uses no sample weights.

The test split is available only through the explicit `--confirm-final` flag.

## Post-final nested diagnostics

The original final result remains immutable. Subsequent artifacts follow these rules:

1. Inner importance and candidate generation use only 2017–2020.
2. Outer feature-count/path selection uses only 2021–2022.
3. Outer candidate fits use the exact locked 1,500-tree evaluation learner.
4. Forward-time and station/time ranking paths are generated independently and unioned before any outer labels are read.
5. Progressive elimination, when enabled, creates bridge sizes from the current and requested counts; it never drops more features in one refit than the requested retained checkpoint.
6. The already-consumed 2023–2025 split is diagnostic only and every manifest sets `unbiased_sota_eligible` to false.
7. Dirty runs record the parent commit, dirty status, tracked-diff hash, and a per-file content hash manifest covering runtime source and environment locks.
8. Artifact files are atomically replaced and a hash-bearing completion marker is written last; restart reuse requires every recorded file to verify.

No path uses feature names, hand lists, family quotas, or bypasses for selection.
