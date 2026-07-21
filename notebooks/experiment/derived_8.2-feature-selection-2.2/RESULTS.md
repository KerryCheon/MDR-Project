# Results: derived_8.2-feature-selection-2.2

**Status:** the original `artifacts/final` run is frozen as a negative result.
The nested architectural revision requested afterward is stored separately and
all of its 2023–2025 numbers are explicitly retrospective diagnostics. They are
not eligible for a new unbiased SOTA claim.

## Selected sets

| Dataset/scope | Features | Stopping reason |
|---|---:|---|
| derived_8.0 global | 50 | minimum grouped-OOF upper confidence bound |
| derived_8.2 global | 100 | minimum grouped-OOF upper confidence bound |
| derived_8.2 regime 0 | 100 (delta 0) | no delta with positive paired LCB |
| derived_8.2 regime 1 | 100 (delta 0) | no delta with positive paired LCB |

The regime mechanism behaved as designed: neither specialist was forced to produce an independently pruned set when additions lacked reproducible paired gain.

## Development validation (2021–2022)

| Dataset/model | β | R² |
|---|---:|---:|
| derived_8.0 hand MDR-v25 | 0.0 | 0.8783 |
| derived_8.0 2.2 global | 0.0 | 0.8306 |
| derived_8.0 hand MDR-v25 | 0.2 | 0.8819 |
| derived_8.0 2.2 global | 0.2 | 0.8319 |
| derived_8.2 V3 | 0.0 | 0.6804 |
| derived_8.2 2.1 c1 | 0.0 | 0.6752 |
| derived_8.2 2.2 global | 0.0 | **0.7375** |
| derived_8.2 2.2 K=2 shared + delta | 0.0 | 0.7097 |

## One-shot final test (2023–2025)

| Gate | 2.2 R² | Required | Result |
|---|---:|---:|---|
| derived_8.0 global, β=0.2 | 0.7735 | > 0.82535 | **FAIL** |
| derived_8.0 global, β=0.0 | 0.7859 | > 0.8222 | **FAIL** |
| derived_8.2 global, β=0.0 | 0.6047 | > 0.6648 | **FAIL** |
| derived_8.2 K=2 shared + delta, β=0.0 | 0.6207 | > 0.6672 | **FAIL** |

The locked controls remained substantially stronger on derived_8.2: V3 reached 0.6537 and 2.1 c1 reached 0.6605 in this environment. The small differences from 2.1's published controls are consistent with the recorded execution device and environment; they do not affect the failed-gate conclusion.

## Fundamental diagnosis

2.2 fixed two real problems in the earlier pipeline:

1. station and time are now part of importance validation rather than discarded before selection; and
2. regime deltas are conditional on a shared backbone and require paired confidence, preventing one-feature specialist collapse.

However, it exposed a deeper selection layer problem. The same grouped 2019–2022 folds were used repeatedly to rank roughly 500 candidates, refit after each reduction, and select feature count. 
That is out-of-fold for each fitted model but **not nested for the feature-search procedure**. 
The search therefore adapted to its validation folds: derived_8.2 appeared excellent on development (R² 0.7375) and then fell to 0.6047 on the untouched test.

The nested revision below therefore separates:

- inner station/time folds used to estimate conditional feature utility;
- an outer forward-time fold used only to choose candidate size/stopping; and
- the already-consumed 2023–2025 test, which is labeled retrospective only.

This is the main architectural conclusion from 2.2. Group-aware scoring is necessary, but without nesting it can still overfit the importance-estimation folds more severely than the original simpler selector.

## Nested architectural revision

The revision separates 2017–2020 inner ranking from 2021–2022 outer selection.
It also evaluates every candidate with the locked 1,500-tree final learner, rather than assuming that the 160-tree importance proxy preserves feature-count ordering.

The first nested artifacts used feature text as the final tie-break whenever permutation confidence and mean utility were exactly equal. 
This contradicted the selector's no-name contract and made early pruning of constant or unused features name-dependent. Those results are superseded. 
The regenerated, position-stable nested search still selects 40 features for derived_8.0 and 65 for derived_8.2, but both derived_8.2 regimes now select an empty delta.

The next diagnostic therefore separates candidate generation into two axes:

1. pure forward-time folds with all stations available; and
2. joint forward-time plus held-out-station folds.

Candidate lists from both mathematical rankings are unioned before the locked joint outer scorer sees any labels. 
With position-stable ties, the crossed outer scorer selects the 40-feature station/time path for derived_8.0 and the 50-feature forward-time path for derived_8.2.

Retrospective testing shows that the crossed outer improvement is not stable through the next temporal shift:

| Dataset/path | Selected | beta | Retrospective R2 |
|---|---:|---:|---:|
| derived_8.0 joint-only | 40 | 0.0 | 0.7644 |
| derived_8.0 crossed | 40 | 0.0 | 0.7644 |
| derived_8.0 hand MDR-v25 | 38 | 0.0 | **0.8246** |
| derived_8.2 joint-only | 65 | 0.0 | 0.6352 |
| derived_8.2 crossed | 50 | 0.0 | 0.6090 |
| derived_8.2 2.1 c1 | - | 0.0 | **0.6605** |

The complete crossed-path diagnostic evaluates all 14 candidates from both sources. 
At beta 0.0, the retrospective ceilings are 0.8002 for derived_8.0 and 0.6351 for derived_8.2. 
The earlier 0.6813 derived_8.2 ceiling was itself name-dependent and is superseded. 
Since the test is already consumed, these observations diagnose the search but cannot authorize choosing a new feature set as an unbiased result.

Progressive elimination tests whether the initial roughly 500-to-150 pruning step suppresses correlated substitutes. 
Its bridge sizes are derived from the current and requested counts, not a hand-tuned drop fraction. 
This raises the derived_8.0 retrospective candidate ceiling from 0.8002 to 0.8209 at 80 features, but remains below hand MDR-v25 at 0.8248. 
Large one-shot pruning is therefore a real source of lost ceiling, but fixing it is insufficient to meet the derived_8.0 gate.

## MoE architecture ablation

Retrospective 2023–2025 results isolate the feature-selection effect from the two-expert architecture:

| derived_8.2 model | beta | R2 |
|---|---:|---:|
| nested global, 65 features | 0.0 | **0.6352** |
| K=2, frozen train router, shared features only | 0.0 | 0.6279 |
| K=2, frozen train router, shared + selected deltas | 0.0 | 0.6279 |
| K=2, refit router, shared + selected deltas | 0.0 | 0.6252 |

Both regime deltas are empty after deterministic tie handling, so shared-only and shared-plus-delta MoE results are identical. Hard routing and fragmented expert training cost about 0.0072 R2 relative to the global model, while router refitting costs another 0.0028. 
The larger architecture losses reported by the superseded artifacts partly reflected name-dependent feature identities. 
The next MoE change should still address gating and expert shrinkage/shared learning, not restore hard-coded features.

## Reproducible artifacts

- `artifacts/final/selection_summary.json`
- `artifacts/final/<dataset>/<scope>/selected_features.json`
- `artifacts/final/<dataset>/<scope>/fold_metrics.csv`
- `artifacts/final/validation_eval/`
- `artifacts/final/final_test_eval/`
- `artifacts/nested/` (nested revision; retrospective test labels)
- `artifacts/nested_locked_outer/` (locked-learner outer comparison)
- `artifacts/crossed_candidates_locked_outer/` (resumable crossed candidates)

Each active revision artifact contains the exact configuration, split hashes, fold definitions, confidence statistics, stopping reason, dirty-source state, runtime source-file hashes, and a last-written completion marker that verifies every required output. 
The frozen original predates this provenance format and remains a historical negative result rather than being retroactively relabeled.
