# Paper 2 claims ledger — assembly 1.0

This ledger accompanies `report.md`. “Confirmed” means faithful to a saved artifact under its stated development protocol; it does **not** mean untouched independent test confirmation. Main temporal/LOSO values come from final-evidence 1.0, all ECE results from 1.1, and older runs remain historical or sensitivity evidence. The source manifest records file hashes.

| ID | Claim or result to carry into the paper | Evidence status and boundary | Source |
| --- | --- | --- | --- |
| C0 | The submitted first paper describes the multi-source pipeline and a global XGBoost estimate; its R² 0.822 used a different five-station protocol. | Historical context; not a ranked baseline for `derived_8.4`. | {{SOURCE_PAPER1}} |
| C1 | Internal three-regime/oracle analysis motivated a routing study. | Internal prior work; the submitted first-paper PDF does not report those oracle/gate comparisons. | `writeup/sections/three_regime/` and {{SOURCE_OUTLINE}} |
| C2 | Guarded no-delta temporal R² {{GUARD_R2}}, RMSE {{GUARD_RMSE}} over 30 expert seeds. | Confirmed development-split result, final-evidence 1.0; seed uncertainty only. | {{SOURCE_MAIN_SUMMARY}} |
| C3 | Guarded and unguarded Backbone differ by {{GUARD_BACKBONE_DIFF}} temporal R². | Direct paired in-harness guard ablation; near-tie in size. | {{SOURCE_MAIN_SEED}} |
| C4 | Guarded LOSO R² {{GUARD_LOSO}} versus paired Backbone {{BACKBONE_LOSO}}, difference {{GUARD_LOSO_DIFF}}. | Direct paired five-seed/seven-fold comparison; {{FOLD_GAINS}} fold gains and {{FOLD_TIES}} ties are descriptive. | {{SOURCE_LOSO_SEED}} and {{SOURCE_T8}} |
| C5 | Guarded temporal/global gap {{GUARD_GLOBAL_GAP}} and LOSO/global gap {{GUARD_GLOBAL_LOSO_GAP}}. | Sensitivity-anchored cross-harness gaps; **not** paired Guarded-versus-global statistics. | {{SOURCE_MAIN_SUMMARY}} and {{SOURCE_FORMAL_LOSO}} |
| C6 | The tested target-derived `Trained_Gating_k2` has temporal R² {{GATE_R2}}. | Sensitivity reference for this specific foil; no claim about all supervised routers. | {{SOURCE_MAIN_SUMMARY}} |
| C7 | Earlier V0 no-delta versus global seed-paired temporal difference is {{V0_GLOBAL_PAIRED}}. | Formal-eval 1.0 **V0** statistic only; never relabel as Guarded. | {{SOURCE_FORMAL_PAIR}} |
| C8 | Earlier test-selected V0 delta-feature bootstrap difference is {{DELTA_BOOTSTRAP}}. | Test-selected sensitivity only; omit from Guarded headline and abstract. | {{SOURCE_FORMAL_BOOTSTRAP}} |
| C9 | K=2 trainval partition separates Spokane/SourdoughGulch from the other five known WA stations. | Descriptive Washington partition; Guarded purity on known sites is by construction. | {{SOURCE_COMPOSITION}} and {{SOURCE_K_SWEEP}} |
| C10 | Eastern-station holdouts change specialist-training composition, coinciding with Guarded LOSO gains. | Mechanism supported by fold ARI and paired station table; not a causal climate claim. | {{SOURCE_FOLD_AGREEMENT}} and {{SOURCE_T8}} |
| C11 | The primary router has no in-situ target labels but uses SMAP-derived soil-moisture proxies and `station_id` for known-station consistency. | Confirmed method/provenance disclosure. | {{SOURCE_FEATURES}} and {{SOURCE_MAIN_CONFIG}} |
| C12 | The 54-feature backbone was selected with a test-era objective. | Confirmed selection-bias limitation; all test scores are development evidence. | {{SOURCE_FEATURES}} |
| C13 | Pre-guard out-of-state no-delta models have negative station-mean R² and no reliable two-expert transfer advantage. | Limitations only; **not** a Guarded out-of-state result. | {{SOURCE_OOS}} |
| C14 | The older ECE published-router V0/global station-mean ΔRMSE was {{ECE_LEGACY_DIFF}} with {{ECE_LEGACY_WINS}} wins. | Earlier policy question; do not combine with 1.1 missingness-aware scores. | {{SOURCE_ECE_LEGACY}} |
| C15 | ECE v3 has {{ECE_ROWS}} evaluation rows at {{ECE_STATIONS}} disjoint in-situ stations; every SMAP router-input block is missing and gated. | Confirmed 1.1 input-availability diagnostic; short window. | {{SOURCE_ECE_AUDIT}} and {{SOURCE_ECE_SPLIT}} |
| C16 | Guarded's selected G_API class-0→local-c0 alignment scored lower masked-WA-validation RMSE than class-0→local-c1. | Confirmed Washington-only calibration; ECE targets were not used to select the mapping. | {{SOURCE_ECE_CALIBRATION}} and {{SOURCE_ECE_AUDIT}} |
| C17 | Guarded ECE `as_routed` RMSE {{ECE_ROUTED_RMSE}}, `auto_hard` RMSE {{ECE_AUTO_RMSE}}, direct global RMSE {{ECE_GLOBAL_RMSE}}. | Confirmed five-seed 1.1 diagnostic, not deployment-wide or SOTA evidence. | {{SOURCE_ECE_POLICY}} |
| C18 | On this ECE set, Guarded `auto_hard` matches local-c0 forcing, while its approved dry-assigned oracle is local c1. | Confirmed family-specific index convention; both `c0_only`/`c1_only` are non-deployable. | {{SOURCE_ECE_CROSSWALK}} and {{SOURCE_ECE_AUDIT}} |
| C19 | ECE station-level errors and site descriptors vary. | Descriptive only; geography/precipitation does not causally explain error. | {{SOURCE_ECE_STATION}} and {{SOURCE_ECE_SITES}} |
| C20 | Median within-day ECE sensor range {{HOURLY_RANGE}} exceeds median absolute daily-target step {{DAILY_STEP}} in the hourly focus windows. | Benchmark-validity diagnostic; does not imply sensor fault. | {{SOURCE_HOURLY}} |
| C21 | Direct same-harness Guarded-versus-global LOSO inference and the global fallback policy were not evaluated. | Unresolved future work; no number should be inferred. | {{SOURCE_OUTLINE}} |

## Reporting constraints

The 2023–2025 test period was reused during backbone, hyperparameter, and later design selection. The seven-station LOSO analysis is low power. Expert-seed intervals exclude feature/model/router-selection uncertainty. ECE calibration is Washington-only and the five-station late-summer evaluation is evaluation-only. Local expert numbers, G_API class numbers, and dry-assigned aliases refer to distinct conventions; no local index is a universal physical class.
