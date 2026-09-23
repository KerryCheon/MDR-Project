# Second Paper Outline (v7) — Covariate-Defined Regional Experts for Soil-Moisture Estimation

Status: active draft outline for PI · Split: `derived_8.4` only · Scope: Washington (regional) · CS/IEEE venue TBD
Historical predecessors: `outline-v2.md`–`outline-v6.md` (all kept unchanged) · Status of this file: active draft · v7 updates the ECE terminology and reports the calibrated missingness-aware routing evidence without turning the five-station diagnostic into a general deployment claim.

This document is the technical-report backbone for writing the full paper without manually digging through experiment notebooks. Every number below is already present in an executed report notebook README (stdout tables), in paper 1, or in repo docs explicitly marked as historical/prior-work sources; the claims ledger (W1) freezes the mapping before drafting.

v7 diff vs v6 (ECE terminology and evidence pass):
- §7 and related references: distinguish the five ECE collaboration in-situ stations from the canonical `derived_8.4_ece_v3` evaluation split; distinguish G_API class-to-local-expert mapping from the dry-assigned oracle label convention.
- ECE calibration: preserve salvage 2.0 routing for V0/Backbone; select Guarded's G_API class alignment only on synthetic-SMAP-masked Washington validation; ECE targets remain evaluation-only.
- §§0/5/7/8/9/10: remove broad geography/covariate and physical dry/wet-transfer claims; state family-specific results, mapping provenance, small-sample and short-window limits.
- T7/T9/F5, C9, W4′/W8, risks, and source map: point to the executed `paper2-final-evidence-1.1` CSVs and notebook outputs.
- Terminology correction: the v5 dry/lowland and wet-mountain names remain historical shorthand, not physical labels for family-local expert indices; `auto_hard`/`auto_soft` are availability-gated, Washington-calibrated G_API policies, not a universal router fix.

v6 diff vs v5 (historical coherence pass):
- §3: add one concise data-to-evaluation flow and treat `derived_8.4` as the canonical study split, without describing dataset evolution.
- §§0/3/4/6.4/9: make reuse of the 2023–2025 test period for backbone and model selection an explicit limitation; distinguish pre-specified guard criteria from independent confirmation.
- Abstract/§6.1/C3: remove the test-delta V0 bootstrap from the Guarded primary claim; retain it only as a clearly labeled sensitivity result or omit it.
- ECE ledger: normalize the three result groups to C9a/C9b/C9c and mark Guarded-on-ECE evidence as pending until its run is complete. *(v7: superseded by the completed 1.1 evidence and family-specific G_API calibration.)*
- §3/§4: define `trainval` and antecedent precipitation index (API) on first use.
- §4: record the shared XGBoost expert settings and disclose their test-era selection provenance.
- §6/C0/R5: label paper 1's R² 0.822 as historical context from a different dataset and protocol, not a directly comparable result; remove the dataset-evolution ledger claim.

v4 diff vs v3 (complete list; all other lines unchanged in substance):
- D9 (§0): primary = `Guarded_Backbone54_k2 (0,0)`; V0 = sensitivity; routing features shared-54-only; W2 pre-registration disclosed.
- §2 abstract: primary router named; headline numbers re-pointed (temporal 0.8118, LOSO 0.638 — both Guarded, both tie V0).
- §3: one line that the router uses the identical 54-feature backbone (no separate router feature source to explain).
- §4: provenance table reworked — Guarded primary row with 3-line guard semantics + station_id/unseen-path disclosure; V0 row demoted to sensitivity appendix.
- §5: purity reframed (by-construction on known stations; diagnostic on unseen) + 3-sentence fold-instability mechanism.
- §6.1/6.2: Guarded headline numbers + Guarded-vs-Backbone ablation; new T8 per-station LOSO-diff table; V0-vs-global 6/7 retained as sensitivity-anchored row.
- §8: design lesson + pre-registration note. §9: station_id disclosure, training-composition caveat, anchor-correction footnote.
- Ledger: C2/C4 re-pointed to Guarded; new C14 (ablation), C15 (mechanism). Figure plan: +T8. Risks: +R12/R13/R14. Source map: +`routing-optimize-1.0`.

v5 diff vs v4 (complete list; all other lines unchanged in substance):
- D10 (§0.4 + §7): ECE reframed from conditional tie/weak-benchmark to diagnostic routing-failure exhibit (discussion + supplementary panel, explicitly NOT a SOTA-beating claim); `c0_only` labeled manual oracle ceiling, deployable claim restricted to `auto_hard/auto_soft`.
- Terminology (§4/§5/§7): normalized to "dry/lowland expert vs wet-mountain expert" with eastern/western station parenthetical on first use; label-flip box (salvage `C0=dry` vs Guarded canonical `c1=drier`) recorded.
- §5: eastern/drier group linked to lowland ECE site descriptors; R10 counterweight softened to mixed/conditional transfer.
- §8/§9/§10: deployment interpretation, ECE caveats, and multi-year ECE future work updated for the new framing.
- Ledger: C9 split into C9a/C9b/C9c; new C16 (covariate-proximity-governs). Figure plan: T7 promoted from optional to supplementary-required + new T9 panel spec. Risks: R3 downgraded, R10 updated, +R15 (oracle misread)/R16 (label-flip). Source map: +`ece-router-salvage-1.1/2.0` + new Guarded-on-ECE run.
- Works: W1/W3 sources expanded; W4 default = include as discussion + supp (cut only for strict 6-page venue); W4′ scope = Guarded-on-ECE run (accepted); +W4″ terminology pass; W5/W8 point at v5.

---

## 0. Locked framing decisions

1. **Claim is multi-regime vs single-regime, NOT "K=2 is universal."**
   - Our data: K=2 is best among tested K ∈ {1,2,3,4} (K=1 = global baseline).
   - Deployment elsewhere may need larger K; K-selection guidance is a contribution, not a fixed answer.
2. **No regime-specific feature selection.** Every configuration (global and all routers) uses the identical shared 54-feature backbone. Delta-feature arms (c0/c1) are OUT of the main claim; at most a one-paragraph sensitivity note that the historical test-selected delta changed R² from 0.8118 to 0.8126 while validation-selected deltas failed. The shared backbone, XGBoost hyperparameters, and some other model choices were selected using the same 2023–2025 test period, so the reported scores are development-split evidence, not independent confirmation; sharing features does not remove that selection bias.
3. **Venue-agnostic outline.** Sections written so the same material works for:
   (a) ML conference (NeurIPS/ICML workshops, AAAI/ACL-app tracks, KDD application),
   (b) AI-for-science / geoscience ML venue (AGU Fall Meeting abstract → paper, IEEE IGARSS/CIKM application track),
   (c) IEEE AIIoT-style applied venue (same family as paper 1).
   Length knobs marked per section (short = 6–8 pp; full = 10–12 pp).
4. **ECE collaboration stations: diagnostic discussion + supplementary panel.** Default = extended Discussion subsection (§7, ~0.75–1 page) + T7/T9 panel, framed as a small, short-window evaluation of Washington-calibrated availability-gated routing under missing satellite router inputs — NOT a SOTA claim or general deployment validation. Use `c0_only`/`c1_only` as explicitly non-deployable, family-indexed oracle/diagnostic conventions; distinguish them from the G_API class-to-local-expert map used by deployable `auto_hard/auto_soft`. Report calibration provenance and family-specific outcomes; do not infer physical regimes or causal geography effects from these five stations. Cut §7 + supp panel only for a strict 6-page venue (see W4).
5. **OOS/out-of-state: Limitations only** (one paragraph + numbers), per prior decision.
6. **Sequel positioning to paper 1 (NEW in v2).**
   - Paper 1 = *Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather-Driven Machine Learning* (submitted IEEE manuscript, `paper/`).
   - Cite paper 1 for foundations: dataset preparation, feature-selection pipeline, preprocessing, global single-regime baseline performance, and (if present in the submitted PDF — verify in W-new) the prior three-regime/oracle analysis and gating limitations.
   - This paper does NOT re-litigate those results; it answers the open question paper 1 left: *what routing signal actually makes regime specialization deployable?*
   - Historical internal experiments (gating v19–v25, threshold recalibrations, adapted-training) are prior-work context only — never mixed into our result tables as comparable rows across dataset versions.
7. **Design principle as a first-class contribution (NEW in v2).**
   - Target-derived gates can be circular or unavailable at inference when they require the in-situ target; the current in-paper foil is specifically `Trained_Gating_k2`.
   - The tested `Trained_Gating_k2`, seasonal, and single-index routers are weaker in this benchmark; do not generalize that result to supervised or heuristic routers as classes.
   - The primary KMeans router uses the shared covariate backbone and does not fit on the in-situ target, but it does use satellite soil-moisture proxies. Describe station groups and their feature associations as descriptive, not as discovered causal regimes.
   - Headline framing: "route without the in-situ target, and disclose proxy provenance."
8. **Prior-attempt depth in the paper body (NEW in v2, default).**
   - Intro: one tight paragraph on the genealogy (global ceiling → 3-regime MoE idea → deployable routing failed → paper 1 shipped global).
   - Optional appendix/context table ("prior design attempts", dataset-labeled, not ranked against current results) — include only if venue has room; default = omit from main results, keep in outline as available material.
   - All historical numbers attributed to paper 1 or clearly marked as internal prior work with dataset version tags (e.g. `derived_9.0`, `derived_8.1_pos`).
9. **Guarded primary router with pre-specified promotion bars (NEW in v4).**
   - Primary config = `Guarded_Backbone54_k2 (0,0)`: KMeans on the shared 54-feature backbone plus a deterministic station-consistency guard (fit-frame majority vote, §4). No hand-picked router feature subset, no legacy 50-feature import.
   - The guard design and its promotion bars (temporal within ±0.003 of 0.8118; LOSO within ±0.010 of 0.6372) were specified before the follow-up training run; both pass (temporal +0.00004, LOSO +0.0007). Because the benchmark test period had already informed feature/model selection, this pre-specification limits within-run tuning but does not make the test an independent confirmation (R12).
   - `Clustering_V0_Full_k2` is demoted to a sensitivity reference (appendix row): same full `trainval` (training + validation) partition (ARI = 1.0), marginally different LOSO behavior under per-fold refits. It is never co-ranked as a competing method.
   - Motivation depth is middle-path by decision: full method spec (reproducibility) + compact motivation (one paragraph + T8 mechanism table). No multi-paragraph design genealogy; no silent rename. The guard is recomputable from the shared feature backbone; this is implementation portability, not evidence of cross-region performance transfer.

---

### v3 clarifications retained, v4 deltas marked

- **Controlled router design, not novel architecture:** describe the implemented model as a hard-gated, cluster-wise XGBoost mixture-of-experts formulation. The contribution is the controlled router comparison, not a new MoE architecture. *(v4: the guard is a deterministic consistency rule inside the router, not an architecture claim — say so once in §4.)*
- **Primary routing provenance:** the main covariate router is in-situ-target-label-free, but it is not proxy-free: the shared 54-feature backbone includes current, lagged, and rolling SMAP-derived soil-moisture features.
- **Empirical comparison:** hold the expert type, shared backbone, split, and evaluation protocol fixed while comparing target-derived, heuristic, and covariate-defined routers. *(v4: Guarded vs unguarded-Backbone ablation is same-harness and paired; Guarded vs global/heuristic rows are sensitivity-anchored to `formal-eval-1.0` — never present cross-harness numbers as paired.)*
- **Narrow supervised-router claim:** treat the v2 supervised/heuristic failure language as historical motivation. The current negative result applies specifically to the tested `Trained_Gating_k2` router trained from an in-situ target threshold; do not generalize it to supervised routers as a class.
- **Terminology and interpretation:** use “covariate-defined regional strata” for the current partition. Station purity, east/west composition, and feature separation are descriptive associations; they do not establish causal climate or hydrologic mechanisms without subject-matter validation. *(v4: purity on known stations is by-construction under the guard, so it is a weaker diagnostic than in v3 — say so in §5; the transferable evidence is the LOSO behavior on unseen stations.)*
- **v7 terminology override:** retain the v5 dry/lowland and wet-mountain names only as historical shorthand. ECE reporting must identify the family-local expert index and the selected G_API class-to-local-expert map separately; neither index is itself a physical dry/wet label. The approved dry-assigned oracle convention remains family-specific and non-deployable, including Guarded local c1 where Washington means disagree.
- **Default venue posture:** retain the v2 venue options, while using the CS/IEEE-leaning presentation and minimizing geoscience-process interpretation unless the target venue and reviewers support expanding it.

---

## 1. Title candidates — v3 framing retained

### Recommended

1. "Covariate-Defined Regional Experts for Soil-Moisture Estimation: A Controlled Study of Hard-Gated Routing"

### Alternatives

2. "When Does Regional Specialization Help? Router Design for Hard-Gated Soil-Moisture Experts"
3. "Hard-Gated Cluster-Wise XGBoost for Regional Soil-Moisture Estimation"
4. "Inference-Available Routing for Regional Soil-Moisture Estimation with Specialized XGBoost Experts"
5. "Beyond a Single Regional Model: A Router Comparison for Soil-Moisture Estimation"

Avoid titles that claim a novel MoE architecture, universal covariate-defined regional strata, or general failure of learned routers. *(v4: no title change — the guard does not alter the paper's claim level.)*

---

## 2. Abstract skeleton (6 sentences — v4: guarded primary)

1. Problem: regional soil-moisture data combine stations with different environmental conditions, site characteristics, and covariate distributions. A single pooled estimator uses one shared mapping across these groups, while expert specialization requires a routing rule that identifies which mapping to use for each prediction. That rule must operate without the contemporaneous in-situ soil-moisture target—the quantity being estimated—and instead rely on inputs available at deployment.
2. Approach: compare target-derived, heuristic, and inference-available covariate routers under a fixed hard-gated XGBoost expert design; the primary router is KMeans on the shared 54-feature backbone with a station-consistency guard, in-situ-target-label-free but satellite-soil-moisture-informed.
3. Result temporal: on the 2023–2025 development test period, Guarded (0,0) reports mean R² 0.8118 (seed SD 0.0014) and RMSE 0.0442 over 30 expert seeds; the global values (R² 0.7798, RMSE 0.0478) are sensitivity-anchored, not an in-harness paired Guarded comparison.
4. Result spatial (in-state LOSO): Guarded mean R² is 0.638; global 0.580 is sensitivity-anchored, while the paired Guarded-vs-Backbone comparison improves 3/7 folds and ties 4/7 (descriptive).
5. Analysis: the guard enforces station consistency for known stations; per-sample KMeans can change specialist-training composition under eastern-station holdouts, while held-out assignment remains similar. The tested target-derived `Trained_Gating_k2` foil underperforms; these results concern this benchmark and these tested routers.
6. Scope: this is a Washington regional study, not evidence that K=2 or this router transfers universally; test-guided feature/model selection makes the reported test scores development evidence, and out-of-state transfer fails for all tested models.

---

## 3. Section-by-section outline

Each section lists: goal · claims · numbers to quote · source (experiment path + table) · figures · venue knobs · open writing tasks.

### §1 Introduction [all venues] — router-design problem genealogy

- **Beat 1 — What paper 1 established.** Dataset preparation, feature pipeline, and a strong single-regime global XGBoost model established the baseline; prior error analysis motivated specialization. Cite paper 1 only for claims verified in the submitted manuscript.
- **Beat 2 — Why router provenance matters.** Earlier target-threshold and target-derived routing attempts show why oracle specialization is not the same as deployable specialization: labels derived from the in-situ target are unavailable at inference and hard gates amplify boundary errors.
- **Beat 3 — What this paper studies.** On the clean 7-station `derived_8.4` split, hold the prediction backbone and XGBoost experts fixed while comparing target-derived, seasonal/antecedent-precipitation-index (API), dynamic, and covariate-defined routers. *(v4: add one sentence — the primary covariate router carries a station-consistency guard so the model is describable purely from the shared backbone, with no legacy feature set to port to new regions.)*
- **Contributions:**
  - C-A: A hard-gated, cluster-wise XGBoost formulation for regional soil-moisture estimation, treated as an MoE formulation rather than a novel MoE architecture.
  - C-B: A controlled comparison of router provenance under shared features, multi-seed temporal evaluation, and in-state LOSO; block bootstrap is reported only for a test-selected V0 sensitivity arm.
  - C-C: In the no-delta development-split comparison, Guarded reports temporal R² 0.8118 and is close to the unguarded Backbone (0.8117; paired difference +0.0001). In paired LOSO, its mean R² is 0.638 vs 0.620, with gains on 3/7 folds and ties on 4/7; comparison with global is sensitivity-anchored.
  - C-D: A failure analysis showing that the tested target-derived gate is not deployable and underperforms; this does not generalize to all supervised routers.
  - C-E: A reporting framework covering feature availability, satellite-soil-moisture proxy use, K selection, partition stability, and deployment limits.
  - *(v4: C-C's partition is the guarded one; add no new contribution bullet — the guard is a device inside C-B/C-C, not a standalone claim.)*
- **Scope boundary:** station-group and feature-separation results are descriptive associations, not causal hydrologic explanations.
- Sources: paper 1 PDF; `writeup/sections/three_regime/*`; `docs/gating.md`; `docs/plans/20260703-why-MoE.md`; `derived_8.4-gating-analysis-1.0`.
- Knobs: ML/IEEE venues emphasize router provenance and evaluation; geoscience venues may expand interpretation only after subject-matter review.

### §2 Related work [all venues] — unchanged from v3

- **Pillar 0 — Positioning against paper 1.** Distinguish the submitted paper’s global model and prior regime analysis from this paper’s controlled router comparison.
- **Pillar 1 — Direct soil-moisture MoE papers.** Discuss Yang et al. (2025), “An Interpretable framework of Soil moisture estimation based on Mixture-of-Experts (ISMoE)” (*Journal of Hydrology* 661, 133763), which uses CNN-LSTM experts and adaptive routing across Tibetan Plateau climate zones; Zhang et al. (2026), “An Antecedent-Precipitation-Informed Soil Water Balance and Time-Aware Mamba–MoE Framework for Surface Soil Moisture Forecasting” (*Remote Sensing* 18, 3125), which combines a physical prior, time-aware Mamba, and context-based MoE; Singh et al. (2026), “Weak Physics-Guided Multi-Agent Learning for Surface to Subsurface Moisture Estimation Across Diverse Climate and Soil Conditions” (*JGR: Machine Learning and Computation*), with dry/intermediate/wet specialists; and Wu et al. (2026), “An RTE-guided mixture-of-experts framework for AMSR2 passive microwave soil moisture retrieval” (*International Journal of Applied Earth Observation and Geoinformation*).
- **Pillar 2 — Clustered and regional soil-moisture models.** Discuss Chakrabarti et al. (2016), “Disaggregation of Remotely Sensed Soil Moisture in Heterogeneous Landscapes Using Holistic Structure-Based Models” (*IEEE TGRS* 54), which uses soft clustering followed by cluster-specific kernel regressors; Ding et al. (2024), “Addressing spatial gaps in ESA CCI soil moisture product: A hierarchical reconstruction approach using deep learning model” (*International Journal of Applied Earth Observation and Geoinformation* 132), which uses KMeans-defined climate subregions with specialized gap-filling models; and Stahl & McColl (2022), “The Seasonal Cycle of Surface Soil Moisture” (*Journal of Climate* 35), which uses KMeans to identify global surface-soil-moisture seasonal regimes.
- **Pillar 3 — Hydrologic MoE and router selection.** Discuss Moges et al. (2016), “Hierarchical mixture of experts and diagnostic modeling approach to reduce hydrologic model structural uncertainty” (*Water Resources Research*), on indicator-driven hierarchical mixtures, and Rong et al. (2025), “Mixture of experts leveraging Informer and LSTM variants for enhanced daily streamflow forecasting” (*Journal of Hydrology*), on heterogeneous LSTM/GRU/Informer experts with RF/LSTM/Transformer routers.
- **Pillar 4 — Hybrid model terminology.** Distinguish LSTM+XGBoost feature fusion or residual stacking from MoE routing: different model types alone do not constitute a mixture of experts unless a router selects or weights expert outputs.
- **Pillar 5 — Evaluation practice.** Position LOSO, multi-seed inference, configuration-specific block-bootstrap sensitivity, feature provenance, target-label leakage, and deployment-availability checks as the gap addressed here.
- **Positioning sentence:** the reviewed literature establishes that soil-moisture MoE and cluster-specific modeling already exist; this paper studies how router provenance affects hard-gated regional XGBoost performance under a fixed protocol that audits target availability and discloses test-set reuse.
- **Open task W2:** create verified BibTeX records and preserve DOI/venue metadata for every direct comparator; no citations invented in the outline.

### §3 Problem setup & data [all venues] — compact end-to-end workflow

- **End-to-end workflow:** Station observations are requested or loaded, parsed, cleaned, and merged; satellite/site predictors are added, hourly weather is aggregated to daily rows, predictor gaps are filled, configured smoothing is applied, and temporal, seasonal, and meteorological features are derived. The resulting prepared data are represented by the canonical `derived_8.4` modeling table and its time-based train/validation/test partitions. The shared 54-feature backbone is then used by the router and XGBoost experts; the router assigns each row to an expert, and performance is assessed on the later temporal period and with station-held-out LOSO. Keep this as a short overview and cite the established preprocessing/feature pipeline description rather than repeating its implementation details.

- `derived_8.4`: 7 WA stations; train 2017–20 (9,803) / val 2021–22 (4,805) / test 2023–25 (6,620). Define `trainval` as train + validation (14,608 rows) when first used. Include a concise station list or table with one-line descriptors (east/west, elevation).
- Stations: BeaverPass_WA_990, CayusePass_WA, Darrington, Paradise_WA, Quinault, SourdoughGulch_WA_985, Spokane.
- Feature backbone: 54 shared features (identical for every model) — name the source (`derived_8.4-feature-selection-2.0` / eval-1.1 `selected_features.json`); foundations of feature engineering cite paper 1. State that this backbone was selected using the same 2023–2025 test period and then shared across arms: sharing controls the comparison but does not remove selection bias.
- *(v4 NEW one-liner):* the primary router uses this identical backbone — there is no separate router feature source to justify or port (the legacy 50-feature V0 set survives only as a sensitivity reference, §4).
- Target: `soil_moisture_5cm`; metrics: R², RMSE, MAE, bias (+ Pearson where useful).
- Sources: `data/splits/derived_8.4/split_meta.json`; `src/pipeline/README.md` and `src/pipeline/main.py` for the processing sequence; `writeup/sections/shared/preprocessing.tex` and `features.tex` for the publication-level description; `derived_8.4-formal-eval-1.0` protocol; `derived_8.4-gating-analysis-1.0` feature-list tables; paper 1 §data/features.

### §4 Method: hard-gated cluster-wise XGBoost (MoE formulation) [all venues]

- Architecture: router → hard assignment → one XGBoost expert per assigned group; inference activates one expert and a small router. The formulation is an MoE, but the paper does not claim a new MoE architecture. The station-consistency guard is a deterministic rule inside the router, not an architecture claim.
- **Router alternatives and provenance:**
  - A1: Target-derived threshold gate on `soil_moisture_5cm < 0.16` — the in-paper `Trained_Gating_k2` foil; unavailable without the in-situ target and not deployable as an estimator router.
  - A2: Heuristic routers — `Seasonal_Binary_k2` and `Univariate_G_API_k2`; inference-available but limited state/season rules.
  - A3: Covariate routers — `Clustering_Dynamic_k2` and unguarded `Clustering_Backbone54_k2`; fitted without the in-situ target and applied to inference-available inputs. Unguarded-Backbone is the in-harness ablation for the guard (same features, same protocol, paired seeds).
  - A4 (primary): `Guarded_Backbone54_k2` — KMeans on the shared 54-feature backbone plus a station-consistency guard. Guard semantics in three lines: (1) fit KMeans on the fit frame only (mean-impute → StandardScaler → KMeans(2, seed 42)); canonicalize label ids so c1 has the lower trainval mean of `SMAP_sm_pm_interp_rollmean30` (a feature-indexing rule, not a physical dry/wet label); (2) known stations receive their fit-frame majority label (deterministic tie-break); (3) unseen stations (held-out LOSO folds, new deployments), missing ids, and availability-gated rows fall back to the per-sample KMeans prediction, with a fit-frame-calibrated margin flag available for global-expert fallback. Label it **in-situ-target-label-free, satellite-soil-moisture-informed covariate routing**, not proxy-free routing.
  - Sensitivity (appendix row, not a competing method): `Clustering_V0_Full_k2` on the legacy 50-feature set — same full `trainval` partition (ARI = 1.0), kept only as a sensitivity reference.
- **Router provenance table:**

| Family | Configuration | In-situ target used to fit router? | SMAP-derived inputs? | Deployment disclosure |
|---|---|---:|---:|---|
| Target-derived | `Trained_Gating_k2` | Yes | Not required | Diagnostic foil only; unavailable without target labels |
| Seasonal heuristic | `Seasonal_Binary_k2` | No | No | Calendar rule; deployable where date is available |
| API heuristic | `Univariate_G_API_k2` | No | No | Deployable if API inputs are available |
| Dynamic covariate | `Clustering_Dynamic_k2` | No | Yes, lagged SMAP feature | Requires the same satellite product/latency assumptions |
| Full covariate (ablation) | `Clustering_Backbone54_k2` (unguarded) | No | Yes, current/lagged/rolling SMAP features | Same-harness guard ablation; satellite-SM-informed |
| Full covariate (primary) | `Guarded_Backbone54_k2` | No | Yes, current/lagged/rolling SMAP features | Primary result; explicitly satellite-SM-informed; guard recomputed per region from trainval |
| Legacy sensitivity | `Clustering_V0_Full_k2` (appendix) | No | Partial | Same partition (ARI 1.0); legacy 50-feature provenance, not portable |
| Global baseline | `Global_Single_54` | No router | Yes in prediction backbone | Single shared model |

- **Station_id disclosure (NEW, R13):** known-station majority voting uses `station_id` at inference; this enforces consistency of an unsupervised partition and does not touch the target. LOSO held-out stations are unseen by construction (no leakage into means/scaler/centroids/majority/thresholds), so the spatial evaluation is uncontaminated. New-region deployment recomputes all fit-frame quantities locally.
- **Pre-specified promotion note (R12):** the guard design and promotion bars were fixed before the follow-up training run (temporal ±0.003, LOSO ±0.010), and both pass. State that the 2023–2025 test period had already informed feature/model selection, so this pre-specification does not create an independent confirmation set.
- Router families compared: the six v3 configurations plus the guarded primary; V0 appears only as the appendix sensitivity row.
- Training protocol: routers fit on trainval only; experts per group; 30 temporal seeds; LOSO 5 seeds × 7 folds with per-fold refitting. State exactly which components are stochastic.
- **Shared XGBoost expert settings:** XGBoost 3.2.0 with 2,500 trees, learning rate 0.005, max depth 9, minimum child weight 8, gamma 0, α=0.03 and λ=0.75 regularization, and row/column subsampling 0.9/0.8 (`tree_method=hist`). These settings are shared across the global and routed experts; they came from test-era tuning and are part of the test-guided model-selection limitation (§9).
- Statistics: report seed-level intervals and tests, BH-FDR, block bootstrap, and LOSO summaries with each comparison's configuration and harness identified. Seed variation measures expert fitting stochasticity only; it does not include feature/model-selection uncertainty. Global/gating comparisons are sensitivity-anchored to `formal-eval-1.0`; the only paired in-harness comparison is Guarded vs unguarded Backbone. The cited `ΔR² +0.035` sample bootstrap is specifically V0 with test-selected `c0=0, c1=10` and is sensitivity-only, not evidence for the Guarded no-delta primary.
- **K-selection disclosure:** the recommended protocol selects K using train/validation or predeclared clustering diagnostics and reserves test for final evaluation. In this study, the test period was also used in feature/model selection; disclose this reuse when presenting results. *(The routing-optimize K-sweep reproduces gating-analysis indices exactly on trainval and provides a per-region recomputable diagnostic.)*
- **Feature-availability disclosure:** distinguish in-situ target labels from satellite soil-moisture proxies and state whether each input exists at intended deployment time.
- No claim is made that hard routing is superior to soft routing; the paper compares the implemented hard-gated router families only. Margin/availability fallback machinery is reported as implemented-but-advisory in one sentence (it never triggers on WA data).
- Sources: `derived_8.4-formal-eval-1.0` (including its `config.yaml` and README for exact XGBoost settings and test-era tuning provenance); `derived_8.4-routing-optimize-1.0` (guard spec, agreement + training runs and `config.yaml`); `derived_8.4-gating-analysis-1.0`; `eval11/routers.py`; `derived_8.4-regime-interpretation-1.1`; direct literature listed in §2.

### §5 Partition diagnostics: covariate-defined regional strata

[ML/IEEE venues: 0.5–1 page · geoscience venues: expand only with subject-matter review]

- K=2 purity = 1.000 trainval and test **on known stations by construction under the guard**; report this as an enforced consistency property, not as discovered regime evidence. The transferable diagnostic is unseen-station behavior (LOSO), not purity itself.
- Partition composition: cluster 1 = Spokane + SourdoughGulch; cluster 0 = five western/mountain stations. If these historical shorthand labels are retained, identify them as descriptive and experiment-specific; state the family-local index convention in §4 and use the ECE crosswalk. Do not treat them as physical classes. The five ECE collaboration stations have lowland site descriptors (51–157m elevation); describe these measurements and their relationship to the WA training stations without treating proximity as a causal explanation of error.
- **Fold-instability mechanism (NEW, compact — the motivation exhibit):** per-sample KMeans on either feature set fragments stations when an eastern station is held out (fold-trainval V0-vs-Backbone ARI 0.13/0.19 on Sourdough/Spokane folds; identical ARI = 1.0 on western-holdout folds), while held-out assignment is near-identical across routers (≤15 rows, R3b) — so the LOSO gap comes from specialist-training composition, not misrouting. The guard restores station-pure training sets by construction (Guarded-vs-Backbone fold ARI 0.33–0.35 on those folds). Three sentences max; T8 carries the numbers.
- Top separating features include static/site descriptors, SMAP rolling statistics, and LST/NDMI windows. Treat these as associated partition indicators, not causal drivers.
- Correlation drift, including `SMAP_sm_pm_interp_rollmean30`, is evidence that the pooled mapping may differ across strata; it does not establish a physical mechanism.
- Quality indices: Calinski-Harabasz and Davies-Bouldin favor K=2 while silhouette is marginally higher at K=3. Internal indices alone do not select the meaningful partition.
- Explicit caveat (R10): station purity partly reflects site descriptors or station identity proxies, and the guard enforces it explicitly — purity diagnoses specialization on this dataset but may reduce transferability to unseen stations. ECE is a conditional diagnostic of unseen-site performance under missing SMAP router inputs; it does not establish that a named physical expert transfers because of geography or covariate proximity (see §7).
- Contrast with the target-derived gate: target labels create wet/dry separation but do not yield stable station groups and are unavailable at inference.
- Sources: `derived_8.4-gating-analysis-1.0`; `derived_8.4-routing-optimize-1.0` (R2/R3/R3b/R4); `derived_8.4-regime-interpretation-1.0/1.1/1.2`; feature provenance in §4.

### §6 Results [core of every venue]

#### 6.1 Temporal (primary) — no-delta only

- Headline table (rank by RMSE): at minimum
  - `Guarded_Backbone54_k2` (0,0): mean R² 0.8118 (seed SD 0.0014), RMSE 0.0442 *(NEW primary; 30 seeds, slurm job 2155747)*
  - `Global_Single_54`: mean R² 0.7798 (seed SD 0.0013), RMSE 0.0478 *(sensitivity-anchored to formal-eval-1.0)*
  - `Trained_Gating_k2` (0,0): mean R² 0.7354 (seed SD 0.0011) *(sensitivity-anchored to formal-eval-1.0)*
  - plus Dynamic / Seasonal / Univariate no-delta rows for the full router comparison *(all sensitivity-anchored to formal-eval-1.0)*
  - Appendix row: `Clustering_V0_Full_k2` (0,0) 0.8118 (same rounded value; ties Guarded within 0.0001) and unguarded `Backbone54` (0,0) 0.8117 (in-harness ablation: Guarded better on 30/30 seeds by +0.0001, honestly a tie).
- Pairwise sensitivities from `formal-eval-1.0`: V0 no-delta vs global ΔR² +0.032 [0.031, 0.033], p < 1e-12, 100% seeds; V0 no-delta vs trained gating +0.089. These are sensitivity-anchored, not paired Guarded-vs-global tests.
- The sample bootstrap ΔR² +0.035 [0.019, 0.055], p = 0.0005, and RMSE −0.0040, p = 0.0005, belong to V0 with test-selected `c0=0, c1=10`; report only as a sensitivity result, never as evidence for the Guarded no-delta primary. Omit from the abstract and main headline table.
- **Router-design sentence:** specialization is not sufficient by itself; the sensitivity-anchored V0 no-delta comparison favors the two-regime model over the global baseline, while the tested target-derived gate underperforms. For Guarded, the direct in-harness comparison is against unguarded Backbone, with a near-tie temporally and gains on 3/7 LOSO folds. These are development-split findings, not a universal ranking of supervised versus unsupervised routers.
- Seed-42 note (one line): Guarded seed-42 reproduces eval-1.1's V0-backbone 0.814334 exactly — the guard recovers the legacy-partition number on shared-54 features.
- Optional footnote only (not a table row): paper 1's R² 0.822 is a historical result on a different dataset and protocol; include it only as context, not as a directly comparable baseline or ranked result (R5).
- Source: `derived_8.4-routing-optimize-1.0` T1 (primary + ablation) + `formal-eval-1.0` temporal seed table (sensitivity rows).
- Figure F1: bar/dot temporal R² with 95% intervals across ~8 configs; label these as seed-level intervals that reflect expert random-state variation, not sampling or selection uncertainty (NEW figure notebook).

#### 6.2 In-state LOSO (secondary, descriptive)

- No-delta: `Guarded_Backbone54` 0.638 vs `Global_Single_54` 0.580 (sensitivity-anchored) and `Baseline_V0_50` 0.591; unguarded `Backbone54` 0.620 in-harness. V0 sensitivity 0.637.
- **T8 (NEW, the mechanism table):** Guarded-minus-Backbone per-held-out-station mean R² diff — SourdoughGulch +0.059, Spokane +0.051, Paradise +0.015, all other folds 0.000; Guarded wins 5/5 seeds overall.
- 6/7-vs-global figure retained ONLY as the V0-anchored sensitivity row (sign p = 0.125 — state underpowered); the in-harness claim is Guarded-vs-Backbone (gains on 3/7 folds, ties on 4/7). If PI wants direct Guarded-vs-global win counts, that is a flagged follow-up run (Global_Single_54 was not in-harness in routing-optimize-1.0).
- Station difficulty table (7 stations, median LOSO R² across configs): Darrington 0.70, BeaverPass 0.69, Spokane 0.65, Paradise 0.59, Quinault 0.56, SourdoughGulch 0.43, CayusePass 0.39. *(unchanged; from formal-eval-1.0/eval-1.3)*
- Explicit sentence: LOSO supports consistency, not statistical significance at n = 7.
- Source: `derived_8.4-routing-optimize-1.0` L1 + T8; `formal-eval-1.0` LOSO summary (sensitivity rows); `derived_8.4-eval-1.3` for difficulty narrative if needed.
- Figure F2: per-station paired R² (existing `loso_pair` / `loso_station_bars` PNGs).

#### 6.3 Router comparison (the multi-regime story)

- Ranking message: among the no-delta configurations, Guarded reports the strongest temporal result and ties the legacy V0 result without its legacy feature set. Comparisons against global/heuristic rows are sensitivity-anchored, not paired Guarded comparisons; the in-harness paired ablation shows consistency enforcement raises mean LOSO R² from 0.620 to 0.638, with gains on 3/7 folds and ties on 4/7. Do not generalize the tested target-derived foil to supervised routing as a class.
- Optional single table merging 6.1 + 6.2 columns (temporal R² | LOSO R²).

#### 6.4 Robustness note (1 paragraph, not a section)

- Guarded no-delta (0.8118) is close to the historical V0 test-selected delta arm (0.8126); treat this only as a descriptive sensitivity comparison, not evidence independent of test-guided selection.
- Val-selected deltas historically UNDERPERFORM global (0.735 vs 0.780) → selection instability exists in this benchmark; our simplified no-delta protocol avoids it.
- **Required selection caveat:** the shared 54-feature backbone and XGBoost hyperparameters were selected using 2023–2025 test-era evidence, and some router/model choices were also made after examining this period. Treat all test scores as development-split evidence rather than independent confirmation; shared features improve comparability but do not eliminate selection bias. Seed intervals reflect expert random-state variation, not uncertainty from feature, router, or model selection.
- Source: `derived_8.4-formal-eval-1.0` delta-robustness table + caveat bullets.

#### 6.5 (Optional, venue-dependent) Prior-attempts context material

- NOT a ranked results row. If included: small appendix table listing prior design attempts (threshold hard/soft, supervised classifier, adapted training, threshold recalibration, dataset expansion) with **dataset version tags** and qualitative outcome (worked / failed / partially), sourced from paper 1 + internal docs — purpose is narrative honesty, not benchmarking.
- Default: omit from main paper; keep outline entry for PI option (fold into W4-style venue decision).

### §7 ECE collaboration station diagnostic (v7 terminology)

[Default INCLUDE as a short Discussion subsection + supplementary T7/T9 panel; cut both only for a strict 6-page venue — see W4. This is a five-station, short-window diagnostic, not a SOTA claim or a general deployment validation.]

- Terminology: call these the **five ECE collaboration in-situ stations** when referring to the sites and `derived_8.4_ece_v3` when referring to the held-out evaluation split. The split contains 150 daily target observations over 2026-07-20–08-19. Keep “station,” “sensor deployment,” and “evaluation row” distinct; do not imply five observations per sensor or a complete seasonal study.
- Protocol: fit static routers and main experts on 7 Washington trainval stations (14,608 rows, 2017–2022). For synthetic-SMAP-mask calibration, fit auxiliary experts on WA train and use WA validation to choose soft temperature and Guarded's G_API class alignment; freeze those choices before evaluation on the five disjoint ECE stations. Their rows have native-missing SMAP router inputs, so the availability gate activates and routes them through the SMAP-free G_API auxiliary router. ECE targets are used only for final scoring.
- Routing terminology: `auto_hard` means availability gate → G_API hard class on gated rows, otherwise the family’s static route. `auto_soft` uses the same mapped G_API class as the anchor for the WA-calibrated blend. G_API class numbers do not have an inherent dry/wet meaning. Preserve salvage 2.0's class-to-local-index mapping for V0/Backbone; choose Guarded’s class alignment between both expert indices using synthetic-SMAP-masked Washington validation RMSE, and report both candidate scores. Do not call these policies a universal “router fix.”
- Expert labels: retain the approved family-specific `c0_only` oracle convention (salvage C0 for V0/Backbone; Guarded local c1 under its canonicalization) for historical comparability. These names are label conventions, not an independent physical classification. Report local indices, canonical-feature/target means, and the separate G_API mapping in the crosswalk. Both `c0_only` and `c1_only` are non-deployable diagnostics.
- Results: present the published-router tie (`formal-eval-2.1-ece-v3`, ΔRMSE +0.00025, 2/5 station wins, not significant) separately from the 1.1 missingness-aware ECE evaluation. In the executed T7 table, `auto_hard` and `auto_soft` have pooled RMSE 0.057768 (seed SD 0.000617), bias 0.028654–0.028655, and ubRMSE 0.050158 for each routed family; the direct Global reference has RMSE 0.058634 (SD 0.000735), bias 0.015122, and ubRMSE 0.056635. The selected Guarded map is G_API class 0 → local c0 (masked WA-validation RMSE 0.073965; the alternative class 0 → c1 scores 0.117180). All 150 ECE rows trigger the availability gate and have a fully missing SMAP router block. All 750 Guarded prediction rows have G_API class 0, so `auto_hard` coincides with the complementary local-c0 diagnostic, while the approved dry-assigned oracle is local c1; the chart crosswalk keeps those outputs and deployability labels separate. Interpret this index-dependent pattern neither as physical dry/wet transfer nor as a causal geography effect.
- Benchmark validity: hourly analysis found a diurnal cycle (median hourly range 0.035 ≈ 14× median daily step 0.0025); overnight rain and midday drying are hidden by daily aggregation; hour-scale corr(rain, Δsensor) is near zero at lags 0–6h (`ece-input-diagnose-1.0`). Describe the short daily series as limited for ranking daily models, while useful for examining routing behavior under this input-availability shift. Do not claim sensors ignore rain or are faulty.
- What we must NOT say: that two-regime modeling wins on ECE as a SOTA claim; that the five sites establish broad dry/wet regimes or causal geographic effects; that an oracle is deployable; or that these 30 days generalize to winter or a wet season.
- Decision rule W4: keep §7 and its supplement by default; cut both only for a strict 6-page venue.
- Sources: `derived_8.4-formal-eval-2.1-ece-v3` for the published-router tie; `derived_8.4-ece-router-salvage-2.0` for the historical V0/Backbone salvage mapping; `paper2-final-evidence-1.1/` for the rerun, WA mapping calibration, audit, and T7/T9/F5 notebook; `ece-input-diagnose-1.0` for hourly benchmark-validity analysis.

### §8 Discussion

- **What the controlled router comparison establishes:**
  1. A router trained from an in-situ target threshold is unavailable at inference and performs poorly in this benchmark; this is a result about the tested target-derived gate.
  2. Inference-available covariate partitions can produce stable station-group specialists; the no-delta global comparison is sensitivity-anchored, while the paired Guarded ablation against unguarded Backbone is near-tied temporally and improves mean LOSO R².
  3. The primary router uses SMAP-derived soil-moisture proxies, so the correct claim is in-situ-target-label-free and satellite-SM-informed, not proxy-free.
  4. Station purity on known stations is enforced by the guard, so it is a weaker diagnostic than previously presented; the transferable evidence is unseen-station LOSO behavior, and purity/associations do not prove causal climate or hydrologic regimes.
  5. Oracle upper bounds diagnose specialization potential, not deployability.
  6. *(v4 NEW)* The guard's value is implementation portability, not demonstrated performance transfer: it recovers the legacy-partition result on shared-54 features with a per-region recomputable rule, removing the one non-automatic element (the 50-feature legacy set) from the deployment story. Guard design and promotion bars were specified before the follow-up training run; explain that this did not make the already-used test period independent confirmation.
- **Why the target-derived gate loses in-paper:** target-threshold labels define wet/dry states that are unavailable at prediction time and may assign heterogeneous station groups to the same expert. Avoid saying that all learned or supervised routers fail.
- **Why the guard helps where it helps (compact):** eastern-holdout refits redraw specialist training sets; majority voting stabilizes them. It does not improve held-out routing (already near-identical) and does not help unseen stations directly — it protects training composition.
- **Regional strata, not universal regimes:** select K using deployment-region diagnostics, partition stability, group balance, and validation performance. Expect the useful number and meaning of strata to change across regions. New regions recompute imputation means, scaler, centroids, majority map, and margin threshold from local trainval; K=2 is not exported.
- **Deployment interpretation:** the OOS results remain essential counterevidence (regional specialization can help in-region while failing under station or distribution shift). ECE tests routing when native-missing SMAP inputs activate an availability gate on five held-out stations. Report each family’s Washington-calibrated G_API mapping and ECE result; neither a local expert index nor site descriptors alone establish a physical dry/wet regime, causal geography effect, or broad transfer claim.
- **Regional scope by design:** the contribution is an empirical router-design result for Washington soil-moisture estimation, not a general hydrologic law.

### §9 Limitations [required]

- n = 7 LOSO, low power; partial 2025 coverage; seed variation does not cover router stochasticity (routers fixed).
- **Test-set reuse / selection bias:** the 54-feature backbone and shared XGBoost hyperparameters, along with some router/model choices, were selected or promoted after inspecting the same 2023–2025 period used for evaluation. The test results are therefore development-split evidence, not untouched confirmatory estimates; both absolute scores and apparent winner advantages may be optimistic. Seed-level intervals do not capture this selection uncertainty. Independent confirmation requires a new time period or external dataset.
- **Guard-specific limitations (NEW):** (i) majority voting uses `station_id` at inference for known stations — legitimate as a consistency rule over an unsupervised partition, but it is station-identity-adjacent and must not be described as discovering regimes; (ii) the guard acts on training composition, not on unseen-station routing — unseen rows use per-sample predictions, so no direct transfer gain should be claimed beyond what LOSO measures; (iii) `StationMeanV-B` (0.634) trails `GuardedV-A` (0.638) and is reported as a challenger null, not a second method.
- **OOS paragraph:** train on 7 WA → 10 out-of-state stations: ALL models degrade (station-mean R² negative for every config; pooled global 0.206 vs best two-regime ~0.11–0.14; two-regime worse than global, e.g. V0_Full no-delta wins 2/10). Frame: regime specialization is a within-region prior; outside the region both experts and router are OOD — consistent with regional scope, not a hidden result. Source: `derived_8.4-formal-eval-2.0` summary + Table 1 (no-delta rows only). *(v4: OOS rows predate the guard; do not imply guard numbers there.)*
- **Snowpack / feature-space boundary:** stations under alpine snowpack dynamics removed/out-of-scope because the feature set lacks SWE/snow depth; algorithm gating cannot fix missing physics (cite station-removal plan). Future work: SWE integration.
- ECE caveats if §7 kept: 30 daily observations over a late-summer one-sided window, tiny target variance (rank RMSE/bias/ubRMSE; do not rank by R²), daily aggregation hiding diurnal/event response, and only five held-out stations (limited power; report uncertainty plainly). Native-missing SMAP router features differ from Washington calibration inputs, motivating the availability gate; validation-based mapping calibration does not guarantee transfer. Treat `c0_only`/`c1_only` as non-deployable, family-indexed diagnostics and report Washington feature and target means beside the approved family-specific oracle convention: salvage local C0 remains the V0/Backbone dry-assigned oracle despite lower WA canonical-feature and target means at local c1, while Guarded uses local c1. Treat these as index conventions, not physical labels. The 1.1 results are family-specific and do not support other stations or seasons.
- **Router provenance limitation:** the primary router is in-situ-target-label-free but uses SMAP-derived soil-moisture proxies; deployment requires documenting product availability, latency, missingness, and sensor/domain mismatch.
- **Interpretation limitation:** static/site-driven purity and feature correlations are descriptive; causal geoscience interpretation requires subject-matter validation.
- **Anchor-correction footnote (NEW, transparency):** training-run anchors for the guards point at eval-1.1's V0-backbone 0.814334 (the partition the guards provably reproduce, §R2 ARI = 1.0), not Backbone's 0.814205; both values are prior published numbers, and the correction is recorded in the experiment `config.yaml`, not fitted to results.
- Historical/attribution limitation (optional one line): some prior routing results live only in an internal tech report / pre-camera-ready analysis — we cite paper 1 where possible and do not over-claim unpublished numbers.

### §10 Conclusion & future work

- Restate design lesson in one line (on this development split and under a fixed hard-gated XGBoost setup, the tested inference-available covariate router performed better than the tested target-derived gate; Guarded was near-tied temporally with unguarded Backbone and improved mean LOSO R². A station-consistency guard makes the router recomputable from the shared backbone but does not establish cross-region performance transfer).
- Future work bullets (not claims): automated K selection; OOD/ambiguity fallback to global expert (diagnostics exist in `derived_8.4-formal-eval-2.0` Table 4; fallback-mask machinery from routing-optimize-1.0 is implemented but unevaluated as a policy); station-count / training-set composition ablation (explicitly future); snowpack-aware features (SWE) + expanded station types; multi-year and multi-season evaluation with the ECE collaboration once a full season is available, plus independent validation of how G_API classes map to local experts across deployments; revisit 3-way partitions only with a routing signal that is not derived from y; direct Guarded-vs-global LOSO win counts remain a separate follow-up.

---

## 4. Figure & table plan (all via notebook, `nb execute --uv`)

| ID | Item | Status | Source |
|----|------|--------|--------|
| T0 | (Optional) Prior-attempts context table, dataset-labeled | Optional appendix | Paper 1 + verified internal docs; §6.5 |
| T1 | Router × feature × protocol configuration | New, from method text | Gating-analysis and evaluation configs |
| T2 | Temporal seed-level results (no-delta) | Guarded primary + paired ablation; sensitivity rows labeled | `routing-optimize-1.0` T1; `formal-eval-1.0` |
| T3 | Temporal pairwise comparisons and bootstrap | Sensitivity-only; bootstrap is V0 `c0=0, c1=10` | `formal-eval-1.0` |
| T4 | LOSO summary and per-station comparisons | Guarded/Backbone paired; global rows sensitivity-anchored | `routing-optimize-1.0` L1/T8; `formal-eval-1.0` |
| T5 | K-sweep partition stability/quality | Existing; reproduced using trainval only | `gating-analysis-1.0`; `routing-optimize-1.0` R4 |
| T6 | Regional-strata station composition and associated features | Existing | Gating-analysis and regime-interpretation reports |
| T7 | ECE v3 RMSE/bias/ubRMSE by family and policy, with local-index and G_API mapping crosswalk | Supplementary-required; v1.1 complete | `paper2-final-evidence-1.1/ece_guarded/*.csv`; `formal-eval-2.1-ece-v3` |
| T8 | Guarded-minus-Backbone per-held-out-station LOSO difference and fold agreement | Existing notebook output | `routing-optimize-1.0` R3/R3b/L1 |
| T9 | ECE v3 diagnostic panel: per-station errors, family-specific routing shares, and descriptive site attributes | New supplement; v1.1 complete | v1.1 policy/prediction/site CSVs and `station_static_features.csv` |
| F1 | Temporal multi-seed dot/bar with intervals | New | `paper2-figures-1.0` notebook |
| F2 | LOSO per-station paired results | Reuse existing figure | `formal-eval-1.0` |
| F3 | Washington regional-strata map | New or reuse existing map | Gating geographic figures |
| F4 | Delta-robustness comparison (test/validation/none) | Optional; disclose test selection | `formal-eval-1.0` |
| F5 | ECE v3 time series for Guarded (`as_routed/auto_hard/c0_only/c1_only`), with non-deployable diagnostic curves marked | Supplement; v1.1 complete | `paper2-final-evidence-1.1/ece_guarded/predictions_v3.csv` and daily overlay CSV |
| F6 | Router-family concept diagram | Optional | Method narrative |

---

## 5. Claims ledger (excerpt — full ledger = separate file in works)

| # | Claim | Config / source | Number | Artifact |
|---|-------|-----------------|--------|----------|
| C0 | Prior-paper context: global baseline + regime motivation | paper 1 | R² 0.822; different dataset and protocol, so not directly comparable to `derived_8.4` | `paper/` PDF |
| C1 | Prior work: specialization potential under oracle; deployable routing was the gap | paper 1 (if in PDF) / internal prior | oracle ≫ global; e2e routing weak | paper 1 §three_regime (verify) |
| C2 | Two-regime > global temporal (OURS; development-split evidence) | Guarded primary; global comparison sensitivity-anchored to V0 no-delta | 0.8118 vs 0.7798; Δ +0.032 and seed-level p < 1e-12 are from V0 no-delta vs global, not a paired Guarded comparison | `routing-optimize-1.0` T1 + `formal-eval-1.0` |
| C3 | Sensitivity-only sample-level bootstrap (not Guarded primary) | V0 `c0=0, c1=10` (test-selected delta) vs global | ΔR² +0.035 [0.019, 0.055], p = 0.0005; RMSE −0.0040, p = 0.0005 | `formal-eval-1.0` bootstrap |
| C4 | Two-regime > global LOSO | Guarded (0,0) | 0.638 vs 0.580 (sensitivity-anchored); in-harness Guarded-vs-Backbone gains 3/7 folds | `routing-optimize-1.0` L1/T8 + `formal-eval-1.0` LOSO |
| C5 | Target-derived gating inferior temporal (specific in-paper foil) | (0,0) | 0.735 vs 0.812 (sensitivity-anchored) | `formal-eval-1.0` |
| C6 | K=2 yields a stable station-group partition; K>2 fragments it | Backbone54 | purity 1.000 / 0.833 / 0.695 (trainval-only indices reproduced) | `gating-analysis-1.0` + `routing-optimize-1.0` R4 |
| C7 | Partition is descriptively eastern/drier vs western/mountain station groups | composition table | Spokane + Sourdough = C1 | `gating-analysis-1.0` |
| C8 | OOS: no model transfers; 2-regime ≤ global | no-delta arms (pre-guard) | mean R² < 0 all; Δ −0.17 (2/10) | `formal-eval-2.0` |
| C9a | ECE published-router tie + metric realism (diagnostic baseline) | no-delta `as_routed` | ΔRMSE +0.00025 ns (2/5); R² invalid at σ_y ∈ [0.003, 0.008] | `formal-eval-2.1` Tables 1–2 |
| C9b | ECE v3 single-expert diagnostic spread (manual, non-deployable) | `c0_only` is the approved dry-assigned oracle; `c1_only` is its complement, using family-local indices | Pooled RMSE: V0/Backbone `c0_only` 0.057768, `c1_only` 0.192536; Guarded reverses (dry-assigned local c1 = 0.192536; complement local c0 = 0.057768). All oracle/diagnostic rows are `deployable=false`. WA means show local c1 lower on both the canonical SMAP feature (0.180078 vs 0.435981) and target (0.209373 vs 0.221648); preserve the approved salvage/Guarded family convention without treating labels as physical classes. | v1.1 T7 stdout, policy crosswalk, and routing audit |
| C9c | Historical ECE missingness-aware V0/Backbone policies | salvage 2.0 `auto_hard/auto_soft`, τ=0.10, G_API class mapped directly to local index | Historical pooled RMSE ~0.058 on salvage C0 convention; not a Guarded or universal router result | `derived_8.4-ece-router-salvage-2.0` |
| C9d | ECE v3 family-specific availability-gated evaluation, including Guarded | v1.1 `auto_hard/auto_soft`; Guarded G_API alignment selected on synthetic-SMAP-masked WA validation | V0/Backbone/Guarded: RMSE 0.057768 (SD 0.000617); selected Guarded class 0 → local c0 (0.073965 vs 0.117180 candidate RMSE); 150/150 rows gated with SMAP fully missing; ECE targets evaluation-only | v1.1 summary, `wa_calibration.csv`, `routing_audit.csv`/`.json` |
| C16 | ECE stations test missing-input routing under a narrow deployment sample (diagnostic, not causal) | Five disjoint ECE collaboration stations with native-missing SMAP router inputs | All five sites/150 rows are evaluation-only; site descriptors are descriptive and do not explain model error causally | v1.1 ECE gate audit + `station_static_features.csv` |
| C10 | ECE daily means hide diurnal/event response | n/a | hourly range 0.035 vs daily step 0.0025 | `ece-input-diagnose-1.0` |
| C12 | Design lesson: router provenance matters under the fixed benchmark | C5 + method narrative | qualitative; target-derived foil is specific, not universal | §8.1 |
| C13 | Primary router is in-situ-target-label-free but satellite-SM-informed | shared backbone + router table | SMAP features present; deployment availability required | §3/§4 |
| C14 | (NEW) Guarded ablation: consistency enforcement closes the Backbone→V0 gap | Guarded vs Backbone, same harness, paired | temporal +0.0001 (30/30); LOSO +0.018 (5/5) | `routing-optimize-1.0` T1/L1 |
| C15 | (NEW) Mechanism localized to eastern-holdout training composition | per-fold ARI + held-out shares + per-station diffs | fold ARI 0.13/0.19; held-out ≤15 rows; diffs +0.059/+0.051/+0.015 | `routing-optimize-1.0` R3/R3b/T8 |

Every draft sentence must map to a ledger row; no orphan numbers. C0/C1 must be checked against the actual submitted PDF before citation (W-new). Cross-harness rows (Guarded vs global/gating) are sensitivity-anchored, never paired — the only paired in-harness comparison is C14.

---

## 6. Venue adaptability matrix

| Element | ML conf (6–8p) | Geo/AGU-style | Applied IEEE (AIIoT-like) |
|---------|----------------|---------------|---------------------------|
| §1 genealogy beats | full (2–3) | short | short, cite paper 1 |
| §4 design-alternatives | full | medium | medium |
| §4 guard spec | full (reproducibility) | medium | medium |
| §5 mechanism (T8) | full | short | short (1 para) |
| §6.5 prior-attempts table | omit or appendix | omit | optional appendix |
| §7 ECE | discussion + supp panel T7/T9 (diagnostic, not SOTA); cut both only for strict 6-page | discussion + supp (deployment venue: emphasize ECE collaboration) | include (motivation + collaboration differentiator) |
| Stats detail | protocol emphasis | metrics emphasis | protocol lighter |
| OOS limitation | 3 sentences | 1 short para | 1 short para |
| Related work | MoE-heavy | SM-heavy | balanced |

---

## 7. Works to do (ordered)

- **W0′** Freeze v7 claim scope: primary = Guarded (0,0) no-delta; V0 = appendix sensitivity; middle-path motivation depth; ECE = diagnostic discussion + supplementary panel (NOT a SOTA claim); use 1.1 family-specific routing/calibration evidence and leave `outline-v2.md` through `outline-v6.md` unchanged. *(decision, no code)*
- **W-new** Open `paper/*.pdf` and confirm whether three-regime/oracle/limitation content is in the **submitted** manuscript (vs only `writeup/`); record which C0/C1 sentences are safely citable to paper 1.
- **W1** Write full claims ledger → `writeup2/claims-ledger.md` from `routing-optimize-1.0` (T1/L1/T8/R3/R3b) + `formal-eval-1.0` / `2.0` / `2.1` + `ece-router-salvage-1.1/2.0` + `paper2-final-evidence-1.1` + `gating-analysis` + `ece-input-diagnose` READMEs (stdout-only numbers for our results; paper-1 citations for C0/C1). Use C9a–C9d + C16; distinguish sensitivity-anchored from paired rows; mark every oracle/diagnostic `deployable=false`; document the WA-selected Guarded G_API mapping and its validation provenance.
- **W2** Related-work search + BibTeX: include the direct soil-moisture MoE papers, clustered soil-moisture models, seasonal-regime work, and hydrology MoE/router papers listed in §2; verify DOI/venue metadata and tag venue-fit. Add one positioning sentence for the ECE collaboration differentiator (live in-situ deployment no SOTA benchmark offers) without overstating novelty.
- **W3** Figure notebooks: (a) main `notebooks/experiment/paper2-final-evidence-1.0/` producing F1/T8; (b) `paper2-final-evidence-1.1/notebooks/paper2_ece_supplement.ipynb` producing T7/T9/F5 from v1.1 CSVs + `station_static_features.csv`; execute with `nb execute --uv`, retain figures in the evidence bundle, and fill outline numbers only from executed notebook stdout.
- **W4** ECE inclusion decision (PI, v5 reframing): default = INCLUDE §7 (extended to ~0.75–1 page) + supplementary T7/T9 panel as diagnostic exhibit. Cut both only for a strict 6-page venue — then the paper falls back to the v4 tie-only caveat paragraph. Same pass: include T0/§6.5 or not.
- **W4′** *(completed in `paper2-final-evidence-1.1`)* Re-run the four-family ECE v3 suite with five seeds and the comparable policies; preserve the V0/Backbone salvage G_API mapping and include `Global_Single_54` as a direct reference; select Guarded’s G_API class-to-local-expert mapping on synthetic-SMAP-masked WA validation only, and export the ECE summary/predictions/calibration/audit plus T7/T9/F5 figures. ECE targets are evaluation-only. Direct Guarded-vs-global paired LOSO remains a separate request.
- **W4″** *(terminology updated in v7)* Name the five ECE collaboration in-situ stations and `derived_8.4_ece_v3` split explicitly; keep local c0/c1 separate from G_API class labels and from physical dry/wet interpretation.
- **W5** Create/refresh `writeup2/` skeleton: use `outline-v7.md` as the active outline; retain `outline-v2.md` through `outline-v6.md` unchanged for history; add `claims-ledger.md`, `figures/`, and later `main.tex` or `main.md`.
- **W6** Draft §3–§6 (method + results) first — fully supported subject to the stated pending items; then §1 genealogy + §2, §5, §8–10. Guard motivation kept to one paragraph + T8 in all drafts; ECE diagnostic kept to §7 + supp (never promoted into §6 headline tables); expand only on PI request.
- **W7** PI review pass against ledger + title-route pick (A/B/C); then venue-specific trimming via matrix §6. Explicit PI question: keep supp ECE panel for all venues except strict 6-page, or cut everywhere?
- **W8** Reproducibility and framing gate before submission: re-execute `routing-optimize-1.0` (routing + T1/L1 cells), verify both evidence notebooks and CSV lineage, and confirm every outline result equals notebook stdout; audit router feature availability, WA-only mapping calibration, K-selection provenance, and test-set reuse disclosures; verify `outline-v2.md` through `outline-v6.md` are unchanged; run `make lint` / `make test` if src is touched (expected: no).

Not doing (this paper): presenting ECE as a SOTA-beating claim; presenting `c0_only` as deployable; station-count ablation, OOS full section, broad architecture redesign, edits to `writeup/` or `paper/`, archived notebooks, re-running historical v19–v25 gates, a second guard variant campaign (StationMeanV-B stays a reported null).

---

## 8. Risks

- **R3 (v5 DOWNGRADED):** ECE section distracts or invites reviewer attack — mitigate: ECE is now explicitly diagnostic + supplementary (never in §6 headline tables, never a SOTA claim); oracle/deployable split stated in §7 lead; n = 5 + 30-day one-sided caveats written into the subsection, not buried; cut option retained only for strict 6-page venues (W4).
- **R1:** LOSO non-significance misrepresented — mitigate: fixed wording in T4 caption.
- **R2:** K=2 or “regimes” read as universal — mitigate: use regional strata terminology, pretest K-selection disclosure, and explicit regional scope.
- **R4:** Backbone and some model choices selected using test — reviewers may challenge absolutes and winner ranking — mitigate: mandatory selection-bias caveat in §3, §6.4, and §9; no claim that shared features preserve unbiased relative comparisons.
- **R8:** Direct literature overlap makes an architecture-novelty claim untenable — mitigate: frame the paper as a controlled router-provenance study and cite direct soil-moisture MoE work in §2.
- **R9:** SMAP features make “target/proxy-free” inaccurate — mitigate: use in-situ-target-label-free/satellite-SM-informed language and include the provenance table.
- **R10 (v7 updated):** Site descriptors or station identity proxies can look like transferable regimes — mitigate: describe ECE location and errors descriptively; do not attribute model performance to geography or covariate proximity without independent study.
- **R15 (NEW in v5):** ECE oracle (`c0_only`) misread as deployable — mitigate: mark all manual one-expert oracle and complementary diagnostic rows `deployable=false`; describe `auto_hard/auto_soft` as availability-gated G_API policies with family-specific WA calibration, not a general fix.
- **R16 (v7 updated):** Family-local expert labels may be conflated with G_API class numbers or physical wet/dry states — mitigate: report both mapping crosswalks, display validation source and canonical/target means, and use index-qualified or family-convention wording.
- **R17 (NEW in v7):** Guarded mapping selection on Washington validation may overfit that calibration sample — mitigate: disclose both candidate scores and the selection rule, keep ECE strictly evaluation-only, and avoid claims beyond these five stations.
- **R11:** Causal geoscience interpretation exceeds team expertise — mitigate: report descriptive associations only and require subject-matter validation before strengthening claims.
- **R5 (v3):** Cross-dataset number confusion (paper 1's 0.822 vs `derived_8.4`) — label the prior-paper value as historical context from a different dataset and protocol; do not compare or co-rank it with current results.
- **R6 (v3):** Over-attributing unpublished internal failure numbers as if peer-reviewed — mitigate: cite paper 1 for what it actually contains (W-new); label internal numbers "prior/internal analysis" or omit from camera-ready.
- **R7 (v3):** Sequel narrative reads as dumping on paper 1 — mitigate: tone = "open question left by prior work," contributions credit paper 1 for pipeline + diagnosis.
- **R12 (NEW in v4):** Guard reads as post-hoc engineering to recover V0 numbers — mitigate: state that promotion bars were fixed before the follow-up run, but the benchmark test had already informed feature/model choices; the pre-specification limits within-run tuning and does not make results independent confirmation. Use ablation + mechanism tables to show where the guard acts.
- **R13 (NEW in v4):** Guard reads as station-identity routing — mitigate: §4 station_id disclosure (consistency over an unsupervised partition, target untouched) + LOSO-unseen defense (held-out stations never in fit quantities); never claim regime discovery from purity.
- **R14 (NEW in v4):** Anchor correction (guards → 0.814334) reads as moving goalposts — mitigate: §9 footnote with the pre-specified comparison rationale (guards provably reproduce the V0 test partition, §R2 ARI = 1.0, established before the follow-up training); both anchors are prior published numbers recorded in `config.yaml`. Keep this separate from the test-reuse limitation in §9.

---

## 9. Primary source map (where every number lives)

| Source | Role in paper |
|--------|---------------|
| `paper/*.pdf` | Paper 1 — foundations citation; prior regime analysis (verify content) |
| `writeup/sections/three_regime/*` | Detailed prior 3-regime narrative (cross-check vs submitted PDF) |
| `writeup/sections/base_model/*`, `writeup/sections/shared/*` | Paper-1 pipeline/features/data prose to cite, not copy |
| `src/pipeline/README.md`, `src/pipeline/main.py` | Current ingestion-to-feature-processing order for the compact §3 workflow summary |
| `docs/gating.md` | Internal history of threshold/hard/soft/classifier gates (understanding; cite only if venue accepts tech report) |
| `docs/plans/20260703-why-MoE.md` | Motivation background only (re-verify any number before quoting) |
| `docs/plans/20260721-remove-incomplete-stations.md` | Dataset pruning: incomplete stations |
| `docs/plans/20260726-stations-removal.md` | Station inclusion/exclusion decisions |
| `notebooks/experiment/derived_8.4-formal-eval-1.0/` | Sensitivity rows: temporal + LOSO formal stats vs global/gating/deltas (V0 arms) |
| `notebooks/experiment/derived_8.4-routing-optimize-1.0/` | Primary results: guard spec, R1–R4 agreement/K-sweep, T1 temporal, L1/T8 LOSO (slurm job 2155747) |
| `notebooks/experiment/derived_8.4-gating-analysis-1.0/` | K-sweep, purity, regime composition, separating features |
| `notebooks/experiment/derived_8.4-regime-interpretation-1.1/` | Method narrative, physical interpretation of regimes |
| `notebooks/experiment/derived_8.4-eval-1.3/` | LOSO protocol provenance, station difficulty (secondary) |
| `notebooks/experiment/derived_8.4-formal-eval-2.0/` | OOS numbers for Limitations only (pre-guard rows) |
| `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/` | ECE tie rows + metric realism (§7 baseline half) |
| `notebooks/experiment/derived_8.4-ece-router-salvage-1.1/` | ECE oracle/poison rows C9b + v3 station tables + routing audit (supp panel) |
| `notebooks/experiment/derived_8.4-ece-router-salvage-2.0/` | ECE deployable rows C9c + WA-only gate/temperature calibration (supp panel) |
| `notebooks/experiment/paper2-final-evidence-1.1/` | Re-run missingness-aware ECE v3 results for V0/Backbone/Guarded plus the direct Global reference; WA-only Guarded G_API mapping calibration, audit, and executed T7/T9/F5 notebook |
| `data/splits/derived_8.4_ece_v3/station_static_features.csv` | ECE-vs-train site-descriptor strip (elevation/precip) for T9 |
| `notebooks/experiment/ece-input-diagnose-1.0/` | ECE sensor/benchmark validity diagnosis (§7 if kept) |
| `data/splits/derived_8.4/split_meta.json` | Split definition for §3 |
| External direct-overlap literature | Related-work records listed in §2; verify DOI/venue metadata in W2 |
