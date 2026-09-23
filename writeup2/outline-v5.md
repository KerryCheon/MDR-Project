# Second Paper Outline (v5) — Covariate-Defined Regional Experts for Soil-Moisture Estimation

Status: active draft outline for PI · Split: `derived_8.4` only · Scope: Washington (regional) · CS/IEEE venue TBD
Historical predecessors: `outline-v2.md`, `outline-v3.md`, `outline-v4.md` (all kept unchanged) · Status of this file: active draft · v5 keeps the v4 guarded backbone router as primary and reframes §7 ECE as a diagnostic routing-failure exhibit (discussion + supplementary panel, explicitly NOT a SOTA claim), adds the Guarded-on-ECE run to W4′ scope, and normalizes dry/lowland vs wet-mountain terminology.

This document is the technical-report backbone for writing the full paper without manually digging through experiment notebooks. Every number below is already present in an executed report notebook README (stdout tables), in paper 1, or in repo docs explicitly marked as historical/prior-work sources; the claims ledger (W1) freezes the mapping before drafting.

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
2. **No regime-specific feature selection.** Every configuration (global and all routers) uses the identical shared 54-feature backbone. Delta-feature arms (c0/c1) are OUT of the main claim; at most a one-paragraph robustness note that historical test-selected deltas barely change results (0.8126 vs 0.8118) and val-selected deltas fail.
3. **Venue-agnostic outline.** Sections written so the same material works for:
   (a) ML conference (NeurIPS/ICML workshops, AAAI/ACL-app tracks, KDD application),
   (b) AI-for-science / geoscience ML venue (AGU Fall Meeting abstract → paper, IEEE IGARSS/CIKM application track),
   (c) IEEE AIIoT-style applied venue (same family as paper 1).
   Length knobs marked per section (short = 6–8 pp; full = 10–12 pp).
4. **ECE sensors: diagnostic discussion + supplementary panel (v5 reframing of the v4 conditional).** Default = extended Discussion subsection (§7, ~0.75–1 page) + supplementary results panel (T7-promoted/T9), explicitly framed as a routing-failure diagnostic enabled by the interdisciplinary ECE collaboration — NOT a SOTA-beating claim. Core message: geography misleads, covariate proximity governs (lowland ECE is covariate-close to the eastern dry expert despite western geography). `c0_only` is reported strictly as a manual oracle ceiling (`deployable=false`); the only deployable ECE claim is the WA-calibrated availability-gated `auto_hard/auto_soft` fix. Cut §7 + supp panel entirely only for a strict 6-page venue (see W4).
5. **OOS/out-of-state: Limitations only** (one paragraph + numbers), per prior decision.
6. **Sequel positioning to paper 1 (NEW in v2).**
   - Paper 1 = *Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather-Driven Machine Learning* (submitted IEEE manuscript, `paper/`).
   - Cite paper 1 for foundations: dataset preparation, feature-selection pipeline, preprocessing, global single-regime baseline performance, and (if present in the submitted PDF — verify in W-new) the prior three-regime/oracle analysis and gating limitations.
   - This paper does NOT re-litigate those results; it answers the open question paper 1 left: *what routing signal actually makes regime specialization deployable?*
   - Historical internal experiments (gating v19–v25, threshold recalibrations, adapted-training) are prior-work context only — never mixed into our result tables as comparable rows across dataset versions.
7. **Design principle as a first-class contribution (NEW in v2).**
   - Failed family A: route on the target or a proxy of it (thresholds on true/predicted SM, classifier trained to reproduce regime labels derived from y) → circular + misrouting compounds through hard gates.
   - Failed family B: supervised/heuristic routers on weak signals (learned classifier, seasonal-only, single-index) → unstable across splits (our in-paper foil: `Trained_Gating_k2`).
   - Working family C: **unsupervised routing on the shared covariate backbone** (KMeans) → never sees the target at routing time; yields station-pure, physically interpretable regimes.
   - Headline framing: "route on covariates, not on the target."
8. **Prior-attempt depth in the paper body (NEW in v2, default).**
   - Intro: one tight paragraph on the genealogy (global ceiling → 3-regime MoE idea → deployable routing failed → paper 1 shipped global).
   - Optional appendix/context table ("prior design attempts", dataset-labeled, not ranked against current results) — include only if venue has room; default = omit from main results, keep in outline as available material.
   - All historical numbers attributed to paper 1 or clearly marked as internal prior work with dataset version tags (e.g. `derived_9.0`, `derived_8.1_pos`).
9. **Guarded primary router with pre-registered bars (NEW in v4).**
   - Primary config = `Guarded_Backbone54_k2 (0,0)`: KMeans on the shared 54-feature backbone plus a deterministic station-consistency guard (fit-frame majority vote, §4). No hand-picked router feature subset, no legacy 50-feature import.
   - The guard design and its promotion bars (temporal within ±0.003 of 0.8118; LOSO within ±0.010 of 0.6372) were frozen in the W2 spec *before* the training runs; both bars pass (temporal +0.00004, LOSO +0.0007). State this once, in §4, to pre-empt post-hoc-engineering readings (R12).
   - `Clustering_V0_Full_k2` is demoted to a sensitivity reference (appendix row): same full-trainval partition (ARI = 1.0), marginally different LOSO behavior under per-fold refits. It is never co-ranked as a competing method.
   - Motivation depth is middle-path by decision: full method spec (reproducibility) + compact motivation (one paragraph + T8 mechanism table). No multi-paragraph design genealogy; no silent rename.

---

### v3 clarifications retained, v4 deltas marked

- **Controlled router design, not novel architecture:** describe the implemented model as a hard-gated, cluster-wise XGBoost mixture-of-experts formulation. The contribution is the controlled router comparison, not a new MoE architecture. *(v4: the guard is a deterministic consistency rule inside the router, not an architecture claim — say so once in §4.)*
- **Primary routing provenance:** the main covariate router is in-situ-target-label-free, but it is not proxy-free: the shared 54-feature backbone includes current, lagged, and rolling SMAP-derived soil-moisture features.
- **Empirical comparison:** hold the expert type, shared backbone, split, and evaluation protocol fixed while comparing target-derived, heuristic, and covariate-defined routers. *(v4: Guarded vs unguarded-Backbone ablation is same-harness and paired; Guarded vs global/heuristic rows are sensitivity-anchored to `formal-eval-1.0` — never present cross-harness numbers as paired.)*
- **Narrow supervised-router claim:** treat the v2 supervised/heuristic failure language as historical motivation. The current negative result applies specifically to the tested `Trained_Gating_k2` router trained from an in-situ target threshold; do not generalize it to supervised routers as a class.
- **Terminology and interpretation:** use “covariate-defined regional strata” for the current partition. Station purity, east/west composition, and feature separation are descriptive associations; they do not establish causal climate or hydrologic mechanisms without subject-matter validation. *(v4: purity on known stations is by-construction under the guard, so it is a weaker diagnostic than in v3 — say so in §5; the transferable evidence is the LOSO behavior on unseen stations.)* *(v5: normalized expert labels are “dry/lowland expert vs wet-mountain expert” — eastern stations Spokane + SourdoughGulch vs five western/mountain stations on first use, short labels after. Record the label-flip once in §4: salvage notebooks call the dry expert `C0`, while the Guarded canonicalization calls the drier regime `c1` (lower `SMAP_sm_pm_interp_rollmean30`); remap before any cross-experiment comparison.)*
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
3. Result temporal: R² 0.812 vs 0.780 (RMSE 0.044 vs 0.048), 30 seeds, p < 1e-12, sample bootstrap ΔR² +0.035 [0.019, 0.055], p = 0.0005. *(v4: Guarded (0,0) 0.8118 ± 0.0014 — ties the legacy V0 partition result within 0.0001; seed-42 reproduces eval-1.1's V0-backbone 0.814334 exactly.)*
4. Result spatial (in-state LOSO): mean R² 0.64 vs 0.58, wins 6/7 stations (descriptive). *(v4: Guarded 0.638; 6/7 figure is the V0-anchored sensitivity row — Guarded-vs-global win counts were not run in-harness; report Guarded-vs-Backbone instead: gains on 3/7 folds, ties on 4/7, T8.)*
5. Analysis: the selected K=2 partition is stable and station-pure on this dataset; per-sample KMeans fragments stations under eastern-holdout refits while the guard restores station-pure specialist training sets; the target-derived `Trained_Gating_k2` foil underperforms, while covariate-defined regional strata produce the strongest current result.
6. Scope: this is a Washington regional study, not evidence that K=2 or this router transfers universally; out-of-state transfer fails for all tested models.

---

## 3. Section-by-section outline

Each section lists: goal · claims · numbers to quote · source (experiment path + table) · figures · venue knobs · open writing tasks.

### §1 Introduction [all venues] — router-design problem genealogy

- **Beat 1 — What paper 1 established.** Dataset preparation, feature pipeline, and a strong single-regime global XGBoost model established the baseline; prior error analysis motivated specialization. Cite paper 1 only for claims verified in the submitted manuscript.
- **Beat 2 — Why router provenance matters.** Earlier target-threshold and target-derived routing attempts show why oracle specialization is not the same as deployable specialization: labels derived from the in-situ target are unavailable at inference and hard gates amplify boundary errors.
- **Beat 3 — What this paper studies.** On the clean 7-station `derived_8.4` split, hold the prediction backbone and XGBoost experts fixed while comparing target-derived, seasonal/API, dynamic, and covariate-defined routers. *(v4: add one sentence — the primary covariate router carries a station-consistency guard so the model is describable purely from the shared backbone, with no legacy feature set to port to new regions.)*
- **Contributions:**
  - C-A: A hard-gated, cluster-wise XGBoost formulation for regional soil-moisture estimation, treated as an MoE formulation rather than a novel MoE architecture.
  - C-B: A controlled comparison of router provenance under shared features, multi-seed temporal evaluation, in-state LOSO, and block bootstrap.
  - C-C: An empirical result that the selected covariate-defined regional partition improves temporal performance and wins most in-state LOSO station comparisons.
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
- **Pillar 5 — Evaluation practice.** Position LOSO, multi-seed inference, block bootstrap, feature provenance, target-label leakage, and deployment-availability checks as the gap addressed here.
- **Positioning sentence:** the reviewed literature establishes that soil-moisture MoE and cluster-specific modeling already exist; this paper studies how router provenance affects hard-gated regional XGBoost performance under a fixed, leakage-audited evaluation protocol.
- **Open task W2:** create verified BibTeX records and preserve DOI/venue metadata for every direct comparator; no citations invented in the outline.

### §3 Problem setup & data [all venues] — one added line

- `derived_8.4`: 7 WA stations; train 2017–20 (9,803) / val 2021–22 (4,805) / test 2023–25 (6,620); station list with one-line descriptors (east/west, elevation).
- Stations: BeaverPass_WA_990, CayusePass_WA, Darrington, Paradise_WA, Quinault, SourdoughGulch_WA_985, Spokane.
- **Dataset evolution paragraph:** earlier expansion roughly doubled station count; many added stations removed for data-quality issues and because the current feature set lacks snow/SWE variables needed for snowpack-dominated sites (alpine stations out of scope; cite `docs/plans/20260721-remove-incomplete-stations.md`, `docs/plans/20260726-stations-removal.md`). Result: clean, deployment-relevant 7-station split. One sentence: snowpack regimes are future work (SWE/Snow Depth features), not a silent omission.
- Feature backbone: 54 shared features (identical for every model) — name the source (`derived_8.4-feature-selection-2.0` / eval-1.1 `selected_features.json`); foundations of the selection pipeline cite paper 1; state that the backbone was fixed before this comparison and shared across arms (relative claims).
- *(v4 NEW one-liner):* the primary router uses this identical backbone — there is no separate router feature source to justify or port (the legacy 50-feature V0 set survives only as a sensitivity reference, §4).
- Target: `soil_moisture_5cm`; metrics: R², RMSE, MAE, bias (+ Pearson where useful).
- **Cross-dataset caution (R5):** any comparison to paper 1's 0.822 must note different split/dataset version/protocol; never place it in the same ranking table as `derived_8.4` rows without a footnote.
- Sources: `data/splits/derived_8.4/split_meta.json`; `derived_8.4-formal-eval-1.0` protocol section; `derived_8.4-gating-analysis-1.0` feature-list tables; paper 1 §data/features; station-removal plans.

### §4 Method: hard-gated cluster-wise XGBoost (MoE formulation) [all venues]

- Architecture: router → hard assignment → one XGBoost expert per assigned group; inference activates one expert and a small router. The formulation is an MoE, but the paper does not claim a new MoE architecture. The station-consistency guard is a deterministic rule inside the router, not an architecture claim.
- **Router alternatives and provenance:**
  - A1: Target-derived threshold gate on `soil_moisture_5cm < 0.16` — the in-paper `Trained_Gating_k2` foil; unavailable without the in-situ target and not deployable as an estimator router.
  - A2: Heuristic routers — `Seasonal_Binary_k2` and `Univariate_G_API_k2`; inference-available but limited state/season rules.
  - A3: Covariate routers — `Clustering_Dynamic_k2` and unguarded `Clustering_Backbone54_k2`; fitted without the in-situ target and applied to inference-available inputs. Unguarded-Backbone is the in-harness ablation for the guard (same features, same protocol, paired seeds).
  - A4 (primary): `Guarded_Backbone54_k2` — KMeans on the shared 54-feature backbone plus a station-consistency guard. Guard semantics in three lines: (1) fit KMeans on the fit frame only (mean-impute → StandardScaler → KMeans(2, seed 42)); canonicalize labels so c1 is the drier regime; (2) known stations receive their fit-frame majority label (deterministic tie-break); (3) unseen stations (held-out LOSO folds, new deployments), missing ids, and availability-gated rows fall back to the per-sample KMeans prediction, with a fit-frame-calibrated margin flag available for global-expert fallback. Label it **in-situ-target-label-free, satellite-soil-moisture-informed covariate routing**, not proxy-free routing.
  - Sensitivity (appendix row, not a competing method): `Clustering_V0_Full_k2` on the legacy 50-feature set — same full-trainval partition (ARI = 1.0), kept only to show the primary result does not depend on the legacy features.
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
- **Pre-registration note (NEW, R12):** the guard design and its promotion bars were frozen before the training runs (temporal ±0.003, LOSO ±0.010); both pass. One sentence, in §4, then move on.
- Router families compared: the six v3 configurations plus the guarded primary; V0 appears only as the appendix sensitivity row.
- Training protocol: routers fit on trainval only; experts per group; 30 temporal seeds; LOSO 5 seeds × 7 folds with per-fold refitting. State exactly which components are stochastic.
- Statistics: retain seed-level confidence intervals, paired tests, BH-FDR, station-month bootstrap, LOSO win counts, and the explicit limitation that seed variation measures expert fitting stochasticity only. *(v4: bootstrap/pairwise rows against global and gating are sensitivity-anchored to `formal-eval-1.0`'s V0 arms — Guarded ties V0 within 0.0001, so rounded values transfer; the only paired in-harness test is Guarded vs unguarded-Backbone.)*
- **K-selection disclosure:** select K using train/validation or predeclared clustering diagnostics, then reserve test for final evaluation. *(v4: the routing-optimize K-sweep reproduces gating-analysis indices exactly on trainval — cite as the per-region recomputable procedure.)*
- **Feature-availability disclosure:** distinguish in-situ target labels from satellite soil-moisture proxies and state whether each input exists at intended deployment time.
- No claim is made that hard routing is superior to soft routing; the paper compares the implemented hard-gated router families only. Margin/availability fallback machinery is reported as implemented-but-advisory in one sentence (it never triggers on WA data).
- Sources: `derived_8.4-formal-eval-1.0`; `derived_8.4-routing-optimize-1.0` (guard spec, agreement + training runs); `derived_8.4-gating-analysis-1.0`; `eval11/routers.py`; `derived_8.4-regime-interpretation-1.1`; direct literature listed in §2.

### §5 Partition diagnostics: covariate-defined regional strata

[ML/IEEE venues: 0.5–1 page · geoscience venues: expand only with subject-matter review]

- K=2 purity = 1.000 trainval and test **on known stations by construction under the guard**; report this as an enforced consistency property, not as discovered regime evidence. The transferable diagnostic is unseen-station behavior (LOSO), not purity itself.
- Partition composition: cluster 1 = Spokane + SourdoughGulch; cluster 0 = five western/mountain stations. Use descriptive labels "dry/lowland expert (eastern stations Spokane + SourdoughGulch)" and "wet-mountain expert (five western/mountain stations)" on first use, short labels after. *(v5: link once to ECE site descriptors — ECE Bellevue/Renton sensors at 51–157m elevation, `bio12` ~1018–1227mm are lowland deployment sites covariate-close to the dry/lowland expert despite western geography, and far from the alpine western-mountain training stations BeaverPass 1125m / CayusePass 1588m / Paradise 1564m; full elevation/precip numbers live in the supp panel, not the main text.)*
- **Fold-instability mechanism (NEW, compact — the motivation exhibit):** per-sample KMeans on either feature set fragments stations when an eastern station is held out (fold-trainval V0-vs-Backbone ARI 0.13/0.19 on Sourdough/Spokane folds; identical ARI = 1.0 on western-holdout folds), while held-out assignment is near-identical across routers (≤15 rows, R3b) — so the LOSO gap comes from specialist-training composition, not misrouting. The guard restores station-pure training sets by construction (Guarded-vs-Backbone fold ARI 0.33–0.35 on those folds). Three sentences max; T8 carries the numbers.
- Top separating features include static/site descriptors, SMAP rolling statistics, and LST/NDMI windows. Treat these as associated partition indicators, not causal drivers.
- Correlation drift, including `SMAP_sm_pm_interp_rollmean30`, is evidence that the pooled mapping may differ across strata; it does not establish a physical mechanism.
- Quality indices: Calinski-Harabasz and Davies-Bouldin favor K=2 while silhouette is marginally higher at K=3. Internal indices alone do not select the meaningful partition.
- Explicit caveat (strengthened, R10; v5 softened to mixed transfer): station purity partly reflects site descriptors or station identity proxies, and the guard now enforces it explicitly — purity diagnoses specialization on this dataset but may reduce transferability to unseen stations. ECE provides mixed rather than purely negative counterevidence: the dry/lowland expert transfers to lowland ECE under forced routing while the static router misroutes there (see §7) — so keep OOS failures as the hard limitation and ECE as the conditional-transfer diagnostic.
- Contrast with the target-derived gate: target labels create wet/dry separation but do not yield stable station groups and are unavailable at inference.
- Sources: `derived_8.4-gating-analysis-1.0`; `derived_8.4-routing-optimize-1.0` (R2/R3/R3b/R4); `derived_8.4-regime-interpretation-1.0/1.1/1.2`; feature provenance in §4.

### §6 Results [core of every venue]

#### 6.1 Temporal (primary) — no-delta only

- Headline table (rank by RMSE): at minimum
  - `Guarded_Backbone54_k2` (0,0): R² 0.8118 ± 0.0014, RMSE 0.0442 *(NEW primary; 30 seeds, slurm job 2155747)*
  - `Global_Single_54`: 0.7798 ± 0.0013, RMSE 0.0478 *(sensitivity-anchored to formal-eval-1.0)*
  - `Trained_Gating_k2` (0,0): 0.7354 ± 0.0011 *(sensitivity-anchored to formal-eval-1.0)*
  - plus Dynamic / Seasonal / Univariate no-delta rows for the full router comparison *(all sensitivity-anchored to formal-eval-1.0)*
  - Appendix row: `Clustering_V0_Full_k2` (0,0) 0.8118 (same rounded value; ties Guarded within 0.0001) and unguarded `Backbone54` (0,0) 0.8117 (in-harness ablation: Guarded better on 30/30 seeds by +0.0001, honestly a tie).
- Pairwise (sensitivity-anchored to `formal-eval-1.0` V0 arms; Guarded ties V0 within 0.0001 so rounded values transfer): clustering vs global ΔR² +0.032 [0.031, 0.033], p < 1e-12, 100% seeds; clustering vs trained gating +0.089. Bootstrap (station, month): ΔR² +0.035 [0.019, 0.055] p = 0.0005; RMSE −0.0040 p = 0.0005.
- **Router-design sentence:** specialization is not sufficient by itself; under this fixed expert/evaluation setup, the tested covariate-defined regional partition outperforms the global baseline, while the target-derived gate underperforms. This is evidence about router provenance on this benchmark, not a universal ranking of supervised versus unsupervised routers.
- Seed-42 note (one line): Guarded seed-42 reproduces eval-1.1's V0-backbone 0.814334 exactly — the guard recovers the legacy-partition number on shared-54 features.
- Optional footnote only (not a table row): paper 1 reported 0.822 under a different dataset/split — not directly comparable (R5).
- Source: `derived_8.4-routing-optimize-1.0` T1 (primary + ablation) + `formal-eval-1.0` temporal seed table (sensitivity rows).
- Figure F1: bar/dot temporal R² with 95% CI across ~8 configs (NEW figure notebook).

#### 6.2 In-state LOSO (secondary, descriptive)

- No-delta: `Guarded_Backbone54` 0.638 vs `Global_Single_54` 0.580 (sensitivity-anchored) and `Baseline_V0_50` 0.591; unguarded `Backbone54` 0.620 in-harness. V0 sensitivity 0.637.
- **T8 (NEW, the mechanism table):** Guarded-minus-Backbone per-held-out-station mean R² diff — SourdoughGulch +0.059, Spokane +0.051, Paradise +0.015, all other folds 0.000; Guarded wins 5/5 seeds overall.
- 6/7-vs-global figure retained ONLY as the V0-anchored sensitivity row (sign p = 0.125 — state underpowered); the in-harness claim is Guarded-vs-Backbone (gains on 3/7 folds, ties on 4/7). If PI wants direct Guarded-vs-global win counts, that is a flagged follow-up run (Global_Single_54 was not in-harness in routing-optimize-1.0).
- Station difficulty table (7 stations, median LOSO R² across configs): Darrington 0.70, BeaverPass 0.69, Spokane 0.65, Paradise 0.59, Quinault 0.56, SourdoughGulch 0.43, CayusePass 0.39. *(unchanged; from formal-eval-1.0/eval-1.3)*
- Explicit sentence: LOSO supports consistency, not statistical significance at n = 7.
- Source: `derived_8.4-routing-optimize-1.0` L1 + T8; `formal-eval-1.0` LOSO summary (sensitivity rows); `derived_8.4-eval-1.3` for difficulty narrative if needed.
- Figure F2: per-station paired R² (existing `loso_pair` / `loso_station_bars` PNGs).

#### 6.3 Router comparison (the multi-regime story)

- Ranking message: the guarded full-covariate partition is strongest in this dataset (ties the legacy V0 result with no legacy features); heuristics and the target-derived gate are weaker under the same protocol. The ablation reading: consistency enforcement, not a new signal, closes the Backbone→V0 gap (+0.018 LOSO). Do not generalize this single target-derived foil to supervised routing as a class.
- Optional single table merging 6.1 + 6.2 columns (temporal R² | LOSO R²).

#### 6.4 Robustness note (1 paragraph, not a section)

- No-delta ≈ historical test-delta winner (0.8118 vs 0.8126) → headline is not a delta-selection artifact.
- Val-selected deltas historically UNDERPERFORM global (0.735 vs 0.780) → selection instability exists in this benchmark; our simplified no-delta protocol avoids it.
- Optional: one sentence that the shared backbone was itself test-era-selected (shared by all arms; absolute numbers optimistic, relative comparison intact).
- Source: `derived_8.4-formal-eval-1.0` delta-robustness table + caveat bullets.

#### 6.5 (Optional, venue-dependent) Prior-attempts context material

- NOT a ranked results row. If included: small appendix table listing prior design attempts (threshold hard/soft, supervised classifier, adapted training, threshold recalibration, dataset expansion) with **dataset version tags** and qualitative outcome (worked / failed / partially), sourced from paper 1 + internal docs — purpose is narrative honesty, not benchmarking.
- Default: omit from main paper; keep outline entry for PI option (fold into W4-style venue decision).

### §7 Deployment diagnostic — ECE sensors (v5 reframing; was "deployment stress test" in v3/v4)

[Default INCLUDE as extended Discussion subsection (~0.75–1 page) + supplementary results panel T7/T9; cut both only for a strict 6-page venue — see W4. Explicitly diagnostic, NOT a SOTA-beating claim.]

- Purpose: (i) a real new-deployment transfer attempt on unseen in-situ sensors made possible by the ECE collaboration, (ii) what breaks when you evaluate there, and (iii, NEW in v5) a routing-failure isolation exhibit: geography misleads, covariate proximity governs. The geographically-near western-mountain expert is OOD for lowland ECE; the geographically-far eastern dry/lowland expert transfers under forced routing.
- What we can honestly say:
  1. Protocol: train on 7 WA trainval (14,608 rows, 2017–2022); evaluate 5 ECE stations on the canonical `derived_8.4_ece_v3` split (150 rows, 2026-07-20..08-19, ~30-day late-summer dry-down window, strict native-NaN SMAP). ECE strictly unseen in all fits, thresholds, and gate calibration.
  2. Outcome, two parts: (a) published-router configuration ties: two-regime ≈ single-regime on RMSE (`formal-eval-2.1-ece-v3` Table 2: `Clustering vs Global ΔRMSE +0.00025, 2/5 wins, ns`; all R² deeply negative because σ_y ∈ [0.003, 0.008] — rank by RMSE/bias/ubRMSE, not R²); (b) routing-isolation configuration separates sharply (`ece-router-salvage-1.1` v3 stdout: pooled RMSE `as_routed 0.165/0.166 → c0_only dry 0.059 → c1_only wet 0.191`; deployable WA-calibrated `auto_hard/auto_soft 0.058` ties the oracle; per-station e.g. `BBG_Lost_Meadow 0.197→0.050`, `BBG_Main 0.190→0.051`, `Renton_Home 0.238→0.090`). Static KMeans sends `Lost_Meadow 100%` + `Renton_Home ~90%` to the wet-mountain expert, while G_API/dynamic/seasonal send 100% of ECE rows to the dry expert — experts transfer, the static router poisons.
  3. Oracle vs deployable discipline (NEW): `c0_only` is a MANUAL oracle ceiling (`deployable=false` in `salvage-2.0`), kept only to quantify the gap. The paper-claimable deployable fix is the availability-gated `auto_hard` (SMAP-block-missing OR miss-rate > 0.10 → SMAP-free `G_API` label; WA-val-tuned, ECE never touched) or `margin_fallback → Global_Single_54` in 1.x. Never present `c0_only` as a method.
  4. Sensor/benchmark diagnosis (retained from v4): hourly data show strong diurnal cycle (median hourly range 0.035 ≈ 14× median daily step 0.0025); rain falls overnight and dries by midday, so 24-h means look flat; hour-scale corr(rain, Δsensor) ≈ 0 at all lags 0–6h (`ece-input-diagnose-1.0`). So: NOT "sensors ignore rain" but "daily aggregation + short dry-down window make this set a weak benchmark for ranking daily models — but a strong instrument for isolating routing failure."
  5. Predictions collapse toward a common value across sensors with very different target means (0.018–0.198) under `as_routed` — line-chart observation (supp panel, ≤5 lines per chart).
  6. Takeaway for the paper (v5): sensor → benchmark validity checks (variance floor, event-response check, aggregation choice) should precede model ranking; AND covariate-aware + availability-aware routing should precede geographic-deployment assumptions. The ECE collaboration is the paper's differentiator vs SOTA-chasing baselines: no public benchmark offers live lowland in-situ sensors to expose this geography-vs-covariate gap.
- What we must NOT say: that two-regime wins on ECE as a SOTA claim (n = 5, ns; 30-day one-sided dry window, all transplants collapse to 100% dry); that `c0_only` is deployable; that sensors are broken; any causal claim about sensor hardware; any winter/wet-season ECE generalization.
- Decision rule W4 (v5): default = keep §7 + supp panel. Cut both only for a strict 6-page ML venue — nothing else depends on them.
- Sources: `derived_8.4-formal-eval-2.1-ece-v3` README (Tables 1–4 + time-series figs, for the tie rows); `ece-router-salvage-1.1` README (v3 pooled + station tables, for oracle/poison rows); `ece-router-salvage-2.0` README (WA calibration + `auto_hard/auto_soft` deployable rows); `ece-input-diagnose-1.0` README §4–5 + figures; new Guarded-on-ECE run (W4′, same stdout discipline) once executed.

### §8 Discussion

- **What the controlled router comparison establishes:**
  1. A router trained from an in-situ target threshold is unavailable at inference and performs poorly in this benchmark; this is a result about the tested target-derived gate.
  2. Inference-available covariate partitions can produce stable station-group specialists and improve temporal performance under the current regional split.
  3. The primary router uses SMAP-derived soil-moisture proxies, so the correct claim is in-situ-target-label-free and satellite-SM-informed, not proxy-free.
  4. Station purity on known stations is enforced by the guard, so it is a weaker diagnostic than previously presented; the transferable evidence is unseen-station LOSO behavior, and purity/associations do not prove causal climate or hydrologic regimes.
  5. Oracle upper bounds diagnose specialization potential, not deployability.
  6. *(v4 NEW)* The guard's value is portability, not performance: it recovers the legacy-partition result on shared-54 features with a per-region recomputable rule, removing the one non-automatic element (the 50-feature legacy set) from the deployment story. Guard design and promotion bars were pre-registered before training.
- **Why the target-derived gate loses in-paper:** target-threshold labels define wet/dry states that are unavailable at prediction time and may assign heterogeneous station groups to the same expert. Avoid saying that all learned or supervised routers fail.
- **Why the guard helps where it helps (compact):** eastern-holdout refits redraw specialist training sets; majority voting stabilizes them. It does not improve held-out routing (already near-identical) and does not help unseen stations directly — it protects training composition.
- **Regional strata, not universal regimes:** select K using deployment-region diagnostics, partition stability, group balance, and validation performance. Expect the useful number and meaning of strata to change across regions. New regions recompute imputation means, scaler, centroids, majority map, and margin threshold from local trainval; K=2 is not exported.
- **Deployment interpretation:** the OOS results remain essential counterevidence (regional specialization can help in-region while failing under station or distribution shift). ECE is now mixed evidence rather than pure counterevidence: the static router fails on lowland ECE (geography-misled routing into the wet-mountain expert), while the dry/lowland expert itself transfers there and a WA-calibrated availability-gated fallback recovers oracle-level error — so ECE isolates a fixable routing/availability failure, not an untransferable-expert failure. State both halves every time ECE is cited.
- **Regional scope by design:** the contribution is an empirical router-design result for Washington soil-moisture estimation, not a general hydrologic law.

### §9 Limitations [required]

- n = 7 LOSO, low power; partial 2025 coverage; seed variation does not cover router stochasticity (routers fixed); shared backbone/hyperparameters test-era-selected.
- **Guard-specific limitations (NEW):** (i) majority voting uses `station_id` at inference for known stations — legitimate as a consistency rule over an unsupervised partition, but it is station-identity-adjacent and must not be described as discovering regimes; (ii) the guard acts on training composition, not on unseen-station routing — unseen rows use per-sample predictions, so no direct transfer gain should be claimed beyond what LOSO measures; (iii) `StationMeanV-B` (0.634) trails `GuardedV-A` (0.638) and is reported as a challenger null, not a second method.
- **OOS paragraph:** train on 7 WA → 10 out-of-state stations: ALL models degrade (station-mean R² negative for every config; pooled global 0.206 vs best two-regime ~0.11–0.14; two-regime worse than global, e.g. V0_Full no-delta wins 2/10). Frame: regime specialization is a within-region prior; outside the region both experts and router are OOD — consistent with regional scope, not a hidden result. Source: `derived_8.4-formal-eval-2.0` summary + Table 1 (no-delta rows only). *(v4: OOS rows predate the guard; do not imply guard numbers there.)*
- **Snowpack / feature-space boundary:** stations under alpine snowpack dynamics removed/out-of-scope because the feature set lacks SWE/snow depth; algorithm gating cannot fix missing physics (cite station-removal plan). Future work: SWE integration.
- ECE caveats if §7 kept: 30-day late-summer dry-down window (one-sided; all dry-routed transplants collapse to 100% dry), tiny target variance (rank RMSE/bias/ubRMSE, never R²), daily aggregation hiding diurnal/event response, n = 5 stations (low power, report ns plainly), and mismatch between the trained SMAP-informed static router and unseen native-NaN SMAP in-situ inputs (hence the availability gate). `c0_only` numbers are oracle ceilings, not deployable results; Guarded-on-ECE rows (W4′) must be labeled pre-guard vs guarded and must verify the dry-expert index after canonicalization.
- **Router provenance limitation:** the primary router is in-situ-target-label-free but uses SMAP-derived soil-moisture proxies; deployment requires documenting product availability, latency, missingness, and sensor/domain mismatch.
- **Interpretation limitation:** static/site-driven purity and feature correlations are descriptive; causal geoscience interpretation requires subject-matter validation.
- **Anchor-correction footnote (NEW, transparency):** training-run anchors for the guards point at eval-1.1's V0-backbone 0.814334 (the partition the guards provably reproduce, §R2 ARI = 1.0), not Backbone's 0.814205; both values are prior published numbers, and the correction is recorded in the experiment `config.yaml`, not fitted to results.
- Historical/attribution limitation (optional one line): some prior routing results live only in an internal tech report / pre-camera-ready analysis — we cite paper 1 where possible and do not over-claim unpublished numbers.

### §10 Conclusion & future work

- Restate design lesson in one line (under a fixed hard-gated XGBoost setup, inference-available covariate-defined regional strata made specialization useful in-region, whereas the tested target-derived gate did not; a station-consistency guard keeps that result portable to shared-backbone features).
- Future work bullets (not claims): automated K selection; OOD/ambiguity fallback to global expert (diagnostics exist in `derived_8.4-formal-eval-2.0` Table 4; fallback-mask machinery from routing-optimize-1.0 is implemented but unevaluated as a policy); station-count / training-set composition ablation (explicitly future); snowpack-aware features (SWE) + expanded station types; multi-year multi-season ECE evaluation with the ECE team once sensors accumulate a full season (including wet-season routing behavior, which the current 30-day dry window cannot address) + lowland-vs-dry-expert assignment validation; revisit 3-way partitions only with a routing signal that is not derived from y; direct Guarded-vs-global LOSO win counts + Guarded-on-ECE rows (both W4′ follow-ups).

---

## 4. Figure & table plan (all via notebook, `nb execute --uv`)

| ID | Item | Status | Source |
|----|------|--------|--------|
| T0 | (optional) Prior-attempts context table, dataset-labeled | new from paper1+docs | §6.5 / appendix |
| T1 | Config table (routers × features × protocol) | new LaTeX from method text | gating-analysis + eval config |
| T2 | Temporal seed-level results (no-delta) | routing-optimize T1 (primary+ablation) + formal-eval-1.0 (sensitivity rows) | `routing-optimize-1.0` + `formal-eval-1.0` |
| T3 | Temporal pairwise (global, gating) | sensitivity-anchored rows | `formal-eval-1.0` |
| T4 | LOSO summary + win counts (no-delta) | routing-optimize L1 + formal-eval-1.0 (sensitivity rows) | `routing-optimize-1.0` + `formal-eval-1.0` |
| T5 | K-sweep partition stability/quality | existing (reproduced on trainval) | `gating-analysis-1.0` + `routing-optimize-1.0` R4 |
| T6 | Regional-strata station composition + associated features | existing | gating-analysis / regime-interpretation |
| T7 | ECE routing-isolation table (supplementary-required in v5; was optional): pooled RMSE `as_routed vs c0_only (oracle) vs c1_only (poison) vs auto_hard/auto_soft (deployable)` by family (V0/Backbone/Guarded once W4′ lands) + station RMSE columns | `ece-router-salvage-1.1` (oracle/poison) + `ece-router-salvage-2.0` (deployable) + new Guarded-on-ECE run + `formal-eval-2.1-ece-v3` Tables 1–2 (tie rows) | `formal-eval-2.1-ece-v3` |
| T9 | (NEW in v5, supplementary) ECE diagnostic panel: (a) policy RMSE bars by family, (b) per-station RMSE matrix, (c) routing-assignment strip (% rows to dry expert + WA-vs-ECE margin/ambiguity contrast), (d) elevation/precip site-descriptor strip (ECE lowland vs train-east vs train-west-mountain) | new from salvage CSVs (`summary/station_metrics/routing_audit`) + `station_static_features.csv` | supp panel |
| T8 | (NEW) Guarded-minus-Backbone per-held-out-station LOSO diff + fold-agreement summary | existing (notebook stdout) | `routing-optimize-1.0` R3/R3b/L1 |
| F1 | Temporal multi-seed dot/bar + CI | NEW | `paper2-figures-1.0` notebook |
| F2 | LOSO per-station pairs | reuse PNG | `formal-eval-1.0` |
| F3 | Regional-strata geographic map (WA) | NEW or reuse | gating geographic PNGs exist — pick/crop |
| F4 | Delta-robustness slope (test/val/none) | NEW, optional | `formal-eval-1.0` |
| F5 | ECE time-series line charts (supplementary in v5): reuse ≤5-line figs + one new Guarded overlay (observed + Guarded `as_routed/auto_hard/c0_only` + Global-54) | reuse + new (W4′) | `formal-eval-2.1-ece-v3` + salvage `predictions_v3` |
| F6 | (opt) Concept figure: router provenance families vs chosen | NEW, simple diagram | method narrative only |

---

## 5. Claims ledger (excerpt — full ledger = separate file in works)

| # | Claim | Config / source | Number | Artifact |
|---|-------|-----------------|--------|----------|
| C0 | Prior work: global baseline + regime motivation | paper 1 | R² 0.822 (paper-1 protocol) | `paper/` PDF |
| C1 | Prior work: specialization potential under oracle; deployable routing was the gap | paper 1 (if in PDF) / internal prior | oracle ≫ global; e2e routing weak | paper 1 §three_regime (verify) |
| C2 | Two-regime > global temporal (OURS) | Guarded (0,0) vs Global_54 | 0.8118 vs 0.7798; Δ +0.032 p < 1e-12 (sensitivity-anchored) | `routing-optimize-1.0` T1 + `formal-eval-1.0` |
| C3 | Sample-level significance | same | ΔR² +0.035 [0.019, 0.055] p = 0.0005 (sensitivity-anchored) | `formal-eval-1.0` bootstrap |
| C4 | Two-regime > global LOSO | Guarded (0,0) | 0.638 vs 0.580 (sensitivity-anchored); in-harness Guarded-vs-Backbone gains 3/7 folds | `routing-optimize-1.0` L1/T8 + `formal-eval-1.0` LOSO |
| C5 | Target-derived gating inferior temporal (specific in-paper foil) | (0,0) | 0.735 vs 0.812 (sensitivity-anchored) | `formal-eval-1.0` |
| C6 | K=2 yields a stable station-group partition; K>2 fragments it | Backbone54 | purity 1.000 / 0.833 / 0.695 (trainval-only indices reproduced) | `gating-analysis-1.0` + `routing-optimize-1.0` R4 |
| C7 | Partition is descriptively eastern/drier vs western/mountain station groups | composition table | Spokane + Sourdough = C1 | `gating-analysis-1.0` |
| C8 | OOS: no model transfers; 2-regime ≤ global | no-delta arms (pre-guard) | mean R² < 0 all; Δ −0.17 (2/10) | `formal-eval-2.0` |
| C9 | ECE published-router tie + metric realism (diagnostic baseline) | no-delta `as_routed` | ΔRMSE +0.00025 ns (2/5); R² invalid at σ_y ∈ [0.003, 0.008] | `formal-eval-2.1` Tables 1–2 |
| C9b | (NEW in v5) ECE routing isolation: dry-expert oracle vs wet-expert poison (manual, non-deployable) | `c0_only` vs `c1_only` vs `as_routed`, frozen experts | v3 pooled RMSE `as_routed 0.165/0.166 → dry 0.059 → wet 0.191`; e.g. Lost_Meadow `0.197→0.050`, Renton_Home `0.238→0.090` | `ece-router-salvage-1.1` stdout/CSVs |
| C9c | (NEW in v5) ECE deployable availability-gated fix ties oracle (WA-only calibration) | `auto_hard/auto_soft` (gate τ=0.10, T WA-tuned, ECE unseen) | pooled RMSE `0.058` (= oracle `0.059` within seed noise); `c0_only` marked `deployable=false` | `ece-router-salvage-2.0` stdout/CSVs |
| C16 | (NEW in v5) Covariate proximity governs ECE transfer, not geography (diagnostic interpretation, not SOTA claim) | dry/lowland expert (Spokane+Sourdough) vs wet-mountain expert on lowland ECE (51–157m) | qualitative + C9b/C9c numbers; static router misroutes Lost_Meadow 100% / Renton_Home ~90% to wet expert while dry-routed policies send 100% to dry expert | salvage routing audits + `station_static_features.csv` |
| C10 | ECE daily means hide diurnal/event response | n/a | hourly range 0.035 vs daily step 0.0025 | `ece-input-diagnose-1.0` |
| C11 | Dataset evolution: expansion → prune → 7 stations (quality + snow fit) | plans | 5→13→7 (approx.; verify exact counts) | removal plans + split_meta |
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

- **W0′** Freeze v5 claim scope: primary = Guarded (0,0) no-delta; V0 = appendix sensitivity; middle-path motivation depth; ECE = diagnostic discussion + supplementary panel (NOT a SOTA claim); leave `outline-v2.md`, `outline-v3.md`, and `outline-v4.md` unchanged. *(decision, no code)*
- **W-new** Open `paper/*.pdf` and confirm whether three-regime/oracle/limitation content is in the **submitted** manuscript (vs only `writeup/`); record which C0/C1 sentences are safely citable to paper 1. Also confirm exact station counts over time for C11 (5→13→7) from split history before freezing numbers.
- **W1** Write full claims ledger → `writeup2/claims-ledger.md` from `routing-optimize-1.0` (T1/L1/T8/R3/R3b) + `formal-eval-1.0` / `2.0` / `2.1` + `ece-router-salvage-1.1` (C9b oracle/poison rows) + `ece-router-salvage-2.0` (C9c deployable rows + WA calibration) + new Guarded-on-ECE run (once executed) + `gating-analysis` + `ece-input-diagnose` READMEs (stdout-only numbers for our results; paper-1 citations for C0/C1). Split C9 → C9/C9b/C9c + C16; mark cross-harness rows as sensitivity-anchored vs paired; mark every `c0_only` row `deployable=false`.
- **W2** Related-work search + BibTeX: include the direct soil-moisture MoE papers, clustered soil-moisture models, seasonal-regime work, and hydrology MoE/router papers listed in §2; verify DOI/venue metadata and tag venue-fit. Add one positioning sentence for the ECE collaboration differentiator (live in-situ deployment no SOTA benchmark offers) without overstating novelty.
- **W3** Figure notebooks: (a) main `notebooks/experiment/paper2-figures-1.0/` producing F1 (+F4/F6 if kept) with Guarded primary row + T8 panel; (b, NEW in v5) supplementary ECE diagnostic panel producing T7/T9/F5-supp from salvage CSVs (`summary/station_metrics/routing_audit/predictions_v3`) + `station_static_features.csv`; both via `nb execute --uv`; figures saved under the experiment dir(s); tables in writeup2 filled only from notebook stdout.
- **W4** ECE inclusion decision (PI, v5 reframing): default = INCLUDE §7 (extended to ~0.75–1 page) + supplementary T7/T9 panel as diagnostic exhibit. Cut both only for a strict 6-page venue — then the paper falls back to the v4 tie-only caveat paragraph. Same pass: include T0/§6.5 or not.
- **W4′** (v5: ACCEPTED, expanded) Guarded-on-ECE run: clone `ece-router-salvage-2.0` harness as a new versioned experiment dir adding `Guarded_Backbone54_k2 (c0_0_c1_0, shared-54, no deltas)` as third family on canonical v3 (frozen WA-trainval experts/routers/thresholds; ECE strictly unseen; 5 seeds; policies `as_routed/auto_hard/auto_soft/c0_only-oracle/c1_only-diagnostic` + `Global_Single_54 direct` ref). Verify dry-expert index after Guarded canonicalization (`c1`=drier) before comparing to salvage `C0`=dry; re-run the WA synthetic-SMAP-mask calibration row for the guarded family; update the cloned `tests/test_ece_router_salvage*.py` counts. ALSO still open (optional): direct Guarded-vs-global LOSO win counts in-harness (Global_Single_54 was not in routing-optimize-1.0) — do only if PI wants paired win counts in the main table; otherwise keep T8 + V0-anchored 6/7.
- **W4″** (NEW in v5) Terminology pass: enforce "dry/lowland expert vs wet-mountain expert" (+ station parenthetical on first use) across §4/§5/§7/§8/figures; record the salvage-vs-Guarded label-flip box once in §4; never describe purity as regime discovery.
- **W5** Create/refresh `writeup2/` skeleton: use `outline-v5.md` as the active outline; retain `outline-v2.md`, `outline-v3.md`, and `outline-v4.md` unchanged for history; add `claims-ledger.md`, `figures/`, and later `main.tex` or `main.md`.
- **W6** Draft §3–§6 (method + results) first — fully supported; then §1 genealogy + §2, §5, §8–10. Guard motivation kept to one paragraph + T8 in all drafts; ECE diagnostic kept to §7 + supp (never promoted into §6 headline tables); expand only on PI request.
- **W7** PI review pass against ledger + title-route pick (A/B/C); then venue-specific trimming via matrix §6. Explicit PI question: keep supp ECE panel for all venues except strict 6-page, or cut everywhere?
- **W8** Reproducibility and framing gate before submission: re-execute `routing-optimize-1.0` (routing + T1/L1 cells), the new Guarded-on-ECE run, and both figure notebooks; confirm every table row = stdout; audit router feature availability and K-selection provenance; verify `outline-v2.md`, `outline-v3.md`, and `outline-v4.md` are unchanged; run `make lint` / `make test` if src is touched (expected: no).

Not doing (this paper): presenting ECE as a SOTA-beating claim; presenting `c0_only` as deployable; station-count ablation, OOS full section, broad architecture redesign, edits to `writeup/` or `paper/`, archived notebooks, re-running historical v19–v25 gates, a second guard variant campaign (StationMeanV-B stays a reported null).

---

## 8. Risks

- **R3 (v5 DOWNGRADED):** ECE section distracts or invites reviewer attack — mitigate: ECE is now explicitly diagnostic + supplementary (never in §6 headline tables, never a SOTA claim); oracle/deployable split stated in §7 lead; n = 5 + 30-day one-sided caveats written into the subsection, not buried; cut option retained only for strict 6-page venues (W4).
- **R1:** LOSO non-significance misrepresented — mitigate: fixed wording in T4 caption.
- **R2:** K=2 or “regimes” read as universal — mitigate: use regional strata terminology, pretest K-selection disclosure, and explicit regional scope.
- **R4:** Backbone selected on test — reviewers may challenge absolutes — mitigate: shared-backbone sentence in §3 + §6.4; never compare to literature absolutes without note.
- **R8:** Direct literature overlap makes an architecture-novelty claim untenable — mitigate: frame the paper as a controlled router-provenance study and cite direct soil-moisture MoE work in §2.
- **R9:** SMAP features make “target/proxy-free” inaccurate — mitigate: use in-situ-target-label-free/satellite-SM-informed language and include the provenance table.
- **R10 (v5 updated):** Perfect station purity may reflect static station descriptors rather than transferable regimes — mitigate: purity is disclosed as guard-enforced (weaker diagnostic); OOS failures stay the hard limitation; ECE is mixed evidence (static-router failure + dry-expert conditional transfer under forced/deployable-gated routing) rather than a pure counterweight; transferable evidence is LOSO on unseen stations + ECE routing-isolation numbers with ns disclosed.
- **R15 (NEW in v5):** ECE oracle (`c0_only`) misread as a deployable method — mitigate: every C9b row/table/figure caption carries `manual oracle ceiling, deployable=false`; deployable claim restricted to C9c `auto_hard/auto_soft` with WA-only calibration stated.
- **R16 (NEW in v5):** Salvage-vs-Guarded label flip (`C0`=dry vs Guarded canonical `c1`=drier) causes inverted expert comparison — mitigate: §4 label-flip box + dry-expert-index verification step in W4′; supp panel uses "dry/lowland" / "wet-mountain" words, never bare c0/c1, outside Methods.
- **R11:** Causal geoscience interpretation exceeds team expertise — mitigate: report descriptive associations only and require subject-matter validation before strengthening claims.
- **R5 (v3):** Cross-dataset number confusion (paper 1's 0.822 vs our 0.780/0.812) — mitigate: never co-rank; footnote-only comparisons; dataset version tags on any historical table (T0).
- **R6 (v3):** Over-attributing unpublished internal failure numbers as if peer-reviewed — mitigate: cite paper 1 for what it actually contains (W-new); label internal numbers "prior/internal analysis" or omit from camera-ready.
- **R7 (v3):** Sequel narrative reads as dumping on paper 1 — mitigate: tone = "open question left by prior work," contributions credit paper 1 for pipeline + diagnosis.
- **R12 (NEW in v4):** Guard reads as post-hoc engineering to recover V0 numbers — mitigate: W2 spec with promotion bars frozen before training (cite §4, one sentence); ablation + mechanism tables show *where* it acts, not just that it ties.
- **R13 (NEW in v4):** Guard reads as station-identity routing — mitigate: §4 station_id disclosure (consistency over an unsupervised partition, target untouched) + LOSO-unseen defense (held-out stations never in fit quantities); never claim regime discovery from purity.
- **R14 (NEW in v4):** Anchor correction (guards → 0.814334) reads as moving goalposts — mitigate: §9 footnote with the pre-registered justification (guards provably reproduce the V0 test partition, §R2 ARI = 1.0, established before training); both anchors are prior published numbers recorded in `config.yaml`.

---

## 9. Primary source map (where every number lives)

| Source | Role in paper |
|--------|---------------|
| `paper/*.pdf` | Paper 1 — foundations citation; prior regime analysis (verify content) |
| `writeup/sections/three_regime/*` | Detailed prior 3-regime narrative (cross-check vs submitted PDF) |
| `writeup/sections/base_model/*`, `writeup/sections/shared/*` | Paper-1 pipeline/features/data prose to cite, not copy |
| `docs/gating.md` | Internal history of threshold/hard/soft/classifier gates (understanding; cite only if venue accepts tech report) |
| `docs/plans/20260703-why-MoE.md` | Motivation background only (re-verify any number before quoting) |
| `docs/plans/20260721-remove-incomplete-stations.md` | Dataset pruning: incomplete stations |
| `docs/plans/20260726-stations-removal.md` | Dataset pruning: snowpack/out-of-scope stations |
| `notebooks/experiment/derived_8.4-formal-eval-1.0/` | Sensitivity rows: temporal + LOSO formal stats vs global/gating/deltas (V0 arms) |
| `notebooks/experiment/derived_8.4-routing-optimize-1.0/` | Primary results: guard spec, R1–R4 agreement/K-sweep, T1 temporal, L1/T8 LOSO (slurm job 2155747) |
| `notebooks/experiment/derived_8.4-gating-analysis-1.0/` | K-sweep, purity, regime composition, separating features |
| `notebooks/experiment/derived_8.4-regime-interpretation-1.1/` | Method narrative, physical interpretation of regimes |
| `notebooks/experiment/derived_8.4-eval-1.3/` | LOSO protocol provenance, station difficulty (secondary) |
| `notebooks/experiment/derived_8.4-formal-eval-2.0/` | OOS numbers for Limitations only (pre-guard rows) |
| `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/` | ECE tie rows + metric realism (§7 baseline half) |
| `notebooks/experiment/derived_8.4-ece-router-salvage-1.1/` | ECE oracle/poison rows C9b + v3 station tables + routing audit (supp panel) |
| `notebooks/experiment/derived_8.4-ece-router-salvage-2.0/` | ECE deployable rows C9c + WA-only gate/temperature calibration (supp panel) |
| `notebooks/experiment/derived_8.4-ece-router-salvage-1.2-guarded/` (NEW, W4′ — create on execution) | Guarded-on-ECE rows (same stdout discipline; pre-guard vs guarded labeled) |
| `data/splits/derived_8.4_ece_v3/station_static_features.csv` | ECE-vs-train site-descriptor strip (elevation/precip) for T9 |
| `notebooks/experiment/ece-input-diagnose-1.0/` | ECE sensor/benchmark validity diagnosis (§7 if kept) |
| `data/splits/derived_8.4/split_meta.json` | Split definition for §3 |
| External direct-overlap literature | Related-work records listed in §2; verify DOI/venue metadata in W2 |

(End of file - total 420 lines)
