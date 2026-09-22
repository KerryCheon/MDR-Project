# Second Paper Outline (v2) — Regional Multi-Regime MoE for Soil Moisture

Status: draft outline for PI · Split: `derived_8.4` only · Scope: Washington (regional) · CS venue TBD
Supersedes: `outline-v1.md` (kept for history) · New in v2: sequel framing to paper 1, routing-design narrative, dataset-evolution context, revised title routes.

This document is the technical-report backbone for writing the full paper without manually digging through experiment notebooks. Every number below is already present in an executed report notebook README (stdout tables), in paper 1, or in repo docs explicitly marked as historical/prior-work sources; the claims ledger (W1) freezes the mapping before drafting.

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
4. **ECE sensors: conditional inclusion** (see §7 + works-to-do W4). Default = short "deployment stress test" subsection with heavy caveats, not a headline result. Drop entirely if reviewers/PI prefer a pure CS story.
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

---

## 1. Title candidates (three routes — pick at drafting)

### Route A — Routing-contrast (encodes the failure→fix story)

1. "Routing Is the Bottleneck: Unsupervised Regime Assignment for Soil Moisture Mixture-of-Experts"
2. "From Threshold Gates to Feature-Space Clusters: Reliable Regime Routing for Regional Soil Moisture Estimation"
3. "When Learned Routers Fail: Covariate-Space Regime Routing for Soil Moisture Prediction"

### Route B — Sequel / revisit framing

4. "Revisiting Regime-Based Soil Moisture Modeling: Unsupervised Routing over Threshold and Supervised Gates"
5. "Beyond the Single Global Model: A Regional Study of Unsupervised Mixture-of-Experts for Soil Moisture"

### Route C — Neutral descriptive (safer for IEEE-style)

6. "Unsupervised Climate-Regime Mixture-of-Experts for Regional Soil Moisture Estimation"
7. "Hard-Gated Two-Expert Soil Moisture Modeling with Unsupervised Station-Interpretable Routing"

(No ban-list enforced yet; refine after venue pick.)

---

## 2. Abstract skeleton (6 sentences — revised for sequel arc)

1. Problem: a single global tabular model must compromise across wet/dry and east/west soil-moisture behaviors; our prior work established the pipeline and a strong global baseline but left regime specialization undeployable because realistic routers misroute (regional, not continental, target).
2. Approach: we revisit hard-gated MoE with routers that never see the target — KMeans on the same 54-feature backbone the single model uses — plus a formal multi-seed + block-bootstrap protocol; no regime-specific feature selection.
3. Result temporal: R² 0.812 vs 0.780 (RMSE 0.044 vs 0.048), 30 seeds, p < 1e-12, sample bootstrap ΔR² +0.035 [0.019, 0.055], p = 0.0005.
4. Result spatial (in-state LOSO): mean R² 0.64 vs 0.58, wins 6/7 stations (descriptive).
5. Analysis: K=2 is the largest K with station-pure regimes on this dataset; regimes = semi-arid east vs maritime west; supervised gating (our foil) is consistently worse — supporting covariate-space over target/heuristic routing.
6. Scope: regional model; out-of-state transfer fails for all models (honest limitation); guidance for K selection elsewhere.

---

## 3. Section-by-section outline

Each section lists: goal · claims · numbers to quote · source (experiment path + table) · figures · venue knobs · open writing tasks.

### §1 Introduction [all venues] — rewritten as problem genealogy

- **Beat 1 — What paper 1 established.** Dataset prep + feature pipeline + strong single-regime global XGBoost (paper 1 reports R² 0.822 under its protocol); residual diagnostics / error structure across the target space motivated explicit regime modeling (paper 1 three-regime writeup sections: non-uniform error, oracle upper bound promising).
- **Beat 2 — Why the obvious MoE failed.** Prior three-regime dry/transition/wet design with threshold/classifier routing: oracle routing showed headroom, but deployable routers misroute; transition class is poorly separable; misrouting compounds through hard gates (cite paper 1 limitations if present in PDF; otherwise attribute as prior work / internal analysis with dataset tag). Paper 1 therefore reported the single-regime global model.
- **Beat 3 — What this paper does.** (i) Attempted scale-up of training data, then rigorous station pruning for quality + snowpack-feature fit (→ clean 7-station `derived_8.4`); (ii) attempted threshold reassignment — did not close the gap; (iii) **unsupervised covariate-space routing** with shared backbone + formal evaluation → deployable specialization.
- **Contributions (bullet list):**
  - C-A: Hard-gated two-expert MoE with KMeans router on the shared feature backbone (target-blind routing).
  - C-B: Multi-seed temporal + LOSO + block-bootstrap protocol with selection-leakage hygiene (no delta features in the headline).
  - C-C: Empirical result: significant in-region temporal gain + consistent in-state LOSO wins over the global baseline.
  - C-D: Analysis: K-selection guidance (multi-regime, not "K=2 universal"); station-pure east/west regimes; supervised gating inferior.
  - C-E (framing): Design lesson — route on covariates, not on the target/proxy; prior threshold/supervised routing families fail on this problem class.
- Claims refs: C0–C2 (historical, paper-1-attributed), C1–C6 (ours). See §5.
- Sources: paper 1 PDF (`paper/`); `writeup/sections/three_regime/*` (verify overlap with submitted PDF — W-new); `docs/gating.md` (internal, for our own understanding only unless venue allows citing tech report); `docs/plans/20260703-why-MoE.md`; `derived_8.4-gating-analysis-1.0` correlation-drift figures.
- Knobs: science-venue expands physical motivation; ML venue expands routing-design story (Beats 2–3); applied IEEE keeps Beat 1 short and cites paper 1 for pipeline.

### §2 Related work [all venues] — NEW WRITING (expanded)

- Pillar 0 (NEW): positioning vs paper 1 — one short paragraph, not a survey; what is new here vs the submitted manuscript.
- Pillars: (i) regional vs continental/global soil-moisture ML; (ii) MoE / hard vs soft gating / **learned vs unsupervised routers** (expand: gating-network MoE, cluster-based or product-of-experts routing, regime/threshold partitioning in hydrology); (iii) gradient-boosted tabular baselines for geoscience; (iv) evaluation practice (LOSO, multi-seed, selection leakage).
- Open task W2: literature search + BibTeX (no citations invented in outline).

### §3 Problem setup & data [all venues] — expanded with dataset evolution

- `derived_8.4`: 7 WA stations; train 2017–20 (9,803) / val 2021–22 (4,805) / test 2023–25 (6,620); station list with one-line descriptors (east/west, elevation).
- Stations: BeaverPass_WA_990, CayusePass_WA, Darrington, Paradise_WA, Quinault, SourdoughGulch_WA_985, Spokane.
- **Dataset evolution paragraph (NEW):** earlier expansion roughly doubled station count; many added stations removed for data-quality issues and because the current feature set lacks snow/SWE variables needed for snowpack-dominated sites (alpine stations out of scope; cite `docs/plans/20260721-remove-incomplete-stations.md`, `docs/plans/20260726-stations-removal.md`). Result: clean, deployment-relevant 7-station split. One sentence: snowpack regimes are future work (SWE/Snow Depth features), not a silent omission.
- Feature backbone: 54 shared features (identical for every model) — name the source (`derived_8.4-feature-selection-2.0` / eval-1.1 `selected_features.json`); foundations of the selection pipeline cite paper 1; state that the backbone was fixed before this comparison and shared across arms (relative claims).
- Target: `soil_moisture_5cm`; metrics: R², RMSE, MAE, bias (+ Pearson where useful).
- **Cross-dataset caution (R5):** any comparison to paper 1's 0.822 must note different split/dataset version/protocol; never place it in the same ranking table as `derived_8.4` rows without a footnote.
- Sources: `data/splits/derived_8.4/split_meta.json`; `derived_8.4-formal-eval-1.0` protocol section; `derived_8.4-gating-analysis-1.0` feature-list tables; paper 1 §data/features; station-removal plans.

### §4 Method: multi-regime hard-gated MoE [all venues] — + design-alternatives subsection

- Architecture: router → hard assignment → per-regime XGBoost expert; inference cost = single model + tiny router.
- **§4.x Design alternatives tried before (NEW, short):**
  - A1: Threshold gates on true/predicted soil moisture (dry/transition/wet; hard + soft) — prior work / paper 1 era; circular routing signal, boundary misrouting.
  - A2: Supervised classifier router trained to predict regime membership — in-paper instance: `Trained_Gating_k2` (losing row in §6); historical e2e classifiers similarly weak (cite carefully).
  - A3: Heuristic routers (seasonal, single-index) — in-paper: `Seasonal_Binary_k2`, `Univariate_G_API_k2`.
  - A4 (chosen): **Unsupervised KMeans on the shared 54-feature backbone** — routing never observes the target; stable assignment; station-interpretable clusters.
- Router families compared (same 54 feats, no deltas):
  1. `Global_Single_54` (K=1 baseline)
  2. `Clustering_V0_Full_k2` / `Clustering_Backbone54_k2` (unsupervised KMeans; V0 and Backbone54 give ARI = 1.0000 partition — pick ONE paper name, report the other as equivalent) — PRIMARY
  3. `Clustering_Dynamic_k2` (3-feature light router)
  4. `Seasonal_Binary_k2` (heuristic)
  5. `Univariate_G_API_k2` (heuristic)
  6. `Trained_Gating_k2` (supervised) — the foil
- Training protocol: routers fit on trainval only; experts per regime; 30 seeds temporal (seed scope = expert `random_state` only — state this); LOSO 5 seeds × 7 folds with per-fold router refit (no held-out-station leakage into routing).
- Statistics: seed-level mean ± std + t-CI; paired t / Wilcoxon + BH-FDR; sample-level paired cluster bootstrap over (station, month); LOSO win counts + sign test. Seed-level inference = fitting stochasticity ONLY (must appear in the paper).
- K-selection subsection: sweep K = 2, 3, 4 (+ quality indices K = 2..6); deployment guidance ("choose largest K with coherent, whole-group regimes on YOUR region") — not "always 2." Tie to genealogy: K=3 on *target thresholds* failed historically; K=3 on *covariates* is testable but fragments station purity here — different failure modes, state both.
- Sources: `derived_8.4-formal-eval-1.0` README protocol + stats; `derived_8.4-gating-analysis-1.0` K-sweep tables; `eval11/routers.py`; `derived_8.4-regime-interpretation-1.1` Part A (method narrative); paper 1 for prior architecture description.

### §5 Why regimes exist on this dataset (interpretability)

[science venues: full · ML venues: 0.5–1 page]

- K=2 purity = 1.000 trainval AND test; K=3 mean purity 0.833; K=4 0.695 — larger K fragments stations (table).
- Regime composition: cluster 1 = Spokane + SourdoughGulch (semi-arid east); cluster 0 = 5 western/mountain stations. Geographic map figure.
- Top separating features: `J_bio_bio13` (annual precip 391 vs 70), SMAP 30-day roll means, LST/NDMI windows (top-15 table).
- Correlation drift example: `SMAP_sm_pm_interp_rollmean30` flips sign across regimes — the "why specialization can help" evidence (also why a single global mapping compromises).
- Quality indices: Calinski-Harabasz and Davies-Bouldin best at K=2; silhouette marginally higher at K=3 (admit internal indices alone don't decide K — purity/interpretability does).
- Contrast with target-threshold regimes (NEW sentence): splitting on y produces regimes that are *definitional* but not *station-stable*; covariate clusters recover geographic/climate partitions that generalize across splits.
- Sources: `derived_8.4-gating-analysis-1.0` (all tables/figs); `derived_8.4-regime-interpretation-1.0/1.1/1.2`.

### §6 Results [core of every venue]

#### 6.1 Temporal (primary) — no-delta only

- Headline table (rank by RMSE): at minimum
  - `Clustering_V0_Full_k2` (0,0): R² 0.8118 ± 0.0014, RMSE 0.0442
  - `Global_Single_54`: 0.7798 ± 0.0013, RMSE 0.0478
  - `Trained_Gating_k2` (0,0): 0.7354 ± 0.0011
  - plus Dynamic / Seasonal / Univariate no-delta rows for the full router comparison
- Pairwise: clustering vs global ΔR² +0.032 [0.031, 0.033], p < 1e-12, 100% seeds; clustering vs trained gating +0.089. Bootstrap (station, month): ΔR² +0.035 [0.019, 0.055] p = 0.0005; RMSE −0.0040 p = 0.0005.
- **Genealogy sentence (NEW):** our results confirm the prior-work hypothesis that specialization helps, while showing that *routing choice* (unsupervised covariate clusters vs supervised/heuristic/threshold) is what makes specialization deployable — `Trained_Gating_k2` underperforming global is the in-paper demonstration.
- Optional footnote only (not a table row): paper 1 reported 0.822 under a different dataset/split — not directly comparable (R5).
- Source: `derived_8.4-formal-eval-1.0` temporal seed table (filter `delta_source=none`) + focused pairwise + bootstrap tables.
- Figure F1: bar/dot temporal R² with 95% CI across ~8 configs (NEW figure notebook).

#### 6.2 In-state LOSO (secondary, descriptive)

- No-delta: `Clustering_V0_Full` 0.637 / `Backbone54` 0.621 vs `Global_Single_54` 0.580 (and `Baseline_V0_50` 0.591). Wins 6/7 vs global (sign p = 0.125 — state underpowered).
- Station difficulty table (7 stations, median LOSO R² across configs): Darrington 0.70, BeaverPass 0.69, Spokane 0.65, Paradise 0.59, Quinault 0.56, SourdoughGulch 0.43, CayusePass 0.39.
- Explicit sentence: LOSO supports consistency, not statistical significance at n = 7.
- Source: `derived_8.4-formal-eval-1.0` LOSO summary + `loso_pairwise_focused` (filter none); `derived_8.4-eval-1.3` for difficulty narrative if needed.
- Figure F2: per-station paired R² (existing `loso_pair` / `loso_station_bars` PNGs).

#### 6.3 Router comparison (the multi-regime story)

- Ranking message: unsupervised climate clustering ≈ or > heuristics > supervised gating on this dataset; global sits in the middle temporally, near bottom on LOSO.
- Optional single table merging 6.1 + 6.2 columns (temporal R² | LOSO R²).

#### 6.4 Robustness note (1 paragraph, not a section)

- No-delta ≈ historical test-delta winner (0.8118 vs 0.8126) → headline is not a delta-selection artifact.
- Val-selected deltas historically UNDERPERFORM global (0.735 vs 0.780) → selection instability exists in this benchmark; our simplified no-delta protocol avoids it.
- Optional: one sentence that the shared backbone was itself test-era-selected (shared by all arms; absolute numbers optimistic, relative comparison intact).
- Source: `derived_8.4-formal-eval-1.0` delta-robustness table + caveat bullets.

#### 6.5 (Optional, venue-dependent) Prior-attempts context material

- NOT a ranked results row. If included: small appendix table listing prior design attempts (threshold hard/soft, supervised classifier, adapted training, threshold recalibration, dataset expansion) with **dataset version tags** and qualitative outcome (worked / failed / partially), sourced from paper 1 + internal docs — purpose is narrative honesty, not benchmarking.
- Default: omit from main paper; keep outline entry for PI option (fold into W4-style venue decision).

### §7 Deployment stress test — ECE sensors

[OPTIONAL subsection; default INCLUDE as ~0.5 page under Discussion or a short §7]

- Purpose: not to show superiority — all models are similar/bad — but (i) a real new-deployment transfer attempt on unseen in-situ sensors, (ii) what breaks when you evaluate there.
- What we can honestly say:
  1. Protocol: train on 7 WA trainval; evaluate 5 ECE stations (`derived_8.4_ece_v3`, 150 rows, 2026-07-20..08-19, ~30-day window).
  2. Outcome: two-regime ≈ single-regime on RMSE (ΔRMSE +0.00025, 2/5 wins, ns); all R² deeply negative because σ_y ∈ [0.003, 0.008] — rank by RMSE, not R² (metric-realism lesson).
  3. Predictions collapse toward a common value across sensors with very different target means (0.018–0.198) — line-chart observation.
  4. Sensor diagnosis (`ece-input-diagnose-1.0`): hourly data show strong diurnal cycle (median hourly range 0.035 ≈ 14× median daily step 0.0025); rain falls overnight and dries by midday, so 24-h means look flat; hour-scale corr(rain, Δsensor) ≈ 0 at all lags 0–6h. So: NOT "sensors ignore rain" but "daily aggregation + short dry-down window make this set a weak benchmark for daily models."
  5. Takeaway for the paper: sensor → benchmark validity checks (variance floor, event-response check, aggregation choice) should precede model ranking.
- What we must NOT say: that two-regime wins on ECE; that sensors are broken; any causal claim about sensor hardware.
- Decision rule W4: if PI wants a pure methods/CS paper with no deployment narrative, cut §7 entirely — nothing else depends on it.
- Sources: `derived_8.4-formal-eval-2.1-ece-v3` README (tables 1–3 + time-series figs); `ece-input-diagnose-1.0` README §4–5 + figures; `derived_8.4-ece-error-analysis-1.0` if needed.

### §8 Discussion

- **§8.1 Lessons from failed routing (NEW, core):**
  1. Routing on the target (or its proxy) is circular and boundary-fragile — hard gates turn tiny errors into step-function losses.
  2. The transition class (3-way on y) is ill-posed on this region: heterogeneous, low-separability; collapsing to climate-coherent partitions is more usable than finer *target* slices.
  3. Supervised routers optimize the wrong objective when regime labels themselves are noisy/threshold-derived; unsupervised covariate clusters are stable and station-interpretable.
  4. Oracle upper bounds (prior work) diagnose *potential*, not deployability — report oracle only as motivation, never as our result.
- Why learned gating loses in-paper: routing noise + expert co-adaptation gap; unsupervised router is stable and station-interpretable.
- Multi-regime, not two-regime: decision procedure for choosing K on a new region (purity/coherence, quality indices, physical interpretability); expect K > 2 where climate zones multiply.
- Regional scope by design (PI position: remote sensing / domain-specific models are legitimate; cite paper 1 scope statement if appropriate).

### §9 Limitations [required]

- n = 7 LOSO, low power; partial 2025 coverage; seed variation does not cover router stochasticity (routers fixed); shared backbone/hyperparameters test-era-selected.
- **OOS paragraph:** train on 7 WA → 10 out-of-state stations: ALL models degrade (station-mean R² negative for every config; pooled global 0.206 vs best two-regime ~0.11–0.14; two-regime worse than global, e.g. V0_Full no-delta wins 2/10). Frame: regime specialization is a within-region prior; outside the region both experts and router are OOD — consistent with regional scope, not a hidden result. Source: `derived_8.4-formal-eval-2.0` summary + Table 1 (no-delta rows only).
- **Snowpack / feature-space boundary (NEW):** stations under alpine snowpack dynamics removed/out-of-scope because the feature set lacks SWE/snow depth; algorithm gating cannot fix missing physics (cite station-removal plan). Future work: SWE integration.
- ECE caveats if §7 kept: 30-day window, tiny target variance, daily aggregation.
- Historical/attribution limitation (optional one line): some prior routing results live only in an internal tech report / pre-camera-ready analysis — we cite paper 1 where possible and do not over-claim unpublished numbers.

### §10 Conclusion & future work

- Restate design lesson in one line (covariate-space routing made specialization deployable where threshold/supervised routing did not).
- Future work bullets (not claims): automated K selection; OOD/ambiguity fallback to global expert (diagnostics exist in `derived_8.4-formal-eval-2.0` Table 4); station-count / training-set composition ablation (explicitly future); snowpack-aware features (SWE) + expanded station types; multi-year ECE evaluation once sensors accumulate a full season; revisit 3-way partitions only with a routing signal that is not derived from y.

---

## 4. Figure & table plan (all via notebook, `nb execute --uv`)

| ID | Item | Status | Source |
|----|------|--------|--------|
| T0 | (optional) Prior-attempts context table, dataset-labeled | new from paper1+docs | §6.5 / appendix |
| T1 | Config table (routers × features × protocol) | new LaTeX from method text | gating-analysis + eval config |
| T2 | Temporal seed-level results (no-delta) | filter existing table | `formal-eval-1.0` |
| T3 | Temporal pairwise (global, gating) | existing focused rows | `formal-eval-1.0` |
| T4 | LOSO summary + win counts (no-delta) | filter existing | `formal-eval-1.0` |
| T5 | K-sweep purity/quality | existing | `gating-analysis-1.0` |
| T6 | Regime station composition + top features | existing | gating-analysis / regime-interpretation |
| T7 | (opt) ECE RMSE pairwise + station means | existing | `formal-eval-2.1-ece-v3` |
| F1 | Temporal multi-seed dot/bar + CI | NEW | `paper2-figures-1.0` notebook |
| F2 | LOSO per-station pairs | reuse PNG | `formal-eval-1.0` |
| F3 | Regime geographic map (WA) | NEW or reuse | gating geographic PNGs exist — pick/crop |
| F4 | Delta-robustness slope (test/val/none) | NEW, optional | `formal-eval-1.0` |
| F5 | (opt) ECE time-series line charts | reuse ≤5-line figs | `formal-eval-2.1-ece-v3` |
| F6 | (opt) Concept figure: routing families A/B/C vs chosen | NEW, simple diagram | method narrative only |

---

## 5. Claims ledger (excerpt — full ledger = separate file in works)

| # | Claim | Config / source | Number | Artifact |
|---|-------|-----------------|--------|----------|
| C0 | Prior work: global baseline + regime motivation | paper 1 | R² 0.822 (paper-1 protocol) | `paper/` PDF |
| C1 | Prior work: specialization potential under oracle; deployable routing was the gap | paper 1 (if in PDF) / internal prior | oracle ≫ global; e2e routing weak | paper 1 §three_regime (verify) |
| C2 | Two-regime > global temporal (OURS) | V0_Full (0,0) vs Global_54 | 0.8118 vs 0.7798; Δ +0.032 p < 1e-12 | `formal-eval-1.0` |
| C3 | Sample-level significance | same | ΔR² +0.035 [0.019, 0.055] p = 0.0005 | `formal-eval-1.0` bootstrap |
| C4 | Two-regime > global LOSO | V0_Full (0,0) | 0.637 vs 0.580; 6/7 wins | `formal-eval-1.0` LOSO |
| C5 | Supervised gating inferior temporal (in-paper foil) | (0,0) | 0.735 vs 0.812 | `formal-eval-1.0` |
| C6 | K=2 station-pure; K>2 fragments | Backbone54 | purity 1.000 / 0.833 / 0.695 | `gating-analysis-1.0` |
| C7 | Regimes = east vs west | composition table | Spokane + Sourdough = C1 | `gating-analysis-1.0` |
| C8 | OOS: no model transfers; 2-regime ≤ global | no-delta arms | mean R² < 0 all; Δ −0.17 (2/10) | `formal-eval-2.0` |
| C9 | ECE: models tie; R² invalid at low σ_y | no-delta | ΔRMSE +0.00025 ns | `formal-eval-2.1` |
| C10 | ECE daily means hide diurnal/event response | n/a | hourly range 0.035 vs daily step 0.0025 | `ece-input-diagnose-1.0` |
| C11 | Dataset evolution: expansion → prune → 7 stations (quality + snow fit) | plans | 5→13→7 (approx.; verify exact counts) | removal plans + split_meta |
| C12 | Design lesson: covariate routing > target/heuristic/supervised (framing) | C5 + method narrative | qualitative; supported by C5/C6/C7 | §8.1 |

Every draft sentence must map to a ledger row; no orphan numbers. C0/C1 must be checked against the actual submitted PDF before citation (W-new).

---

## 6. Venue adaptability matrix

| Element | ML conf (6–8p) | Geo/AGU-style | Applied IEEE (AIIoT-like) |
|---------|----------------|---------------|---------------------------|
| §1 genealogy beats | full (2–3) | short | short, cite paper 1 |
| §4 design-alternatives | full | medium | medium |
| §5 interpretability | short | full | medium |
| §6.5 prior-attempts table | omit or appendix | omit | optional appendix |
| §7 ECE | optional/omit | include (deployment) | include (motivation) |
| Stats detail | protocol emphasis | metrics emphasis | protocol lighter |
| OOS limitation | 3 sentences | 1 short para | 1 short para |
| Related work | MoE-heavy | SM-heavy | balanced |

---

## 7. Works to do (ordered)

- **W0** Freeze claim scope: primary configs = no-delta only; router name paper-wide (recommend `Clustering_V0_Full_k2` or rename to `Clustering_k2` in text; Backbone54 noted ARI=1). *(decision, no code)*
- **W-new** Open `paper/*.pdf` and confirm whether three-regime/oracle/limitation content is in the **submitted** manuscript (vs only `writeup/`); record which C0/C1 sentences are safely citable to paper 1. Also confirm exact station counts over time for C11 (5→13→7) from split history before freezing numbers.
- **W1** Write full claims ledger → `writeup2/claims-ledger.md` from `formal-eval-1.0` / `2.0` / `2.1` + `gating-analysis` + `ece-input-diagnose` READMEs (stdout-only numbers for our results; paper-1 citations for C0/C1).
- **W2** Related-work search + BibTeX (venue chosen later; collect ~40 candidates across the five pillars incl. pillar 0 positioning note; tag venue-fit).
- **W3** New figure notebook `notebooks/experiment/paper2-figures-1.0/` producing F1 (+F4/F6 if kept) from existing CSVs; `nb execute --uv`; figures saved under the experiment dir; tables in writeup2 filled only from notebook stdout.
- **W4** ECE inclusion decision (PI): include §7 vs cut. Default plan includes it. Same pass: include T0/§6.5 or not.
- **W5** Create/refresh `writeup2/` skeleton: this outline + `claims-ledger.md` + `figures/` + (later) `main.tex` or `main.md`; compile script only after venue pick.
- **W6** Draft §3–§6 (method + results) first — fully supported; then §1 genealogy + §2, §5, §8–10.
- **W7** PI review pass against ledger + title-route pick (A/B/C); then venue-specific trimming via matrix §6.
- **W8** Reproducibility gate before any submission: re-execute `formal-eval-1.0` report notebook + `paper2-figures` notebook; confirm every table row = stdout; `make lint` / `make test` if src touched (expected: no).

Not doing (this paper): station-count ablation, OOS full section, new training runs, edits to `writeup/` or `paper/`, archived notebooks, re-running historical v19–v25 gates.

---

## 8. Risks

- **R1:** LOSO non-significance misrepresented — mitigate: fixed wording in T4 caption.
- **R2:** "two regimes" read as universal — mitigate: §4 K-selection + §8 discussion.
- **R3:** ECE section distracts or invites reviewer attack — mitigate: W4 cut option; caveats written into the subsection, not buried.
- **R4:** Backbone selected on test — reviewers may challenge absolutes — mitigate: shared-backbone sentence in §3 + §6.4; never compare to literature absolutes without note.
- **R5 (NEW):** Cross-dataset number confusion (paper 1's 0.822 vs our 0.780/0.812) — mitigate: never co-rank; footnote-only comparisons; dataset version tags on any historical table (T0).
- **R6 (NEW):** Over-attributing unpublished internal failure numbers as if peer-reviewed — mitigate: cite paper 1 for what it actually contains (W-new); label internal numbers "prior/internal analysis" or omit from camera-ready.
- **R7 (NEW):** Sequel narrative reads as dumping on paper 1 — mitigate: tone = "open question left by prior work," contributions credit paper 1 for pipeline + diagnosis.

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
| `notebooks/experiment/derived_8.4-formal-eval-1.0/` | Temporal + LOSO formal stats (core results) |
| `notebooks/experiment/derived_8.4-gating-analysis-1.0/` | K-sweep, purity, regime composition, separating features |
| `notebooks/experiment/derived_8.4-regime-interpretation-1.1/` | Method narrative, physical interpretation of regimes |
| `notebooks/experiment/derived_8.4-eval-1.3/` | LOSO protocol provenance, station difficulty (secondary) |
| `notebooks/experiment/derived_8.4-formal-eval-2.0/` | OOS numbers for Limitations only |
| `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/` | ECE transfer results (§7 if kept) |
| `notebooks/experiment/ece-input-diagnose-1.0/` | ECE sensor/benchmark validity diagnosis (§7 if kept) |
| `data/splits/derived_8.4/split_meta.json` | Split definition for §3 |
