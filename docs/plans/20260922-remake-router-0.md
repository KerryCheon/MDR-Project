## Finding (why this risk is real, and smaller than it looks)

* Full-trainval temporal: `V0_Full_k2` vs `Backbone54_k2` are the **same partition** — `gating-analysis-1.0/README.md:81-83`: group sizes 10,624/3,984, ARI=1.0000, purity 1.000/1.000. Temporal no-delta ties: `0.8118 vs 0.8117` (`formal-eval-1.0/README.md:62-66`).
* Divergence appears only on **LOSO per-fold refits** (router refit on 6 stations, `eval-1.4/README.md:35-50`): `V0 0.634 vs Backbone 0.617` no-delta (`eval-1.4/README.md:181-187`), `0.637 vs 0.621` in formal-eval. Experts are identical 54-backbone, so gap isolates routing. Only 18/50+54 features overlap.
* ECE salvage confirms OOD flip-flop without pooled winner (`ece-router-salvage-1.1`: pooled tie, per-station flips).
* Provenance trap:
  * V0-50 = `# exact copy of derived_8.2's global c1 feature set` (`dataset_metadata.py:6-7`), routed through `fs2.0/V0Router` fit on train only. Cleaner test-provenance (`audit.md:14`) but **not automatic**: new region must explain/port 50 legacy features.
  * Backbone-54 = 6-round greedy optimizing `unweighted_pooled_test_r2_2023_2025` (`selected-features.md:5`, `fs2.0/README.md:3`). Cleanest narrative ("router = experts = global model") but test-selected — needs R4 disclosure either way.
* No station-guard exists today: all routers in `eval_formal/routers.py:26-202` are per-row `predict(frame)`; purity is diagnostic only. Only reusable precedent is `SalvagedKMeansRouter` (`formal-eval-2.1-ece-v3/eval_formal/routers.py:170-312`) with availability gate + margin fallback.

Given your pick (`Backbone54 primary`), recommended direction:

## Plan

### W0 — Freeze narrative (decision, no code)
* Primary: `Clustering_Backbone54_k2 (0,0)` — "KMeans on the shared 54-feature backbone; in-situ-target-label-free, satellite-SM-informed."
* `V0_Full_k2` demoted to robustness appendix: "legacy-router sensitivity: same full-trainval partition (ARI=1), +0.016 LOSO on n=7 ns."
* Outline-v3 §4 router table + §5-§6 wording updated accordingly; never co-rank paper-1 0.822 (R5).

### W1 — Router-agreement audit (reproducible notebook, not scratch)
* New `notebooks/experiment/derived_8.4-router-agreement-1.0/` reusing `eval14/routers.py` + `analyze_router_agreement.py` logic:
  1. Feature overlap (expect 18 shared; list both directions).
  2. Full-trainval ARI/crosstab (reproduce ARI=1).
  3. Per-LOSO-fold ARI/agreement + cluster sizes (explain 0.634→0.617).
  4. Trainval-only K-selection table (CH/DB/silhouette + purity + balance; test purity labeled "confirmation only" per `audit.md:23-24`).
* `nb execute --uv`, README tables from stdout only. This is the evidence that LOSO gap = refit instability, not a different regime concept.

### W2 — Guarded-Backbone54 variant (the "automatic" fix, minimal new model campaign)
* Implement `StationMajorityBackbone54Router` wrapping `Backbone54Router`:
  * `fit(train)`: fit KMeans on trainval as today + record per-station majority label + WA-p5 margin threshold (reuse salvage pattern).
  * `predict(frame)`: if `station_id` known (train/val/LOSO-train): broadcast station majority → guarantees station-pure by construction; if unseen (test held-out / ECE / new region): per-sample KMeans predict, with `margin < p5 → Global_Single_54` fallback (already exists as `margin_fallback` policy).
  * Alternative to test in same notebook: station-mean routing (one vector per station → one label, broadcast). Compare both on agreement + LOSO; pick one.
* Why this preserves narrative: routing features stay exactly the 54 backbone ("Full covariate routing"); guard is a deterministic post-processing rule, no hand-picked feature subset, fully recomputable per region (KMeans + majority + p5, all trainval-only).

### W3 — Minimal evaluation (no full re-campaign)
* Temporal 30-seed + LOSO 5-seed for **one** new config (`Guarded_Backbone54_k2 (0,0)`) under `formal-eval-1.0` protocol (pinned config, seed-42 replication check vs `0.814205/0.6171` baselines).
* If `Guarded ≈ V0 (±0.005)` temporal and closes LOSO gap while keeping purity=1.0 by construction → promote to primary; else keep unguarded Backbone54 primary and report guard as null result (still supports "automatic" claim as method, not as win).
* Do NOT rerun delta grids, K=3/4, OOS/ECE full sweeps — cite existing `eval-1.4` K-sweep + `formal-eval-2.0/2.1` as limitations.

### W4 — Writeup updates
* §4: provenance table row becomes `Backbone54 (+guard)` primary; V0 row → "legacy sensitivity".
* §5: purity = guard guarantee on train + diagnostic on test, with R10 caveat (static-descriptor proxy).
* §6: headline `Guarded_Backbone54 (0,0)` vs `Global_54`; footnote V0 sensitivity; §6.4 + §9 keep shared-backbone test-selection disclosure (R4) + OOS/ECE failures.
* `claims-ledger.md` + F1/F2 regenerated from notebook stdout; `outline-v2.md` untouched.

### Risks / gates
* Guard may not recover +0.016 (LOSO n=7 underpowered anyway; sign test 6/7 p=0.125). Pre-commit to narrative win even on tie: automatability > 0.016.
* New-region claim stays procedural (fit KMeans + majority + p5 on local trainval, K-selection guidance) — K=2 not universal, snow/SWE sites out-of-scope, SMAP latency documented.
* Repro gate: `nb execute --uv` must pass sequentially; `make lint/test` if `src` touched (expected: no).

Want me to flesh out W2 guard spec (majority vs station-mean, unseen-station fallback thresholds) as the next step?
