# Paper 2 technical report assembly 1.4

This directory contains the reader-facing technical report. Start with [report.md](report.md); [claims-ledger.md](claims-ledger.md) records the evidence status and exact source paths for its claims. [references.bib](references.bib) and [literature-verification.md](provenance/literature-verification.md) contain reference metadata and DOI links.

`report.pdf` is a local build artifact, not a committed file (see `.gitignore`): generate it with `scripts/export_pdf.py` for reading or review. The validator checks the PDF when present and skips those checks on a fresh clone.

## Evidence and provenance

The report presents the Washington temporal and held-out-station evaluation alongside the ECE station evaluation as one final evidence batch. The Washington results are fully seed-paired: the multi-regime models ran in `notebooks/experiment/paper2-final-evidence-1.0` and the single-regime global model was refit under the same protocol in `notebooks/experiment/paper2-final-evidence-1.2` (same seed lists, same splits, same features and hyperparameters). The paired-global temporal refit reproduces the earlier sensitivity values exactly; for LOSO this report uses 0.579737, the mean of the saved seed rows in both harnesses, rather than the stale 0.5795 in an older summary file. The source manifest and claims ledger preserve the exact run folders, inputs, source precedence, and SHA-256 hashes behind each result. No formal-eval-1.0 sources remain; earlier experiments appear only as scope boundaries.

The station network labels are derived from the versioned Washington station configuration and checked by the report-data script. The temporal/LOSO figure is regenerated from the paired evaluation tables by the versioned notebook at [notebooks/report_figures.ipynb](notebooks/report_figures.ipynb); the ECE figures are unchanged. This revision does not require model fitting or split regeneration.

## Reproduce and validate

Run these commands from the repository root. Python commands use the `notebooks/` uv environment. PDF rendering adds pinned `weasyprint==66.0` and `mistune==3.1.4` for the command; validation adds `pypdf==6.0.0`. On this cluster, load Pango before rendering.

```bash
cd notebooks
uv run --no-sync python ../writeup2/report-assemble-1.4/scripts/assemble_report.py
module load GCCcore/12.3.0 Pango/1.50.14  # if unavailable, GCCcore/11.2.0 Pango/1.48.8 also renders with weasyprint 66 (verified on this cluster 2026-10-04)
uv run --no-sync --with 'weasyprint==66.0' --with 'mistune==3.1.4' python ../writeup2/report-assemble-1.4/scripts/export_pdf.py
uv run --no-sync --with 'pypdf==6.0.0' python ../writeup2/report-assemble-1.4/scripts/validate_assembly.py
```

Regenerate the figures from the `notebooks/` uv environment (the venv `bin` must be on `PATH` so the kernel entry point resolves):

```bash
cd notebooks
PATH="$PWD/.venv/bin:$PATH" nb execute ../writeup2/report-assemble-1.4/notebooks/report_figures.ipynb --kernel python3 -t 300
```

Assembly inserts values from saved CSV/JSON artifacts. The validator checks the report against its templates, source hashes, paired seed/station coverage and Global seed alignment, station networks, ECE mapping provenance, local links, figures, and PDF content. If figure data or presentation change, create a new versioned report-figure notebook whose output target is this directory; keep the existing versioned notebook unchanged.

## Reading guide

There is no up-front model glossary. Each model is named and described the first time it matters: §1 names the method under test and the two models it is compared against, §3 states that every model shares one target, one 54-feature set, and one XGBoost configuration, and §4 is where the shared-feature cluster-routed multi-regime model family, the station-majority rule (all observations of one station held inside a single regime), the model without that guarantee, and the existing single-regime global model are all defined. Both flagship gaps in Tables 1–2 are seed-paired differences: station-majority versus the same model without the guarantee (isolating the rule) and station-majority versus the paired global refit (isolating grouping itself). The four simpler grouping rows sit alongside as labeled context from a different saved run, with per-row evidence marked in §4 and in the tables. The previous paper's R² of 0.822 is a different-protocol anchor, never a baseline. There is no background-variant appendix. Group labels describe this Washington station sample; the report does not treat them as universal climate classes. The ECE section states the SMAP shortfall first, then the Washington-selected precipitation fallback, including the fact that all 150 ECE rows took the same fallback branch.

## What changed since 1.3

- The global comparator is now a seed-paired refit, not sensitivity-anchored context: `paper2-final-evidence-1.2` refit `Global_Single` under the same protocol with the same 30 temporal and 5 LOSO seeds (temporal bit-identical to the old values; LOSO 0.579737, the seed-row mean in both harnesses). Tables 1–2 carry three paired flagship rows plus four labeled context rows; §§1/5/6 report two paired differences (versus twin, versus global) with seed win/tie/loss counts, including three LOSO folds that favor the global model.
- Dropped all `derived_8.4-formal-eval-1.0` sources and the simpler-grouping context rows (seasonal, precipitation-index, three-feature, target-threshold) and the earlier-variant appendix: every Washington number in the report is now `paired_in_harness`.
- Expanded §4 into a fuller methods core: the fit→predict chain in words, group sizes (10624/3984) with minimum station purity 1.000 on the full-trainval fit, explicit drier-group canonicalization, and the two pairings stated with what each isolates.
- Split the old open-comparison ledger entry: the same-holdout Guarded-versus-global comparison is resolved (C20); an evaluated global fallback for ECE routing remains future work (C21).
- Restored full 1.3 model coverage with no new compute: the four simpler grouping rules (seasonal, precipitation-index, three-feature, target-threshold) return as labeled context rows in Tables 1–2 (temporal from saved sensitivity rows, LOSO no-delta rows from the saved formal summary) plus a per-row Evidence column in the §4 table; the flagship paired results are untouched.

## Earlier changes (1.2 → 1.3)

- Deleted the opening glossary that listed every model before any results. Its content is now woven in where it is needed: the shared target/features/XGBoost statement moved to §3, the family and station-majority rule definitions to §4, the global-comparator caveat to §4, the router fit recipe to §4, and the ECE-station definition to §1 and §8.
- Retired the intermediary alias that stood in for the headline model, as well as the one-word label used for its ablation.
- Renamed every model to one descriptive, conflict-free name (this bullet is the name registry). The §4 family is the **shared-feature cluster-routed multi-regime model**; the headline is the **station-majority shared-feature cluster-routed multi-regime model**; its paired ablation is the **shared-feature cluster-routed multi-regime model without station consistency guarantee**; the comparator is the **Existing single-regime global model** (the out-of-state row keeps its 54-feature run qualifier). Table rows, figure labels, and the claims ledger follow these names, and the validator fails if any retired alias — the earlier generic headline/ablation/family wording or the old global-model wording — reappears in the report, ledger, or this file. Names stay free of hard-coded feature or seed counts so the method can be described without pinning the paper to this run's backbone.
