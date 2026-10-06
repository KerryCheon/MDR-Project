# Paper 2 technical report assembly 1.5

This directory contains the reader-facing technical report. Start with [report.md](report.md); [claims-ledger.md](claims-ledger.md) records the evidence status and exact source paths for its claims. [references.bib](references.bib) and [literature-verification.md](provenance/literature-verification.md) contain reference metadata and DOI links.

`report.pdf` is a local build artifact, not a committed file (see `.gitignore`): generate it with `scripts/export_pdf.py` for reading or review. The validator checks the PDF when present and skips those checks on a fresh clone.

## Evidence and provenance

The report presents the Washington temporal and held-out-station evaluation alongside the ECE station evaluation as one final evidence batch. The Washington results are fully seed-paired: the multi-regime models ran in `notebooks/experiment/paper2-final-evidence-1.0` and the single-regime global model was refit under the same protocol in `notebooks/experiment/paper2-final-evidence-1.2` (same seed lists, same splits, same features and hyperparameters). The paired-global temporal refit reproduces the earlier sensitivity values exactly; for LOSO this report uses 0.579737, the mean of the saved seed rows in both harnesses, rather than the stale 0.5795 in an older summary file. The source manifest and claims ledger preserve the exact run folders, inputs, source precedence, and SHA-256 hashes behind each result. No formal-eval-1.0 sources remain; earlier experiments appear only as scope boundaries.

The station network labels are derived from the versioned Washington station configuration and checked by the report-data script. The temporal/LOSO figure and the ECE policy/site figure are regenerated from the paired evaluation tables by the versioned notebook at [notebooks/report_figures.ipynb](notebooks/report_figures.ipynb). This revision does not require model fitting or split regeneration.

## Reproduce and validate

Run these commands from the repository root. Python commands use the `notebooks/` uv environment. PDF rendering adds pinned `weasyprint==66.0` and `mistune==3.1.4` for the command; validation adds `pypdf==6.0.0`. On this cluster, load Pango before rendering.

```bash
cd notebooks
uv run --no-sync python ../writeup2/report-assemble-1.5/scripts/assemble_report.py
module load GCCcore/12.3.0 Pango/1.50.14  # if unavailable, GCCcore/11.2.0 Pango/1.48.8 also renders with weasyprint 66 (verified on this cluster 2026-10-04)
uv run --no-sync --with 'weasyprint==66.0' --with 'mistune==3.1.4' python ../writeup2/report-assemble-1.5/scripts/export_pdf.py
uv run --no-sync --with 'pypdf==6.0.0' python ../writeup2/report-assemble-1.5/scripts/validate_assembly.py
```

Regenerate the figures from the `notebooks/` uv environment (the venv `bin` must be on `PATH` so the kernel entry point resolves):

```bash
cd notebooks
PATH="$PWD/.venv/bin:$PATH" nb execute ../writeup2/report-assemble-1.5/notebooks/report_figures.ipynb --kernel python3 -t 300
```

Assembly inserts values from saved CSV/JSON artifacts. The validator checks the report against its templates, source hashes, paired seed/station coverage and Global seed alignment, station networks, ECE mapping provenance, local links, figures, and PDF content. If figure data or presentation change, create a new versioned report-figure notebook whose output target is this directory; keep the existing versioned notebook unchanged.

## Reading guide

There is no up-front model glossary. Each model is named and described the first time it matters: §1 names the method under test and the two models it is compared against, §3 states that every model shares one target, one 54-feature set, and one XGBoost configuration, and §4 is a self-contained method description of the shared-feature cluster-routed multi-regime model family: the routing idea, the K-means router and its canonicalization, the station-majority rule (all observations of one station held inside a single regime), the model without that guarantee, the existing single-regime global model, the availability and margin gates, and the fit/predict procedure. §4 explains the method only and reports no results; the paired differences (station-majority versus the same model without the guarantee, and station-majority versus the paired global refit) are discussed in §§5–6. The four simpler grouping rows sit alongside as labeled context from a different saved run, marked in the tables. The previous paper's R² of 0.822 is a different-protocol anchor, never a baseline. There is no background-variant appendix. Group labels describe this Washington station sample; the report does not treat them as universal climate classes. The ECE section states the SMAP shortfall first, then the Washington-selected precipitation fallback, including the fact that all 150 ECE rows took the same fallback branch.

## What changed since 1.4

- Rewrote §4 as a concept-only method core, framed as an original design. It now gives the routing idea, the K-means router recipe and label canonicalization, k=2 as a Washington scope choice, the station-majority rule as a controlled comparison, assignment for unseen/missing/gated rows with the fixed 0.10 availability gate and fit-frame margin fallback, an explicit temporal and leave-one-station-out fit/predict procedure, and a provenance note that the router reuses the global model's 54-feature set without claiming it is optimal for routing.
- Removed all performance gaps and win/tie/loss counts from §4; those results remain in §§5–6, which are unchanged. No new evidence, numbers, or model fits: §1–§3 and §5–§10 are unchanged from 1.4.
- Retargeted the versioned report-figures notebook to this directory; the temporal/LOSO and ECE policy/site figures are unchanged.
- Added a §2 **Contribution relative to the first paper** paragraph that frames the work as a controlled study of the grouping/routing step (answering a question the first paper left as future work) rather than a new architecture.
- Added a §7 discussion of specialization versus data: each specialist is fitted on a subset of the rows yet matches or beats the all-data global predictor on its own group, additional inputs did not help the larger/ saturated group, and the global model carries larger pooled bias. Every figure comes from saved artifacts (`temporal_seed_cluster.csv`, the paired-global temporal rows, and the per-regime feature-addition `delta_grid.csv`), the point is labeled as interpretation, no controlled data-quantity experiment is claimed, and no station set is compared across dataset versions.
- Overhauled §8 from a results recap into an interpretation of the regional labels. It now reads the Washington split as a wet/maritime versus drier/seasonal contrast (separation index 1.00 on longitude and wettest-month precipitation, 0.93 on satellite soil moisture, only 0.21 on elevation) with elevation a secondary axis, and shows that at the new lowland ECE sites the error follows the expert a row is routed to rather than the site location. It removes the daily-overlay figure and keeps the ECE policy/site figure, so the report has two figures; the ECE tables and all numbers are unchanged, and the elevation/hydroclimate and ECE site values are anchored in `report_data.py` from the group profile and site descriptors. It also states that routing stays automatic — the input-availability gate applies the fallback and the class-to-expert direction is fixed once on Washington validation — so the per-site comparisons are diagnostics, not a manual regime-selection step.

## Provenance additions (1.5 revision)

- Added a §3 **Data sources and preparation** paragraph that names the data sources (NOAA USCRN and USDA ISMN SNOTEL, SMAP, Sentinel-1, Sentinel-2, MODIS, ERA5-Land, and static SRTM/WorldClim/WorldCover/HWSD descriptors) and states that the dataset reuses the first paper's multi-source pipeline: acquisition, temporal alignment, ensemble imputation, smoothing, and feature engineering.
- Added a §3 **Feature selection** paragraph describing the shared 54-feature backbone as continuing the project's feature-selection and validation pipeline, with a matching claims-ledger row (C28).

## Earlier changes (1.2 → 1.3)

- Deleted the opening glossary that listed every model before any results. Its content is now woven in where it is needed: the shared target/features/XGBoost statement moved to §3, the family and station-majority rule definitions to §4, the global-comparator caveat to §4, the router fit recipe to §4, and the ECE-station definition to §1 and §8.
- Retired the intermediary alias that stood in for the headline model, as well as the one-word label used for its ablation.
- Renamed every model to one descriptive, conflict-free name (this bullet is the name registry). The §4 family is the **shared-feature cluster-routed multi-regime model**; the headline is the **station-majority shared-feature cluster-routed multi-regime model**; its paired ablation is the **shared-feature cluster-routed multi-regime model without station consistency guarantee**; the comparator is the **Existing single-regime global model** (the out-of-state row keeps its 54-feature run qualifier). Table rows, figure labels, and the claims ledger follow these names, and the validator fails if any retired alias — the earlier generic headline/ablation/family wording or the old global-model wording — reappears in the report, ledger, or this file. Names stay free of hard-coded feature or seed counts so the method can be described without pinning the paper to this run's backbone.
