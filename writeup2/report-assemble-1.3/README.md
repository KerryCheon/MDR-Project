# Paper 2 technical report assembly 1.3

This directory contains the reader-facing technical report. Start with [report.md](report.md); [claims-ledger.md](claims-ledger.md) records the evidence status and exact source paths for its claims. [references.bib](references.bib) and [literature-verification.md](provenance/literature-verification.md) contain reference metadata and DOI links.

`report.pdf` is a local build artifact, not a committed file (see `.gitignore`): generate it with `scripts/export_pdf.py` for reading or review. The validator checks the PDF when present and skips those checks on a fresh clone.

## Evidence and provenance

The report presents the Washington temporal and held-out-station evaluation alongside the ECE station evaluation as one final evidence batch. These evaluations use different stations and protocols. The source manifest and claims ledger preserve the exact run folders, inputs, source precedence, and SHA-256 hashes behind each result. All numbers are identical to assembly 1.2 (same saved artifacts); only the prose, model names, table labels, and figure labels changed.

The station network labels are derived from the versioned Washington station configuration and checked by the report-data script. The figures are regenerated from the same saved evidence by the versioned notebook at [notebooks/report_figures.ipynb](notebooks/report_figures.ipynb); only their labels changed. This revision does not require model fitting or split regeneration.

## Reproduce and validate

Run these commands from the repository root. Python commands use the `notebooks/` uv environment. PDF rendering adds pinned `weasyprint==66.0` and `mistune==3.1.4` for the command; validation adds `pypdf==6.0.0`. On this cluster, load Pango before rendering.

```bash
cd notebooks
uv run --no-sync python ../writeup2/report-assemble-1.3/scripts/assemble_report.py
module load GCCcore/12.3.0 Pango/1.50.14
uv run --no-sync --with 'weasyprint==66.0' --with 'mistune==3.1.4' python ../writeup2/report-assemble-1.3/scripts/export_pdf.py
uv run --no-sync --with 'pypdf==6.0.0' python ../writeup2/report-assemble-1.3/scripts/validate_assembly.py
```

Regenerate the figures from the `notebooks/` uv environment (the venv `bin` must be on `PATH` so the kernel entry point resolves):

```bash
cd notebooks
PATH="$PWD/.venv/bin:$PATH" nb execute ../writeup2/report-assemble-1.3/notebooks/report_figures.ipynb --kernel python3 -t 300
```

Assembly inserts values from saved CSV/JSON artifacts. The validator checks the report against its templates, source hashes, station networks, result coverage, ECE mapping provenance, local links, figures, and PDF content. If figure data or presentation change, create a new versioned report-figure notebook whose output target is this directory; keep the existing versioned notebook unchanged.

## Reading guide

There is no up-front model glossary. Each model is named and described the first time it matters: §1 names the method under test and the two models it is compared against, §3 states that every model shares one target, one 54-feature set, and one XGBoost configuration, and §4 is where the shared-feature cluster-routed multi-regime model family, the station-majority rule (all observations of one station held inside a single regime), the model without that guarantee, and the existing single-regime global model are all defined. Only the gap between the station-majority shared-feature cluster-routed multi-regime model and the same model without the station consistency guarantee is a paired test; seasonal, precipitation-index, three-feature K-means, target-threshold, and global rows are contemporary context from a different saved run. The previous paper's R² of 0.822 is a different-protocol anchor, never a baseline. Appendix B confines the earlier 50-feature V0 variant. Group labels describe this Washington station sample; the report does not treat them as universal climate classes. The ECE section states the SMAP shortfall first, then the Washington-selected precipitation fallback, including the fact that all 150 ECE rows took the same fallback branch.

## What changed since 1.2

- Deleted the opening glossary that listed every model before any results. Its content is now woven in where it is needed: the shared target/features/XGBoost statement moved to §3, the family and station-majority rule definitions to §4, the global-comparator caveat to §4, the router fit recipe to §4, and the ECE-station definition to §1 and §8.
- Retired the intermediary alias that stood in for the headline model, as well as the one-word label used for its ablation.
- Renamed every model to one descriptive, conflict-free name (this bullet is the name registry). The §4 family is the **shared-feature cluster-routed multi-regime model**; the headline is the **station-majority shared-feature cluster-routed multi-regime model**; its paired ablation is the **shared-feature cluster-routed multi-regime model without station consistency guarantee**; the comparator is the **Existing single-regime global model** (the out-of-state row keeps its 54-feature run qualifier). The three-input router is now **Three-feature K-means grouping**, the ECE single-predictor references are **Single regime predictor 0/1 (reference)**, the ECE station table columns are labeled by assignment rather than by model, and the appendix background model is the **Earlier V0 multi-regime model** (short form in prose: the V0 model). Table rows, figure labels, and the claims ledger follow these names, and the validator fails if any retired alias — the earlier generic headline/ablation/family wording, the old global-model wording, the old three-input grouping wording, or the old V0/out-of-state wording — reappears in the report, ledger, or this file. Names stay free of hard-coded feature or seed counts so the method can be described without pinning the paper to this run's backbone.

## Earlier changes (1.1 → 1.2)

- Terms defined before use: two-regime family → station-majority model → model without the guarantee → global; one name per concept, internal codenames (`Guarded/Backbone/V0`, `c0/c1`, `trainval`) translated or removed from prose.
- Section 4 lists every grouping rule explicitly: seasonal months (May–Oct vs Nov–Apr), `G_API` fit-median split, dynamic K-means 3-feature list, target-threshold classifier (0.16 m³/m³ gate), 54-feature K-means recipe plus station-majority rule.
- Removed the `Comparison role` / paired-vs-historical-reference taxonomy from Tables 1–2; replaced with paired-result vs contemporary-context vs previous-paper-anchor framing. Table 8 shows the station-majority model plus global and single-predictor references (the rows of the model without the guarantee are bit-identical and stated in text; soft blend matches at reported precision). Out-of-state stays a scope boundary; V0 stays in Appendix B.
- Reworked §4 into the methods core: states the multi-regime-vs-single-regime claim, demotes k=2 to a Washington setting (other regions must re-tune; Table 4 caption says so), surfaces group membership and fit-frame group means, justifies hard assignment via the `auto_soft` run, and slims the grouping table to inputs + rule (SMAP/target columns folded in). Membership repeats removed from §§7–8.
- Appendix A groups the 54 features into 7 themes with per-feature glosses plus a name-prefix legend (all 54 names still verbatim). §3 station table drops the `Trainval local group` column (no hand-assigned groups anywhere). Table 2 drops the `Earlier global model` row. Table 4 caption defines Calinski–Harabasz/Davies–Bouldin.
