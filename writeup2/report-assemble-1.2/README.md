# Paper 2 technical report assembly 1.2

This directory contains the reader-facing technical report. Start with [report.md](report.md); [claims-ledger.md](claims-ledger.md) records the evidence status and exact source paths for its claims. [references.bib](references.bib) and [literature-verification.md](provenance/literature-verification.md) contain reference metadata and DOI links.

`report.pdf` is a local build artifact, not a committed file (see `.gitignore`): generate it with `scripts/export_pdf.py` for reading or review. The validator checks the PDF when present and skips those checks on a fresh clone.

## Evidence and provenance

The report presents the Washington temporal and held-out-station evaluation alongside the ECE station evaluation as one final evidence batch. These evaluations use different stations and protocols. The source manifest and claims ledger preserve the exact run folders, inputs, source precedence, and SHA-256 hashes behind each result. All numbers are identical to assembly 1.1 (same saved artifacts); only the prose, table framing, and figure captions changed.

The station network labels are derived from the versioned Washington station configuration and checked by the report-data script. The figures are unchanged carry-forward outputs; their source notebook is linked at [notebooks/report_figures.ipynb](notebooks/report_figures.ipynb). This revision does not require model fitting, split regeneration, or notebook edits.

## Reproduce and validate

Run these commands from the repository root. Python commands use the `notebooks/` uv environment. PDF rendering adds pinned `weasyprint==66.0` and `mistune==3.1.4` for the command; validation adds `pypdf==6.0.0`. On this cluster, load Pango before rendering.

```bash
cd notebooks
uv run --no-sync python ../writeup2/report-assemble-1.2/scripts/assemble_report.py
module load GCCcore/12.3.0 Pango/1.50.14
uv run --no-sync --with 'weasyprint==66.0' --with 'mistune==3.1.4' python ../writeup2/report-assemble-1.2/scripts/export_pdf.py
uv run --no-sync --with 'pypdf==6.0.0' python ../writeup2/report-assemble-1.2/scripts/validate_assembly.py
```

Assembly inserts values from saved CSV/JSON artifacts. The validator checks the report against its templates, source hashes, station networks, result coverage, ECE mapping provenance, local links, figures, and PDF content. If figure data or presentation change, create a new versioned report-figure notebook whose output target is this directory; keep the existing versioned notebook unchanged.

## Reading guide

The three main models (two-regime family, primary regional model with station-majority rule, unguarded twin, single-regime global model) are defined before any results. Only the primary-vs-unguarded gap is a paired test; seasonal, precipitation-index, dynamic-feature, target-threshold, and global rows are contemporary context from a different saved run. The previous paper's R² of 0.822 is a different-protocol anchor, never a baseline. Appendix B confines the earlier 50-feature V0 variant. Group labels describe this Washington station sample; the report does not treat them as universal climate classes. The ECE section states the SMAP shortfall first, then the Washington-selected precipitation fallback, including the fact that all 150 ECE rows took the same fallback branch.

## What changed since 1.1

- Terms defined before use: two-regime family → primary (station-majority) → unguarded twin → global; one name per concept, internal codenames (`Guarded/Backbone/V0`, `c0/c1`, `trainval`) translated or removed from prose.
- Section 4 lists every grouping rule explicitly: seasonal months (May–Oct vs Nov–Apr), `G_API` fit-median split, dynamic K-means 3-feature list, target-threshold classifier (0.16 m³/m³ gate), 54-feature K-means recipe plus station-majority rule.
- Removed the `Comparison role` / paired-vs-historical-reference taxonomy from Tables 1–2; replaced with paired-result vs contemporary-context vs previous-paper-anchor framing. Table 8 shows the primary model plus global and single-predictor references (unguarded deployable rows are bit-identical and stated in text; soft blend matches at reported precision). Out-of-state stays a scope boundary; V0 stays in Appendix B.
- Reworked §4 into the methods core: states the multi-regime-vs-single-regime claim, demotes k=2 to a Washington setting (other regions must re-tune; Table 4 caption says so), surfaces group membership and fit-frame group means, justifies hard assignment via the `auto_soft` run, and slims the grouping table to inputs + rule (SMAP/target columns folded in). Membership repeats removed from §§7–8.
- Appendix A groups the 54 features into 7 themes with per-feature glosses plus a name-prefix legend (all 54 names still verbatim). §3 station table drops the `Trainval local group` column (no hand-assigned groups anywhere). Table 2 drops the `Earlier global model` row. Table 4 caption defines Calinski–Harabasz/Davies–Bouldin. ECE stations defined in the read-first box.
