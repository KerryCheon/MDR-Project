# Paper 2 technical report assembly 1.1

This directory contains the reader-facing technical report and its formatted PDF. Start with [report.md](report.md); [claims-ledger.md](claims-ledger.md) records the evidence status and exact source paths for its claims. [references.bib](references.bib) and [literature-verification.md](provenance/literature-verification.md) contain reference metadata and DOI links.

## Evidence and provenance

The report presents the Washington temporal and held-out-station evaluation alongside the ECE station evaluation as one final evidence batch. These evaluations use different stations and protocols. The source manifest and claims ledger preserve the exact run folders, inputs, source precedence, and SHA-256 hashes behind each result.

The station network labels are derived from the versioned Washington station configuration and checked by the report-data script. The figures are unchanged carry-forward outputs; their source notebook is linked at [notebooks/report_figures.ipynb](notebooks/report_figures.ipynb). This revision does not require model fitting, split regeneration, or notebook edits.

## Reproduce and validate

Run these commands from the repository root. Python commands use the `notebooks/` uv environment. PDF rendering adds pinned `weasyprint==66.0` and `mistune==3.1.4` for the command; validation adds `pypdf==6.0.0`. On this cluster, load Pango before rendering.

```bash
cd notebooks
uv run --no-sync python ../writeup2/report-assemble-1.1/scripts/assemble_report.py
module load GCCcore/12.3.0 Pango/1.50.14
uv run --no-sync --with 'weasyprint==66.0' --with 'mistune==3.1.4' python ../writeup2/report-assemble-1.1/scripts/export_pdf.py
uv run --no-sync --with 'pypdf==6.0.0' python ../writeup2/report-assemble-1.1/scripts/validate_assembly.py
```

Assembly inserts values from saved CSV/JSON artifacts. The validator checks the report against its templates, source hashes, station networks, result coverage, ECE mapping provenance, local links, figures, and PDF content. If figure data or presentation change, create a new versioned report-figure notebook whose output target is this directory; keep the existing versioned notebook unchanged.

## Reading guide

The primary regional model and its comparison with the existing global model are introduced before the detailed methods. Appendix B preserves the earlier V0 model and its feature-set results as historical context. Group labels describe this Washington station sample; the report does not treat them as universal climate classes. The ECE section explains the regional models at new stations, then compares the assignment options available for that evaluation.
