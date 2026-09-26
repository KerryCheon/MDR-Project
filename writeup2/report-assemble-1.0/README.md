# Paper 2 technical report assembly 1.0

This directory is the PI handoff. Read [report.md](report.md) for methods, results, interpretation, limits, and paper guidance, or [report.pdf](report.pdf) for the formatted reading copy. [claims-ledger.md](claims-ledger.md) marks the evidentiary status of each paper-ready claim. [references.bib](references.bib) and [literature-verification.md](provenance/literature-verification.md) hold reference metadata and verification links.

## Evidence boundary

- `writeup2/outline-v7.md` supplies framing.
- `paper2-final-evidence-1.0` supplies the main temporal and paired Guarded–Backbone LOSO results.
- `paper2-final-evidence-1.1` supplies **all** ECE policy, mapping, calibration, and site results. It supersedes the preliminary ECE account in 1.0.
- Earlier saved experiments supply only partition diagnostics, sensitivity comparisons, out-of-state limits, and historical context. The report labels their comparison role explicitly.
- No model fitting, split regeneration, or edits to existing evidence notebooks are part of this assembly.

[source_manifest.json](provenance/source_manifest.json) lists every input path and SHA-256 hash, the source precedence, split identifiers, seed counts, and the extraction rule. [report_data.py](scripts/report_data.py) is the executable specification of extracted values and schema/count checks. The report and ledger are generated from Markdown templates; their numbers and comparisons are computed from saved CSV/JSON, including values embedded in interpretation paragraphs. The submitted first-paper R² is a separately cited historical figure from its PDF.

## Reproduce

Run these commands at the repository root. `nb` is the project notebook CLI described in `.agents/skills/notebook-cli/SKILL.md`; Python commands use the `notebooks/` uv environment. PDF rendering adds pinned `weasyprint==66.0` and `mistune==3.1.4` to that environment for the command only. The validator adds `pypdf==6.0.0` to check the PDF's text and embedded figures. On this cluster, load the Pango module shown below before rendering.

```bash
cd notebooks
nb execute experiment/paper2-report-assemble-1.0/report_figures.ipynb --uv --timeout 120
uv run --no-sync python ../writeup2/report-assemble-1.0/scripts/assemble_report.py
module load GCCcore/12.3.0 Pango/1.50.14
uv run --no-sync --with 'weasyprint==66.0' --with 'mistune==3.1.4' python ../writeup2/report-assemble-1.0/scripts/export_pdf.py
uv run --no-sync --with 'pypdf==6.0.0' python ../writeup2/report-assemble-1.0/scripts/validate_assembly.py
```

The source notebook is [report_figures.ipynb](../../notebooks/experiment/paper2-report-assemble-1.0/report_figures.ipynb); [the report-local notebook link](notebooks/report_figures.ipynb) points to it. `nb execute --uv` must use the actual path under `notebooks/` so uv finds the notebook environment. The notebook reads saved evidence tables and writes the three files in `figures/`. It prints the key temporal, LOSO, and ECE values used for cross-checking the report.

The assembly validator compares the generated Markdown and manifest to the current templates and source hashes, checks source coverage and mapping audits, validates all local links and figure files, and verifies the PDF exists. A change to frozen evidence requires reviewing the resulting diff and its interpretation, then executing the sequence above again.

## Reading cautions

The strongest new comparison is the paired Guarded–Backbone ablation. The apparent Guarded–global gaps use sensitivity anchors from another harness. The shared feature backbone and other choices saw the 2023–2025 test era, so the metrics are development evidence. ECE contains five disjoint sites and 150 late-summer daily rows; its forced one-expert policies are diagnostic and non-deployable. A direct paired Guarded–global LOSO run and evaluated global fallback remain open work.
