# Summer research showcase poster

One landscape poster, **36 inches wide × 24 inches high**, based on the two-regime work summarized in [technical report 1.5](../report-assemble-1.5/report.md). The supplied reference determines the Seattle University branding, red section bars, white background, and three-column structure.

## Deliverables

- [Editable PowerPoint](summer_showcase.pptx): text, credit placeholders, and the method diagram are native editable objects. Charts and the map are embedded at 300 dpi.
- [Print PDF](summer_showcase.pdf): one page at the physical poster size, with embedded fonts and vector charts/map.
- [Poster preview](summer_showcase_preview.png).
- [Executed figure/build notebook](poster_figures.ipynb).
- [Validation results](artifacts/validation.json), [captured notebook stdout](artifacts/notebook_stdout.txt), and [source manifest](artifacts/source_manifest.json).

Replace the `[Author names]` and `[Mentor name, credentials]` text in PowerPoint before printing. After manual edits, export the edited PowerPoint to a new PDF at its existing slide size. The supplied PDF reflects the reproducible source and initially contains those placeholders too.

The poster uses DejaVu Sans, a close sans-serif match to the template. Its regular/bold fonts and license are included in `assets/fonts/`; install them if your PowerPoint machine does not have this family. The two university logos are copied from the user-supplied PowerPoint, preserving the source image resolution. Rebuilding uses these bundled assets and never requires the temporary attachment folder.

## Results printed by the notebook

The tables below are copied from the executed evidence cell's stdout, also saved as [readme_results.md](artifacts/readme_results.md). R² results compare the station-majority two-regime model with the global predictor under matched inputs, settings, seeds, and evaluation rows.

| Evaluation | Global | Two-regime |
| --- | ---: | ---: |
| Later-year R² | 0.7798 | 0.8118 |
| Held-out-station mean R² | 0.5797 | 0.6379 |

| ECE assignment | Pooled RMSE (m³/m³) |
| --- | ---: |
| Standard routing | 0.1674 |
| Precipitation fallback | 0.0578 |
| Global predictor | 0.0586 |

The full station table and group-profile medians are printed in the notebook and saved in `artifacts/`. Temporal results cover 30 matched fitting seeds. Station-holdout results cover five seeds at each of seven stations: four station-level gains and three losses. Seed repetition does not establish uncertainty across independently sampled sites or seasons.

ECE results cover 150 daily observations at five new sensor sites, 20 July–19 August 2026. All fallback rows took one specialist branch; its error is close to the global model's error. The poster therefore presents this as a routing/availability lesson, without claiming full-season transfer or general superiority on these sites. The first paper's different-protocol R² is not used as a comparator.

## Reproduce

Run from the repository root. The existing `notebooks/` uv environment supplies scientific dependencies; `requirements.txt` pins the added export dependencies. No model training or split generation is performed.

```bash
cd notebooks
timeout 180s uv pip install --python .venv/bin/python -r ../writeup2/poster-summer-1.0/requirements.txt
timeout 30s uv run --no-sync python -m ipykernel install --user --name mdr-poster-uv --display-name "MDR poster (uv)"
UV_NO_SYNC=1 PATH="$PWD/.venv/bin:$PATH" timeout 240s nb execute ../writeup2/poster-summer-1.0/poster_figures.ipynb --uv --kernel mdr-poster-uv -t 180
```

The explicit kernel name prevents a stale user-wide `python3` kernel from selecting another project's environment. `UV_NO_SYNC=1` preserves the pinned poster packages installed above while `nb --uv` launches the notebook. When opening the notebook interactively, select **MDR poster (uv)**.

The dependency cell also checks the export versions and installs missing ones into the active notebook environment. For the captured stdout artifact, redirect the final command's output to `../writeup2/poster-summer-1.0/artifacts/notebook_stdout.txt` (the included `artifacts/` directory already exists).

All figures are generated through notebook execution. Tracked modules separate evidence extraction (`scripts/poster_data.py`), scientific figures (`scripts/poster_plots.py`), and the shared poster layout/export (`scripts/build_poster.py`). The layout uses inches and measured font widths for both formats. SVG/PDF versions of all four figures are saved alongside their PNG versions in `figures/`.

To edit reproducible copy or geometry, change `scripts/build_poster.py` and execute the notebook again. Credit placeholders are deliberately editable without requiring code changes. Do not modify earlier versioned experiment or report notebooks.

## Sources and checks

- The report's existing extractor validates its saved evidence and source coverage before plotting. The poster separately joins temporal seeds and LOSO seed/station pairs and checks the report's displayed headline values.
- Main two-regime results: `notebooks/experiment/paper2-final-evidence-1.0/`; matched global results: `paper2-final-evidence-1.2/`; ECE results and routing audits: `paper2-final-evidence-1.1/`.
- Map coordinates/memberships and group profiles come from the report's declared `derived_8.4-gating-analysis-1.0` tables. The cached Natural Earth admin-1 file is explicitly filtered to `iso_3166_2 == US-WA`. No continuous geographic regime boundaries are inferred.
- [Branding provenance](assets/branding_provenance.json) records the template hash, selected picture names, dimensions, and copied-asset hashes. [Source manifest](artifacts/source_manifest.json) records evidence and map/report hashes.
- Export validation checks one slide/page, 36″ × 24″ dimensions, embedded PDF fonts, all seven station names, all headline metric values, editable credit text, measured text fit, and the total rendered word count including chart labels.
- The PDF preview was visually reviewed for clipping, label collisions, and reading order. PowerPoint was structurally validated; no PowerPoint/LibreOffice rendering engine is installed on this host. Both outputs use the same measured layout, with editable native shapes in the PowerPoint.

The reproduction code and output notebook form this poster's source of record; the existing report/evidence directories remain unchanged.
