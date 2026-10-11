# Summer research poster 1.1

A 36″ wide × 24″ high poster about the two-regime work summarized in [technical report 1.5](../report-assemble-1.5/report.md). Version 1.1 keeps the Seattle University branding and four scientific graphics, with fuller explanations of the research question, methods, and results.

## Deliverables

- [Editable PowerPoint](summer_showcase.pptx): native text, credit placeholders, method shapes, and hard-routing equation; scientific figures at 300 dpi.
- [Print-ready PDF](summer_showcase.pdf): one page at the physical poster size, with embedded fonts and vector figures.
- [PNG preview](summer_showcase_preview.png) and [larger print-detail rendering](artifacts/print_detail.png).
- [Executed notebook](poster_figures.ipynb), [captured stdout](artifacts/notebook_stdout.txt), [validation record](artifacts/validation.json), and [source manifest](artifacts/source_manifest.json).

Replace `[Author names]` and `[Mentor name, credentials]` in PowerPoint before printing. The PDF contains the same placeholders. After manual edits, export the edited PowerPoint at its existing slide size. For reproducible edits, update the shared layout source and execute the notebook again.

## Revision choices

The opening asks "Why soil moisture estimation?" Methods and results explain how specialists are selected, how the global comparison is controlled, which periods are used for fitting and testing, and what repeated seeds and the ECE evaluation can establish. The target is 850 to 950 words, including headings and chart labels. Main paragraphs use 22 pt type, with smaller supporting labels and captions.

The feature-count and target-depth badges and citation footer are removed. The target depth appears in the methods, and the ECE panel acknowledges the collaboration. The compact method diagram leaves room for numbered explanations, an editable equation, and figure captions. Source references are retained below.

"Wet / Mountains" and "Drier / Lowlands" are display names for the report's existing groups, defined in `scripts/poster_data.py`. The map, diagram, and profiles use the same names. Station memberships remain:

- Wet / Mountains: Beaver Pass, Cayuse Pass, Darrington, Paradise, Quinault.
- Drier / Lowlands: Spokane, Sourdough Gulch.

Report 1.5 §§7 and 8 shows that the wet group includes lowland and mountain stations and that elevation alone weakly separates the groups. The profile values are medians of input conditions; they do not define universal elevation classes or thresholds on measured soil moisture. The map shows station assignments without geographic regime boundaries.

The language polish specifies the observed quantities: Sentinel-1 radar backscatter from the land surface, Sentinel-2 surface reflectance and vegetation/canopy-water indices, and MODIS land-surface temperature and vegetation indices. It also explains daily assignment at new stations and the role of random seeds in repeated training runs. Section headings use sentence case and date ranges use "to". Results, memberships, and the routing equation are unchanged.

## Results from executed notebook stdout

These tables are copied from the evidence cell's stdout, also saved as [readme_results.md](artifacts/readme_results.md). The comparisons use the station-majority shared-feature cluster-routed model and the matched global predictor.

| Evaluation | Global | Two-regime |
| --- | ---: | ---: |
| Later-year R² | 0.7798 | 0.8118 |
| Held-out-station mean R² | 0.5797 | 0.6379 |

| ECE assignment | Pooled RMSE (m³/m³) |
| --- | ---: |
| Standard routing | 0.1674 |
| Precipitation fallback | 0.0578 |
| Global predictor | 0.0586 |

Temporal results cover 30 matched seeds, with higher two-regime R² in all 30 runs. Station holdouts use five seeds at each of seven stations, with four mean gains and three mean losses. Differences calculated from unrounded paired values are +0.0320 temporal and +0.0581 station-holdout R². The station chart and stdout record all seven station differences. Repeated seeds measure variation between fits; they exclude uncertainty from independently sampled sites, seasons, and model selection.

The ECE evaluation covers 150 daily observations at five separate sensor sites from July 20 to August 19, 2026. All expected SMAP inputs are missing. Washington validation data determined the automatic precipitation-based mapping, without ECE target labels. All fallback predictions use the same specialist, so the test evaluates that specialist over a short summer period. Full-season superiority of the complete two-specialist system remains untested. The first paper uses a different evaluation protocol, so its R² is excluded from the comparison.

## Reproduce

Run from the repository root using the existing `notebooks/` uv environment. Scientific dependencies are locked there; `requirements.txt` pins the additional export libraries. The notebook does not retrain predictors, fit new routers, or regenerate splits.

```bash
cd notebooks
timeout 180s uv pip install --python .venv/bin/python -r ../writeup2/poster-summer-1.1/requirements.txt
timeout 30s uv run --no-sync python -m ipykernel install --user --name mdr-poster-uv --display-name "MDR poster (uv)"
UV_NO_SYNC=1 PATH="$PWD/.venv/bin:$PATH" timeout 240s nb execute ../writeup2/poster-summer-1.1/poster_figures.ipynb --uv --kernel mdr-poster-uv -t 180 > ../writeup2/poster-summer-1.1/artifacts/notebook_stdout.txt 2>&1
```

The explicit kernel name selects this project's environment and avoids a stale user-wide `python3` kernelspec. `UV_NO_SYNC=1` preserves the pinned export libraries installed above. For interactive use, select MDR poster (uv), also stored in the notebook metadata.

All scientific figures are generated solely through notebook execution. The tracked modules handle evidence validation, plotting, and a shared layout measured in inches for both exports. Every graphic is saved as 300 dpi PNG and vector PDF/SVG. Edit `scripts/build_poster.py` for copy or geometry and rerun the notebook.

Branding and DejaVu Sans regular/bold fonts with their license are bundled in `assets/`. The university logos retain the resolution provided in the reference PowerPoint; [branding provenance](assets/branding_provenance.json) records the original template and selected pictures. Rebuilding does not depend on temporary attachment paths. Install the bundled fonts on a PowerPoint machine if that font family is unavailable.

## Sources and verification

1. Balkovec et al. (2026), *Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather-Driven Machine Learning*, IEEE AIIoT, [doi:10.1109/AIIoT68874.2026.11569136](https://doi.org/10.1109/AIIoT68874.2026.11569136). Background for the existing multi-source pipeline and global model.
2. MDR Project (2026), [technical report assembly 1.5](../report-assemble-1.5/report.md). Source of the current methods, matched evaluations, regime interpretation, and ECE evidence.

The report's extractor validates saved evidence before plotting. Main two-regime results come from `paper2-final-evidence-1.0`, matched global results from `paper2-final-evidence-1.2`, and ECE results/audits from `paper2-final-evidence-1.1`. Memberships and profiles use the report's declared `derived_8.4-gating-analysis-1.0` tables. The tracked Natural Earth admin-1 cache is filtered to exactly one `US-WA` feature. The manifest records the precise input paths and hashes.

The notebook checks all headline values, seed/station joins, seven map labels, three consistent appearances of each regime name, numbered captions, revised opening, removal of the footer and badges, and the word budget. Export checks cover one slide/page, exact 36″ × 24″ dimensions, embedded PDF fonts, editable credits and equation, measured text fit, and print margins. Numerical results and coordinates/memberships are checked against version 1.0; its files are hashed before and after building to verify preservation.

The final sequential execution passed and reports 932 rendered words in [validation.json](artifacts/validation.json) and notebook stdout. Checks confirm embedded fonts, editable placeholders and equation, and preservation of version 1.0. Both preview sizes were reviewed; no clipping or overlapping text or labels were found.

PDF previews are inspected for clipping, overlap, and reading order. PowerPoint receives structural checks and uses the same measured layout; no native PowerPoint/LibreOffice rendering engine is available on this host. Public in-situ observations come from NOAA USCRN and SNOTEL; the new sensor-site evaluation is in collaboration with Seattle University's ECE team.
