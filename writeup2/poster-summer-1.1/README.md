# Summer research poster 1.1

One **36″ wide × 24″ high** research showcase poster, using the two-regime work summarized in [technical report 1.5](../report-assemble-1.5/report.md). This revision preserves the original Seattle University branding and four evidence graphics, with more explanation of the motivation, modeling steps, evaluation, and interpretation.

## Deliverables

- [Editable PowerPoint](summer_showcase.pptx): native text, credit placeholders, method shapes, and hard-routing equation; scientific figures at 300 dpi.
- [Print-ready PDF](summer_showcase.pdf): one page at the physical poster size, with embedded fonts and vector figures.
- [PNG preview](summer_showcase_preview.png) and [larger print-detail rendering](artifacts/print_detail.png).
- [Executed notebook](poster_figures.ipynb), [captured stdout](artifacts/notebook_stdout.txt), [validation record](artifacts/validation.json), and [source manifest](artifacts/source_manifest.json).

Replace `[Author names]` and `[Mentor name, credentials]` in PowerPoint before printing. The PDF contains the same placeholders. After manual edits, export the edited PowerPoint at its existing slide size. For reproducible edits, update the shared layout source and execute the notebook again.

## Revision choices

The opening now asks **“Why soil moisture estimation?”** and explains the project's purpose before introducing the summer research question. Methods and results include the controlled comparison, station consistency, fitting/test periods, and limits of the seed and ECE evaluations. The 850–950-word budget counts the title, all text, and chart labels. Main paragraphs use 22 pt type; smaller labels and captions support the graphics.

The large feature-count and target-depth badges and the bottom citation footer are removed. The target depth appears once in ordinary methods prose, the ECE acknowledgment is part of its panel, and sources are retained below. The method diagram has less unused vertical space, allowing numbered method explanations, an editable equation, and numbered figure captions.

**Regime names are user-selected descriptive labels, not new clustering results.** “Wet / Mountains” and “Drier / Lowlands” replace the earlier poster's terminology throughout the map, diagram, and profiles. They are defined once in `scripts/poster_data.py`. The report's saved five-versus-two memberships are unchanged:

- Wet / Mountains: Beaver Pass, Cayuse Pass, Darrington, Paradise, Quinault.
- Drier / Lowlands: Spokane, Sourdough Gulch.

Report 1.5 §§7–8 notes that the wet group contains lowland as well as mountain stations and that elevation alone weakly separates the groups. The poster briefly makes that qualification. The profile medians describe input conditions, not universal elevation classes or target-moisture thresholds. No geographic regime boundaries are inferred.

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

Temporal results cover 30 matched seeds; the two-regime predictor leads in all 30. Station holdouts cover five seeds at each of seven stations, with four mean gains and three mean losses. Differences are calculated from unrounded paired values: +0.0320 temporal and +0.0581 station-holdout R². The station chart and full stdout retain all seven station differences. Seed repetition describes fitting variability, not uncertainty over independently sampled sites, seasons, or model selection.

ECE evaluation covers 150 daily observations at five separate sensor sites, July 20–August 19, 2026. All expected SMAP inputs are missing. The automatic precipitation-based mapping was selected using Washington validation, without ECE target labels. All fallback rows exercise one specialist; the short window therefore supports a routing/availability lesson, without establishing full-season superiority of the complete two-specialist system. The first paper's differently evaluated R² is not used in the comparison.

## Reproduce

Run from the repository root using the existing `notebooks/` uv environment. Scientific dependencies are locked there; `requirements.txt` pins the additional export libraries. The notebook does not retrain predictors, fit new routers, or regenerate splits.

```bash
cd notebooks
timeout 180s uv pip install --python .venv/bin/python -r ../writeup2/poster-summer-1.1/requirements.txt
timeout 30s uv run --no-sync python -m ipykernel install --user --name mdr-poster-uv --display-name "MDR poster (uv)"
UV_NO_SYNC=1 PATH="$PWD/.venv/bin:$PATH" timeout 240s nb execute ../writeup2/poster-summer-1.1/poster_figures.ipynb --uv --kernel mdr-poster-uv -t 180 > ../writeup2/poster-summer-1.1/artifacts/notebook_stdout.txt 2>&1
```

The explicit kernel uses this project's environment rather than a stale user-wide `python3` kernelspec. `UV_NO_SYNC=1` preserves the pinned export libraries installed above. For interactive use, select **MDR poster (uv)**, also stored in the notebook metadata.

All scientific figures are generated solely through notebook execution. The tracked modules handle evidence validation, plotting, and a shared layout measured in inches for both exports. Every graphic is saved as 300 dpi PNG and vector PDF/SVG. Edit `scripts/build_poster.py` for copy or geometry and rerun the notebook.

Branding and DejaVu Sans regular/bold fonts with their license are bundled in `assets/`. The university logos retain the resolution provided in the reference PowerPoint; [branding provenance](assets/branding_provenance.json) records the original template and selected pictures. Rebuilding does not depend on temporary attachment paths. Install the bundled fonts on a PowerPoint machine if that font family is unavailable.

## Sources and verification

1. Balkovec et al. (2026), *Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather-Driven Machine Learning*, IEEE AIIoT, [doi:10.1109/AIIoT68874.2026.11569136](https://doi.org/10.1109/AIIoT68874.2026.11569136). Background for the existing multi-source pipeline and global model.
2. MDR Project (2026), [technical report assembly 1.5](../report-assemble-1.5/report.md). Source of the current methods, matched evaluations, regime interpretation, and ECE evidence.

The report's extractor validates saved evidence before plotting. Main two-regime results come from `paper2-final-evidence-1.0`, matched global results from `paper2-final-evidence-1.2`, and ECE results/audits from `paper2-final-evidence-1.1`. Memberships and profiles use the report's declared `derived_8.4-gating-analysis-1.0` tables. The tracked Natural Earth admin-1 cache is filtered to exactly one `US-WA` feature. The manifest records the precise input paths and hashes.

The notebook checks all headline values, seed/station joins, seven map labels, three consistent appearances of each regime name, numbered captions, revised opening, removal of the footer and badges, and the word budget. Export checks cover one slide/page, exact 36″ × 24″ dimensions, embedded PDF fonts, editable credits and equation, measured text fit, and print margins. Numerical results and coordinates/memberships are checked against version 1.0; its files are hashed before and after building to verify preservation.

The final sequential notebook execution passed. Its stdout reports **853 rendered words**, one slide/page, embedded PDF fonts, editable placeholders/equation, and unchanged version 1.0. Both the preview and larger print-detail rendering were reviewed; no clipping or label/text collisions were found.

PDF previews are inspected for clipping, overlap, and reading order. PowerPoint receives structural checks and uses the same measured layout; no native PowerPoint/LibreOffice rendering engine is available on this host. Public in-situ observations come from NOAA USCRN and SNOTEL; the new sensor-site evaluation is in collaboration with Seattle University's ECE team.
