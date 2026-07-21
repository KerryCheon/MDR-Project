# Continuation handoff

The simplified reproduction workflow is implemented through `run_all.py` and `pipeline.ipynb`.

- The current clean rebuild is interrupted during the `selection` stage; its exact state is in `artifacts/run_state.json`.
- Resume with `cd notebooks && nb execute experiment/derived_8.2-feature-selection-2.2/pipeline.ipynb --uv --timeout 86400`.
- A completed run deletes and recreates the full artifact tree when the pipeline is run again. Smoke artifacts are intentionally not recreated.
- `RESULTS.md` is absent until `generate_results.py` completes after every full stage. Do not restore old manually copied metrics.

All 2023–2025 evaluations produced by the canonical runner are retrospective diagnostics and are not eligible for an unbiased SOTA claim.
