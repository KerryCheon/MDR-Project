"""Populate README.md tables from executed v3 salvage outputs (CSV/JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent


def main() -> None:
    summary = pd.read_csv(EXP_DIR / "summary.csv")
    station = pd.read_csv(EXP_DIR / "station_metrics.csv")
    with (EXP_DIR / "routing_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)

    pivot = station.groupby(["family", "ece_input", "policy", "station"],
                            sort=False)["rmse"].mean().unstack("station")

    lines = [
        "# Experiment: `derived_8.4-ece-router-salvage-1.1`",
        "",
        "Router-only salvage for `Clustering_V0_Full_k2` and `Clustering_Backbone54_k2`",
        "(`c0_0_c1_0`, 54 backbone features, no deltas) on the canonical",
        "`derived_8.4_ece_v3` split, plus single-regime global baselines",
        "(`Global_Single_54` on 54 backbone features, `Global_Single_50` on 50 V0",
        "features, `policy=direct`). Experts and baselines are frozen (fit on WA",
        "`trainval` only); routing policies are inference-time label overrides.",
        "ECE targets are used for evaluation only; the margin threshold comes from",
        "WA `trainval` only and falls back to the `Global_Single_54` expert.",
        "",
        f"WA margin thresholds: `{audit['wa_thresholds']}`.",
        "",
        "## Datasets",
        "",
        "Two versioned splits (see `config.yaml:9-11`):",
        "",
        "1. Training — `data/splits/derived_8.4` (7 WA reference stations).",
        "   `train.csv` + `val.csv` concatenated as `trainval` (14,608 rows,",
        "   2017–2022). Routers, experts, AND baselines fit here only.",
        "   The WA `test.csv` (2023–2025) is NOT used.",
        "2. ECE eval — `data/splits/derived_8.4_ece_v3`, `test.csv` only",
        "   (150 rows: 5 stations x 30 days, 2026-07-20–08-19; `train.csv` /",
        "   `val.csv` are empty). 30-day warmup scaffold (Jun 20–Jul 19), strict",
        "   native-NaN SMAP (82 value cols NaN, 3 masks 0, zero `0.0`s), MODIS",
        "   NDVI 16-day fallback. Evaluation only — never used for fitting.",
        "",
        "Every reported number is (2 families x 7 policies + 2 baselines) x 5 seeds",
        "on the single `v3` input. `rmse_change_vs_as_routed` is NaN for baselines",
        "(no `as_routed` exists) — compare them directly by `rmse_mean`.",
        "",
        "## Routing policies",
        "",
        "All regime policies reuse the SAME frozen experts; only the per-row expert",
        "assignment changes. C0 is the dry specialist, C1 the wet-mountain specialist.",
        "",
        "1. `as_routed` (baseline, no fix): the family's own static KMeans router.",
        "2. `c0_only` (force dry): every v3 row goes to the dry expert.",
        "3. `c1_only` (force wet, diagnostic): every row goes to the wet expert.",
        "4. `gapi_transplant`: labels from the `G_API` router, predictions from",
        "   the family's frozen experts.",
        "5. `dynamic_transplant`: labels from the dynamic 3-feature KMeans router,",
        "   predictions from the family's frozen experts.",
        "6. `seasonal`: calendar router (May–Oct dry); the v3 window is all dry.",
        "7. `margin_fallback`: confident rows keep the static decision, ambiguous",
        "   rows (margin below WA 5th percentile) fall back to `Global_Single_54`.",
        "8. `direct` (baselines only): single-regime prediction, no routing.",
        "",
        "## Pooled summary (mean over seeds)",
        "",
        summary.to_markdown(index=False),
        "",
        "## Station RMSE (mean over seeds)",
        "",
        pivot.to_markdown(floatfmt=".6f"),
        "",
        "## Per-station prediction line charts",
        "",
        "Seed-mean observed vs predicted trajectories (`predictions_v3.csv`).",
        "Every panel shows at most 5 lines. Family panels pair each regime model",
        "with its most relevant baseline (V0 with Global-50, Backbone with",
        "Global-54, the `margin_fallback` target).",
        "",
        "![Baseline showdown overlay](figures/timeseries_v3_baselines_overlay.png)",
        "",
        "Per-station family panels:",
        "",
        "- V0: `timeseries_v3_<STATION>_v0.png` (observed, V0 as_routed / c0_only /",
        "  margin_fallback, Global-50 direct)",
        "- Backbone: `timeseries_v3_<STATION>_backbone.png` (observed, Backbone",
        "  as_routed / c0_only / margin_fallback, Global-54 direct)",
        "",
        "with `<STATION>` in `ECE_BBG_Lost_Meadow`, `ECE_BBG_Main_St`,",
        "`ECE_Renton_Garden_North`, `ECE_Renton_Garden_Shed`, `ECE_Renton_Home`.",
        "",
        "## Reproduction",
        "",
        "From `notebooks/`, run:",
        "",
        "```powershell",
        "nb execute experiment/derived_8.4-ece-router-salvage-1.1/derived_8.4-ece-router-salvage-1.1.ipynb --uv --timeout 3600",
        "```",
        "",
        "Or run the tracked script directly (same code the notebook imports):",
        "",
        "```powershell",
        "uv run --project . python notebooks/experiment/derived_8.4-ece-router-salvage-1.1/run_salvage.py",
        "```",
        "",
        "Tables above are transcribed from the executed notebook stdout / CSVs.",
        "Versioned outputs are `summary.csv`, `seed_metrics.csv`,",
        "`station_metrics.csv`, `predictions_v3.csv`, `routing_audit.json`,",
        "and `figures/timeseries_v3_*.png` (11 line charts).",
        "",
    ]
    (EXP_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote README.md ({len(summary)} summary rows)")


if __name__ == "__main__":
    main()
