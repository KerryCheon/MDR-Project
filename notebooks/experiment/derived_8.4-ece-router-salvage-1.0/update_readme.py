"""Populate README.md tables from executed salvage outputs (CSV/JSON)."""

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
        "# Experiment: `derived_8.4-ece-router-salvage-1.0`",
        "",
        "Router-only salvage for `Clustering_V0_Full_k2` and `Clustering_Backbone54_k2`",
        "(`c0_0_c1_0`, 54 backbone features, no deltas) on the 5 in-situ ECE stations.",
        "Experts are frozen (fit on WA `trainval` only); routing policies are",
        "inference-time label overrides. ECE targets are used for evaluation only;",
        "the margin threshold comes from WA `trainval` only.",
        "",
        f"WA margin thresholds: `{audit['wa_thresholds']}`.",
        "",
        "## Datasets",
        "",
        "Three versioned splits (see `config.yaml:9-12`):",
        "",
        "1. Training — `data/splits/derived_8.4` (7 WA reference stations:",
        "   BeaverPass, CayusePass, Darrington, Paradise, Quinault,",
        "   SourdoughGulch, Spokane). `train.csv` + `val.csv` concatenated as",
        "   `trainval` (14,608 rows, 2017–2022). Routers AND experts fit here",
        "   only. The WA `test.csv` (2023–2025) is NOT used in this experiment.",
        "2. ECE eval, zero-filled — `data/splits/derived_8.4-ece`, `test.csv`",
        "   only (150 rows, 5 stations, 2026-07-20–08-19; its `train.csv` /",
        "   `val.csv` are empty). Missing SMAP stored as physical `0.0`.",
        "   Evaluation only — never used for fitting.",
        "3. ECE eval, native-missing —",
        "   `data/splits/derived_8.4-ece-v2-native-missing`, `test.csv` only",
        "   (same 150 rows, same 499-column order). Missing SMAP stays `NaN`",
        "   with observation masks at 0. Evaluation only.",
        "",
        "Every reported number is 2 families x 2 ECE inputs (`zero` vs `native`)",
        "x 7 policies x 5 seeds. The `c0_0_c1_0` variants share the identical",
        "54-feature backbone with no delta additions, so the families differ",
        "only in router.",
        "",
        "## Routing policies",
        "",
        "All policies reuse the SAME frozen experts (fit on WA `trainval` only);",
        "only the per-row expert assignment changes at inference time.",
        "C0 is the dry specialist, C1 the wet-mountain specialist.",
        "",
        "1. `as_routed` (baseline, no fix): the family's own static KMeans router",
        "   decides per row. This is the published-model failure — it sends",
        "   `Lost_Meadow` (100%) and `Renton_Home` (90%) to the wet expert.",
        "2. `c0_only` (force dry): every ECE row goes to the dry expert.",
        "   Blunt but effective; recovers to dynamic-router error levels.",
        "3. `c1_only` (force wet, diagnostic): every row goes to the wet expert.",
        "   Deliberately terrible — proves C1 is the poison and the experts",
        "   themselves are transferable.",
        "4. `gapi_transplant`: labels from the `G_API` rainfall-index router",
        "   (fit on WA only), predictions from the family's frozen experts.",
        "   July–August is bone dry, so all 150 rows land in C0.",
        "5. `dynamic_transplant`: labels from the dynamic 3-feature KMeans router",
        "   (`SMAP_sm_pm_interp_lag1`, `G_API`, `LST_modis`), predictions from the",
        "   family's frozen experts. Also 100% C0 on this window.",
        "6. `seasonal`: calendar router (May–Oct dry, Nov–Apr wet). The ECE window",
        "   falls entirely in the dry season, so all rows land in C0 — the",
        "   cheapest possible rule, using no features at all.",
        "7. `margin_fallback`: keep the static decision when confident, otherwise",
        "   fall back to the `Global_Single_54` expert (also fit on WA only).",
        "   Confidence is the KMeans margin (gap between nearest and",
        "   second-nearest centroid); rows below the WA 5th percentile fall back.",
        "   Best mean RMSE on native-missing input for both families.",
        "",
        "Note: transplant cluster labels index the host family's experts, whose",
        "cluster semantics differ by construction — this mismatch is documented,",
        "not hidden. R2 stays negative throughout (variance compression,",
        "`Var(y) ~ 1e-05`); judge policies by RMSE / bias / ubRMSE.",
        "",
        "## Pooled summary (mean over seeds)",
        "",
        summary.to_markdown(index=False),
        "",
        "## Station RMSE (mean over seeds)",
        "",
        pivot.to_markdown(floatfmt=".6f"),
        "",
        "## Reproduction",
        "",
        "From `notebooks/`, run:",
        "",
        "```powershell",
        "nb execute experiment/derived_8.4-ece-router-salvage-1.0/derived_8.4-ece-router-salvage-1.0.ipynb --uv --timeout 3600",
        "```",
        "",
        "Or run the tracked script directly (same code the notebook imports):",
        "",
        "```powershell",
        "uv run --project . python notebooks/experiment/derived_8.4-ece-router-salvage-1.0/run_salvage.py",
        "```",
        "",
        "Tables above are transcribed from the executed notebook stdout / CSVs.",
        "Versioned outputs are `summary.csv`, `seed_metrics.csv`,",
        "`station_metrics.csv`, and `routing_audit.json`.",
        "",
    ]
    (EXP_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote README.md ({len(summary)} summary rows)")


if __name__ == "__main__":
    main()
