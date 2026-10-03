"""Extract the Paper 2 report from immutable, versioned evidence artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


ASSEMBLY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ASSEMBLY_ROOT.parents[1]
EVIDENCE_10 = "notebooks/experiment/paper2-final-evidence-1.0"
EVIDENCE_11 = "notebooks/experiment/paper2-final-evidence-1.1"
FORMAL_10 = "notebooks/experiment/derived_8.4-formal-eval-1.0"

SOURCES = {
    "outline": "writeup2/outline-v7.md",
    "paper1": "paper/Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather- Driven Machine Learning.pdf",
    "split": "data/splits/derived_8.4/split_meta.json",
    "station_config": "data/splits/derived_8.4/config.yaml",
    "wa_static": "data/splits/derived_8.4/station_static_features.csv",
    "ece_split": "data/splits/derived_8.4_ece_v3/test.csv",
    "features": "notebooks/experiment/derived_8.4-feature-selection-2.0/artifacts/selected_features.json",
    "main_manifest": f"{EVIDENCE_10}/run_manifest.json",
    "main_seed": f"{EVIDENCE_10}/main_temporal_seed_comparison.csv",
    "main_summary": f"{EVIDENCE_10}/main_temporal_summary.csv",
    "loso_seed": f"{EVIDENCE_10}/loso_seed_station.csv",
    "t8": f"{EVIDENCE_10}/t8_loso_station_summary.csv",
    "fold_agreement": f"{EVIDENCE_10}/loso_fold_agreement.csv",
    "agreement": f"{EVIDENCE_10}/agreement_summary.csv",
    "k_sweep": f"{EVIDENCE_10}/k_sweep_quality.csv",
    "composition": "notebooks/experiment/derived_8.4-gating-analysis-1.0/regime_station_composition_Clustering_Backbone54_k2.csv",
    "profile": "notebooks/experiment/derived_8.4-gating-analysis-1.0/regime_profile_summary_Clustering_Backbone54_k2.csv",
    "formal_pair": f"{FORMAL_10}/temporal_pairwise_focused.csv",
    "formal_loso": f"{FORMAL_10}/loso_config_summary.csv",
    "formal_delta": f"{FORMAL_10}/delta_robustness_summary.csv",
    "oos": "notebooks/experiment/derived_8.4-formal-eval-2.0/spatial_focused_no_delta_summary.csv",
    "ece_legacy": "notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/spatial_focused_no_delta_pairwise.csv",
    "ece_manifest": f"{EVIDENCE_11}/run_manifest.json",
    "ece_seed": f"{EVIDENCE_11}/ece_guarded/seed_metrics.csv",
    "ece_policy": f"{EVIDENCE_11}/ece_guarded/ece_policy_summary.csv",
    "ece_station": f"{EVIDENCE_11}/ece_guarded/ece_station_policy_summary.csv",
    "ece_crosswalk": f"{EVIDENCE_11}/ece_guarded/policy_crosswalk.csv",
    "ece_audit": f"{EVIDENCE_11}/ece_guarded/routing_audit.json",
    "ece_calibration": f"{EVIDENCE_11}/ece_guarded/wa_calibration.csv",
    "ece_sites": f"{EVIDENCE_11}/ece_guarded/site_descriptors.csv",
    "ece_shares": f"{EVIDENCE_11}/ece_guarded/ece_routing_shares_by_station.csv",
    "ece_overlay": f"{EVIDENCE_11}/ece_guarded/f5_guarded_daily_overlay.csv",
    "hourly": "notebooks/experiment/ece-input-diagnose-1.0/smoothing_summary.csv",
}

STATION_CONFIG_KEYS = {
    "Spokane": "spokane_17_ssw",
    "Quinault": "quinault_4_ne",
    "Darrington": "darrington_21_nne",
    "SourdoughGulch_WA_985": "sourdough_gulch",
    "CayusePass_WA": "cayuse_pass_wa",
    "Paradise_WA": "paradise_wa",
    "BeaverPass_WA_990": "beaver_pass_wa",
}

EXTRA_PROVENANCE = {
    "main_notebook": f"{EVIDENCE_10}/notebooks/paper2_main_figures.ipynb",
    "ece_notebook": f"{EVIDENCE_11}/notebooks/paper2_ece_supplement.ipynb",
    "main_config": f"{EVIDENCE_10}/config.yaml",
    "ece_config": f"{EVIDENCE_11}/ece_guarded/config.yaml",
    "ece_runner": f"{EVIDENCE_11}/ece_guarded/run_guarded.py",
}


def source_path(key: str) -> Path:
    return REPO_ROOT / SOURCES[key]


# Appendix A groups the shared backbone by theme. Each feature keeps its exact
# pipeline name (backticked) with a one-line gloss adapted from the tracked
# per-feature glossary in docs/selected-features.md ("derived_8.4 - 54 Features").
# Name prefixes follow the derived-split convention (docs/features.md): V_ =
# rolling-window statistics, A_ = changes/slopes, C_ = lags/memory, D_ =
# seasonal/spectral transforms, E_ = radar physics, G_ = hydrologic/weather,
# J_ = static site descriptors, F_ = optical indices, bare SMAP_ = satellite
# soil moisture; _kobsK windows cover the last K valid observations, not days.
FEATURE_GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("Precipitation and hydrologic memory (6 features)",
     "Rain actually observed at the station, plus decaying memory of past rain.",
     [
         ("precip_mm", "daily ERA5-Land precipitation at the station point (mm)"),
         ("G_API", "antecedent precipitation index: exponentially decayed rain memory"),
         ("G_DSLR", "days since last rain: dry-down duration"),
         ("G_rain_sum_3d", "calendar 3-day cumulative rainfall (true days)"),
         ("G_rain_sum_7d", "calendar 7-day cumulative rainfall (true days)"),
         ("V_rollrng_G_API_kobs7", "rolling range of API over the last 7 observations"),
     ]),
    ("Satellite soil moisture - SMAP (12 features)",
     "Coarse satellite moisture and its recent history, the grouping's main signal.",
     [
         ("SMAP_sm_pm_interp", "SMAP evening-pass soil moisture, gap-filled to daily"),
         ("SMAP_sm_pm_interp_lag7", "evening SMAP value shifted back 7 days"),
         ("SMAP_sm_pm_interp_lag30", "evening SMAP value shifted back 30 days"),
         ("SMAP_sm_pm_interp_rollrange7", "7-day rolling range of evening SMAP"),
         ("SMAP_sm_pm_interp_rollmean30", "30-day rolling mean of evening SMAP"),
         ("SMAP_sm_pm_interp_rollrange30", "30-day rolling range of evening SMAP"),
         ("SMAP_sm_interp_rollrange7", "7-day rolling range of combined AM+PM SMAP"),
         ("SMAP_ampm_diff_interp", "morning-minus-evening SMAP: diurnal dry-down signal"),
         ("A_d_SMAP_sm_interp_kobs30", "30-observation first difference of combined SMAP: wetting/drying step"),
         ("V_rollmin_SMAP_sm_interp_kobs14", "rolling minimum of combined SMAP over 14 observations"),
         ("V_rollmin_SMAP_sm_interp_kobs30", "rolling minimum of combined SMAP over 30 observations"),
         ("SMAP_x_year", "SMAP-by-year interaction: long-term sensor drift term"),
     ]),
    ("Radar - Sentinel-1 SAR (10 features)",
     "Microwave backscatter physics and its recent volatility.",
     [
         ("E_SAR_ratio", "VV/VH backscatter ratio: vegetation/soil-moisture sensitive"),
         ("A_d_E_SAR_ratio_kobs30", "30-observation first difference of the SAR ratio"),
         ("V_rollmax_E_SAR_ratio_kobs7", "rolling maximum of the SAR ratio over 7 observations"),
         ("V_rollmin_E_SAR_ratio_kobs30", "rolling minimum of the SAR ratio over 30 observations"),
         ("V_rollmax_E_SAR_ratio_kobs30", "rolling maximum of the SAR ratio over 30 observations"),
         ("A_grad_E_SAR_diff_kobs30", "linear slope of the VV-minus-VH difference over 30 observations"),
         ("V_rollmax_E_SAR_diff_kobs14", "rolling maximum of the SAR difference over 14 observations"),
         ("V_rollrng_E_SAR_diff_kobs30", "rolling range of the SAR difference over 30 observations"),
         ("V_rollmax_E_SAR_diff_kobs30", "rolling maximum of the SAR difference over 30 observations"),
         ("E_rough_s1_vh_kobs14", "surface-roughness proxy: rolling variability of VH backscatter over 14 observations"),
     ]),
    ("Vegetation and optical - Sentinel-2 (12 features)",
     "Greenness, canopy water, and shortwave-infrared moisture bands.",
     [
         ("s2_b4", "Sentinel-2 Red band B4 (665 nm) surface reflectance"),
         ("s2_b8", "Sentinel-2 near-infrared band B8 (842 nm) surface reflectance"),
         ("A_grad_s2_b11_kobs30", "slope of SWIR1 B11 (1610 nm) over 30 observations: moisture trend"),
         ("V_rollrng_s2_b11_kobs30", "rolling range of B11 over 30 observations"),
         ("V_rollmin_s2_b11_kobs30", "rolling minimum of B11 over 30 observations"),
         ("V_rollmin_s2_b12_kobs30", "rolling minimum of SWIR2 B12 (2190 nm) over 30 observations"),
         ("V_rollmax_F_NDMI_kobs30", "rolling maximum of NDMI (NIR-SWIR canopy-water index) over 30 observations"),
         ("V_rollmax_F_NDVI_kobs14", "rolling maximum of NDVI (red-NIR greenness) over 14 observations"),
         ("V_rollmax_F_NDVI_kobs30", "rolling maximum of NDVI over 30 observations"),
         ("V_ema_F_NDVI_kobs30", "exponential moving average of NDVI over 30 observations"),
         ("C_lag_F_NDVI_kobs30", "NDVI value lagged 30 observations"),
         ("D_z_F_NDMI", "seasonal z-score anomaly of NDMI vs its monthly climatology"),
     ]),
    ("Land-surface temperature (5 features)",
     "MODIS temperature level, anomaly, and periodicity.",
     [
         ("V_rollmin_LST_modis_kobs30", "rolling minimum of MODIS land-surface temperature over 30 observations: cold extreme"),
         ("V_ema_LST_modis_kobs30", "exponential moving average of land-surface temperature over 30 observations"),
         ("D_z_LST_modis", "seasonal z-score anomaly of land-surface temperature"),
         ("D_fft_dom_LST_modis_kobs30", "dominant Fourier frequency of temperature over 30 observations: periodicity"),
         ("D_fft_ent_LST_modis_kobs30", "spectral entropy of temperature over 30 observations: signal complexity"),
     ]),
    ("Calendar and seasonal cycle (4 features)",
     "Where the day sits in the annual and multi-year cycle.",
     [
         ("D_sin_DOY", "sine of day-of-year: seasonal cycle phase"),
         ("D_cos_DOY", "cosine of day-of-year: seasonal cycle quadrature"),
         ("sin_year", "sine of fractional year: multi-year cyclic trend"),
         ("cos_year", "cosine of fractional year: multi-year quadrature"),
     ]),
    ("Static site descriptors (5 features)",
     "Terrain, climate normals, land cover, and soil at the station (no time variation).",
     [
         ("J_aspect_deg", "SRTM terrain aspect in degrees at the station"),
         ("J_bio_bio02", "WorldClim mean diurnal temperature range (1970-2000 normals)"),
         ("J_bio_bio13", "WorldClim precipitation of wettest month (1970-2000 normals)"),
         ("J_lc_code", "land-cover class code (ESA WorldCover/NLCD)"),
         ("J_soil_texture_usda_b0", "FAO HWSD USDA soil-texture class at the surface"),
     ]),
]


def feature_groups_block(features: list[str]) -> str:
    """Render the themed Appendix A block; every backbone feature appears once."""
    seen: list[str] = []
    parts = []
    for title, summary, items in FEATURE_GROUPS:
        parts.append(f"**{title}.** {summary}")
        parts.append("")
        parts.extend(f"- `{name}`: {gloss}" for name, gloss in items)
        parts.append("")
        seen.extend(name for name, _ in items)
    if sorted(seen) != sorted(features) or len(seen) != 54:
        raise ValueError("Appendix A theme mapping does not cover the 54 shared features exactly once")
    return "\n".join(parts).rstrip("\n")


def ledger_source_link(relative: str, label: str | None = None) -> str:
    digest = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
    return f"[{label or relative}](../../{relative.replace(' ', '%20')}) (SHA-256: `{digest}`)"


def read_csv(key: str) -> list[dict[str, str]]:
    with source_path(key).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(key: str) -> dict:
    return json.loads(source_path(key).read_text(encoding="utf-8"))


def number(value: str | float | int) -> float:
    return float(value)


def fmt(value: str | float | int, digits: int = 4) -> str:
    result = number(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite report value: {value}")
    return f"{result:.{digits}f}"


def signed(value: str | float | int, digits: int = 4) -> str:
    return f"{number(value):+.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(lines)


def one(rows: list[dict], **terms: str) -> dict:
    matches = [row for row in rows if all(row.get(key) == value for key, value in terms.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {terms}, found {len(matches)}")
    return matches[0]


def mean(rows: list[dict], field: str) -> float:
    return statistics.mean(number(row[field]) for row in rows)


# Absolute tolerances here are intentionally tighter than display rounding
# (report values print at 4-6 decimals): they pin the exact saved artifacts,
# so any upstream regeneration shows up as a loud failure, not silent drift.
def check_close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label}: {actual} != {expected}")


def station_networks(station_ids: list[str]) -> dict[str, str]:
    config = yaml.safe_load(source_path("station_config").read_text(encoding="utf-8"))["stations"]
    if set(station_ids) != set(STATION_CONFIG_KEYS):
        raise ValueError("station-to-configuration map does not cover the Washington split")

    networks = {}
    for station_id, config_key in STATION_CONFIG_KEYS.items():
        station_config = config.get(config_key)
        if station_config is None:
            raise ValueError(f"missing station configuration for {station_id}: {config_key}")
        parse = station_config.get("parse", {})
        request = station_config.get("request", {})
        if parse.get("snotel_mode") is True:
            networks[station_id] = "SNOTEL"
        elif "uscrn" in str(request.get("base_url", "")).lower():
            networks[station_id] = "NOAA USCRN"
        else:
            raise ValueError(f"unrecognized data network for {station_id}")
    if sum(network == "SNOTEL" for network in networks.values()) != 4:
        raise ValueError("expected four SNOTEL stations in the Washington study")
    if sum(network == "NOAA USCRN" for network in networks.values()) != 3:
        raise ValueError("expected three NOAA USCRN stations in the Washington study")
    return networks


def validate_sources() -> None:
    for key, relative in SOURCES.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"{key}: {path}")
    for key, relative in EXTRA_PROVENANCE.items():
        if not (REPO_ROOT / relative).is_file():
            raise FileNotFoundError(f"{key}: {relative}")

    split = read_json("split")
    if split["rows"] != {"train": 9803, "val": 4805, "test": 6620} or len(split["stations"]) != 7:
        raise ValueError("Washington split metadata changed")
    station_networks(split["stations"])
    features = read_json("features")
    if len(features["global_features"]) != 54 or features["selection_goal"] != "unweighted_pooled_test_r2_2023_2025":
        raise ValueError("shared feature backbone/provenance changed")

    main_manifest = read_json("main_manifest")
    ece_manifest = read_json("ece_manifest")
    if main_manifest["status"] != "completed" or ece_manifest["status"] != "complete":
        raise ValueError("evidence bundles are not complete")
    if ece_manifest["ece_run"]["fitting_and_evaluation"]["ece_targets_used_for_fitting_or_mapping"]:
        raise ValueError("ECE targets used for model selection")

    main_seed = read_csv("main_seed")
    main_summary = read_csv("main_summary")
    for row in main_summary:
        group = [item for item in main_seed if item["strategy_name"] == row["strategy_name"] and item["evidence_role"] == row["evidence_role"]]
        if len(group) != 30 or len({item["seed"] for item in group}) != 30:
            raise ValueError(f"temporal seed coverage: {row['strategy_name']}")
        check_close(mean(group, "r2"), number(row["mean_r2"]), f"temporal R2 {row['strategy_name']}")
        check_close(mean(group, "rmse"), number(row["mean_rmse"]), f"temporal RMSE {row['strategy_name']}")
    if len(main_summary) != 8 or len(main_seed) != 240:
        raise ValueError("main temporal table coverage changed")
    guard = one(main_summary, strategy_name="Guarded_Backbone54_k2")
    backbone = one(main_summary, strategy_name="Clustering_Backbone54_k2")
    if guard["evidence_role"] != "paired_in_harness" or backbone["evidence_role"] != "paired_in_harness":
        raise ValueError("guard ablation lost paired provenance")
    check_close(number(guard["mean_r2"]), 0.8118429663371349, "main notebook stdout Guarded R2")
    check_close(number(backbone["mean_r2"]), 0.8117237199127312, "main notebook stdout Backbone R2")

    loso = read_csv("loso_seed")
    if len(loso) != 105 or len({row["station"] for row in loso}) != 7:
        raise ValueError("LOSO coverage changed")
    for strategy in {row["strategy_name"] for row in loso}:
        group = [row for row in loso if row["strategy_name"] == strategy]
        if len(group) != 35 or len({row["seed"] for row in group}) != 5:
            raise ValueError(f"LOSO seeds/folds changed for {strategy}")
    guard_loso = [row for row in loso if row["strategy_name"] == "Guarded_Backbone54_k2"]
    check_close(mean(guard_loso, "r2"), 0.637877, "main notebook stdout Guarded LOSO R2", tolerance=1e-6)
    t8 = read_csv("t8")
    if len(t8) != 7:
        raise ValueError("T8 station count changed")
    for row in t8:
        station = row["station"]
        guard_rows = [item for item in loso if item["station"] == station and item["strategy_name"] == "Guarded_Backbone54_k2"]
        backbone_rows = [item for item in loso if item["station"] == station and item["strategy_name"] == "Clustering_Backbone54_k2"]
        guard_by_seed = {item["seed"]: number(item["r2"]) for item in guard_rows}
        backbone_by_seed = {item["seed"]: number(item["r2"]) for item in backbone_rows}
        check_close(statistics.mean(guard_by_seed[seed] - backbone_by_seed[seed] for seed in guard_by_seed), number(row["mean_r2_difference"]), f"T8 {station}")

    ece_split = read_csv("ece_split")
    if len(ece_split) != 150 or len({row["station_id"] for row in ece_split}) != 5:
        raise ValueError("ECE split coverage changed")
    if {sum(row["station_id"] == station for row in ece_split) for station in {row["station_id"] for row in ece_split}} != {30}:
        raise ValueError("ECE per-station row coverage changed")
    if {row["station_id"] for row in ece_split} & set(split["stations"]):
        raise ValueError("ECE/WA stations overlap")
    ece_seed = read_csv("ece_seed")
    ece_policy = read_csv("ece_policy")
    if len(ece_seed) != 80 or len(ece_policy) != 16 or len({row["seed"] for row in ece_seed}) != 5:
        raise ValueError("ECE seed/policy coverage changed")
    for row in ece_policy:
        group = [item for item in ece_seed if item["family"] == row["family"] and item["policy"] == row["policy"]]
        if len(group) != 5:
            raise ValueError(f"ECE seed coverage {row['family']} {row['policy']}")
        check_close(mean(group, "rmse"), number(row["rmse_mean"]), f"ECE RMSE {row['family']} {row['policy']}")
        if row["policy"] in {"c0_only", "c1_only"} and row["deployable"] != "False":
            raise ValueError("oracle incorrectly marked deployable")
    guard_auto = one(ece_policy, family="Guarded_Backbone54_k2", policy="auto_hard")
    check_close(number(guard_auto["rmse_mean"]), 0.0577680985766914, "ECE notebook stdout Guarded auto RMSE")
    # Table 8 shows the station-majority regional model only because the rows
    # of the same model without the station consistency guarantee are identical
    # on this ECE set; enforce that identity here so the
    # "bit-identical" prose cannot go stale if the artifacts change.
    for policy in ("as_routed", "auto_hard"):
        guard_row = one(ece_policy, family="Guarded_Backbone54_k2", policy=policy)
        backbone_row = one(ece_policy, family="Clustering_Backbone54_k2", policy=policy)
        for field in ("rmse_mean", "rmse_std", "mae_mean", "bias_mean", "ubrmse_mean"):
            check_close(number(guard_row[field]), number(backbone_row[field]), f"ECE identity Guarded==Backbone {policy} {field}")
    # The soft-blend fallback is omitted from Table 8 as matching the hard
    # fallback at reported (6-decimal) precision; enforce that too.
    guard_soft = one(ece_policy, family="Guarded_Backbone54_k2", policy="auto_soft")
    check_close(number(guard_soft["rmse_mean"]), number(guard_auto["rmse_mean"]), "ECE Guarded auto_soft vs auto_hard RMSE", tolerance=1e-6)
    audit = read_json("ece_audit")
    if audit["ece_rows_used_for_fitting"] or audit["ece_rows_used_for_mapping_or_calibration"]:
        raise ValueError("ECE fitting/calibration leakage")
    for family in ("Clustering_V0_Full_k2", "Clustering_Backbone54_k2", "Guarded_Backbone54_k2"):
        gate = audit["ece_gate_audit"][family]
        if gate["n_rows"] != 150 or gate["gate_rows"] != 150 or gate["mean_smap_miss_rate"] != 1.0:
            raise ValueError(f"ECE gate audit changed for {family}")
    maps = audit["aux_expert_mappings"]
    if maps["Guarded_Backbone54_k2"]["gapi_class_0_local_expert"] != 0:
        raise ValueError("Guarded G_API mapping changed")
    if audit["semantic_expert_indices"]["Guarded_Backbone54_k2"]["dry_expert_index"] != 1:
        raise ValueError("Guarded oracle convention changed")
    calibration = [row for row in read_csv("ece_calibration") if row["family"] == "Guarded_Backbone54_k2" and row["setting"] == "smap_masked_val" and row["policy"].startswith("aux_hard_candidate_")]
    if len(calibration) != 2 or sum(row["selected_for_ece"] == "True" for row in calibration) != 1:
        raise ValueError("Guarded mapping selection audit changed")
    selected = next(row for row in calibration if row["selected_for_ece"] == "True")
    check_close(number(selected["rmse"]), 0.07396518184352849, "Guarded WA validation candidate")
    if (number(selected["gapi_class_0_local_expert"]), number(selected["gapi_class_1_local_expert"])) != (0.0, 1.0):
        raise ValueError("Guarded selected mapping is not G_API class 0→local c0")
    alternative = next(row for row in calibration if row["selected_for_ece"] == "False")
    check_close(number(alternative["rmse"]), 0.11717976676399142, "Guarded rejected WA validation candidate")
    crosswalk = one(read_csv("ece_crosswalk"), family="Guarded_Backbone54_k2", policy_id="auto_hard")
    if (number(crosswalk["gapi_class_0_local_expert"]), number(crosswalk["dry_assigned_local_expert_index"])) != (0.0, 1.0):
        raise ValueError("Guarded crosswalk contradicts the selected mapping or dry index")
    if crosswalk["ece_used_for_mapping_or_calibration"] != "False" or crosswalk["deployable"] != "True":
        raise ValueError("Guarded policy deployability/calibration provenance changed")


def source_manifest() -> dict:
    inputs = {**SOURCES, **EXTRA_PROVENANCE}
    return {
        "assembly": "report-assemble-1.3",
        "source_precedence": {
            "final_evaluation_batch": {
                "main_temporal_loso": EVIDENCE_10,
                "ece_results": EVIDENCE_11,
            },
            "outline": SOURCES["outline"],
            "historical_sources": "context or explicitly labeled sensitivity only",
        },
        "extraction": "scripts/report_data.py reads saved CSV/JSON; no model fitting or raw-data inference",
        "inputs": {
            key: {
                "path": relative,
                "sha256": hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(),
            }
            for key, relative in sorted(inputs.items())
        },
        "protocol": {
            "washington_split": "derived_8.4",
            "ece_split": "derived_8.4_ece_v3",
            "temporal_expert_seeds": 30,
            "loso_expert_seeds": 5,
            "loso_held_out_stations": 7,
            "ece_expert_seeds": [42, 7, 13, 101, 123],
            "ece_target_use": "final scoring only",
        },
    }


def build_context() -> dict[str, str]:
    validate_sources()
    split = read_json("split")
    features = read_json("features")["global_features"]
    main = read_csv("main_summary")
    main_seed = read_csv("main_seed")
    loso = read_csv("loso_seed")
    t8 = read_csv("t8")
    ece = read_csv("ece_policy")
    audit = read_json("ece_audit")
    calibration = read_csv("ece_calibration")

    context: dict[str, str] = {}
    context["TRAIN_N"] = str(split["rows"]["train"])
    context["VAL_N"] = str(split["rows"]["val"])
    context["TEST_N"] = str(split["rows"]["test"])
    context["TRAINVAL_N"] = str(split["rows"]["train"] + split["rows"]["val"])
    context["FEATURE_COUNT"] = str(len(features))
    context["FEATURE_GROUPS"] = feature_groups_block(features)
    ece_split_rows = read_csv("ece_split")
    context["ECE_ROWS"] = str(len(ece_split_rows))
    context["ECE_STATIONS"] = str(len({row["station_id"] for row in ece_split_rows}))
    context["ECE_PER_STATION"] = str(len(ece_split_rows) // len({row["station_id"] for row in ece_split_rows}))
    context["ECE_FIRST_DATE"] = min(row["date"] for row in ece_split_rows)
    context["ECE_LAST_DATE"] = max(row["date"] for row in ece_split_rows)

    static = {row["station_id"]: row for row in read_csv("wa_static")}
    networks = station_networks(split["stations"])
    composition = {row["station_id"]: row for row in read_csv("composition")}
    station_rows = []
    station_display_names = {
        "Spokane": "Spokane",
        "Quinault": "Quinault",
        "Darrington": "Darrington",
        "SourdoughGulch_WA_985": "Sourdough Gulch",
        "CayusePass_WA": "Cayuse Pass",
        "Paradise_WA": "Paradise",
        "BeaverPass_WA_990": "Beaver Pass",
    }
    for name in split["stations"]:
        row = static[name]
        cluster = composition[name]
        station_rows.append([station_display_names[name], networks[name], fmt(row["latitude"], 2), fmt(row["longitude"], 2), fmt(row["J_elev_m"], 0), cluster["n"]])
    context["WA_STATION_TABLE"] = md_table(["Station", "Data network", "Lat.", "Lon.", "Elevation m", "Trainval rows"], station_rows)

    name_map = {
        "Guarded_Backbone54_k2": "station-majority regional model",
        "Clustering_Backbone54_k2": "station-majority regional model without station consistency guarantee",
        "Clustering_V0_Full_k2": "Earlier V0 regional model",
        "Global_Single": "Existing global model",
        "Clustering_Dynamic_k2": "Changing-covariate grouping",
        "Seasonal_Binary_k2": "Seasonal grouping",
        "Univariate_G_API_k2": "Precipitation-index grouping",
        "Trained_Gating_k2": "Target-threshold grouping",
    }
    temporal_rows = []
    for row in sorted(main, key=lambda item: number(item["mean_rmse"])):
        group = [item for item in main_seed if item["strategy_name"] == row["strategy_name"] and item["evidence_role"] == row["evidence_role"]]
        if row["strategy_name"] == "Clustering_V0_Full_k2":
            continue
        temporal_rows.append([
            name_map[row["strategy_name"]], row["n_seeds"],
            fmt(row["mean_r2"]), fmt(row["seed_sd_r2"]),
            f"[{fmt(row['seed_ci95_low'])}, {fmt(row['seed_ci95_high'])}]",
            fmt(row["mean_rmse"]), fmt(mean(group, "mae")), fmt(mean(group, "bias")),
        ])
    context["TEMPORAL_TABLE"] = md_table(["Model", "Seeds", "R²", "Seed SD", "95% seed CI", "RMSE", "MAE", "Bias"], temporal_rows)
    guard = one(main, strategy_name="Guarded_Backbone54_k2")
    backbone = one(main, strategy_name="Clustering_Backbone54_k2")
    global_row = one(main, strategy_name="Global_Single")
    gate_row = one(main, strategy_name="Trained_Gating_k2")
    context["GUARD_R2"] = fmt(guard["mean_r2"], 6)
    context["GUARD_RMSE"] = fmt(guard["mean_rmse"], 6)
    context["GUARD_SD"] = fmt(guard["seed_sd_r2"], 6)
    context["BACKBONE_R2"] = fmt(backbone["mean_r2"], 6)
    context["GLOBAL_R2"] = fmt(global_row["mean_r2"], 6)
    context["GATE_R2"] = fmt(gate_row["mean_r2"], 6)
    context["GUARD_BACKBONE_DIFF"] = signed(number(guard["mean_r2"]) - number(backbone["mean_r2"]), 6)
    context["GUARD_GLOBAL_GAP"] = signed(number(guard["mean_r2"]) - number(global_row["mean_r2"]), 4)
    context["GUARD_GATE_GAP"] = signed(number(guard["mean_r2"]) - number(gate_row["mean_r2"]), 4)

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in loso:
        by_strategy[row["strategy_name"]].append(row)
    formal_loso = read_csv("formal_loso")
    loso_rows = []
    loso_values: dict[str, float] = {}
    for strategy in ("Guarded_Backbone54_k2", "Clustering_Backbone54_k2"):
        value = mean(by_strategy[strategy], "r2")
        loso_values[strategy] = value
        loso_rows.append([name_map[strategy], "5 × 7", fmt(value), fmt(mean(by_strategy[strategy], "rmse"))])
    for config_id, label in (
        ("Global_Single_54", "Existing global model"),
    ):
        row = one(formal_loso, config_id=config_id)
        loso_values[config_id] = number(row["loso_mean_r2"])
        loso_rows.append([label, "5 × 7", fmt(row["loso_mean_r2"]), fmt(row["loso_mean_rmse"])])
    context["LOSO_TABLE"] = md_table(["Model", "Seeds × held-out sites", "Station-mean R²", "Station-mean RMSE"], loso_rows)
    context["GUARD_LOSO"] = fmt(loso_values["Guarded_Backbone54_k2"], 6)
    context["BACKBONE_LOSO"] = fmt(loso_values["Clustering_Backbone54_k2"], 6)
    context["GLOBAL_LOSO"] = fmt(loso_values["Global_Single_54"], 6)
    context["GUARD_LOSO_DIFF"] = signed(loso_values["Guarded_Backbone54_k2"] - loso_values["Clustering_Backbone54_k2"], 6)
    context["GUARD_GLOBAL_LOSO_GAP"] = signed(loso_values["Guarded_Backbone54_k2"] - loso_values["Global_Single_54"], 4)
    v0_temporal = one(main, strategy_name="Clustering_V0_Full_k2")
    v0_loso = one(formal_loso, config_id="Clustering_V0_Full_k2_c0_0_c1_0")
    context["V0_CONTEXT_TABLE"] = md_table(["Historical model", "Temporal R²", "LOSO R²", "Role"], [[
        "V0 regional model", fmt(v0_temporal["mean_r2"]), fmt(v0_loso["loso_mean_r2"]), "Historical reference"
    ]])
    fold_agreement = {row["held_out"]: row for row in read_csv("fold_agreement")}
    t8_rows = []
    for row in sorted(t8, key=lambda item: number(item["mean_r2_difference"]), reverse=True):
        station = row["station"]
        fold = fold_agreement[station]
        t8_rows.append([station_display_names.get(station, station), fmt(row["guarded_mean_r2"]), fmt(row["backbone_mean_r2"]), signed(row["mean_r2_difference"]),
                        f"{row['winning_seeds']}/{row['tied_seeds']}", fmt(fold["ARI_tr_V0_Full_vs_Backbone"], 3),
                        fmt(fold["ARI_tr_Backbone_vs_GuardedV-A"], 3)])
    context["T8_TABLE"] = md_table(["Held-out station", "Station-majority regional model R²", "station-majority regional model without station consistency guarantee R²", "Difference", "Wins / ties (5 seeds)", "Train-group agreement: older 50-feat. vs 54-feat.", "Train-group agreement: with vs without station consistency guarantee"], t8_rows)
    context["FOLD_GAINS"] = str(sum(number(row["mean_r2_difference"]) > 0 for row in t8))
    context["FOLD_TIES"] = str(sum(number(row["mean_r2_difference"]) == 0 for row in t8))

    k_rows = read_csv("k_sweep")
    context["K_TABLE"] = md_table(["Number of groups", "Silhouette", "Calinski–Harabasz", "Davies–Bouldin", "Trainval station agreement", "Smallest group share"], [
        [row["K"], fmt(row["silhouette"], 3), fmt(row["calinski_harabasz"], 1), fmt(row["davies_bouldin"], 3), fmt(row["mean_station_purity_trainval"], 3), fmt(row["min_cluster_share"], 3)] for row in k_rows
    ])
    profile = [row for row in read_csv("profile") if row["feature"] != "soil_moisture_5cm"]
    profile.sort(key=lambda row: number(row["separation_index"]), reverse=True)
    context["PROFILE_TABLE"] = md_table(["Feature", "Separation index", "Median c0", "Median c1"], [
        [row["feature"], fmt(row["separation_index"], 3), fmt(row["median_cluster0"], 3), fmt(row["median_cluster1"], 3)] for row in profile[:10]
    ])

    pair_rows = read_csv("formal_pair")
    no_delta_pair = one(pair_rows, A="Clustering_V0_Full_k2_c0_0_c1_0", B="Global_Single_54", metric="R2")
    context["V0_GLOBAL_PAIRED"] = f"{signed(no_delta_pair['mean_diff'])} [{fmt(no_delta_pair['ci_low'])}, {fmt(no_delta_pair['ci_high'])}], p={number(no_delta_pair['t_p']):.2g}"
    delta = one(read_csv("formal_delta"), strategy="Clustering_V0_Full_k2")
    # Temporal cells carry the source's "mean ± seed SD" strings verbatim;
    # the header says so explicitly instead of implying a bare mean.
    context["DELTA_TABLE"] = md_table(["Feature set", "Temporal R² (mean ± seed SD)", "LOSO R²"], [
        ["Shared features only (V0)", delta["none_temporal_r2"], fmt(delta["none_loso_r2"], 4)],
        ["Washington validation-selected additions (V0)", delta["val_temporal_r2"], fmt(delta["val_loso_r2"], 4)],
    ])

    family_names = {
        "Clustering_V0_Full_k2": "Earlier V0 regional model",
        "Clustering_Backbone54_k2": "station-majority regional model without station consistency guarantee",
        "Guarded_Backbone54_k2": "station-majority regional model",
        "Global_Single_54": "Existing global model",
    }
    policy_names = {
        "as_routed": "Usual fitted assignment",
        "auto_hard": "Precipitation-based assignment",
        "auto_soft": "Precipitation-based blend",
        "direct": "Single global predictor",
    }
    policy_order = {"as_routed": 0, "auto_hard": 1, "auto_soft": 2, "c0_only": 3, "c1_only": 4, "direct": 5}
    crosswalk = read_csv("ece_crosswalk")
    fixed_indices = {
        (row["family"], row["policy_id"]): int(number(row["family_local_expert_index"]))
        for row in crosswalk
        if row["policy_id"] in {"c0_only", "c1_only"}
    }
    ece_rows = []
    # 1.3: show the station-majority regional model plus the global reference only. The
    # same model without the station consistency guarantee has deployable rows
    # that are bit-identical on this ECE set (all 150 rows
    # take the same fallback branch), so repeating them doubles the table
    # without adding information; the text states the identity instead.
    # The soft-blend fallback matches the hard fallback at reported precision.
    for row in sorted(
        (item for item in ece if item["family"] in ("Guarded_Backbone54_k2", "Global_Single_54") and item["policy"] != "auto_soft"),
        key=lambda item: (list(family_names).index(item["family"]), policy_order[item["policy"]]),
    ):
        policy = row["policy"]
        policy_label = policy_names[policy] if policy in policy_names else f"Single regional predictor {fixed_indices[(row['family'], policy)]} (reference)"
        ece_rows.append([family_names[row["family"]], policy_label, "yes" if row["deployable"] == "True" else "no",
                         f"{fmt(row['rmse_mean'], 6)} ± {fmt(row['rmse_std'], 6)}", fmt(row["mae_mean"], 6),
                         signed(row["bias_mean"], 6), fmt(row["ubrmse_mean"], 6)])
    context["ECE_TABLE"] = md_table(["Model", "Assignment or reference", "Usable from observed inputs?", "RMSE ± seed SD", "MAE", "Bias", "ubRMSE"], ece_rows)
    auto = one(ece, family="Guarded_Backbone54_k2", policy="auto_hard")
    routed = one(ece, family="Guarded_Backbone54_k2", policy="as_routed")
    global_ece = one(ece, family="Global_Single_54", policy="direct")
    context["ECE_AUTO_RMSE"] = fmt(auto["rmse_mean"], 6)
    context["ECE_ROUTED_RMSE"] = fmt(routed["rmse_mean"], 6)
    context["ECE_GLOBAL_RMSE"] = fmt(global_ece["rmse_mean"], 6)
    context["ECE_AUTO_GLOBAL_GAP"] = signed(number(auto["rmse_mean"]) - number(global_ece["rmse_mean"]), 6)
    context["ECE_AUTO_ROUTED_GAP"] = signed(number(auto["rmse_mean"]) - number(routed["rmse_mean"]), 6)
    context["ECE_AUTO_ROUTED_REDUCTION"] = fmt(number(routed["rmse_mean"]) - number(auto["rmse_mean"]), 6)
    index_reference_policy = {
        index: policy
        for policy in ("c0_only", "c1_only")
        for index in [fixed_indices[("Guarded_Backbone54_k2", policy)]]
    }
    context["ECE_INDEX0_RMSE"] = fmt(one(ece, family="Guarded_Backbone54_k2", policy=index_reference_policy[0])["rmse_mean"], 6)
    context["ECE_INDEX1_RMSE"] = fmt(one(ece, family="Guarded_Backbone54_k2", policy=index_reference_policy[1])["rmse_mean"], 6)

    cross_rows = []
    for family in ("Clustering_Backbone54_k2", "Guarded_Backbone54_k2"):
        row = one(crosswalk, family=family, policy_id="auto_hard")
        oracle = one(crosswalk, family=family, policy_id="c0_only")
        cross_rows.append([family_names[family], f"index {int(number(row['gapi_class_0_local_expert']))}",
                           f"index {int(number(row['gapi_class_1_local_expert']))}",
                           f"index {int(number(row['dry_assigned_local_expert_index']))}",
                           f"index {int(number(row['complementary_local_expert_index']))}",
                           "no" if oracle["deployable"] == "False" else "yes",
                           "yes" if row["dry_assignment_matches_canonical_feature_minimum"] == "True" else "no"])
    context["ECE_CROSSWALK"] = md_table(["Model", "Precipitation class 0 → predictor", "Class 1 → predictor", "Fixed comparator predictor", "Other predictor", "Comparator usable from inputs?", "Index with lower WA SMAP mean?"], cross_rows)
    # The v1.1 crosswalk records the same fit-frame means for every family.
    cross_guard = one(crosswalk, family="Guarded_Backbone54_k2", policy_id="auto_hard")
    context["WA_SMAP_C0"] = fmt(cross_guard["wa_canonical_smap_mean_local_c0"], 6)
    context["WA_SMAP_C1"] = fmt(cross_guard["wa_canonical_smap_mean_local_c1"], 6)
    context["WA_TARGET_C0"] = fmt(cross_guard["wa_target_mean_local_c0"], 6)
    context["WA_TARGET_C1"] = fmt(cross_guard["wa_target_mean_local_c1"], 6)
    candidate_rows = [row for row in calibration if row["family"] == "Guarded_Backbone54_k2" and row["setting"] == "smap_masked_val" and row["policy"].startswith("aux_hard_candidate_")]
    context["ECE_CALIBRATION_TABLE"] = md_table(["Precipitation class 0 →", "Precipitation class 1 →", "WA validation RMSE", "Chosen for ECE"], [
        [f"index {int(number(row['gapi_class_0_local_expert']))}", f"index {int(number(row['gapi_class_1_local_expert']))}", fmt(row["rmse"], 6), "yes" if row["selected_for_ece"] == "True" else "no"]
        for row in sorted(candidate_rows, key=lambda item: number(item["rmse"]))
    ])

    sites = {row["station_id"]: row for row in read_csv("ece_sites")}
    ece_display_names = {
        "ECE_BBG_Lost_Meadow": "Bellevue Botanical Garden Lost Meadow",
        "ECE_BBG_Main_St": "Bellevue Botanical Garden Main Street",
        "ECE_Renton_Garden_North": "Renton Garden North",
        "ECE_Renton_Garden_Shed": "Renton Garden Shed",
        "ECE_Renton_Home": "Renton Home",
    }
    ece_station = read_csv("ece_station")
    share_rows = read_csv("ece_shares")
    station_rows = []
    for station in sorted(sites):
        routed_station = one(ece_station, family="Guarded_Backbone54_k2", policy="as_routed", station=station)
        auto_station = one(ece_station, family="Guarded_Backbone54_k2", policy="auto_hard", station=station)
        global_station = one(ece_station, family="Global_Single_54", policy="direct", station=station)
        share = one(share_rows, family="Guarded_Backbone54_k2", policy="auto_hard", station_id=station)
        station_rows.append([ece_display_names.get(station, station), fmt(sites[station]["J_elev_m"], 0), fmt(sites[station]["J_bio_bio12"], 0),
                             fmt(routed_station["rmse_mean"], 4), fmt(auto_station["rmse_mean"], 4),
                             fmt(global_station["rmse_mean"], 4), signed(auto_station["bias_mean"], 4),
                             fmt(share["dry_assigned_weight_mean"], 3)])
    context["ECE_STATION_TABLE"] = md_table(["ECE station", "Elev. m", "Annual precip. descriptor mm", "Station-majority usual RMSE", "Station-majority alternate RMSE", "Global RMSE", "Alternate bias", "Weight on comparator predictor"], station_rows)

    legacy = one(read_csv("ece_legacy"), Category="Clustering vs Global", **{"Comparison (A vs B)": "Clustering (V0) vs Global-54"})
    context["ECE_LEGACY_DIFF"] = signed(legacy["Station Mean ΔRMSE (A−B)"], 6)
    context["ECE_LEGACY_WINS"] = legacy["Station Wins (A < B)"]
    context["ECE_LEGACY_SIGN_P"] = fmt(legacy["Binomial Sign Test p"], 3)

    oos = read_csv("oos")
    oos_labels = {
        "Baseline Model (50 V0 feats)": "Earlier global baseline (50 features)",
        "Clustering (54 backbone)": "Regional model (54 shared features)",
        "Clustering (50 V0 features)": "Earlier V0 regional model (50 features)",
        "Seasonal Binary (Summer/Winter)": "Seasonal grouping",
        "Univariate G_API split": "Precipitation-index grouping",
        "Global Single Model (54 feats)": "Existing global model (54 features)",
        "Clustering (Dynamic features)": "Changing-covariate grouping",
        "Trained Gating Classifier": "Target-threshold grouping",
    }
    oos_rows = []
    v0_oos_rows = []
    # Branch on the source architecture key, not the display label: the
    # appendix-bound row is the one KMeans variant fitted on the older
    # 50-feature set, regardless of how its display label is worded.
    v0_oos_architectures = {"Clustering (50 V0 features)"}
    for row in oos:
        architecture = row["Model Architecture"]
        label = oos_labels[architecture]
        target_rows = v0_oos_rows if architecture in v0_oos_architectures else oos_rows
        target_rows.append([label, fmt(row["Station Mean R²"], 3), fmt(row["Station Mean RMSE"], 3), row["Pooled R² (mean ± std)"]])
    oos_headers = ["Earlier model or grouping", "10-station mean R²", "10-station mean RMSE", "Pooled R²"]
    context["OOS_TABLE"] = md_table(oos_headers, oos_rows)
    context["V0_OOS_TABLE"] = md_table(oos_headers, v0_oos_rows)
    hourly = read_csv("hourly")
    if not hourly:
        raise ValueError("hourly sensor summary is empty")
    hourly_range = statistics.median(number(row["hourly_range"]) for row in hourly)
    daily_steps = [abs(number(row["d_target_vs_prev_day"])) for row in hourly if row["d_target_vs_prev_day"] and math.isfinite(number(row["d_target_vs_prev_day"]))]
    if not daily_steps:
        raise ValueError("hourly sensor summary has no finite day-to-day target steps")
    context["HOURLY_RANGE"] = fmt(hourly_range, 4)
    context["DAILY_STEP"] = fmt(statistics.median(daily_steps), 4)

    for key in SOURCES:
        context[f"LEDGER_SOURCE_{key.upper()}"] = ledger_source_link(SOURCES[key])
    for key, relative in EXTRA_PROVENANCE.items():
        context[f"LEDGER_SOURCE_{key.upper()}"] = ledger_source_link(relative)
    return context
