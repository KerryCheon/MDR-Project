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
EVIDENCE_12 = "notebooks/experiment/paper2-final-evidence-1.2"
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
    "k_sweep": f"{EVIDENCE_10}/k_sweep_quality.csv",
    "temporal_seed_cluster": f"{EVIDENCE_10}/temporal_seed_cluster.csv",
    "delta_grid": "notebooks/experiment/derived_8.4-feature-selection-2.0/artifacts/delta_grid.csv",
    "composition": "notebooks/experiment/derived_8.4-gating-analysis-1.0/regime_station_composition_Clustering_Backbone54_k2.csv",
    "profile": "notebooks/experiment/derived_8.4-gating-analysis-1.0/regime_profile_summary_Clustering_Backbone54_k2.csv",
    "global_temporal_seed": f"{EVIDENCE_12}/global_paired_temporal_seed.csv",
    "global_temporal_summary": f"{EVIDENCE_12}/global_paired_temporal_summary.csv",
    "global_loso_seed": f"{EVIDENCE_12}/global_paired_loso_seed_station.csv",
    "global_loso_summary": f"{EVIDENCE_12}/global_paired_loso_summary.csv",
    "global_manifest": f"{EVIDENCE_12}/run_manifest.json",
    "formal_loso": f"{FORMAL_10}/loso_config_summary.csv",
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
    "global_config": f"{EVIDENCE_12}/config.yaml",
    "ece_config": f"{EVIDENCE_11}/ece_guarded/config.yaml",
    "ece_runner": f"{EVIDENCE_11}/ece_guarded/run_guarded.py",
}

# Two-sided 97.5% t quantile, 29 df, for the 30-seed 95% intervals reported
# alongside the saved 1.0 intervals (which use the same construction).
T29_975 = 2.0452296428670278

# Expected LOSO fold sizes: identical derived_8.4 split in both harnesses.
LOSO_FOLD_SIZES = {
    "BeaverPass_WA_990": (12423, 626),
    "CayusePass_WA": (12566, 1081),
    "Darrington": (12560, 999),
    "Paradise_WA": (12419, 1067),
    "Quinault": (12448, 1044),
    "SourdoughGulch_WA_985": (12417, 906),
    "Spokane": (12815, 897),
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
    ("Satellite soil moisture — SMAP (12 features)",
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
    ("Radar — Sentinel-1 SAR (10 features)",
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
    ("Vegetation and optical — Sentinel-2 (12 features)",
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


def check_paired_harness(rows: list[dict], label: str) -> None:
    for row in rows:
        if row.get("evidence_role") != "paired_in_harness":
            raise ValueError(f"{label}: row is not paired_in_harness: {row}")
    harnesses = {row.get("source_harness") for row in rows}
    if len(harnesses) != 1:
        raise ValueError(f"{label}: mixed source harnesses: {harnesses}")


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
    global_manifest = read_json("global_manifest")
    if main_manifest["status"] != "completed" or ece_manifest["status"] != "complete":
        raise ValueError("evidence bundles are not complete")
    if global_manifest["status"] != "completed":
        raise ValueError("paired-global bundle is not complete")
    if ece_manifest["ece_run"]["fitting_and_evaluation"]["ece_targets_used_for_fitting_or_mapping"]:
        raise ValueError("ECE targets used for model selection")

    # 1.0 paired temporal rows: Guarded + Backbone, 30 seeds each.
    main_seed = read_csv("main_seed")
    main_summary = read_csv("main_summary")
    paired_names = ("Guarded_Backbone54_k2", "Clustering_Backbone54_k2")
    for strategy in paired_names:
        group = [item for item in main_seed
                 if item["strategy_name"] == strategy and item["evidence_role"] == "paired_in_harness"]
        if len(group) != 30 or len({item["seed"] for item in group}) != 30:
            raise ValueError(f"temporal seed coverage: {strategy}")
        summary_row = one(main_summary, strategy_name=strategy)
        if summary_row["evidence_role"] != "paired_in_harness":
            raise ValueError(f"temporal summary lost paired provenance: {strategy}")
        check_close(mean(group, "r2"), number(summary_row["mean_r2"]), f"temporal R2 {strategy}")
        check_close(mean(group, "rmse"), number(summary_row["mean_rmse"]), f"temporal RMSE {strategy}")
    guard = one(main_summary, strategy_name="Guarded_Backbone54_k2")
    backbone = one(main_summary, strategy_name="Clustering_Backbone54_k2")
    check_close(number(guard["mean_r2"]), 0.8118429663371349, "main notebook stdout Guarded R2")
    check_close(number(backbone["mean_r2"]), 0.8117237199127312, "main notebook stdout Backbone R2")

    # Heuristic context rows: sensitivity-anchored temporal coverage, 30 seeds each.
    for strategy in ("Clustering_Dynamic_k2", "Seasonal_Binary_k2",
                     "Univariate_G_API_k2", "Trained_Gating_k2"):
        group = [item for item in main_seed
                 if item["strategy_name"] == strategy and item["evidence_role"] == "sensitivity_anchored"]
        if len(group) != 30 or len({item["seed"] for item in group}) != 30:
            raise ValueError(f"heuristic temporal seed coverage: {strategy}")
        summary_row = one(main_summary, strategy_name=strategy)
        check_close(mean(group, "r2"), number(summary_row["mean_r2"]), f"heuristic temporal R2 {strategy}")

    # Heuristic context LOSO rows: no-delta configs from the saved formal summary.
    formal_loso = read_csv("formal_loso")
    for config_id in ("Clustering_Dynamic_k2_c0_0_c1_0", "Seasonal_Binary_k2_c0_0_c1_0",
                      "Univariate_G_API_k2_c0_0_c1_0", "Trained_Gating_k2_c0_0_c1_0"):
        row = one(formal_loso, config_id=config_id)
        if int(row["n_stations"]) != 7:
            raise ValueError(f"heuristic LOSO station coverage: {config_id}")

    # 1.2 paired-global temporal rows: 30 seeds, same seed set as 1.0 Guarded.
    global_temporal = read_csv("global_temporal_seed")
    check_paired_harness(global_temporal, "paired-global temporal")
    if global_temporal[0]["source_harness"] != "paper2-final-evidence-1.2":
        raise ValueError("paired-global temporal is not sourced from paper2-final-evidence-1.2")
    global_seeds = {item["seed"] for item in global_temporal}
    guard_seeds = {item["seed"] for item in main_seed if item["strategy_name"] == "Guarded_Backbone54_k2"}
    if len(global_temporal) != 30 or len(global_seeds) != 30 or global_seeds != guard_seeds:
        raise ValueError("paired-global temporal seed coverage changed")
    for item in global_temporal:
        if item["strategy_name"] != "Global_Single" or item["config_id"] != "Global_Single_54":
            raise ValueError(f"paired-global temporal identity changed: {item}")
        if (int(item["n_train_total"]), int(item["n_test"])) != (14608, 6620):
            raise ValueError("paired-global temporal split sizes changed")
    global_summary = read_csv("global_temporal_summary")
    global_summary_row = one(global_summary, strategy_name="Global_Single")
    check_close(mean(global_temporal, "r2"), number(global_summary_row["mean_r2"]), "paired-global temporal R2")
    check_close(mean(global_temporal, "r2"), 0.7797940453067711, "paired-global temporal R2 anchor")
    check_close(mean(global_temporal, "rmse"), 0.0478022798608347, "paired-global temporal RMSE anchor")

    # 1.0 paired LOSO rows: Guarded + Backbone, 5 seeds x 7 folds each.
    loso = read_csv("loso_seed")
    if len({row["station"] for row in loso}) != 7:
        raise ValueError("LOSO station coverage changed")
    for strategy in paired_names:
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

    # 1.2 paired-global LOSO rows: 5 seeds x 7 folds, same keys as 1.0 Guarded.
    global_loso = read_csv("global_loso_seed")
    check_paired_harness(global_loso, "paired-global LOSO")
    if global_loso[0]["source_harness"] != "paper2-final-evidence-1.2":
        raise ValueError("paired-global LOSO is not sourced from paper2-final-evidence-1.2")
    if len(global_loso) != 35:
        raise ValueError("paired-global LOSO coverage changed")
    for item in global_loso:
        if item["strategy_name"] != "Global_Single":
            raise ValueError(f"paired-global LOSO identity changed: {item}")
        expected_sizes = LOSO_FOLD_SIZES.get(item["station"])
        if expected_sizes is None or (int(item["n_train_total"]), int(item["n_test"])) != expected_sizes:
            raise ValueError(f"paired-global LOSO fold sizes changed: {item['station']}")
    global_keys = {(item["seed"], item["station"]) for item in global_loso}
    guard_keys = {(item["seed"], item["station"]) for item in guard_loso}
    if global_keys != guard_keys:
        raise ValueError("paired-global LOSO seed/station keys do not match Guarded LOSO keys")
    global_loso_summary = read_csv("global_loso_summary")
    global_loso_row = one(global_loso_summary, strategy_name="Global_Single")
    check_close(mean(global_loso, "r2"), number(global_loso_row["loso_mean_r2"]), "paired-global LOSO R2")
    check_close(mean(global_loso, "r2"), 0.5797371923742318, "paired-global LOSO R2 anchor", tolerance=1e-9)
    # NOTE: an older formal-eval summary file reports 0.579500 for this same
    # quantity, but the mean of that file's own seed rows is 0.579737 and the
    # 1.2 refit reproduces those seed rows exactly; 0.579737 is the value used.

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
    # Table 8 shows the station-majority shared-feature cluster-routed
    # multi-regime model only because the rows of the same model without the
    # station consistency guarantee are identical on this ECE set (all 150 rows
    # take the same fallback branch), so repeating them doubles the table
    # without adding information; the text states the identity instead. The
    # soft-blend fallback matches the hard fallback at reported (6-decimal)
    # precision; enforce that too.
    for policy in ("as_routed", "auto_hard"):
        guard_row = one(ece_policy, family="Guarded_Backbone54_k2", policy=policy)
        backbone_row = one(ece_policy, family="Clustering_Backbone54_k2", policy=policy)
        for field in ("rmse_mean", "rmse_std", "mae_mean", "bias_mean", "ubrmse_mean"):
            check_close(number(guard_row[field]), number(backbone_row[field]), f"ECE identity Guarded==Backbone {policy} {field}")
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

    # Section 8 interpretation: the two Washington groups separate strongly on
    # longitude/wetness but only weakly on elevation, while the ECE sites sit far
    # below both group medians. Anchor the profile and site descriptors used in
    # the prose.
    profile_anchor = {row["feature"]: row for row in read_csv("profile")}
    check_close(number(profile_anchor["elev"]["separation_index"]), 0.208, "group elevation separation", tolerance=1e-9)
    check_close(number(profile_anchor["elev"]["median_cluster0"]), 1205.0942, "group 0 median elevation", tolerance=1e-9)
    check_close(number(profile_anchor["elev"]["median_cluster1"]), 1160.5261, "group 1 median elevation", tolerance=1e-9)
    site_rows = read_csv("ece_sites")
    check_close(min(number(row["J_elev_m"]) for row in site_rows), 51.0, "ECE site min elevation", tolerance=1e-9)
    check_close(max(number(row["J_elev_m"]) for row in site_rows), 157.0, "ECE site max elevation", tolerance=1e-9)
    check_close(min(number(row["J_bio_bio12"]) for row in site_rows), 1018.0, "ECE site min precipitation descriptor", tolerance=1e-9)
    check_close(max(number(row["J_bio_bio12"]) for row in site_rows), 1227.0, "ECE site max precipitation descriptor", tolerance=1e-9)
    routed_stations = [row for row in read_csv("ece_station")
                       if row["family"] == "Guarded_Backbone54_k2" and row["policy"] == "as_routed"]
    if len(routed_stations) != 5:
        raise ValueError("ECE routed station coverage changed")
    check_close(min(number(row["rmse_mean"]) for row in routed_stations), 0.0479510908994335, "ECE routed station min RMSE", tolerance=1e-9)
    check_close(max(number(row["rmse_mean"]) for row in routed_stations), 0.24252405030086113, "ECE routed station max RMSE", tolerance=1e-9)

    # Section 7 specialization discussion: per-regime specialist metrics and the
    # per-regime feature-addition grid, both anchored to their saved artifacts.
    cluster_rows = read_csv("temporal_seed_cluster")
    for cluster, anchor_r2, anchor_bias in (
        ("0", 0.7989032555009037, 0.00887969006902288),
        ("1", 0.8419359789045798, -0.0018471804156466936),
    ):
        group = [row for row in cluster_rows
                 if row["config_id"].startswith("Guarded_Backbone54_k2") and row["cluster"] == cluster]
        if len(group) != 30 or len({row["seed"] for row in group}) != 30:
            raise ValueError(f"per-cluster temporal seed coverage changed: cluster {cluster}")
        if {row["n_train"] for row in group} != ({"10624"} if cluster == "0" else {"3984"}):
            raise ValueError(f"per-cluster training size changed: cluster {cluster}")
        check_close(mean(group, "r2"), anchor_r2, f"per-cluster R2 cluster {cluster}", tolerance=1e-9)
        check_close(mean(group, "bias"), anchor_bias, f"per-cluster bias cluster {cluster}", tolerance=1e-9)
    check_close(mean(global_temporal, "bias"), 0.01003860659709445, "paired-global mean bias", tolerance=1e-9)

    delta = {row["candidate_id"]: row for row in read_csv("delta_grid")}
    for candidate, anchor in (
        ("delta_c0_0_c1_0", 0.8143343367764553),
        ("delta_c0_10_c1_0", 0.7890715142526764),
        ("delta_c0_0_c1_10", 0.8150371780573779),
    ):
        if candidate not in delta:
            raise ValueError(f"per-regime feature-addition candidate missing: {candidate}")
        check_close(number(delta[candidate]["pooled_r2"]), anchor, f"feature-addition pooled R2 {candidate}", tolerance=1e-9)


def source_manifest() -> dict:
    inputs = {**SOURCES, **EXTRA_PROVENANCE}
    return {
        "assembly": "report-assemble-1.5",
        "source_precedence": {
            "final_evaluation_batch": {
                "main_temporal_loso": EVIDENCE_10,
                "paired_global": EVIDENCE_12,
                "ece_results": EVIDENCE_11,
            },
            "outline": SOURCES["outline"],
            "historical_sources": "labeled unpaired context (heuristic grouping rows) and scope boundaries (out-of-state, earlier ECE comparison)",
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
            "paired_global_harness": EVIDENCE_12,
            "ece_expert_seeds": [42, 7, 13, 101, 123],
            "ece_target_use": "final scoring only",
        },
    }


def paired_wins(diffs: list[float]) -> tuple[int, int, int]:
    wins = sum(1 for diff in diffs if diff > 0)
    ties = sum(1 for diff in diffs if diff == 0)
    losses = sum(1 for diff in diffs if diff < 0)
    return wins, ties, losses


def build_context() -> dict[str, str]:
    validate_sources()
    split = read_json("split")
    features = read_json("features")["global_features"]
    main = read_csv("main_summary")
    main_seed = read_csv("main_seed")
    loso = read_csv("loso_seed")
    t8 = read_csv("t8")
    global_temporal = read_csv("global_temporal_seed")
    global_loso = read_csv("global_loso_seed")
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
        "Guarded_Backbone54_k2": "station-majority shared-feature cluster-routed multi-regime model",
        "Clustering_Backbone54_k2": "shared-feature cluster-routed multi-regime model without station consistency guarantee",
        "Global_Single": "Existing single-regime global model",
        "Clustering_Dynamic_k2": "Three-feature K-means grouping",
        "Seasonal_Binary_k2": "Seasonal grouping",
        "Univariate_G_API_k2": "Precipitation-index grouping",
        "Trained_Gating_k2": "Target-threshold grouping",
    }
    guard = one(main, strategy_name="Guarded_Backbone54_k2")
    backbone = one(main, strategy_name="Clustering_Backbone54_k2")
    guard_seed = [item for item in main_seed if item["strategy_name"] == "Guarded_Backbone54_k2" and item["evidence_role"] == "paired_in_harness"]
    backbone_seed = [item for item in main_seed if item["strategy_name"] == "Clustering_Backbone54_k2" and item["evidence_role"] == "paired_in_harness"]

    # Paired-global temporal row: same 30 seeds, same protocol, refit in 1.2.
    n_global = len(global_temporal)
    global_mean_r2 = mean(global_temporal, "r2")
    global_sd_r2 = statistics.stdev(number(row["r2"]) for row in global_temporal)
    global_ci_half = T29_975 * global_sd_r2 / math.sqrt(n_global)
    temporal_rows = []
    for label, mean_r2, sd_r2, ci_low, ci_high, mean_rmse, mae, bias, n_seeds in [
        ("station-majority shared-feature cluster-routed multi-regime model",
         number(guard["mean_r2"]), number(guard["seed_sd_r2"]),
         number(guard["seed_ci95_low"]), number(guard["seed_ci95_high"]),
         number(guard["mean_rmse"]), mean(guard_seed, "mae"), mean(guard_seed, "bias"), guard["n_seeds"]),
        ("shared-feature cluster-routed multi-regime model without station consistency guarantee",
         number(backbone["mean_r2"]), number(backbone["seed_sd_r2"]),
         number(backbone["seed_ci95_low"]), number(backbone["seed_ci95_high"]),
         number(backbone["mean_rmse"]), mean(backbone_seed, "mae"), mean(backbone_seed, "bias"), backbone["n_seeds"]),
        ("Existing single-regime global model",
         global_mean_r2, global_sd_r2,
         global_mean_r2 - global_ci_half, global_mean_r2 + global_ci_half,
         mean(global_temporal, "rmse"), mean(global_temporal, "mae"), mean(global_temporal, "bias"), str(n_global)),
    ]:
        temporal_rows.append([
            label, n_seeds,
            fmt(mean_r2), fmt(sd_r2),
            f"[{fmt(ci_low)}, {fmt(ci_high)}]",
            fmt(mean_rmse), fmt(mae), fmt(bias),
        ])
    # Simpler-grouping rows, shown for context.
    for strategy in ("Clustering_Dynamic_k2", "Seasonal_Binary_k2",
                     "Univariate_G_API_k2", "Trained_Gating_k2"):
        summary_row = one(main, strategy_name=strategy)
        group = [item for item in main_seed if item["strategy_name"] == strategy and item["evidence_role"] == "sensitivity_anchored"]
        temporal_rows.append([
            name_map[strategy], "30",
            fmt(summary_row["mean_r2"]), fmt(summary_row["seed_sd_r2"]),
            f"[{fmt(summary_row['seed_ci95_low'])}, {fmt(summary_row['seed_ci95_high'])}]",
            fmt(summary_row["mean_rmse"]), fmt(mean(group, "mae")), fmt(mean(group, "bias")),
        ])
    context["TEMPORAL_TABLE"] = md_table(["Model", "Seeds", "R²", "Seed SD", "95% seed CI", "RMSE", "MAE", "Bias"], temporal_rows)
    gate_row = one(main, strategy_name="Trained_Gating_k2")
    context["GATE_R2"] = fmt(gate_row["mean_r2"], 4)
    three_feature_row = one(main, strategy_name="Clustering_Dynamic_k2")
    context["THREE_FEATURE_R2"] = fmt(three_feature_row["mean_r2"], 4)
    context["GUARD_R2"] = fmt(guard["mean_r2"], 4)
    context["GUARD_RMSE"] = fmt(guard["mean_rmse"], 4)
    context["GUARD_SD"] = fmt(guard["seed_sd_r2"], 4)
    context["BACKBONE_R2"] = fmt(backbone["mean_r2"], 4)
    context["GLOBAL_R2"] = fmt(global_mean_r2, 4)
    context["GLOBAL_SD"] = fmt(global_sd_r2, 4)
    context["GUARD_BACKBONE_DIFF"] = signed(number(guard["mean_r2"]) - number(backbone["mean_r2"]), 4)

    # Seed-paired Guarded-minus-Global temporal differences (joined on seed).
    guard_by_seed = {item["seed"]: number(item["r2"]) for item in guard_seed}
    global_by_seed = {item["seed"]: number(item["r2"]) for item in global_temporal}
    guard_global_diffs = [guard_by_seed[seed] - global_by_seed[seed] for seed in guard_by_seed]
    if set(guard_by_seed) != set(global_by_seed):
        raise ValueError("Guarded/Global temporal seed sets do not match for pairing")
    context["GUARD_GLOBAL_DIFF"] = signed(statistics.mean(guard_global_diffs), 4)
    context["GUARD_GLOBAL_GAP"] = signed(statistics.mean(guard_global_diffs), 4)
    wins, ties, losses = paired_wins(guard_global_diffs)
    context["GUARD_GLOBAL_WINS"] = str(wins)
    context["GUARD_GLOBAL_TIES"] = str(ties)
    context["GUARD_GLOBAL_LOSSES"] = str(losses)
    context["GUARD_GLOBAL_SEEDS"] = str(len(guard_global_diffs))

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in loso:
        by_strategy[row["strategy_name"]].append(row)
    guard_loso_mean = mean(by_strategy["Guarded_Backbone54_k2"], "r2")
    backbone_loso_mean = mean(by_strategy["Clustering_Backbone54_k2"], "r2")
    global_loso_mean = mean(global_loso, "r2")
    loso_rows = [
        [name_map["Guarded_Backbone54_k2"], "5 × 7", fmt(guard_loso_mean), fmt(mean(by_strategy["Guarded_Backbone54_k2"], "rmse"))],
        [name_map["Clustering_Backbone54_k2"], "5 × 7", fmt(backbone_loso_mean), fmt(mean(by_strategy["Clustering_Backbone54_k2"], "rmse"))],
        [name_map["Global_Single"], "5 × 7", fmt(global_loso_mean), fmt(mean(global_loso, "rmse"))],
    ]
    # Simpler-grouping LOSO rows, shown for context.
    formal_loso = read_csv("formal_loso")
    for config_id, strategy in (
        ("Clustering_Dynamic_k2_c0_0_c1_0", "Clustering_Dynamic_k2"),
        ("Seasonal_Binary_k2_c0_0_c1_0", "Seasonal_Binary_k2"),
        ("Univariate_G_API_k2_c0_0_c1_0", "Univariate_G_API_k2"),
        ("Trained_Gating_k2_c0_0_c1_0", "Trained_Gating_k2"),
    ):
        row = one(formal_loso, config_id=config_id)
        loso_rows.append([name_map[strategy], "5 × 7", fmt(row["loso_mean_r2"]), fmt(row["loso_mean_rmse"])])
    context["LOSO_TABLE"] = md_table(["Model", "Seeds × held-out sites", "Station-mean R²", "Station-mean RMSE"], loso_rows)
    context["GUARD_LOSO"] = fmt(guard_loso_mean, 4)
    context["BACKBONE_LOSO"] = fmt(backbone_loso_mean, 4)
    context["GLOBAL_LOSO"] = fmt(global_loso_mean, 4)
    context["GUARD_LOSO_DIFF"] = signed(guard_loso_mean - backbone_loso_mean, 4)
    context["GUARD_GLOBAL_LOSO_DIFF"] = signed(guard_loso_mean - global_loso_mean, 4)
    context["GUARD_GLOBAL_LOSO_GAP"] = signed(guard_loso_mean - global_loso_mean, 4)

    fold_agreement = {row["held_out"]: row for row in read_csv("fold_agreement")}
    t8_rows = []
    for row in sorted(t8, key=lambda item: number(item["mean_r2_difference"]), reverse=True):
        station = row["station"]
        fold = fold_agreement[station]
        t8_rows.append([station_display_names.get(station, station), fmt(row["guarded_mean_r2"]), fmt(row["backbone_mean_r2"]), signed(row["mean_r2_difference"]),
                        f"{row['winning_seeds']}/{row['tied_seeds']}", fmt(fold["ARI_tr_Backbone_vs_GuardedV-A"], 3)])
    context["T8_TABLE"] = md_table(["Held-out station", "Station-majority shared-feature cluster-routed multi-regime model R²", "shared-feature cluster-routed multi-regime model without station consistency guarantee R²", "Difference", "Wins / ties (5 seeds)", "Train-group agreement: with vs without station consistency guarantee"], t8_rows)
    context["FOLD_GAINS"] = str(sum(number(row["mean_r2_difference"]) > 0 for row in t8))
    context["FOLD_TIES"] = str(sum(number(row["mean_r2_difference"]) == 0 for row in t8))

    # Seed-paired Guarded-minus-Global LOSO differences, joined on (seed, station).
    guard_loso_by_key = {(item["seed"], item["station"]): number(item["r2"]) for item in by_strategy["Guarded_Backbone54_k2"]}
    global_loso_by_key = {(item["seed"], item["station"]): number(item["r2"]) for item in global_loso}
    t8_global_rows = []
    global_fold_diffs: list[float] = []
    station_seed_wins: dict[str, list[float]] = defaultdict(list)
    for station in sorted({key[1] for key in guard_loso_by_key}):
        diffs = [guard_loso_by_key[(seed, station)] - global_loso_by_key[(seed, station)] for seed in sorted({key[0] for key in guard_loso_by_key})]
        station_seed_wins[station] = diffs
        global_fold_diffs.append(statistics.mean(diffs))
        wins, ties, losses = paired_wins(diffs)
        guard_means = statistics.mean(guard_loso_by_key[(seed, station)] for seed in sorted({key[0] for key in guard_loso_by_key}))
        global_means = statistics.mean(global_loso_by_key[(seed, station)] for seed in sorted({key[0] for key in guard_loso_by_key}))
        t8_global_rows.append([station_display_names.get(station, station), fmt(guard_means), fmt(global_means),
                               signed(statistics.mean(diffs)), f"{wins}/{ties}/{losses}"])
    t8_global_rows.sort(key=lambda row: float(row[3]), reverse=True)
    context["T8_GLOBAL_TABLE"] = md_table(["Held-out station", "Station-majority shared-feature cluster-routed multi-regime model R²", "Existing single-regime global model R²", "Difference (5-seed mean)", "Wins / ties / losses (5 seeds)"], t8_global_rows)
    context["GFOLD_GAINS"] = str(sum(1 for diffs in station_seed_wins.values() if statistics.mean(diffs) > 0))
    context["GFOLD_TIES"] = str(sum(1 for diffs in station_seed_wins.values() if statistics.mean(diffs) == 0))
    context["GFOLD_LOSSES"] = str(sum(1 for diffs in station_seed_wins.values() if statistics.mean(diffs) < 0))
    all_seed_diffs = [diff for diffs in station_seed_wins.values() for diff in diffs]
    wins, ties, losses = paired_wins(all_seed_diffs)
    context["GFOLD_SEED_WINS"] = str(wins)
    context["GFOLD_SEED_TIES"] = str(ties)
    context["GFOLD_SEED_LOSSES"] = str(losses)

    k_rows = read_csv("k_sweep")
    context["K_TABLE"] = md_table(["Number of groups", "Silhouette", "Calinski–Harabasz", "Davies–Bouldin", "Trainval station agreement", "Smallest group share"], [
        [row["K"], fmt(row["silhouette"], 3), fmt(row["calinski_harabasz"], 1), fmt(row["davies_bouldin"], 3), fmt(row["mean_station_purity_trainval"], 3), fmt(row["min_cluster_share"], 3)] for row in k_rows
    ])
    k2_row = one(k_rows, K="2")
    group_sizes = json.loads(k2_row["cluster_sizes"])
    context["GROUP_C0_N"] = str(int(group_sizes[0]))
    context["GROUP_C1_N"] = str(int(group_sizes[1]))
    context["MIN_STATION_PURITY"] = fmt(min(number(row["purity"]) for row in read_csv("composition")), 3)
    # Section 7 specialization discussion: per-regime specialist metrics
    # (paper-2 Guarded fit) and the per-regime feature-addition grid.
    cluster_rows = read_csv("temporal_seed_cluster")
    for cluster, r2_key, bias_key in (("0", "PERCLUSTER_C0_R2", "C0_BIAS"), ("1", "PERCLUSTER_C1_R2", "C1_BIAS")):
        group = [row for row in cluster_rows
                 if row["config_id"].startswith("Guarded_Backbone54_k2") and row["cluster"] == cluster]
        context[r2_key] = fmt(mean(group, "r2"), 4)
        context[bias_key] = fmt(mean(group, "bias"), 4)
    context["GLOBAL_POOLED_BIAS"] = fmt(mean(global_temporal, "bias"), 4)
    delta = {row["candidate_id"]: row for row in read_csv("delta_grid")}
    context["SAT_BASE_R2"] = fmt(delta["delta_c0_0_c1_0"]["pooled_r2"], 4)
    context["SAT_C0_ADD10_R2"] = fmt(delta["delta_c0_10_c1_0"]["pooled_r2"], 4)
    context["SAT_C1_ADD10_R2"] = fmt(delta["delta_c0_0_c1_10"]["pooled_r2"], 4)
    profile_rows = read_csv("profile")
    profile_anchor = {row["feature"]: row for row in profile_rows}
    profile = [row for row in profile_rows if row["feature"] != "soil_moisture_5cm"]
    profile.sort(key=lambda row: number(row["separation_index"]), reverse=True)
    context["PROFILE_TABLE"] = md_table(["Feature", "Separation index", "Median c0", "Median c1"], [
        [row["feature"], fmt(row["separation_index"], 3), fmt(row["median_cluster0"], 3), fmt(row["median_cluster1"], 3)] for row in profile[:10]
    ])
    context["PROFILE_ELEV_SEPARATION"] = fmt(profile_anchor["elev"]["separation_index"], 3)
    context["PROFILE_ELEV_C0"] = fmt(profile_anchor["elev"]["median_cluster0"], 0)
    context["PROFILE_ELEV_C1"] = fmt(profile_anchor["elev"]["median_cluster1"], 0)

    family_names = {
        "Clustering_V0_Full_k2": "Earlier V0 multi-regime model",
        "Clustering_Backbone54_k2": "shared-feature cluster-routed multi-regime model without station consistency guarantee",
        "Guarded_Backbone54_k2": "station-majority shared-feature cluster-routed multi-regime model",
        "Global_Single_54": "Existing single-regime global model",
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
    # 1.3: show the station-majority shared-feature cluster-routed multi-regime
    # model plus the global reference only. The same model without the station
    # consistency guarantee has deployable rows that are bit-identical on this
    # ECE set (all 150 rows take the same fallback branch), so repeating them
    # doubles the table without adding information; the text states the
    # identity instead. The soft-blend fallback matches the hard fallback at
    # reported precision.
    for row in sorted(
        (item for item in ece if item["family"] in ("Guarded_Backbone54_k2", "Global_Single_54") and item["policy"] != "auto_soft"),
        key=lambda item: (list(family_names).index(item["family"]), policy_order[item["policy"]]),
    ):
        policy = row["policy"]
        policy_label = policy_names[policy] if policy in policy_names else f"Single-specialist reference (index {fixed_indices[(row['family'], policy)]})"
        ece_rows.append([family_names[row["family"]], policy_label, "yes" if row["deployable"] == "True" else "no",
                         f"{fmt(row['rmse_mean'], 4)} ± {fmt(row['rmse_std'], 4)}", fmt(row["mae_mean"], 4),
                         signed(row["bias_mean"], 4), fmt(row["ubrmse_mean"], 4)])
    context["ECE_TABLE"] = md_table(["Model", "Assignment or reference", "Usable from observed inputs?", "RMSE ± seed SD", "MAE", "Bias", "ubRMSE"], ece_rows)
    auto = one(ece, family="Guarded_Backbone54_k2", policy="auto_hard")
    routed = one(ece, family="Guarded_Backbone54_k2", policy="as_routed")
    global_ece = one(ece, family="Global_Single_54", policy="direct")
    context["ECE_AUTO_RMSE"] = fmt(auto["rmse_mean"], 4)
    context["ECE_ROUTED_RMSE"] = fmt(routed["rmse_mean"], 4)
    context["ECE_GLOBAL_RMSE"] = fmt(global_ece["rmse_mean"], 4)
    context["ECE_AUTO_GLOBAL_GAP"] = signed(number(auto["rmse_mean"]) - number(global_ece["rmse_mean"]), 4)
    context["ECE_AUTO_ROUTED_GAP"] = signed(number(auto["rmse_mean"]) - number(routed["rmse_mean"]), 4)
    context["ECE_AUTO_ROUTED_REDUCTION"] = fmt(number(routed["rmse_mean"]) - number(auto["rmse_mean"]), 4)
    index_reference_policy = {
        index: policy
        for policy in ("c0_only", "c1_only")
        for index in [fixed_indices[("Guarded_Backbone54_k2", policy)]]
    }
    context["ECE_INDEX0_RMSE"] = fmt(one(ece, family="Guarded_Backbone54_k2", policy=index_reference_policy[0])["rmse_mean"], 4)
    context["ECE_INDEX1_RMSE"] = fmt(one(ece, family="Guarded_Backbone54_k2", policy=index_reference_policy[1])["rmse_mean"], 4)

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
    context["ECE_CROSSWALK"] = md_table(["Model", "Precipitation class 0 → specialist", "Class 1 → specialist", "Fixed comparator specialist", "Other specialist", "Comparator usable from inputs?", "Index with lower WA SMAP mean?"], cross_rows)
    # The v1.1 crosswalk records the same fit-frame means for every family.
    cross_guard = one(crosswalk, family="Guarded_Backbone54_k2", policy_id="auto_hard")
    context["WA_SMAP_C0"] = fmt(cross_guard["wa_canonical_smap_mean_local_c0"], 4)
    context["WA_SMAP_C1"] = fmt(cross_guard["wa_canonical_smap_mean_local_c1"], 4)
    context["WA_TARGET_C0"] = fmt(cross_guard["wa_target_mean_local_c0"], 4)
    context["WA_TARGET_C1"] = fmt(cross_guard["wa_target_mean_local_c1"], 4)
    candidate_rows = [row for row in calibration if row["family"] == "Guarded_Backbone54_k2" and row["setting"] == "smap_masked_val" and row["policy"].startswith("aux_hard_candidate_")]
    context["ECE_CALIBRATION_TABLE"] = md_table(["Precipitation class 0 →", "Precipitation class 1 →", "WA validation RMSE", "Chosen for ECE"], [
        [f"index {int(number(row['gapi_class_0_local_expert']))}", f"index {int(number(row['gapi_class_1_local_expert']))}", fmt(row["rmse"], 4), "yes" if row["selected_for_ece"] == "True" else "no"]
        for row in sorted(candidate_rows, key=lambda item: number(item["rmse"]))
    ])

    sites = {row["station_id"]: row for row in read_csv("ece_sites")}
    context["ECE_ELEV_MIN"] = fmt(min(number(row["J_elev_m"]) for row in sites.values()), 0)
    context["ECE_ELEV_MAX"] = fmt(max(number(row["J_elev_m"]) for row in sites.values()), 0)
    context["ECE_PRECIP_MIN"] = fmt(min(number(row["J_bio_bio12"]) for row in sites.values()), 0)
    context["ECE_PRECIP_MAX"] = fmt(max(number(row["J_bio_bio12"]) for row in sites.values()), 0)
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
    context["ECE_STATION_TABLE"] = md_table(["ECE station", "Elev. m", "Annual precip. descriptor mm", "Usual-assignment RMSE", "Precipitation-assignment RMSE", "Single-regime global RMSE", "Precipitation-assignment bias", "Weight on comparator specialist"], station_rows)
    routed_station_rmse = [number(one(ece_station, family="Guarded_Backbone54_k2", policy="as_routed", station=station)["rmse_mean"]) for station in sites]
    context["ECE_ROUTED_SITE_MIN"] = fmt(min(routed_station_rmse), 4)
    context["ECE_ROUTED_SITE_MAX"] = fmt(max(routed_station_rmse), 4)

    legacy = one(read_csv("ece_legacy"), Category="Clustering vs Global", **{"Comparison (A vs B)": "Clustering (V0) vs Global-54"})
    context["ECE_LEGACY_DIFF"] = signed(legacy["Station Mean ΔRMSE (A−B)"], 4)
    context["ECE_LEGACY_WINS"] = legacy["Station Wins (A < B)"]
    context["ECE_LEGACY_SIGN_P"] = fmt(legacy["Binomial Sign Test p"], 3)

    oos = read_csv("oos")
    oos_labels = {
        "Clustering (54 backbone)": "shared-feature cluster-routed multi-regime model without station consistency guarantee (54 features)",
        "Seasonal Binary (Summer/Winter)": "Seasonal grouping",
        "Univariate G_API split": "Precipitation-index grouping",
        "Global Single Model (54 feats)": "Existing single-regime global model (54 features)",
        "Clustering (Dynamic features)": "Three-feature K-means grouping",
        "Trained Gating Classifier": "Target-threshold grouping",
    }
    oos_rows = []
    for row in oos:
        architecture = row["Model Architecture"]
        # Only the 54-feature configurations are shown; the older 50-feature
        # variants are outside this report's scope.
        if architecture in {"Clustering (50 V0 features)", "Baseline Model (50 V0 feats)"}:
            continue
        label = oos_labels[architecture]
        oos_rows.append([label, fmt(row["Station Mean R²"], 3), fmt(row["Station Mean RMSE"], 3), row["Pooled R² (mean ± std)"]])
    oos_headers = ["Model or grouping", "10-station mean R²", "10-station mean RMSE", "Pooled R²"]
    context["OOS_TABLE"] = md_table(oos_headers, oos_rows)
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
