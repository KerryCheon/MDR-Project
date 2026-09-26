"""Extract the Paper 2 report from immutable, versioned evidence artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ASSEMBLY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ASSEMBLY_ROOT.parents[1]
EVIDENCE_10 = "notebooks/experiment/paper2-final-evidence-1.0"
EVIDENCE_11 = "notebooks/experiment/paper2-final-evidence-1.1"
FORMAL_10 = "notebooks/experiment/derived_8.4-formal-eval-1.0"

SOURCES = {
    "outline": "writeup2/outline-v7.md",
    "paper1": "paper/Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather- Driven Machine Learning.pdf",
    "split": "data/splits/derived_8.4/split_meta.json",
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
    "formal_bootstrap": f"{FORMAL_10}/temporal_bootstrap.csv",
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

EXTRA_PROVENANCE = {
    "main_notebook": f"{EVIDENCE_10}/notebooks/paper2_main_figures.ipynb",
    "ece_notebook": f"{EVIDENCE_11}/notebooks/paper2_ece_supplement.ipynb",
    "main_config": f"{EVIDENCE_10}/config.yaml",
    "ece_config": f"{EVIDENCE_11}/ece_guarded/config.yaml",
    "ece_runner": f"{EVIDENCE_11}/ece_guarded/run_guarded.py",
}


def source_path(key: str) -> Path:
    return REPO_ROOT / SOURCES[key]


def source_link(key: str, label: str | None = None) -> str:
    return f"[{label or key}](../../{SOURCES[key].replace(' ', '%20')})"


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


def check_close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label}: {actual} != {expected}")


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
    if (selected["gapi_class_0_local_expert"], selected["gapi_class_1_local_expert"]) != ("0.0", "1.0"):
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
        "assembly": "report-assemble-1.0",
        "source_precedence": {
            "main_temporal_loso": EVIDENCE_10,
            "ece_all_results": EVIDENCE_11,
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
    context["FEATURE_LIST"] = ", ".join(f"`{item}`" for item in features)
    ece_split_rows = read_csv("ece_split")
    context["ECE_ROWS"] = str(len(ece_split_rows))
    context["ECE_STATIONS"] = str(len({row["station_id"] for row in ece_split_rows}))
    context["ECE_PER_STATION"] = str(len(ece_split_rows) // len({row["station_id"] for row in ece_split_rows}))
    context["ECE_FIRST_DATE"] = min(row["date"] for row in ece_split_rows)
    context["ECE_LAST_DATE"] = max(row["date"] for row in ece_split_rows)

    static = {row["station_id"]: row for row in read_csv("wa_static")}
    composition = {row["station_id"]: row for row in read_csv("composition")}
    station_rows = []
    for name in split["stations"]:
        row = static[name]
        cluster = composition[name]
        station_rows.append([name, fmt(row["latitude"], 2), fmt(row["longitude"], 2), fmt(row["J_elev_m"], 0), cluster["dominant_cluster"], cluster["n"]])
    context["WA_STATION_TABLE"] = md_table(["Station", "Lat.", "Lon.", "Elevation m", "Trainval local cluster", "Trainval rows"], station_rows)

    name_map = {
        "Guarded_Backbone54_k2": "Guarded backbone (primary)",
        "Clustering_Backbone54_k2": "Unguarded backbone (paired ablation)",
        "Clustering_V0_Full_k2": "Legacy V0 (sensitivity)",
        "Global_Single": "Global single model (sensitivity)",
        "Clustering_Dynamic_k2": "Dynamic covariate (sensitivity)",
        "Seasonal_Binary_k2": "Seasonal (sensitivity)",
        "Univariate_G_API_k2": "G_API (sensitivity)",
        "Trained_Gating_k2": "Target-derived gate (sensitivity)",
    }
    temporal_rows = []
    for row in sorted(main, key=lambda item: number(item["mean_rmse"])):
        group = [item for item in main_seed if item["strategy_name"] == row["strategy_name"] and item["evidence_role"] == row["evidence_role"]]
        temporal_rows.append([
            name_map[row["strategy_name"]], row["evidence_role"].replace("_", " "), row["n_seeds"],
            fmt(row["mean_r2"]), fmt(row["seed_sd_r2"]),
            f"[{fmt(row['seed_ci95_low'])}, {fmt(row['seed_ci95_high'])}]",
            fmt(row["mean_rmse"]), fmt(mean(group, "mae")), fmt(mean(group, "bias")),
        ])
    context["TEMPORAL_TABLE"] = md_table(["Router / expert", "Comparison role", "Seeds", "R²", "Seed SD", "95% seed CI", "RMSE", "MAE", "Bias"], temporal_rows)
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
        loso_rows.append([name_map[strategy], "paired in harness", "5 × 7", fmt(value), fmt(mean(by_strategy[strategy], "rmse"))])
    for config_id, label in (
        ("Clustering_V0_Full_k2_c0_0_c1_0", "Legacy V0 (sensitivity)"),
        ("Global_Single_54", "Global 54 (sensitivity)"),
        ("Baseline_V0_50", "Global 50 (sensitivity)"),
    ):
        row = one(formal_loso, config_id=config_id)
        loso_values[config_id] = number(row["loso_mean_r2"])
        loso_rows.append([label, "sensitivity anchored", "5 × 7", fmt(row["loso_mean_r2"]), fmt(row["loso_mean_rmse"])])
    context["LOSO_TABLE"] = md_table(["Configuration", "Comparison role", "Seeds × held-out sites", "Station-mean R²", "Station-mean RMSE"], loso_rows)
    context["GUARD_LOSO"] = fmt(loso_values["Guarded_Backbone54_k2"], 6)
    context["BACKBONE_LOSO"] = fmt(loso_values["Clustering_Backbone54_k2"], 6)
    context["GLOBAL_LOSO"] = fmt(loso_values["Global_Single_54"], 6)
    context["GUARD_LOSO_DIFF"] = signed(loso_values["Guarded_Backbone54_k2"] - loso_values["Clustering_Backbone54_k2"], 6)
    context["GUARD_GLOBAL_LOSO_GAP"] = signed(loso_values["Guarded_Backbone54_k2"] - loso_values["Global_Single_54"], 4)
    fold_agreement = {row["held_out"]: row for row in read_csv("fold_agreement")}
    t8_rows = []
    for row in sorted(t8, key=lambda item: number(item["mean_r2_difference"]), reverse=True):
        station = row["station"]
        fold = fold_agreement[station]
        t8_rows.append([station, fmt(row["guarded_mean_r2"]), fmt(row["backbone_mean_r2"]), signed(row["mean_r2_difference"]),
                        f"{row['winning_seeds']}/{row['tied_seeds']}", fmt(fold["ARI_tr_V0_Full_vs_Backbone"], 3),
                        fmt(fold["ARI_tr_Backbone_vs_GuardedV-A"], 3)])
    context["T8_TABLE"] = md_table(["Held-out station", "Guarded R²", "Backbone R²", "Guarded − Backbone", "Wins / ties (5 seeds)", "Train ARI V0–Backbone", "Train ARI Backbone–Guarded"], t8_rows)
    context["FOLD_GAINS"] = str(sum(number(row["mean_r2_difference"]) > 0 for row in t8))
    context["FOLD_TIES"] = str(sum(number(row["mean_r2_difference"]) == 0 for row in t8))

    k_rows = read_csv("k_sweep")
    context["K_TABLE"] = md_table(["K", "Silhouette", "Calinski–Harabasz", "Davies–Bouldin", "Trainval station purity", "Smallest cluster share"], [
        [row["K"], fmt(row["silhouette"], 3), fmt(row["calinski_harabasz"], 1), fmt(row["davies_bouldin"], 3), fmt(row["mean_station_purity_trainval"], 3), fmt(row["min_cluster_share"], 3)] for row in k_rows
    ])
    profile = [row for row in read_csv("profile") if row["feature"] != "soil_moisture_5cm"]
    profile.sort(key=lambda row: number(row["separation_index"]), reverse=True)
    context["PROFILE_TABLE"] = md_table(["Feature", "Separation index", "Median c0", "Median c1"], [
        [row["feature"], fmt(row["separation_index"], 3), fmt(row["median_cluster0"], 3), fmt(row["median_cluster1"], 3)] for row in profile[:10]
    ])

    pair_rows = read_csv("formal_pair")
    bootstrap_rows = read_csv("formal_bootstrap")
    no_delta_pair = one(pair_rows, A="Clustering_V0_Full_k2_c0_0_c1_0", B="Global_Single_54", metric="R2")
    test_delta_boot = one(bootstrap_rows, A="Clustering_V0_Full_k2_c0_0_c1_10", B="Global_Single_54", metric="R2")
    context["V0_GLOBAL_PAIRED"] = f"{signed(no_delta_pair['mean_diff'])} [{fmt(no_delta_pair['ci_low'])}, {fmt(no_delta_pair['ci_high'])}], p={number(no_delta_pair['t_p']):.2g}"
    context["DELTA_BOOTSTRAP"] = f"{signed(test_delta_boot['diff_mean'])} [{fmt(test_delta_boot['diff_ci_low'])}, {fmt(test_delta_boot['diff_ci_high'])}], p={fmt(test_delta_boot['bootstrap_p'], 4)}"
    delta = one(read_csv("formal_delta"), strategy="Clustering_V0_Full_k2")
    context["DELTA_TABLE"] = md_table(["Feature addition source", "Configuration", "Temporal R²", "LOSO R²"], [
        ["None", delta["none_config"], delta["none_temporal_r2"], delta["none_loso_r2"]],
        ["Validation selected", delta["val_config"], delta["val_temporal_r2"], delta["val_loso_r2"]],
        ["Test selected (sensitivity only)", delta["test_config"], delta["test_temporal_r2"], delta["test_loso_r2"]],
    ])

    family_names = {
        "Clustering_V0_Full_k2": "V0",
        "Clustering_Backbone54_k2": "Backbone54",
        "Guarded_Backbone54_k2": "Guarded",
        "Global_Single_54": "Global54",
    }
    policy_order = {"as_routed": 0, "auto_hard": 1, "auto_soft": 2, "c0_only": 3, "c1_only": 4, "direct": 5}
    ece_rows = []
    for row in sorted(ece, key=lambda item: (list(family_names).index(item["family"]), policy_order[item["policy"]])):
        ece_rows.append([family_names[row["family"]], row["policy"], "yes" if row["deployable"] == "True" else "no",
                         f"{fmt(row['rmse_mean'], 6)} ± {fmt(row['rmse_std'], 6)}", fmt(row["mae_mean"], 6),
                         signed(row["bias_mean"], 6), fmt(row["ubrmse_mean"], 6)])
    context["ECE_TABLE"] = md_table(["Family", "Policy", "Deployable", "RMSE ± seed SD", "MAE", "Bias", "ubRMSE"], ece_rows)
    auto = one(ece, family="Guarded_Backbone54_k2", policy="auto_hard")
    routed = one(ece, family="Guarded_Backbone54_k2", policy="as_routed")
    global_ece = one(ece, family="Global_Single_54", policy="direct")
    context["ECE_AUTO_RMSE"] = fmt(auto["rmse_mean"], 6)
    context["ECE_ROUTED_RMSE"] = fmt(routed["rmse_mean"], 6)
    context["ECE_GLOBAL_RMSE"] = fmt(global_ece["rmse_mean"], 6)
    context["ECE_AUTO_GLOBAL_GAP"] = signed(number(auto["rmse_mean"]) - number(global_ece["rmse_mean"]), 6)
    context["ECE_AUTO_ROUTED_GAP"] = signed(number(auto["rmse_mean"]) - number(routed["rmse_mean"]), 6)
    context["ECE_AUTO_ROUTED_REDUCTION"] = fmt(number(routed["rmse_mean"]) - number(auto["rmse_mean"]), 6)
    context["ECE_C0_DRY_RMSE"] = fmt(one(ece, family="Guarded_Backbone54_k2", policy="c0_only")["rmse_mean"], 6)
    context["ECE_C1_COMPLEMENT_RMSE"] = fmt(one(ece, family="Guarded_Backbone54_k2", policy="c1_only")["rmse_mean"], 6)

    crosswalk = read_csv("ece_crosswalk")
    cross_rows = []
    for family in ("Clustering_V0_Full_k2", "Clustering_Backbone54_k2", "Guarded_Backbone54_k2"):
        row = one(crosswalk, family=family, policy_id="auto_hard")
        oracle = one(crosswalk, family=family, policy_id="c0_only")
        cross_rows.append([family_names[family], f"c{int(number(row['gapi_class_0_local_expert']))}",
                           f"c{int(number(row['gapi_class_1_local_expert']))}",
                           f"c{int(number(row['dry_assigned_local_expert_index']))}",
                           f"c{int(number(row['complementary_local_expert_index']))}",
                           "no" if oracle["deployable"] == "False" else "yes",
                           "yes" if row["dry_assignment_matches_canonical_feature_minimum"] == "True" else "no"])
    context["ECE_CROSSWALK"] = md_table(["Family", "G_API class 0 →", "G_API class 1 →", "Dry-assigned oracle index", "Complement index", "Oracle deployable", "Oracle index matches WA lower SMAP mean"], cross_rows)
    # The v1.1 crosswalk records the same fit-frame means for every family.
    cross_guard = one(crosswalk, family="Guarded_Backbone54_k2", policy_id="auto_hard")
    context["WA_SMAP_C0"] = fmt(cross_guard["wa_canonical_smap_mean_local_c0"], 6)
    context["WA_SMAP_C1"] = fmt(cross_guard["wa_canonical_smap_mean_local_c1"], 6)
    context["WA_TARGET_C0"] = fmt(cross_guard["wa_target_mean_local_c0"], 6)
    context["WA_TARGET_C1"] = fmt(cross_guard["wa_target_mean_local_c1"], 6)
    candidate_rows = [row for row in calibration if row["family"] == "Guarded_Backbone54_k2" and row["setting"] == "smap_masked_val" and row["policy"].startswith("aux_hard_candidate_")]
    context["ECE_CALIBRATION_TABLE"] = md_table(["G_API class 0 →", "G_API class 1 →", "Masked WA validation RMSE", "Selected for ECE"], [
        [f"c{int(number(row['gapi_class_0_local_expert']))}", f"c{int(number(row['gapi_class_1_local_expert']))}", fmt(row["rmse"], 6), "yes" if row["selected_for_ece"] == "True" else "no"]
        for row in sorted(candidate_rows, key=lambda item: number(item["rmse"]))
    ])

    sites = {row["station_id"]: row for row in read_csv("ece_sites")}
    ece_station = read_csv("ece_station")
    share_rows = read_csv("ece_shares")
    station_rows = []
    for station in sorted(sites):
        routed_station = one(ece_station, family="Guarded_Backbone54_k2", policy="as_routed", station=station)
        auto_station = one(ece_station, family="Guarded_Backbone54_k2", policy="auto_hard", station=station)
        global_station = one(ece_station, family="Global_Single_54", policy="direct", station=station)
        share = one(share_rows, family="Guarded_Backbone54_k2", policy="auto_hard", station_id=station)
        station_rows.append([station, fmt(sites[station]["J_elev_m"], 0), fmt(sites[station]["J_bio_bio12"], 0),
                             fmt(routed_station["rmse_mean"], 4), fmt(auto_station["rmse_mean"], 4),
                             fmt(global_station["rmse_mean"], 4), signed(auto_station["bias_mean"], 4),
                             fmt(share["dry_assigned_weight_mean"], 3)])
    context["ECE_STATION_TABLE"] = md_table(["ECE station", "Elev. m", "Annual precip. descriptor mm", "Guarded static RMSE", "Guarded auto RMSE", "Global RMSE", "Auto bias", "Auto dry-assigned share"], station_rows)

    legacy = one(read_csv("ece_legacy"), Category="Clustering vs Global", **{"Comparison (A vs B)": "Clustering (V0) vs Global-54"})
    context["ECE_LEGACY_DIFF"] = signed(legacy["Station Mean ΔRMSE (A−B)"], 6)
    context["ECE_LEGACY_WINS"] = legacy["Station Wins (A < B)"]
    context["ECE_LEGACY_SIGN_P"] = fmt(legacy["Binomial Sign Test p"], 3)

    oos = read_csv("oos")
    oos_rows = []
    for row in oos:
        oos_rows.append([row["Model Architecture"], fmt(row["Station Mean R²"], 3), fmt(row["Station Mean RMSE"], 3), row["Pooled R² (mean ± std)"]])
    context["OOS_TABLE"] = md_table(["Pre-guard configuration", "10-station mean R²", "10-station mean RMSE", "Pooled R²"], oos_rows)
    hourly = read_csv("hourly")
    hourly_range = statistics.median(number(row["hourly_range"]) for row in hourly)
    daily_steps = [abs(number(row["d_target_vs_prev_day"])) for row in hourly if row["d_target_vs_prev_day"] and math.isfinite(number(row["d_target_vs_prev_day"]))]
    context["HOURLY_RANGE"] = fmt(hourly_range, 4)
    context["DAILY_STEP"] = fmt(statistics.median(daily_steps), 4)

    for key in SOURCES:
        context[f"SOURCE_{key.upper()}"] = source_link(key)
    for key, relative in EXTRA_PROVENANCE.items():
        context[f"SOURCE_{key.upper()}"] = f"[{key}](../../{relative.replace(' ', '%20')})"
    return context
