"""Extract the poster's values from the report's validated, immutable evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REPORT = REPO / "writeup2/report-assemble-1.5"
PRIMARY = "Guarded_Backbone54_k2"
BLUE = "#176E9B"
OCHRE = "#B48024"
RED = "#C00000"
GRAY = "#919AA3"
DARK = "#202B35"
REGIME_NAMES = {0: "Wet / Mountains", 1: "Drier / Lowlands"}
STATION_NAMES = {
    "BeaverPass_WA_990": "Beaver Pass",
    "CayusePass_WA": "Cayuse Pass",
    "Darrington": "Darrington",
    "Paradise_WA": "Paradise",
    "Quinault": "Quinault",
    "SourdoughGulch_WA_985": "Sourdough Gulch",
    "Spokane": "Spokane",
}
BOUNDARY = REPO / "notebooks/experiment/derived_8.4-regime-interpretation-1.2-oos/wa_admin1_boundary.geojson"


def report_module():
    spec = importlib.util.spec_from_file_location("poster_report_data", REPORT / "scripts/report_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evidence() -> dict:
    rd = report_module()
    context = rd.build_context()  # Includes the existing source and paired-coverage audit.
    frames = {key: pd.DataFrame(rd.read_csv(key)) for key in (
        "main_seed", "global_temporal_seed", "loso_seed", "global_loso_seed",
        "composition", "profile", "ece_policy", "ece_sites",
    )}
    temporal = frames["main_seed"].query(
        "strategy_name == @PRIMARY and evidence_role == 'paired_in_harness'"
    ).copy()
    global_temporal = frames["global_temporal_seed"].copy()
    loso = frames["loso_seed"].query("strategy_name == @PRIMARY").copy()
    global_loso = frames["global_loso_seed"].copy()
    for frame in (temporal, global_temporal, loso, global_loso):
        frame["r2"] = pd.to_numeric(frame["r2"])
        frame["rmse"] = pd.to_numeric(frame["rmse"])
    assert len(temporal) == len(global_temporal) == 30
    assert len(loso) == len(global_loso) == 35
    assert set(temporal.seed) == set(global_temporal.seed)
    assert set(zip(loso.seed, loso.station)) == set(zip(global_loso.seed, global_loso.station))
    assert loso.groupby("station").size().eq(5).all()
    paired_t = temporal.merge(global_temporal, on="seed", suffixes=("_two", "_global"), validate="one_to_one")
    paired_l = loso.merge(global_loso, on=["seed", "station"], suffixes=("_two", "_global"), validate="one_to_one")
    paired_l["difference"] = paired_l.r2_two - paired_l.r2_global
    station = paired_l.groupby("station").agg(
        two_r2=("r2_two", "mean"), global_r2=("r2_global", "mean"),
        difference=("difference", "mean"), seed_sd=("difference", "std"),
    ).sort_values("difference", ascending=False)
    station["name"] = station.index.map(STATION_NAMES)
    composition = frames["composition"].copy()
    for column in ["dominant_cluster", "longitude", "latitude", "n", "purity"]:
        composition[column] = pd.to_numeric(composition[column])
    composition["name"] = composition.station_id.map(STATION_NAMES)
    composition["regime_name"] = composition.dominant_cluster.map(REGIME_NAMES)
    assert set(composition.station_id) == set(STATION_NAMES)
    assert composition.purity.eq(1).all()
    assert set(composition.query("dominant_cluster == 1").station_id) == {
        "Spokane", "SourdoughGulch_WA_985",
    }
    ece = frames["ece_policy"]

    def ece_row(family, policy):
        row = ece.query("family == @family and policy == @policy")
        assert len(row) == 1
        return float(row.iloc[0].rmse_mean)

    profile = frames["profile"].set_index("feature")
    # Report Table 6 provides group medians; the field names are source-owned.
    print("Profile fields:", ", ".join(profile.columns))
    metrics = {
        "temporal_global": float(global_temporal.r2.mean()),
        "temporal_two": float(temporal.r2.mean()),
        "temporal_difference": float((paired_t.r2_two - paired_t.r2_global).mean()),
        "temporal_wins": int((paired_t.r2_two > paired_t.r2_global).sum()),
        "loso_global": float(station.global_r2.mean()),
        "loso_two": float(station.two_r2.mean()),
        "loso_difference": float(station.difference.mean()),
        "loso_gains": int((station.difference > 0).sum()),
        "loso_losses": int((station.difference < 0).sum()),
        "ece_standard": ece_row(PRIMARY, "as_routed"),
        "ece_fallback": ece_row(PRIMARY, "auto_hard"),
        "ece_global": ece_row("Global_Single_54", "direct"),
        "features": int(context["FEATURE_COUNT"]),
        "stations": len(composition),
        "ece_rows": int(context["ECE_ROWS"]),
        "ece_stations": int(context["ECE_STATIONS"]),
        "ece_start": context["ECE_FIRST_DATE"],
        "ece_end": context["ECE_LAST_DATE"],
    }
    # Values displayed in the report are independently checked before export.
    expected = {
        "temporal_global": "0.7798", "temporal_two": "0.8118",
        "temporal_difference": "0.0320", "loso_global": "0.5797",
        "loso_two": "0.6379", "loso_difference": "0.0581",
        "ece_standard": "0.1674", "ece_fallback": "0.0578", "ece_global": "0.0586",
    }
    for key, value in expected.items():
        assert f"{metrics[key]:.4f}" == value, (key, metrics[key], value)
    manifest = rd.source_manifest()
    manifest["poster"] = "poster-summer-1.1"
    manifest["display_regime_names"] = REGIME_NAMES
    manifest["regime_naming_note"] = (
        "User-selected descriptive names; fitted five-versus-two station memberships unchanged. "
        "Names do not define elevation thresholds or geographic boundaries."
    )
    previous = REPO / "writeup2/poster-summer-1.0"
    previous_metrics = json.loads((previous / "artifacts/poster_metrics.json").read_text())
    assert metrics == previous_metrics, "Revision changed the underlying results"
    previous_stations = pd.read_csv(previous / "artifacts/map_memberships.csv").set_index("station_id")
    for row in composition.itertuples():
        prior = previous_stations.loc[row.station_id]
        for field in ["dominant_cluster", "latitude", "longitude", "n", "purity"]:
            assert getattr(row, field) == prior[field], (row.station_id, field)
    manifest["preserved_previous_version"] = {
        path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(previous.rglob("*")) if path.is_file() and "__pycache__" not in path.parts
    }
    manifest["additional_inputs"] = {
        "washington_boundary": {
            "path": BOUNDARY.relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(BOUNDARY.read_bytes()).hexdigest(),
        },
        "report_markdown": {
            "path": (REPORT / "report.md").relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256((REPORT / "report.md").read_bytes()).hexdigest(),
        },
    }
    return {"metrics": metrics, "stations": composition, "station_effects": station,
            "profile": profile, "manifest": manifest, "report_context": context}


def write_evidence(data: dict) -> str:
    output = ROOT / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    metrics = data["metrics"]
    (output / "poster_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "source_manifest.json").write_text(json.dumps(data["manifest"], indent=2) + "\n")
    data["stations"].to_csv(output / "map_memberships.csv", index=False)
    data["station_effects"].to_csv(output / "station_effects.csv")
    rows = [
        ("Later-year R²", metrics["temporal_global"], metrics["temporal_two"]),
        ("Held-out-station mean R²", metrics["loso_global"], metrics["loso_two"]),
    ]
    lines = ["| Evaluation | Global | Two-regime |", "| --- | ---: | ---: |"]
    lines.extend(f"| {label} | {global_value:.4f} | {two:.4f} |" for label, global_value, two in rows)
    lines += ["", "| ECE assignment | Pooled RMSE (m³/m³) |", "| --- | ---: |",
              f"| Standard routing | {metrics['ece_standard']:.4f} |",
              f"| Precipitation fallback | {metrics['ece_fallback']:.4f} |",
              f"| Global predictor | {metrics['ece_global']:.4f} |"]
    table = "\n".join(lines)
    (output / "readme_results.md").write_text(table + "\n")
    print(table)
    print("\nAll displayed metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nMap: seven fitted memberships and source coordinates")
    print(data["stations"][["name", "dominant_cluster", "regime_name", "latitude", "longitude", "n"]].to_string(index=False))
    print("\nHeld-out-station differences: two-regime minus matched global")
    print(data["station_effects"][["name", "two_r2", "global_r2", "difference"]].to_string(index=False))
    print("\nProvenance for every metric family:")
    for key in ["main_seed", "global_temporal_seed", "loso_seed", "global_loso_seed",
                "composition", "profile", "ece_policy", "ece_audit", "ece_calibration"]:
        print(f"{key}: {data['manifest']['inputs'][key]['path']}")
    return table
