# Jakob Balkovec
# Report

import json
from pathlib import Path

import pandas as pd

from Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Ranking.score import compute_score


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def make_report(
    run_dir,
    config,
    selected_features,
    metric_rows=None,
    weights=None,
    top_n_features=20,
    model_name=None,
):
    log = get_logger("ranking.report")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = metric_rows or []
    weights = weights or ((config or {}).get("scoring", {}).get("score_weights", {}) or {})

    k = len(selected_features) if selected_features is not None else None

    score_out = None
    if metric_rows:
        score_out = compute_score(metric_rows, weights=weights, k=k, prefer_split="val", metric="r2")

    lines = []
    lines.append("# Feature Selection Run Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Model: {model_name or 'N/A'}")
    lines.append(f"- Selected features: {k}")
    if score_out is not None:
        lines.append(f"- Score: {_safe_float(score_out.get('score'))}")
        lines.append(f"- mean_r2: {_safe_float(score_out.get('mean_r2'))}")
        lines.append(f"- std_r2: {_safe_float(score_out.get('std_r2'))}")
        lines.append(f"- train-val gap: {_safe_float(score_out.get('gap'))}")
    lines.append("")

    lines.append("## Top Selected Features")
    if not selected_features:
        lines.append("- (none)")
    else:
        for f in selected_features[: int(top_n_features)]:
            lines.append(f"- {f}")
    lines.append("")

    if metric_rows:
        df = pd.DataFrame(metric_rows)
        lines.append("## Metrics")
        lines.append("")
        lines.append(df.to_markdown(index=False))
        lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("make_report: wrote %s", report_path)

    if score_out is not None:
        score_path = run_dir / "score.json"
        score_path.write_text(json.dumps(score_out, indent=2), encoding="utf-8")
        log.info("make_report: wrote %s", score_path)

    if selected_features is not None:
        feat_path = run_dir / "selected_features.json"
        feat_path.write_text(json.dumps(list(selected_features), indent=2), encoding="utf-8")
        log.info("make_report: wrote %s", feat_path)

    if config is not None:
        cfg_path = run_dir / "config_snapshot.json"
        try:
            cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            log.info("make_report: wrote %s", cfg_path)
        except Exception:
            log.warning("make_report: could not json-dump config (non-serializable values?)")

    return {
        "report_path": str(report_path),
        "score": score_out,
    }
