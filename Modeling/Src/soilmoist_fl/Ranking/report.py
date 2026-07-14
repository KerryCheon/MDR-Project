# Jakob Balkovec
# Report

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Ranking.score import compute_score


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _fmt_float(x, digits=4):
    val = _safe_float(x)
    if val is None or not pd.notna(val):
        return "N/A"
    return f"{val:.{digits}f}"


def _cfg_get(cfg, *keys, default=None):
    cur = cfg or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur.get(k)
    return cur if cur is not None else default



def make_report(
    run_dir,
    config,
    selected_features,
    metric_rows=None,
    weights=None,
    top_n_features=40,
    model_name=None,
):
    log = get_logger("ranking.report")

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = metric_rows or []
    weights = weights or _cfg_get(config, "scoring", "score_weights", default={})
    top_n_features = int(top_n_features or _cfg_get(config, "report", "top_n_features", default=20))
    model_name = model_name or _cfg_get(config, "model", "name", default="feature_selection")

    k = len(selected_features) if selected_features is not None else None

    score_out = None
    if metric_rows:
        score_out = compute_score(metric_rows, weights=weights, k=k, prefer_split="val", metric="r2")

    data_cfg = (config or {}).get("data", {}) if isinstance(config, dict) else {}
    sel_cfg = (config or {}).get("selection", {}) if isinstance(config, dict) else {}
    run_id = run_dir.name if run_dir is not None else "ad-hoc"
    timestamp = datetime.now().isoformat(timespec="seconds")

    lines = []
    lines.append("# Feature Selection Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- Run ID: {run_id}")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Model: {model_name or 'N/A'}")
    lines.append(f"- Target: {data_cfg.get('target', 'N/A')}")
    lines.append(f"- Time column: {data_cfg.get('time_col', 'N/A')}")
    lines.append(f"- ID columns: {', '.join(data_cfg.get('id_cols', []) or []) or 'N/A'}")
    lines.append("")

    lines.append("## Selection Summary")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Selected features | {k if k is not None else 'N/A'} |")
    lines.append(f"| Stages | {', '.join([str(s.get('kind')) for s in sel_cfg.get('stages', []) if isinstance(s, dict)]) or 'N/A'} |")
    lines.append(f"| Top-k target | {sel_cfg.get('top_k', 'N/A')} |")
    if score_out is not None:
        lines.append(f"| Score | {_fmt_float(score_out.get('score'))} |")
        lines.append(f"| Mean R2 | {_fmt_float(score_out.get('mean_r2'))} |")
        lines.append(f"| Std R2 | {_fmt_float(score_out.get('std_r2'))} |")
        lines.append(f"| Train-Val Gap | {_fmt_float(score_out.get('gap'))} |")
    lines.append("")

    lines.append("## Top Selected Features")
    if not selected_features:
        lines.append("")
        lines.append("_None_")
    else:
        top_feats = list(selected_features)[: int(top_n_features)]
        lines.append("")
        lines.append("| # | Feature |")
        lines.append("| --- | --- |")
        for i, feat in enumerate(top_feats, start=1):
            lines.append(f"| {i} | {feat} |")
    lines.append("")

    if weights:
        lines.append("## Score Weights")
        lines.append("")
        lines.append("| Metric | Weight |")
        lines.append("| --- | --- |")
        for key in sorted(weights.keys()):
            lines.append(f"| {key} | {_fmt_float(weights.get(key), digits=4)} |")
        lines.append("")

    if metric_rows:
        df = pd.DataFrame(metric_rows)
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].map(lambda v: round(v, 4) if pd.notna(v) else v)
        lines.append("## Metrics")
        lines.append("")
        lines.append("> Note: These models have not been tuned or optimized in any way")
        lines.append("")
        lines.append(df.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## Metrics")
        lines.append("")
        lines.append("_No metrics provided._")
        lines.append("")

    report_content = "\n".join(lines)
    report_path = None

    if run_dir is not None:
        report_path = run_dir / "report.md"
        report_path.write_text(report_content, encoding="utf-8")
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
        "report_path": str(report_path) if report_path is not None else None,
        "report_content": report_content,
        "score": score_out,
    }
