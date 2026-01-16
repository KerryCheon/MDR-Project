# Jakob Balkovec
# CLI entrypoints

import argparse
from pathlib import Path

from Modeling.Src.soilmoist_fl.Data.load import load_splits
from Modeling.Src.soilmoist_fl.Data.validate import validate_loaded_data
from Modeling.Src.soilmoist_fl.Features.groups import group_features
from Modeling.Src.soilmoist_fl.Features.leakage import (
    check_feature_name_leakage,
    check_fold_boundary_gap,
    check_no_future_leakage,
)
from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split
from Modeling.Src.soilmoist_fl.Evaluation.metrics import log_metrics_table, metrics_block
from Modeling.Src.soilmoist_fl.Evaluation.robustness import robustness_block, stability_summary
from Modeling.Src.soilmoist_fl.Models.linear import LinearModel
from Modeling.Src.soilmoist_fl.Models.rf import RFModel
from Modeling.Src.soilmoist_fl.Models.xgb import XGBModel
from Modeling.Src.soilmoist_fl.Ranking.report import make_report
from Modeling.Src.soilmoist_fl.Selectors.elasticnet import select_elasticnet
from Modeling.Src.soilmoist_fl.Selectors.mi import select_mi
from Modeling.Src.soilmoist_fl.Selectors.stability import stability_bootstrap_elasticnet
from Modeling.Src.soilmoist_fl.Tracking.artifacts import (
    ensure_run_dir,
    save_json,
    save_metrics_csv,
    save_selected_features,
    save_stage_features,
)
from Modeling.Src.soilmoist_fl.Tracking.registry import list_runs, register_run
from Modeling.Utils.config import load_config
from Modeling.Utils.logging import get_logger, setup_logger


DEFAULT_CONFIG_PATH = Path("/Users/jbalkovec/Desktop/MDR/Modeling/Configs/default.yaml")
DEFAULT_RUNS_DIR = Path("/Users/jbalkovec/Desktop/MDR/Modeling/Runs")


def _build_models(cfg):
    log = get_logger("models.registry")
    model_cfgs = (cfg or {}).get("models", []) if isinstance(cfg, dict) else []

    registry = {
        "linear": LinearModel,
        "rf": RFModel,
        "xgb": XGBModel,
    }

    out = []
    for entry in model_cfgs:
        if isinstance(entry, str):
            kind = entry
            mcfg = {"kind": kind}
        elif isinstance(entry, dict):
            kind = entry.get("kind")
            mcfg = entry
        else:
            log.warning("Unknown model config entry: %s", entry)
            continue

        if not kind:
            log.warning("Model entry missing kind: %s", entry)
            continue

        cls = registry.get(str(kind).lower())
        if cls is None:
            log.warning("Unsupported model kind: %s", kind)
            continue

        name = str(mcfg.get("name", kind))
        out.append((name, cls(config=mcfg)))

    if not out:
        log.warning("No models configured; skipping training.")
    return out


def run_feature_selection(
    config_path=DEFAULT_CONFIG_PATH,
    base_runs_dir=DEFAULT_RUNS_DIR,
    run_dir=None,
    run_id=None,
):
    cfg = load_config(str(config_path))

    if run_dir is None:
        run_dir, run_id = ensure_run_dir(base_runs_dir, run_id=run_id)
    else:
        run_dir = Path(run_dir)
        if run_id is None:
            run_id = run_dir.name

    setup_logger(cfg, run_dir)
    log = get_logger("main")

    register_run(run_dir, run_id=run_id, meta={"config": str(config_path)})

    loaded = load_splits(cfg)
    report = validate_loaded_data(loaded, cfg)
    if not report.ok:
        raise SystemExit("Validation failed. Check logs for details.")

    data_cfg = (cfg or {}).get("data", {})
    target = data_cfg.get("target")

    id_cols = list(data_cfg.get("id_cols", []) or [])
    time_col = data_cfg.get("time_col")

    drop_cols = list(id_cols)
    if time_col:
        drop_cols.append(time_col)

    forbidden = list((cfg or {}).get("selection", {}).get("forbidden_tokens", []) or [])

    fold = loaded.folds[0]

    check_no_future_leakage(fold.train, time_col=time_col)
    check_fold_boundary_gap(fold.train, fold.val, time_col=time_col)

    X_tr, y_tr, _, _ = preprocess_split(fold.train, target, drop_cols=drop_cols)
    X_va, y_va, _, _ = preprocess_split(fold.val, target, drop_cols=drop_cols)
    X_te, y_te, _, _ = preprocess_split(fold.test, target, drop_cols=drop_cols)

    if list(X_tr.columns) != list(X_va.columns) or list(X_tr.columns) != list(X_te.columns):
        raise SystemExit("Feature columns mismatch across splits after preprocessing.")

    group_features(list(X_tr.columns))
    check_feature_name_leakage(list(X_tr.columns), forbidden=forbidden)

    log.info("Ready for selectors: train X=%s y=%s", X_tr.shape, y_tr.shape)

    sel_cfg = (cfg or {}).get("selection", {})
    stages = list(sel_cfg.get("stages", []) or [])
    top_k = int(sel_cfg.get("top_k", 40))

    # defaults
    mi_k = 120
    enet_k = 60
    min_freq = 0.6

    for st in stages:
        kind = str(st.get("kind", "")).lower()
        if kind == "mi":
            mi_k = int(st.get("k", mi_k))
        elif kind == "elasticnet":
            enet_k = int(st.get("k", enet_k))
        elif kind == "stability":
            min_freq = float(st.get("min_freq", min_freq))

    mi_out = select_mi(X_tr, y_tr, k=mi_k)
    mi_feats = mi_out["selected"]
    save_stage_features(run_dir, "mi", mi_feats, ranked=mi_out.get("ranked"), scores=mi_out.get("scores"))
    log.info("Stage MI done: selected=%d", len(mi_feats))

    X_tr_mi = X_tr[mi_feats]
    enet_out = select_elasticnet(X_tr_mi, y_tr, k=enet_k)
    enet_feats = enet_out["selected"]
    save_stage_features(run_dir, "elasticnet", enet_feats, ranked=enet_out.get("ranked"), scores=enet_out.get("scores"))
    log.info("Stage ElasticNet done: selected=%d", len(enet_feats))

    stab_out = stability_bootstrap_elasticnet(
        X_tr_mi,
        y_tr,
        n_boot=int(sel_cfg.get("stability_n_boot", 10)),
        sample_frac=float(sel_cfg.get("stability_sample_frac", 0.8)),
        min_freq=min_freq,
        top_k=top_k,
        random_state=int(sel_cfg.get("random_state", 42)),
        enet_k=enet_k,
        enet_kwargs={},
    )
    stable_feats = stab_out["selected"]
    save_stage_features(run_dir, "stability", stable_feats, ranked=stab_out.get("ranked"), scores=stab_out.get("scores"))
    log.info("Stage Stability done: selected=%d (top_k=%d)", len(stable_feats), top_k)

    final_feats = stable_feats
    log.info("Final selected features (first 20): %s", final_feats[:20])

    missing_va = [f for f in final_feats if f not in X_va.columns]
    missing_te = [f for f in final_feats if f not in X_te.columns]
    if missing_va or missing_te:
        raise SystemExit(
            f"Final feature set missing in splits: val_missing={missing_va[:10]} test_missing={missing_te[:10]}"
        )

    save_selected_features(run_dir, final_feats)

    metric_rows = []
    model_summaries = []
    models = _build_models(cfg)

    if models:
        X_tr_final = X_tr[final_feats]
        X_va_final = X_va[final_feats]
        X_te_final = X_te[final_feats]

        for model_name, model in models:
            log.info("Training model: %s", model_name)
            model.fit(X_tr_final, y_tr)

            yhat_tr = model.predict(X_tr_final)
            yhat_va = model.predict(X_va_final)
            yhat_te = model.predict(X_te_final)

            m_tr = metrics_block("train", y_tr, yhat_tr)
            m_va = metrics_block("val", y_va, yhat_va)
            m_te = metrics_block("test", y_te, yhat_te)

            for m in (m_tr, m_va, m_te):
                m["model"] = model_name
                m["n_features"] = len(final_feats)

            metric_rows.extend([m_tr, m_va, m_te])

            log_metrics_table([m_tr, m_va, m_te], title=f"Metrics: {model_name}")

            robust = robustness_block(m_tr, m_va, m_te)
            stability = stability_summary([m_tr, m_va, m_te], metric="r2")
            model_summaries.append({
                "model": model_name,
                "robustness": robust,
                "stability": stability,
            })

        save_metrics_csv(run_dir / "metrics.csv", metric_rows)
        save_json(run_dir / "model_summaries.json", model_summaries)

    make_report(
        run_dir=run_dir,
        config=cfg,
        selected_features=final_feats,
        metric_rows=metric_rows,
        model_name="feature_selection",
    )

    return {
        "run_dir": str(run_dir),
        "run_id": str(run_id),
        "selected_features": final_feats,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="soilmoist_fl")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run feature selection pipeline")
    run_p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    run_p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument("--run-dir", default=None)

    list_p = sub.add_parser("list-runs", help="List run registry entries")
    list_p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))

    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "run"
    return args


def main(argv=None):
    args = _parse_args(argv)
    if args.command == "list-runs":
        runs = list_runs(args.runs_dir)
        for r in runs:
            print(r)
        return 0

    run_feature_selection(
        config_path=args.config,
        base_runs_dir=args.runs_dir,
        run_dir=args.run_dir,
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
