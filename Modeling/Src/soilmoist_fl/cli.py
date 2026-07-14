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
from Modeling.Src.soilmoist_fl.Selectors.stability import stability_bootstrap, stability_bootstrap_elasticnet
from Modeling.Src.soilmoist_fl.Selectors.correlation import select_correlation
from Modeling.Src.soilmoist_fl.Selectors.rf_importance import select_rf_importance
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


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "Configs" / "default.yaml"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "Runs"


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


def select_features(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    X_test=None,
    y_test=None,
    config=None,
    run_dir=None,
    run_id=None,
    top_k=None,
    stages=None,
    bypass_prefixes=('J_', 'K_', 'D_', 'G_'),
    bypass_exact={'longitude', 'latitude', 'elev', 'slope', 'aspect', 'DOY', 'precip_mm', 'sin_year', 'cos_year'},
    random_state=42,
    verbose=True,
):
    # 1. Resolve config
    if config is None:
        if DEFAULT_CONFIG_PATH.exists():
            cfg = load_config(str(DEFAULT_CONFIG_PATH))
        else:
            cfg = {}
    elif isinstance(config, (str, Path)):
        cfg = load_config(str(config))
    elif isinstance(config, dict):
        cfg = config
    else:
        raise TypeError("config must be None, a path, or a dict")

    # 2. Setup logging if needed
    if verbose:
        setup_logger(cfg, run_dir)
    log = get_logger("select_features")

    # 3. Resolve top_k and stages
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg, dict) else {}
    if top_k is None:
        top_k = int(sel_cfg.get("top_k", 40))
    if stages is None:
        stages = list(sel_cfg.get("stages", []) or [])
        if not stages:
            stages = [
                {"kind": "mi", "k": 300},
                {"kind": "elasticnet", "k": 60},
                {"kind": "stability", "min_freq": 0.6}
            ]

    # 4. Feature checks and prep
    forbidden = list(sel_cfg.get("forbidden_tokens", []) or [])
    check_feature_name_leakage(list(X_train.columns), forbidden=forbidden)
    group_features(list(X_train.columns))

    # Identify bypass features
    if bypass_prefixes is None:
        bypass_prefixes = ()
    if bypass_exact is None:
        bypass_exact = set()
    
    bypass_cols = [
        c for c in X_train.columns 
        if (bypass_prefixes and c.startswith(bypass_prefixes)) or 'year' in c or c in bypass_exact
    ]
    ts_cols = [c for c in X_train.columns if c not in bypass_cols]

    # Initialize current features to TS features
    current_feats = [c for c in ts_cols if c in X_train.columns]
    
    opt_alpha = None
    opt_l1_ratio = None
    last_stage_kind = None
    ranking_input_feats = None

    # Run selection stages
    for i, st in enumerate(stages):
        kind = str(st.get("kind", "")).lower()
        log.info("Running selection stage %d: %s", i + 1, kind)
        
        if kind == "mi":
            mi_k = int(st.get("k", 120))
            mi_out = select_mi(X_train[current_feats], y_train, k=mi_k)
            mi_feats = mi_out["selected"]
            if run_dir:
                save_stage_features(run_dir, "mi", mi_feats, ranked=mi_out.get("ranked"), scores=mi_out.get("scores"))
            
            # Force include bypass cols back in the candidates
            current_feats = list(set(mi_feats + bypass_cols))
            current_feats = [f for f in current_feats if f in X_train.columns]
            log.info("Stage MI done: selected=%d (plus %d bypassed features)", len(mi_feats), len(bypass_cols))
            
        elif kind == "correlation":
            threshold = float(st.get("threshold", 0.95))
            corr_out = select_correlation(X_train[current_feats], y_train, threshold=threshold)
            corr_feats = corr_out["selected"]
            if run_dir:
                save_stage_features(run_dir, "correlation", corr_feats, ranked=corr_out.get("ranked"), scores=corr_out.get("scores"))
            current_feats = corr_feats
            log.info("Stage Correlation done: selected=%d (dropped %d collinear features)", len(current_feats), len(corr_out.get("dropped", [])))
            
        elif kind == "elasticnet":
            enet_k = int(st.get("k", 60))
            ranking_input_feats = list(current_feats)  # Candidates for stability bootstrap
            enet_out = select_elasticnet(X_train[current_feats], y_train, k=enet_k)
            enet_feats = enet_out["selected"]
            if run_dir:
                save_stage_features(run_dir, "elasticnet", enet_feats, ranked=enet_out.get("ranked"), scores=enet_out.get("scores"))
            
            opt_alpha = enet_out["alpha"]
            opt_l1_ratio = enet_out["l1_ratio"]
            current_feats = enet_feats
            log.info("Stage ElasticNet done: selected=%d (alpha=%.6g, l1_ratio=%.3f)", len(current_feats), opt_alpha, opt_l1_ratio)
            
        elif kind == "rf_importance":
            rf_k = int(st.get("k", 60))
            ranking_input_feats = list(current_feats)  # Candidates for stability bootstrap
            rf_out = select_rf_importance(X_train[current_feats], y_train, k=rf_k)
            rf_feats = rf_out["selected"]
            if run_dir:
                save_stage_features(run_dir, "rf_importance", rf_feats, ranked=rf_out.get("ranked"), scores=rf_out.get("scores"))
            current_feats = rf_feats
            log.info("Stage RF Importance done: selected=%d", len(current_feats))
            
        elif kind == "stability":
            min_freq = float(st.get("min_freq", 0.6))
            n_boot = int(st.get("stability_n_boot", sel_cfg.get("stability_n_boot", 100)))
            sample_frac = float(st.get("stability_sample_frac", sel_cfg.get("stability_sample_frac", 0.8)))
            
            # Determine base estimator for stability bootstrap
            base_estimator = str(st.get("base", "")).lower()
            if not base_estimator:
                if last_stage_kind == "rf_importance":
                    base_estimator = "rf"
                else:
                    base_estimator = "elasticnet"
            
            # Use candidates from before the ranking step
            stab_input_feats = ranking_input_feats if ranking_input_feats is not None else current_feats
            X_tr_stab = X_train[stab_input_feats]
            
            # Determine stability k
            stab_k = int(st.get("k", 60))
            
            base_kwargs = {}
            if base_estimator == "elasticnet" and opt_alpha is not None:
                base_kwargs = {"alpha": opt_alpha, "l1_ratio": opt_l1_ratio}
                
            stab_out = stability_bootstrap(
                X=X_tr_stab,
                y=y_train,
                base=base_estimator,
                n_boot=n_boot,
                sample_frac=sample_frac,
                min_freq=min_freq,
                top_k=top_k,
                random_state=int(sel_cfg.get("random_state", random_state)),
                base_k=stab_k,
                base_kwargs=base_kwargs,
            )
            stable_feats = stab_out["selected"]
            if run_dir:
                save_stage_features(run_dir, "stability", stable_feats, ranked=stab_out.get("ranked"), scores=stab_out.get("scores"))
            current_feats = stable_feats
            log.info("Stage Stability done: selected=%d (base=%s, n_boot=%d)", len(current_feats), base_estimator, n_boot)
            
        last_stage_kind = kind

    final_feats = current_feats
    log.info("Final selected features (first 20): %s", final_feats[:20])

    if run_dir:
        save_selected_features(run_dir, final_feats)

    # 5. Model evaluation (only if models configured and val split is provided)
    metric_rows = []
    model_summaries = []
    models = _build_models(cfg)

    # Check if we should/can run model evaluation
    has_splits = X_val is not None and y_val is not None
    has_test = X_test is not None and y_test is not None
    if models and not has_splits:
        log.warning("Models are configured but validation split data was not provided. Skipping downstream model evaluation.")
        models = []

    if models and has_splits and not has_test:
        log.warning(
            "Validation split provided but test split is missing. "
            "The robustness metric 'val_minus_test_r2' will be computed as "
            "val_metrics - val_metrics (= 0) since m_va is used as a stand-in "
            "for the missing test set. This produces an artificially perfect "
            "generalization score and is NOT a valid robustness measure. "
            "Pass a true held-out test set (X_test, y_test) to get a meaningful result."
        )

    if models:
        # Align columns
        if list(X_train.columns) != list(X_val.columns):
            log.warning("Feature columns mismatch between train and val. Aligning val columns to train.")
            X_val = X_val.reindex(columns=X_train.columns)
            
        missing_va = [f for f in final_feats if f not in X_val.columns]
        if missing_va:
            raise ValueError(f"Final feature set missing in val split: {missing_va[:10]}")
        
        X_tr_final = X_train[final_feats]
        X_va_final = X_val[final_feats]
        
        if X_test is not None and y_test is not None:
            if list(X_train.columns) != list(X_test.columns):
                log.warning("Feature columns mismatch between train and test. Aligning test columns to train.")
                X_test = X_test.reindex(columns=X_train.columns)
            missing_te = [f for f in final_feats if f not in X_test.columns]
            if missing_te:
                raise ValueError(f"Final feature set missing in test split: {missing_te[:10]}")
            X_te_final = X_test[final_feats]
        else:
            X_te_final = None

        for model_name, model in models:
            log.info("Training model: %s", model_name)
            model.fit(X_tr_final, y_train)

            yhat_tr = model.predict(X_tr_final)
            yhat_va = model.predict(X_va_final)

            m_tr = metrics_block("train", y_train, yhat_tr)
            m_va = metrics_block("val", y_val, yhat_va)
            
            m_tr["model"] = model_name
            m_tr["n_features"] = len(final_feats)
            m_va["model"] = model_name
            m_va["n_features"] = len(final_feats)

            metrics_list = [m_tr, m_va]

            if X_te_final is not None and y_test is not None:
                yhat_te = model.predict(X_te_final)
                m_te = metrics_block("test", y_test, yhat_te)
                m_te["model"] = model_name
                m_te["n_features"] = len(final_feats)
                metrics_list.append(m_te)
                robust = robustness_block(m_tr, m_va, m_te)
            else:
                m_te = None
                # NOTE: When no test set is provided, m_va is passed as both
                # val and test to robustness_block, causing val_minus_test_r2
                # = 0.0 by construction. See warning logged at top of block.
                robust = robustness_block(m_tr, m_va, m_va)

            metric_rows.extend(metrics_list)
            log_metrics_table(metrics_list, title=f"Metrics: {model_name}")

            stability = stability_summary(metrics_list, metric="r2")
            model_summaries.append({
                "model": model_name,
                "robustness": robust,
                "stability": stability,
            })

        if run_dir:
            save_metrics_csv(run_dir / "metrics.csv", metric_rows)
            save_json(run_dir / "model_summaries.json", model_summaries)

    # 6. Report generation
    report_out = make_report(
        run_dir=run_dir,
        config=cfg,
        selected_features=final_feats,
        metric_rows=metric_rows,
        model_name="feature_selection",
    )

    return {
        "selected_features": final_feats,
        "run_dir": str(run_dir) if run_dir else None,
        "run_id": str(run_id) if run_id else None,
        "metrics": metric_rows,
        "model_summaries": model_summaries,
        "report_content": report_out.get("report_content"),
        "score": report_out.get("score"),
    }


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

    fold = loaded.folds[0]

    check_no_future_leakage(fold.train, time_col=time_col)
    check_fold_boundary_gap(fold.train, fold.val, time_col=time_col)

    X_tr, y_tr, _, _ = preprocess_split(fold.train, target, drop_cols=drop_cols)
    X_va, y_va, _, _ = preprocess_split(fold.val, target, drop_cols=drop_cols)
    X_te, y_te, _, _ = preprocess_split(fold.test, target, drop_cols=drop_cols)

    if list(X_tr.columns) != list(X_va.columns) or list(X_tr.columns) != list(X_te.columns):
        raise SystemExit("Feature columns mismatch across splits after preprocessing.")

    res = select_features(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        config=cfg,
        run_dir=run_dir,
        run_id=run_id,
        verbose=False,
    )

    return {
        "run_dir": res["run_dir"],
        "run_id": res["run_id"],
        "selected_features": res["selected_features"],
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
