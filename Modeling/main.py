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
from Modeling.Src.soilmoist_fl.Selectors.mi import select_mi
from Modeling.Src.soilmoist_fl.Selectors.elasticnet import select_elasticnet
from Modeling.Src.soilmoist_fl.Selectors.stability import stability_bootstrap_elasticnet
from Modeling.Src.soilmoist_fl.Tracking.artifacts import ensure_run_dir
from Modeling.Utils.config import load_config
from Modeling.Utils.logging import setup_logger
from Modeling.Utils.logging import get_logger

cfg = load_config("/Users/jbalkovec/Desktop/MDR/Modeling/Configs/default.yaml")

base_runs_dir = Path("/Users/jbalkovec/Desktop/MDR/Modeling/Runs")
run_dir, run_id = ensure_run_dir(base_runs_dir)
setup_logger(cfg, run_dir)
log = get_logger("main")

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
log.info("Stage MI done: selected=%d", len(mi_feats))

X_tr_mi = X_tr[mi_feats]
enet_out = select_elasticnet(X_tr_mi, y_tr, k=enet_k)
enet_feats = enet_out["selected"]
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
log.info("Stage Stability done: selected=%d (top_k=%d)", len(stable_feats), top_k)

final_feats = stable_feats
log.info("Final selected features (first 20): %s", final_feats[:20])

missing_va = [f for f in final_feats if f not in X_va.columns]
missing_te = [f for f in final_feats if f not in X_te.columns]
if missing_va or missing_te:
    raise SystemExit(
        f"Final feature set missing in splits: val_missing={missing_va[:10]} test_missing={missing_te[:10]}"
    )

log.info("Ready for models: train=%s val=%s test=%s using %d features",
         X_tr.shape, X_va.shape, X_te.shape, len(final_feats))
