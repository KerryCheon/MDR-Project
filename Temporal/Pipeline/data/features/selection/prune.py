# Jakob Balkovec
# Thu Feb 19th
#
# script to prune feature sets with stable permutation importance and yaml config
# pretty console output with green/red deltas, tqdm progress, and checkpoints
#
# to run:
# python prune.py --config config.yaml
# python prune.py --config configs/base_config.yaml

import os
import re
import json
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.inspection import permutation_importance

import yaml

from xgboost import XGBRegressor


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def c_green(s): return f"{GREEN}{s}{RESET}"
def c_red(s): return f"{RED}{s}{RESET}"
def c_yellow(s): return f"{YELLOW}{s}{RESET}"
def c_cyan(s): return f"{CYAN}{s}{RESET}"
def c_dim(s): return f"{DIM}{s}{RESET}"
def c_bold(s): return f"{BOLD}{s}{RESET}"


def log_info(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{c_dim(ts)} {c_bold('[info]')} {msg}")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_text(path, s):
    with open(path, "w") as f:
        f.write(s)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))

    std = float(np.std(y_true))
    nrmse_std = float(rmse / std) if std > 0 else float("nan")

    p90_abs_err = float(np.percentile(np.abs(y_pred - y_true), 90))

    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "pearson": pearson,
        "explained_var": evs,
        "nrmse_std": nrmse_std,
        "p90_abs_err": p90_abs_err,
    }


def load_split(path, parse_dates, date_col):
    if not str(path).lower().endswith(".csv"):
        raise ValueError(f"Only CSV input is supported. Received: {path}")

    log_info(f"Loading CSV: {path}")
    if parse_dates and date_col:
        df = pd.read_csv(path, parse_dates=[date_col])
    else:
        df = pd.read_csv(path)
    log_info(f"Loaded {path} with shape={df.shape}")
    return df


def build_feature_cols(df, target_col, id_cols, drop_cols):
    drops = set([target_col] + list(id_cols) + list(drop_cols or []))
    cols = [c for c in df.columns if c not in drops]
    return cols


def compile_family_rules(rules_dict):
    out = {}
    for fam, pattern in rules_dict.items():
        out[fam] = re.compile(pattern)
    return out


def family_map(feature_cols, compiled_rules):
    fam_cols = {k: [] for k in compiled_rules.keys()}
    other = []
    for c in feature_cols:
        found = False
        for fam, rx in compiled_rules.items():
            if rx.search(c):
                fam_cols[fam].append(c)
                found = True
                break
        if not found:
            other.append(c)
    return fam_cols, other


def select_by_families(fam_cols, include_families):
    out = []
    for f in include_families:
        out.extend(fam_cols.get(f, []))
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def xy(df, feature_cols, target_col):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)
    return X, y


def fit_xgb(Xtr, ytr, Xva, yva, params):
    model = XGBRegressor(**params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


def eval_features(df_train, df_val, df_test, feature_cols, target_col, params):
    Xtr, ytr = xy(df_train, feature_cols, target_col)
    Xva, yva = xy(df_val, feature_cols, target_col)
    Xte, yte = xy(df_test, feature_cols, target_col)

    model = fit_xgb(Xtr, ytr, Xva, yva, params)

    pva = model.predict(Xva)
    pte = model.predict(Xte)

    return {
        "n_features": len(feature_cols),
        "val": compute_metrics(yva, pva),
        "test": compute_metrics(yte, pte),
    }, model


def stable_perm_rank(df_train, df_val, feature_cols, target_col, params, seeds, repeats, scoring):
    Xva, yva = xy(df_val, feature_cols, target_col)
    Xtr, ytr = xy(df_train, feature_cols, target_col)

    merged = None

    for s in seeds:
        p = dict(params)
        p["random_state"] = s

        model = fit_xgb(Xtr, ytr, Xva, yva, p)
        pi = permutation_importance(
            model,
            Xva,
            yva,
            n_repeats=repeats,
            random_state=s,
            scoring=scoring,
            n_jobs=p.get("n_jobs", -1),
        )

        df_imp = pd.DataFrame({
            "feature": feature_cols,
            f"imp_seed_{s}": pi.importances_mean
        }).set_index("feature")

        merged = df_imp if merged is None else merged.join(df_imp, how="outer")

    merged = merged.fillna(0.0)
    cols = [c for c in merged.columns if c.startswith("imp_seed_")]
    merged["importance"] = merged[cols].mean(axis=1)
    out = merged[["importance"]].sort_values("importance", ascending=False).reset_index()
    return out


def format_delta(curr, prev, higher_is_better=True, fmt="{:+.6f}"):
    d = curr - prev
    s = fmt.format(d)
    if d == 0:
        return s
    good = d > 0 if higher_is_better else d < 0
    return c_green(s) if good else c_red(s)

def pretty_row(round_idx, nfeat, m, prev_m):
    r2 = m["r2"]
    rmse = m["rmse"]
    mae = m["mae"]
    p90 = m["p90_abs_err"]

    if prev_m is None:
        return f"{c_bold(str(round_idx).rjust(2))}  n={c_cyan(str(nfeat).rjust(4))}  r2={r2:.6f}  rmse={rmse:.6f}  mae={mae:.6f}  p90={p90:.6f}"

    return (
        f"{c_bold(str(round_idx).rjust(2))}  n={c_cyan(str(nfeat).rjust(4))}  "
        f"r2={r2:.6f}({format_delta(r2, prev_m['r2'], higher_is_better=True)})  "
        f"rmse={rmse:.6f}({format_delta(rmse, prev_m['rmse'], higher_is_better=False)})  "
        f"mae={mae:.6f}({format_delta(mae, prev_m['mae'], higher_is_better=False)})  "
        f"p90={p90:.6f}({format_delta(p90, prev_m['p90_abs_err'], higher_is_better=False)})"
    )


def save_feature_list(out_dir, name, features):
    path = os.path.join(out_dir, name)
    pd.Series(features).to_csv(path, index=False)


def run_stage(df_train, df_val, df_test, target_col, params,
              start_features, stage_name, out_dir,
              target_n, drop_frac, allow_r2_drop,
              seeds, repeats, scoring,
              patience, min_drop_frac,
              save_rankings, save_round_features, print_top_k):

    rows = []
    best_r2 = -1e9
    best_features = list(start_features)
    best_round = 0

    reject_from_n = None
    reject_streak = 0
    MAX_REJECT_STREAK_SAME_N = 2

    prev_metrics = None
    bad = 0
    curr_features = list(start_features)

    last_rejected_transition = None
    rejected_transition_repeats = 0

    ensure_dir(out_dir)
    log_info(
        f"Stage={stage_name} init: n_start={len(curr_features)}, target_n={target_n}, "
        f"drop_frac={drop_frac:.3f}, allow_r2_drop={allow_r2_drop:.4f}, "
        f"patience={patience}, scoring={scoring}, repeats={repeats}, seeds={seeds}"
    )

    eval0, _ = eval_features(df_train, df_val, df_test, curr_features, target_col, params)
    m0 = eval0["val"]
    rows.append({"round": 0, "n_features": len(curr_features), **m0, "tag": f"{stage_name}_start"})

    best_r2 = m0["r2"]
    best_features = list(curr_features)
    best_round = 0

    print(c_bold(f"\n[{stage_name}] start"))
    print(pretty_row(0, len(curr_features), m0, None))

    if save_round_features:
        save_feature_list(out_dir, f"{stage_name}_round_000_features.csv", curr_features)
    save_feature_list(out_dir, f"{stage_name}_best_features.csv", best_features)

    round_idx = 0

    est_steps = max(
        1,
        int(
            math.ceil(
                math.log(
                    max(len(curr_features), 2) / max(target_n, 1),
                    1.0 / max(1e-6, (1.0 - drop_frac))
                )
            )
        )
    )

    pbar = tqdm(total=est_steps, desc=f"{stage_name} pruning", leave=True)

    while len(curr_features) > target_n:
        round_idx += 1

        log_info(f"[{stage_name}] round={round_idx} computing permutation importance for n={len(curr_features)}")
        ranked = stable_perm_rank(df_train, df_val, curr_features, target_col, params, seeds, repeats, scoring)

        n_drop = int(max(1, math.floor(len(curr_features) * drop_frac)))
        n_keep = max(target_n, len(curr_features) - n_drop)
        transition = (len(curr_features), n_keep)

        top_feature = ranked["feature"].iloc[0] if len(ranked) else "N/A"
        top_importance = float(ranked["importance"].iloc[0]) if len(ranked) else float("nan")
        log_info(
            f"[{stage_name}] round={round_idx} ranked complete: top={top_feature} "
            f"importance={top_importance:.6f}, drop={n_drop}, keep={n_keep}"
        )

        keep = ranked["feature"].iloc[:n_keep].tolist()

        eval_next, _ = eval_features(df_train, df_val, df_test, keep, target_col, params)
        m = eval_next["val"]

        rows.append({"round": round_idx, "n_features": len(keep), **m, "tag": stage_name})

        print(pretty_row(round_idx, len(keep), m, prev_metrics if prev_metrics is not None else m0))

        if print_top_k and print_top_k > 0:
            tops = ranked.head(int(print_top_k)).copy()
            tops["importance"] = tops["importance"].map(lambda x: f"{x:.6f}")
            print(c_dim(tops.to_string(index=False)))

        if save_rankings:
            ranked.to_csv(os.path.join(out_dir, f"{stage_name}_round_{round_idx:03d}_ranked.csv"), index=False)

        if save_round_features:
            save_feature_list(out_dir, f"{stage_name}_round_{round_idx:03d}_features.csv", keep)

        prev_metrics = m

        if m["r2"] > best_r2:
            best_r2 = m["r2"]
            best_features = list(keep)
            best_round = round_idx
            log_info(f"[{stage_name}] round={round_idx} new best: r2={best_r2:.6f}, n={len(best_features)}")
            save_feature_list(out_dir, f"{stage_name}_best_features.csv", best_features)

        if m["r2"] >= best_r2 - allow_r2_drop:
            log_info(
                f"[{stage_name}] round={round_idx} accepted: val_r2={m['r2']:.6f}, "
                f"threshold={best_r2 - allow_r2_drop:.6f}"
            )
            curr_features = list(keep)
            bad = 0

            last_rejected_transition = None
            rejected_transition_repeats = 0
            reject_from_n = None
            reject_streak = 0

        else:
            bad += 1
            log_info(
                f"[{stage_name}] round={round_idx} rejected: val_r2={m['r2']:.6f}, "
                f"threshold={best_r2 - allow_r2_drop:.6f}, bad={bad}/{patience}"
            )

            from_n, to_n = transition

            if reject_from_n == from_n:
                reject_streak += 1
            else:
                reject_from_n = from_n
                reject_streak = 1

            if reject_streak >= MAX_REJECT_STREAK_SAME_N:
                print(c_yellow(
                    f"  oscillation guard hit (rejected {reject_streak} times from n={from_n}); "
                    "stopping stage and keeping best features"
                ))
                log_info(
                    f"[{stage_name}] round={round_idx} oscillation guard: rejected {reject_streak} times from n={from_n}; "
                    f"ending stage at best_n={len(best_features)}"
                )
                curr_features = list(best_features)
                break

            drop_frac_changed = False
            if bad >= patience:
                new_drop_frac = max(min_drop_frac, drop_frac * 0.7)
                drop_frac_changed = new_drop_frac < drop_frac
                drop_frac = new_drop_frac
                bad = 0
                print(c_yellow(f"  drop_frac adjusted to {drop_frac:.3f}"))
                log_info(f"[{stage_name}] round={round_idx} drop_frac updated to {drop_frac:.3f}")

                last_rejected_transition = None
                rejected_transition_repeats = 0
                reject_from_n = None
                reject_streak = 0

            if not drop_frac_changed:
                if transition == last_rejected_transition:
                    rejected_transition_repeats += 1
                else:
                    last_rejected_transition = transition
                    rejected_transition_repeats = 1

                if rejected_transition_repeats >= 2:
                    print(c_yellow(
                        f"  oscillation guard hit ({transition[0]} -> {transition[1]}) with no R2 gain; "
                        "stopping stage and keeping best features"
                    ))
                    log_info(
                        f"[{stage_name}] round={round_idx} oscillation guard: repeated rejected transition "
                        f"{transition[0]} -> {transition[1]}; ending stage at best_n={len(best_features)}"
                    )
                    curr_features = list(best_features)
                    break

            curr_features = list(best_features)

        if len(curr_features) <= target_n:
            break

        pbar.update(1)

        if drop_frac <= min_drop_frac and len(curr_features) <= target_n:
            break

    pbar.close()

    hist = pd.DataFrame(rows)
    hist.to_csv(os.path.join(out_dir, f"{stage_name}_history.csv"), index=False)

    summary = {
        "stage": stage_name,
        "best_round": int(best_round),
        "best_r2": float(best_r2),
        "best_n_features": int(len(best_features)),
    }
    save_json(os.path.join(out_dir, f"{stage_name}_summary.json"), summary)

    print(c_bold(f"[{stage_name}] best: round={best_round}  n={len(best_features)}  r2={best_r2:.6f}"))
    return best_features, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    log_info(f"Loaded config from {args.config}")

    out_dir = cfg["paths"]["out_dir"]
    ensure_dir(out_dir)
    log_info(f"Output directory: {out_dir}")
    save_text(os.path.join(out_dir, "config_used.yaml"), yaml.safe_dump(cfg, sort_keys=False))

    parse_dates = bool(cfg["data"].get("parse_dates", False))
    date_col = cfg["data"].get("date_col", "")
    log_info(f"Data settings: parse_dates={parse_dates}, date_col={date_col}")

    df_train = load_split(cfg["paths"]["train"], parse_dates, date_col)
    df_val = load_split(cfg["paths"]["val"], parse_dates, date_col)
    df_test = load_split(cfg["paths"]["test"], parse_dates, date_col)
    log_info(f"Split shapes: train={df_train.shape}, val={df_val.shape}, test={df_test.shape}")

    target_col = cfg["data"]["target_col"]
    id_cols = cfg["data"].get("id_cols", [])
    drop_cols = cfg["data"].get("drop_cols", [])
    log_info(f"Target={target_col}, id_cols={id_cols}, drop_cols={drop_cols}")

    feature_cols = build_feature_cols(df_train, target_col, id_cols, drop_cols)
    log_info(f"Candidate features after drops: {len(feature_cols)}")

    compiled = compile_family_rules(cfg["families"]["rules"])
    fam_cols, other = family_map(feature_cols, compiled)
    family_counts = {k: len(v) for k, v in fam_cols.items()}
    log_info(f"Family counts: {family_counts}")

    include_fams = cfg["families"]["include"]
    start_features = select_by_families(fam_cols, include_fams)
    log_info(f"Included families={include_fams}, start_features={len(start_features)}, unmatched={len(other)}")

    if len(other) > 0:
        save_feature_list(out_dir, "unmatched_features.csv", other)

    save_feature_list(out_dir, "start_features.csv", start_features)

    params = cfg["model"]["params"]
    log_info(f"Model params loaded (n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')})")

    coarse = cfg["prune"]["start"]
    fine = cfg["prune"]["fine"]
    perm = cfg["prune"]["perm"]
    stop = cfg["prune"]["stop"]

    output_cfg = cfg.get("output", {})
    save_rankings = bool(output_cfg.get("save_rankings", True))
    save_round_features = bool(output_cfg.get("save_round_features", True))
    print_top_k = int(output_cfg.get("print_top_features_each_round", 0))

    seeds = perm.get("seeds", [1, 2, 3, 4, 5])
    repeats = int(perm.get("repeats", 5))
    scoring = perm.get("scoring", "r2")
    log_info(f"Permutation settings: seeds={seeds}, repeats={repeats}, scoring={scoring}")

    min_drop_frac = float(stop.get("min_drop_frac", 0.05))

    coarse_out = os.path.join(out_dir, "stage_coarse")
    fine_out = os.path.join(out_dir, "stage_fine")
    log_info("Starting coarse pruning stage")

    coarse_features, _ = run_stage(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        target_col=target_col,
        params=params,
        start_features=start_features,
        stage_name="coarse",
        out_dir=coarse_out,
        target_n=int(coarse["coarse_target"]),
        drop_frac=float(coarse["coarse_drop_frac"]),
        allow_r2_drop=float(coarse["coarse_allow_r2_drop"]),
        seeds=seeds,
        repeats=repeats,
        scoring=scoring,
        patience=int(fine.get("patience", 2)),
        min_drop_frac=min_drop_frac,
        save_rankings=save_rankings,
        save_round_features=save_round_features,
        print_top_k=print_top_k,
    )

    log_info("Starting fine pruning stage")
    final_features, hist = run_stage(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        target_col=target_col,
        params=params,
        start_features=coarse_features,
        stage_name="fine",
        out_dir=fine_out,
        target_n=int(fine["target_n"]),
        drop_frac=float(fine["drop_frac"]),
        allow_r2_drop=float(fine["allow_r2_drop"]),
        seeds=seeds,
        repeats=repeats,
        scoring=scoring,
        patience=int(fine.get("patience", 2)),
        min_drop_frac=min_drop_frac,
        save_rankings=save_rankings,
        save_round_features=save_round_features,
        print_top_k=print_top_k,
    )

    final_eval, _ = eval_features(df_train, df_val, df_test, final_features, target_col, params)

    save_feature_list(out_dir, "FINAL_best_features.csv", final_features)
    save_json(os.path.join(out_dir, "FINAL_metrics.json"), {
        "n_features": final_eval["n_features"],
        "val": final_eval["val"],
        "test": final_eval["test"],
    })
    log_info(
        f"Final metrics saved: n_features={final_eval['n_features']}, "
        f"val_r2={final_eval['val']['r2']:.6f}, test_r2={final_eval['test']['r2']:.6f}"
    )

    print(c_bold("\n[final]"))
    print(f"n_features={c_cyan(str(final_eval['n_features']))}")
    print(f"val_r2={final_eval['val']['r2']:.6f}  test_r2={final_eval['test']['r2']:.6f}")
    print(f"val_rmse={final_eval['val']['rmse']:.6f}  test_rmse={final_eval['test']['rmse']:.6f}")
    print(f"val_p90={final_eval['val']['p90_abs_err']:.6f}  test_p90={final_eval['test']['p90_abs_err']:.6f}")


if __name__ == "__main__":
    main()
