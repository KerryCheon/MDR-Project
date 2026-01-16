# Jakob Balkovec
# Loader

from pathlib import Path
import os

import pandas as pd

from Modeling.Utils.logging import get_logger


class SplitSet:
    def __init__(self, name, train, val, test, paths):
        self.name = name
        self.train = train
        self.val = val
        self.test = test
        self.paths = paths


class LoadedData:
    def __init__(self, folds, meta):
        self.folds = folds
        self.meta = meta


def _resolve_path(p):
    # desc: here if we transition to using .env vars at some point (path issues)
    s = os.path.expandvars(str(p))
    return Path(s).expanduser().resolve()


def _read_df(path):
    log = get_logger("data.load")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file type (expected .csv): {path}")
    log.debug("Reading CSV: %s", path)
    return pd.read_csv(path)


def _require_keys(d, keys, ctx):
    missing = [k for k in keys if k not in d or d[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required keys in {ctx}: {missing}")


def load_splits(config):
    log = get_logger("data.load")

    data_cfg = (config or {}).get("data", {})
    target = data_cfg.get("target")
    if not target:
        raise ValueError("Missing required config: data.target")

    log.info("Loading data splits (target=%s)", target)

    folds = []

    # mode B: explicit folds list
    if "folds" in data_cfg and data_cfg["folds"]:
        for fcfg in data_cfg["folds"]:
            _require_keys(fcfg, ["name", "train", "val", "test"], ctx="data.folds[]")

            name = str(fcfg["name"])
            paths = {
                "train": _resolve_path(fcfg["train"]),
                "val": _resolve_path(fcfg["val"]),
                "test": _resolve_path(fcfg["test"]),
            }

            for split, p in paths.items():
                if not p.exists():
                    raise FileNotFoundError(f"{name}.{split} not found: {p}")
                log.debug("Resolved %s.%s: %s", name, split, p)

            log.info("Loading %s", name)

            train_df = _read_df(paths["train"])
            val_df   = _read_df(paths["val"])
            test_df  = _read_df(paths["test"])

            log.info(
                "%s shapes: train=%s val=%s test=%s",
                name, train_df.shape, val_df.shape, test_df.shape
            )

            folds.append(
                SplitSet(
                    name=name,
                    train=train_df,
                    val=val_df,
                    test=test_df,
                    paths=paths,
                )
            )

    # mode A: single train/val/test
    else:
        splits = data_cfg.get("splits", {})
        _require_keys(splits, ["train", "val", "test"], ctx="data.splits")

        paths = {
            "train": _resolve_path(splits["train"]),
            "val":   _resolve_path(splits["val"]),
            "test":  _resolve_path(splits["test"]),
        }

        for split, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"{split} not found: {p}")
            log.debug("Resolved %s: %s", split, p)

        log.info("Loading single split set (fold0)")

        train_df = _read_df(paths["train"])
        val_df   = _read_df(paths["val"])
        test_df  = _read_df(paths["test"])

        log.info(
            "fold0 shapes: train=%s val=%s test=%s",
            train_df.shape, val_df.shape, test_df.shape
        )

        folds.append(
            SplitSet(
                name="fold0",
                train=train_df,
                val=val_df,
                test=test_df,
                paths=paths,
            )
        )

    meta = {
        "target": target,
        "n_folds": len(folds),
        "fold_names": [f.name for f in folds],
    }

    log.info("Loaded %d fold(s).", len(folds))
    return LoadedData(folds=folds, meta=meta)
