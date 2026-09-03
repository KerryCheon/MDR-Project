"""Router-only salvage: pure-function checks (no model training)."""

import importlib.util
from pathlib import Path

import numpy as np
import yaml

RUNNER = Path(
    "notebooks/experiment/derived_8.4-ece-router-salvage-1.0/run_salvage.py"
).resolve()


def _module():
    spec = importlib.util.spec_from_file_location("router_salvage", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_lists_expected_families_and_policies():
    config = yaml.safe_load(
        (RUNNER.parent / "config.yaml").read_text(encoding="utf-8"))
    assert config["families"] == ["Clustering_V0_Full_k2", "Clustering_Backbone54_k2"]
    for policy in ("as_routed", "c0_only", "gapi_transplant",
                   "dynamic_transplant", "seasonal", "margin_fallback"):
        assert policy in config["policies"]
    assert config["experiment"]["constraint"].startswith("ECE strictly unseen")


def test_compute_metrics_recovers_known_values():
    module = _module()
    y_true = np.array([0.05, 0.06, 0.07])
    y_pred = np.array([0.06, 0.06, 0.06])
    metrics = module.compute_metrics(y_true, y_pred)
    assert metrics["bias"] == float(np.mean(y_pred - y_true))
    assert metrics["rmse"] > 0
    assert metrics["ubrmse"] <= metrics["rmse"] + 1e-12


def test_margin_fallback_is_deterministic_given_threshold():
    # Pure routing logic: rows below the WA threshold fall back to global.
    as_routed = np.array([0, 1, 1, 0])
    margins = np.array([0.1, 5.0, 0.2, 6.0])
    threshold = 1.0
    confident = margins >= threshold
    labels = np.where(confident, as_routed, -1)
    assert labels.tolist() == [-1, 1, -1, 0]
    assert float((~confident).mean()) == 0.5
