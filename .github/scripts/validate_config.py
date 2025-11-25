# Jakob Balkovec
# Nov 25 2025
# .github/scripts/validate_config.py

import yaml
import sys
from pathlib import Path

def fail(msg: str):
    print(f"[CONFIG ERROR] {msg}")
    sys.exit(1)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

CONFIG_PATH = REPO_ROOT / "Temporal" / "Pipeline" / "config.yaml"

if not CONFIG_PATH.exists():
    fail(f"config.yaml not found at expected path: {CONFIG_PATH}")

with CONFIG_PATH.open("r") as f:
    cfg = yaml.safe_load(f)

imputers = cfg.get("imputer", {})
weights = cfg.get("base_weights", {})

active_imputers = [k for k, v in imputers.items() if v.get("active", False)]

for key in weights.keys():
    if key not in imputers:
        fail(f"Weight key '{key}' does not match any imputer defined in config.")

for imp in active_imputers:
    if imp not in weights:
        fail(f"Active imputer '{imp}' is missing from base_weights.")

total_weight = float(sum(weights.values()))
if abs(total_weight - 1.0) > 1e-6:
    fail(f"Weights sum to {total_weight}, expected 1.0.")

print(f"Config validation successful. Loaded from: {CONFIG_PATH}")
