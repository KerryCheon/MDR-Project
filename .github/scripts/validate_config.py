# Jakob Balkovec
# Nov 25 2025
# .github/scripts/validate_config.py

import yaml
import sys

def fail(msg):
    print(f"[CONFIG ERROR] {msg}")
    sys.exit(1)

with open("Temporal/Pipeline/config.yaml", "r") as f:
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

total_weight = sum(weights.values())
if abs(total_weight - 1.0) > 1e-6:
    fail(f"Weights sum to {total_weight}, expected 1.0.")

print("Config validation successful.")
