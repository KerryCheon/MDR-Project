"""Builds the reproducible derived_8.4-ece-router-salvage-1.0.ipynb notebook."""

import json
from pathlib import Path
import uuid

EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = EXP_DIR / "derived_8.4-ece-router-salvage-1.0.ipynb"


def make_cell(cell_type: str, source: str) -> dict:
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {"cell_type": cell_type, "id": str(uuid.uuid4()), "metadata": {},
            "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main() -> None:
    cells = []

    cells.append(make_cell("markdown",
        r"""# Router-Only Salvage: `Clustering_V0_Full_k2` + `Clustering_Backbone54_k2` on ECE
## Freeze experts, fix routing — ECE strictly unseen

The two static-routed MoE models fail on ECE (`RMSE 0.100 / 0.144`, bias `+0.07 / +0.13`)
because the static KMeans router sends `Lost_Meadow (100%)` and `Renton_Home (90%)`
to the wet-mountain expert, while dynamic routers send 100% of ECE rows to the dry
expert. This notebook reuses the SAME frozen experts (fit on WA `trainval` only) and
compares inference-time routing overrides: `as_routed`, `c0_only`, `c1_only`,
`gapi_transplant`, `dynamic_transplant`, `seasonal`, and `margin_fallback`
(ambiguous static rows fall back to the `Global_Single_54` expert).
ECE targets are used for evaluation only. Thresholds come from WA `trainval` only."""))

    cells.append(make_cell("markdown",
        r"""## 1. Load the versioned experiment implementation
The notebook imports the tracked runner instead of duplicating logic, so results
reproduce from a clean checkout. Paths resolve from the repository root."""))

    cells.append(make_cell("code",
        r"""from pathlib import Path
import importlib.util
import sys

cur = Path.cwd().resolve()
while cur != cur.parent:
    if (cur / "data" / "splits").exists() and (cur / "notebooks").exists():
        PROJECT_ROOT = cur
        break
    cur = cur.parent

EXP_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.4-ece-router-salvage-1.0"
sys.path.insert(0, str(EXP_DIR))
RUNNER_PATH = EXP_DIR / "run_salvage.py"
spec = importlib.util.spec_from_file_location("router_salvage", RUNNER_PATH)
salvage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(salvage)
config = salvage.load_configuration()
print(f"Experiment: {config['experiment']['name']}")
print(f"Families: {config['families']}")
print(f"Policies: {config['policies']}")
print(f"Seeds: {config['seeds']}")
print(f"Constraint: {config['experiment']['constraint']}")
"""))

    cells.append(make_cell("markdown",
        r"""## 2. Run the router-only salvage
Fits routers and experts on WA `trainval` only, then scores ECE (`zero` and
`native-missing` inputs) under each routing policy. Per-seed checkpoints resume,
so re-execution is fast when artifacts exist. From a clean checkout this cell
trains (documented timeout 3600s, CPU)."""))

    cells.append(make_cell("code",
        r"""salvage.main([])  # [] = defaults; avoids parsing the kernel's argv
"""))

    cells.append(make_cell("markdown",
        r"""## 3. Routing audit and policy comparison
Checks how many ECE rows each policy redirects, the WA-only margin thresholds,
and the pooled RMSE / bias recovery versus `as_routed`."""))

    cells.append(make_cell("code",
        r"""import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = pd.read_csv(EXP_DIR / "summary.csv")
seed_metrics = pd.read_csv(EXP_DIR / "seed_metrics.csv")
with (EXP_DIR / "routing_audit.json").open(encoding="utf-8") as f:
    audit = json.load(f)
print(f"WA margin thresholds: {audit['wa_thresholds']}")
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (family, sub) in zip(axes, summary.groupby("family")):
    native = sub[sub["ece_input"] == "native"].set_index("policy").reindex(config["policies"])
    ax.bar(native.index, native["rmse_mean"], yerr=native["rmse_std"], capsize=3)
    ax.set_title(f"{family} (native-missing ECE)")
    ax.set_ylabel("pooled RMSE (m3/m3)")
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
(EXP_DIR / "figures").mkdir(exist_ok=True)
fig.savefig(EXP_DIR / "figures" / "policy_rmse_by_family.png", dpi=150)
print("Saved figures/policy_rmse_by_family.png")
"""))

    cells.append(make_cell("markdown",
        r"""## 4. Station-level verdict and stop rule
If `c0_only` does not recover toward the dynamic-router level (`RMSE ~0.05`),
the experts themselves are untransferable and router surgery stops here."""))

    cells.append(make_cell("code",
        r"""import pandas as pd

station = pd.read_csv(EXP_DIR / "station_metrics.csv")
pivot = station.groupby(["family", "ece_input", "policy", "station"], sort=False)["rmse"].mean().unstack("station")
print(pivot.to_string(float_format=lambda v: f"{v:.6f}"))
native = pivot.xs("native", level="ece_input")
for family in config["families"]:
    base = native.loc[(family, "as_routed")].mean()
    fixed = native.loc[(family, "c0_only")].mean()
    print(f"{family}: as_routed mean-station RMSE={base:.6f} -> c0_only={fixed:.6f} "
          f"(delta={fixed - base:+.6f})")
"""))

    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3",
                                            "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
