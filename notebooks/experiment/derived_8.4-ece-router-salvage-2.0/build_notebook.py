"""Builds the reproducible derived_8.4-ece-router-salvage-2.0.ipynb notebook."""

import json
from pathlib import Path
import uuid

EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = EXP_DIR / "derived_8.4-ece-router-salvage-2.0.ipynb"


def make_cell(cell_type: str, source: str) -> dict:
    lines = source.split("\n")
    # Canonical nbformat: one string per line without trailing newlines, and
    # no empty trailing element.
    cell = {"cell_type": cell_type, "id": str(uuid.uuid4()), "metadata": {},
            "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main() -> None:
    cells = []

    cells.append(make_cell("markdown",
        r"""# Automatic MoE Routing Bandaid on v3: availability gate + soft blend, no global fallback
## `derived_8.4-ece-router-salvage-2.0`

Manual routing overrides (`c0_only`, transplants) close the ECE gap but are not
deployable: they peek at the target to pick the winner. This experiment replaces
them with an **automatic, input-only bandaid** that only activates when routing
inputs are missing/unreliable and never leaves the MoE (every prediction is
`w0*E0 + w1*E1` of the same two frozen regime experts). Experts, routers, margin
cutoffs, gate rule, and softmax temperature are all fit on WA `trainval` /
`train`+`val` only; `derived_8.4_ece_v3` targets are eval-only. `c0_only` is kept
purely as a MANUAL oracle ceiling (`deployable=false`)."""))

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

EXP_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.4-ece-router-salvage-2.0"
sys.path.insert(0, str(EXP_DIR))
RUNNER_PATH = EXP_DIR / "run_auto.py"
spec = importlib.util.spec_from_file_location("router_auto_salvage", RUNNER_PATH)
salvage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(salvage)
config = salvage.load_configuration()
print(f"Experiment: {config['experiment']['name']}")
print(f"Families: {config['families']}")
print(f"Policies: {config['policies']}")
print(f"Deployable: {config['deployable']}")
print(f"Oracle: {config['oracle']}")
print(f"Seeds: {config['seeds']}")
print(f"Constraint: {config['experiment']['constraint']}")
"""))

    cells.append(make_cell("markdown",
        r"""## 2. Run the automatic salvage on v3
Fits routers + experts on WA `trainval` only, calibrates temperature and gate on
WA `train`/`val` only (ECE never touched), then scores v3 under each policy.
Per-seed checkpoints resume, so re-execution is fast when artifacts exist. From a
clean checkout this cell trains (documented timeout 3600s, CPU)."""))

    cells.append(make_cell("code",
        r"""salvage.main([])  # [] = defaults; avoids parsing the kernel's argv
"""))

    cells.append(make_cell("markdown",
        r"""## 3. Policy comparison (deployable vs oracle) + WA calibration
Pooled RMSE / bias per policy on v3. `deployable=false` rows (`c0_only`) are the
manual oracle ceiling — the deployable claim is `auto_*`/`soft_static` vs
`as_routed`. The WA calibration table underneath proves `T` and the gate regime
were selected without ECE."""))

    cells.append(make_cell("code",
        r"""import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = pd.read_csv(EXP_DIR / "summary.csv")
seed_metrics = pd.read_csv(EXP_DIR / "seed_metrics.csv")
calibration = pd.read_csv(EXP_DIR / "wa_calibration.csv")
with (EXP_DIR / "routing_audit.json").open(encoding="utf-8") as f:
    audit = json.load(f)
print(f"WA margin thresholds: {audit['wa_thresholds']}")
print(f"WA medians: {audit['wa_medians']}")
print(f"WA temperatures: {audit['temperatures']}")
print(summary.to_string(index=False))
print(calibration.to_string(index=False))

deployable = summary[summary["deployable"] == True]
order = [p for p in config["policies"] if p in set(deployable["policy"])]
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for ax, (family, sub) in zip(axes, deployable.groupby("family")):
    panel = sub.set_index("policy").reindex(order)
    ax.bar(panel.index, panel["rmse_mean"], yerr=panel["rmse_std"], capsize=3)
    oracle = summary.query("family == @family and policy == 'c0_only'")
    if not oracle.empty:
        ax.axhline(float(oracle["rmse_mean"].iloc[0]), linestyle=":", color="gray",
                   label="c0_only oracle (manual)")
    ax.set_title(f"{family} (v3 ECE, deployable)")
    ax.set_ylabel("pooled RMSE (m3/m3)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
fig.tight_layout()
(EXP_DIR / "figures").mkdir(exist_ok=True)
fig.savefig(EXP_DIR / "figures" / "policy_rmse_deployable_v3.png", dpi=150)
print("Saved figures/policy_rmse_deployable_v3.png")
"""))

    cells.append(make_cell("markdown",
        r"""## 4. Station-level verdict + gate/blend audit
Compares deployable `auto_*` against `as_routed` station by station and checks
the gate fired for the right (input-only) reason with MoE-internal weights."""))

    cells.append(make_cell("code",
        r"""import json
import pandas as pd

station = pd.read_csv(EXP_DIR / "station_metrics.csv")
pivot = station.groupby(["family", "ece_input", "policy", "station"], sort=False)["rmse"].mean().unstack("station")
print(pivot.to_string(float_format=lambda v: f"{v:.6f}"))
v3 = pivot.xs("v3", level="ece_input")
for family in config["families"]:
    base = v3.loc[(family, "as_routed")].mean()
    oracle = v3.loc[(family, "c0_only")].mean()
    for policy in config["deployable"]:
        if policy == "as_routed":
            continue
        val = v3.loc[(family, policy)].mean()
        print(f"{family} {policy}: as_routed={base:.6f} -> {policy}={val:.6f} "
              f"(delta={val - base:+.6f}, gap_vs_oracle={val - oracle:+.6f})")

with (EXP_DIR / "routing_audit.json").open(encoding="utf-8") as f:
    audit = json.load(f)
for entry in audit["seeds"]:
    for family in config["families"]:
        auto = entry["policies"][family]["auto_soft"]
        print(f"seed={entry['seed']} {family} auto_soft: gate_share={auto.get('gate_share'):.3f} "
              f"ambiguous_share={auto.get('ambiguous_share'):.3f} mean_w0={auto.get('mean_w0'):.3f} "
              f"c0_share={auto.get('c0_share'):.3f}")
"""))

    cells.append(make_cell("markdown",
        r"""## 5. Per-station prediction line charts (≤5 lines per chart)
Seed-mean observed vs predicted trajectories from `predictions_v3.csv`: 10
family panels (5 stations × V0 / Backbone: observed + as_routed / auto_soft /
auto_hard + c0_only oracle) plus one deployable overlay. Every panel enforces
the 5-line maximum."""))

    cells.append(make_cell("code",
        r"""import importlib.util

PLOT_PATH = EXP_DIR / "plot_timeseries.py"
plot_spec = importlib.util.spec_from_file_location("auto_plots", PLOT_PATH)
plots = importlib.util.module_from_spec(plot_spec)
plot_spec.loader.exec_module(plots)
plots.main([])

from IPython.display import Image, display
display(Image(str(EXP_DIR / "figures" / "timeseries_v3_auto_overlay.png")))
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
