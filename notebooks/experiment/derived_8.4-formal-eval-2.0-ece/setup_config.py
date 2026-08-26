import yaml
from pathlib import Path

exp_dir = Path(__file__).resolve().parent
project_root = exp_dir.parents[2]

# Load base config from formal-eval-2.0
with open(project_root / "notebooks/experiment/derived_8.4-formal-eval-2.0/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Update config for 2.0-ece
cfg["data"].pop("spatial_oos", None)
cfg["data"]["spatial_ece"] = {
    "metadata_path": "data/splits/derived_8.4-ece/dataset_metadata.py",
    "splits": {
        "train": "data/splits/derived_8.4-ece/train.csv",
        "val": "data/splits/derived_8.4-ece/val.csv",
        "test": "data/splits/derived_8.4-ece/test.csv",
    }
}

cfg["seeds"]["spatial"] = list(cfg["seeds"]["temporal"]) # 30 seeds

cfg["spatial"] = {
    "save_predictions": True,
    "predictions_dir": "predictions_spatial",
    "models_dir": "models",
    "n_parallel": 8,
    "data_version": 1,
}

# Remove loso section if present
cfg.pop("loso", None)
if "replication" in cfg and "loso_mean_r2" in cfg["replication"]:
    cfg["replication"].pop("loso_mean_r2", None)

header = """# Configuration for derived_8.4-formal-eval-2.0-ece — In-Situ ECE Formal Statistical Evaluation
#
# Publication-oriented statistical evaluation of the two-regime (KMeans k=2) clustering model
# against global baseline and trained gating, evaluated on in-situ ECE sensor spatial generalization.
#
# Protocol summary:
#   - Models and routers trained strictly on the 7 Washington state stations from derived_8.4
#     (trainval: train 2017-2020 + val 2021-2022, 14,608 rows).
#   - Spatial generalization evaluated on derived_8.4-ece (5 in-situ ECE sensor stations across
#     Bellevue Botanical Garden and Renton, WA; 150 rows across 2026-07-20 to 2026-08-19).
#     derived_8.4-ece is COMPLETELY UNSEEN during training.
#   - Reuses the 20 pinned configurations and feature selections (val_selected_deltas.json) from
#     derived_8.4-formal-eval-1.0 to eliminate redundant computation.
#   - 30 random seeds for both temporal (WA test) and spatial (ECE) evaluation.
#
"""

with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
    f.write(header + yaml.dump(cfg, sort_keys=False, indent=2))

with open(exp_dir / ".gitignore", "w", encoding="utf-8") as f:
    f.write("""models/*
predictions/*
predictions_spatial/*
artifacts/*
*.log
*.out
*.err
__pycache__/
""")

print("Successfully created config.yaml and .gitignore in", exp_dir)
