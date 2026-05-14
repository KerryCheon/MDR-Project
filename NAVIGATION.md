# NAVIGATION.md

> A map of this repo so you don't spend 10 minutes clicking around.
> Everything lives somewhere intentional,  this doc tells you where.

---

## The short version

| I want to... | Go to |
| --- | --- |
| Read the paper | [`paper/`](paper/) |
| Run the data pipeline | [`src/pipeline/`](src/pipeline/) |
| Look at the final model notebooks | [`notebooks/training/`](notebooks/training/) |
| See evaluation results / diagnostics | [`notebooks/evaluation/`](notebooks/evaluation/) |
| Find a specific figure | [`results/figures/`](results/figures/) |
| Read or compile the LaTeX writeup | [`writeup/`](writeup/) |
| Browse EDA and experiment notebooks | [`experiments/`](experiments/) |
| Find the train/val/test data split | [`data/splits/derived_9.0/`](data/splits/derived_9.0/) |
| Run the tests | `pytest tests/` |

---

## Full tree

```
MDR/
├── paper/                    <-- submitted IEEE PDF lives here
├── data/
│   └── splits/
│       ├── derived_9.0/      <-- THE split used for all paper results
│       └── archive/          <-- older split versions (don't use these)
├── src/
│   └── pipeline/             <-- all the data pipeline source code
├── notebooks/
│   ├── training/             <-- best model notebooks (v20.8, v23, v24)
│   │   └── archive/          <-- full version history (v0 → v22)
│   └── evaluation/           <-- eval diagnostics, SHAP, regime analysis
├── experiments/              <-- EDA, correlations, imputation experiments
├── results/
│   └── figures/              <-- every saved figure, organized by type
├── writeup/                  <-- LaTeX source for the paper
├── tests/                    <-- pytest suite
├── models/                   <-- model version history + best artifacts
├── docs/                     <-- (reserved, empty for now)
│
├── README.md                 <-- paper landing page, start here
├── NAVIGATION.md             <-- you are here
├── CITATION.cff              <-- how to cite this work
├── LICENSE                   <-- MIT
├── environment.yml           <-- conda env (name: mdr)
├── pyproject.toml            <-- project metadata + tool config
├── Makefile                  <-- install / train / eval / figures / lint
└── .github/workflows/        <-- CI: tests + lint on every push
```

---

## Where the code lives

> The pipeline that turns raw station data into model-ready features is all in `src/pipeline/`.

```
src/pipeline/
├── main.py           <-- entry point, run this
├── config.yaml       <-- station list and pipeline settings
├── pipes/            <-- processing stages (parse → clean → features → save)
│   ├── parse_pipe.py
│   ├── clean_pipe.py
│   ├── feature_pipe.py
│   ├── satellite_pipe.py
│   ├── weather_pipe.py
│   └── ...
├── imputers/         <-- 12 missing-value imputation strategies
│   ├── knn_temporal.py
│   ├── xgb_model.py
│   ├── voting.py     <-- ensemble imputer
│   └── ...
├── records/          <-- daily observation record handling
├── utils/            <-- config loading, logging, math helpers
├── smoothing/        <-- Whittaker smoother + Fourier transform
└── validation/       <-- input data validation
```

> To run the pipeline:
> ```bash
> PYTHONPATH=. python src/pipeline/main.py                          # all stations
> PYTHONPATH=. python src/pipeline/main.py --station spokane_17_ssw  # one station
> ```

There's also a separate **feature selection framework** at `modeling/` (package: `soilmoist-feature-lab`). That's a standalone tool and has its own `pyproject.toml` and run registry under `modeling/Runs/`.

---

## Where the notebooks live

> Rule of thumb: if it made it into the paper, the notebook is in `notebooks/training/`. Everything else is in `archive/`.

```
notebooks/
├── training/
│   ├── MDR-v24.ipynb                    <-- spatial generalization (final)
│   ├── MDR-v24-main.ipynb               <-- spatial gen. main analysis
│   ├── MDR-v23.1.ipynb                  <-- model survey (XGB vs GB comparison)
│   ├── MDR-v20.8.ipynb                  <-- three-regime + LOSO spatial eval
│   ├── MDR-TemporalSpatial-v2.1.ipynb   <-- temporal-spatial transfer
│   └── archive/                         <-- v0 through v22, all sub-versions
└── evaluation/
    ├── eval.ipynb                        <-- primary eval notebook
    ├── main_eval.ipynb                   <-- full eval pipeline
    ├── regime_separability.ipynb         <-- dry / transition / wet analysis
    ├── regime_feature_importance_top30.ipynb  <-- SHAP by regime
    └── best_model_analysis.ipynb         <-- best checkpoint deep-dive
```

> The full version history (85+ notebooks) is in `notebooks/training/archive/`, organized by version number. You probably don't need those unless you're debugging why a specific version behaved a certain way.

---

## Where the figures live

> All saved figures ended up in `results/figures/`, split by what they show.

```
results/figures/
├── temporal/    <-- time-series predictions, residuals, loss curves, feature importances
├── loso/        <-- leave-one-station-out scatter plots, R² bars, timeseries
├── spatial/     <-- spatial generalization plots (Quinault, TemporalSpatial)
├── shap/        <-- SHAP plots for all 11 feature set iterations (62 files)
└── writeup/     <-- figures used directly in the IEEE manuscript
```

> If you're looking for a specific figure from the paper, check `results/figures/writeup/` first.

---

## Where the data lives

> Processed splits are versioned. Use `derived_9.0`, that's the one the paper used.

```
data/splits/
├── derived_9.0/    <-- canonical split (train: 2017–2020, val: 2021–2022, test: 2023–2025)
└── archive/        <-- base_1.0, base_2.0, derived_1.0 through derived_8.0, unseen ECE data
```

> Raw station data is **not committed** (too large, gitignored). It lives in `Temporal/Pipeline/data/raw/`. USCRN and SNOTEL source files are available from NOAA and NRCS. API caches (satellite + weather JSON) are in `Temporal/Pipeline/data/cache/`.

---

## Where the writeup lives

> The LaTeX source for the paper is in `writeup/`. Figures are pulled from `results/figures/writeup/` at compile time.

```
writeup/
├── main.tex          <-- root document
├── refs.bib          <-- bibliography
├── compile.sh        <-- builds the PDF (runs pdflatex × 3 + bibtex)
├── sections/
│   ├── shared/       <-- data, preprocessing, features (used by both models)
│   ├── base_model/   <-- training, temporal eval, spatial eval, new locations, observations
│   └── three_regime/ <-- training, features, temporal eval, limitations
├── build/            <-- compiled PDF + LaTeX artifacts (gitignored except PDF)
└── pdf/              <-- FinalDraft.pdf snapshot
```

> To recompile: `cd writeup && bash compile.sh`

---

## Where the model artifacts live

> Iterative model development history is in `models/Temporal/`. Each version folder has `model.json` (the XGBoost booster) and `run_metadata.json`.

- Best checkpoint: `models/Temporal/Archive/BEST_MODEL/`, has the actual `.pkl` model weights used for paper results
- Version history with R² scores: [`models/README.md`](models/README.md)
- Final model: **v24** trained on `data/splits/derived_9.0/`

---

## EDA and experiments

> These were one-off investigations during model development. Each has a notebook and usually a PDF report alongside it.

```
experiments/
├── domain_analysis/   <-- rainfall patterns, soil moisture distributions
├── analysis/          <-- general data exploration, heatmaps
├── correlation/       <-- feature-target and inter-feature correlation
├── interpolation/     <-- gap-filling strategy evaluation
└── missing_values/    <-- missingness patterns (ground truth + satellite)
```

---

## Quick start

```bash
# 1. set up the environment
conda env create -f environment.yml
conda activate mdr

# 2. run the pipeline
make train

# 3. run tests
make test

# 4. check linting
make lint
```

> See the root [`Makefile`](Makefile) for all available targets.
> See [`README.md`](README.md) for the paper abstract, key results tables, and citation.
