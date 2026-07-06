## Notebooks

Model notebooks live in `notebooks/training/`. Top-level files are the current best models; versioned subdirectories hold newer experiment lines that bypass `archive/`:

- `MDR-TemporalSpatial-v2.1.ipynb` — temporal-spatial transfer
- `MDR-v25.ipynb` — latest model version
- `TemporalDelta-v0/MDR-TD-v0.ipynb` — temporal-delta modeling, v0
- `TemporalDelta-v1/MDR-TD-v1.0.ipynb`, `MDR-TD-v1.1.ipynb` — temporal-delta modeling, v1
- `Temporal-v20/MDR-v20.4.1-portable.ipynb`, `MDR-v20.4.2.ipynb`, `MDR-v20.5-portable.ipynb` — v20 portable/new variants
- `Temporal-v21/MDR-v21.4-portable.ipynb`, `MDR-v21.5.ipynb` — v21 portable/new variants
- `Temporal-v22/MDR-v22.3-portable.ipynb` — v22 portable variant

The full version history (v1–v24) is in `notebooks/training/archive/`, organized by version number. If a user mentions a version like "v20" or "v22", check both `notebooks/training/Temporal-vXX/` (newer active lines) and `notebooks/training/archive/vXX/` (history). List the directory contents to see what's available; there are nested subdirectories (e.g. `archive/v20/v20.1/MDR-v20.1.ipynb`).

### Experiments
New experiments live it `notebooks/experiment/`. Each experiment should have its own subdirectory (e.g. `derived_8.1-data-exploration/`).

For simplicity, you don't need to use Jupyter notebook for short-running experiments (i.e. less than 5 minutes). In that case, you can just write regular Python scripts to run analysis and generate figures, then present findings in a Markdown file. 

## Working with Notebooks (.ipynb files)

When the user asks to read, edit, execute, or work with .ipynb files, use the notebook-cli skill, which provides the `nb` command-line tool. Do not use the built-in Read/Write tools for `.ipynb` files.
Note: you need to run `nb execute` inside `notebooks/` directory, otherwise the packages might not be able to be found.

## Runtime
The dependencies are managed with `uv`, so don't use `python` directly as the global installation might not have all the packages. You should use `uv run` to run any python scripts.

## Best Practices
### Value user's time:
- Check for potential errors before start long-running executions
- Enable accelerations like CUDA when possible
- For long running scripts (i.e. more than 5 minutes; if you're training models in a loop then it must be exceed that), must create checkpoints and saving results incrementally to prevent lossing all progress.
- Must set timer/timeout then executing code, they will take longer than you expected