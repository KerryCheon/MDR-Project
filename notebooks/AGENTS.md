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

For simplicity, you don't need to use Jupyter notebook; you can just write regular Python scripts to run analysis and generate figures, then present findings in a Markdown file. 

### Interacting with Notebooks

Use `jupyter-mcp-server` for reading, running and modifying notebooks rather than manipulating the string content directly, to ensure proper formatting and metadata handling.

The MCP interface is automatically configured to connect to a local Jupyter Lab. Reuse existing kernel sessions if possible (use `list_kernels` command), avoid starting multiple Jupyter kernels to prevent data out of sync issues.

#### Steps when using MCP to interact with notebooks:

1. `list_kernels` to see if there's an existing kernel session, remember the `id` of the kernel you want to use.
2. `use_notebook` with the notebook path and kernel id to connect to the notebook. This is essentially opening the notebook you want to use.

#### Notes on MCP Commands

- `list_files`: it starts from the `notebooks/` directory, not the project root, so the path should be relative to `notebooks/` (e.g. `training/TemporalDelta-v1/MDR-TD-v1.0.ipynb`). Since there are many notebooks, increase the `limit` parameter to 100 when listing from `/` to ensure you see all notebooks.
- `read_notebook` only works for notebooks you have `use_notebook`d. If it doesn't work, remind the user to check if the jupyter-collaboration extension is enabled.
- Never run `connect_to_jupyter` because it requires a token and can cause confusion.