# Dislocation Modelling Practical

Teaching materials and supporting Python library for an elastic dislocation modelling practical exercise. Students explore how fault geometry and slip parameters control surface deformation patterns, using the Okada (1992) analytical model.

## Repository contents

```
dislocation_models/   Python package (fault geometry, Green's functions, displacement grids)
practical/            Student-facing notebooks and answer pickle files
dislocation_practical.ipynb          Original practical (uses external Binder sandbox)
dislocation_practical_lookup.ipynb   Self-contained version with built-in interactive GUI
```

### Practical notebooks

There are two versions of the practical at the top level:

- **`dislocation_practical.ipynb`** — links out to a hosted interactive sandbox on mybinder.org for Part 1, then runs local dislocation models for Parts 2 and 3.
- **`dislocation_practical_lookup.ipynb`** — fully self-contained; the Part 1 sandbox runs locally via ipywidgets. Use this if Binder is unavailable or slow.

The `practical/` subdirectory contains revised versions of the notebook with improved instructions and formatted answer cells.

### `dislocation_models`

The supporting Python library. See [`dislocation_models/README.md`](dislocation_models/README.md) for full documentation, installation instructions, and a usage example.

## Setup

Install [okada4py](../okada4py/) first, then the `dislocations` package:

```bash
uv pip install ../okada4py
uv pip install -e dislocation_models
```

Then launch JupyterLab:

```bash
jupyter lab
```
