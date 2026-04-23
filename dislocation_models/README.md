# dislocations

Python library for elastic dislocation modelling, wrapping the Okada (1992) surface deformation model. Designed for use with projected coordinates (metres), primarily NZTM.

## Installation

Requires [okada4py](../okada4py/) to be installed first:

```bash
pip install ../okada4py
pip install .
```

Or in development mode:

```bash
pip install -e .
```

## Modules

### `faults`

Core fault geometry classes.

- **`Patch`** — a single rectangular fault patch defined by centroid, strike/dip, length, and width. Supports multiple construction methods (`from_centroid`, `from_top_dimensions`, `from_top_endpoints`). Computes Okada Green's functions via `patch.greens_functions(x, y)`.
- **`ListricFault`** — a stack of `Patch` objects with varying dip, sharing a common strike and along-strike length. Build by adding patches from the top down.
- **`MultiListric`** — an array of `ListricFault` objects tiled along strike. Supports Laplacian smoothing matrix construction.
- **`MultiPatchFault`** — a collection of arbitrary `Patch` objects with subdivision utilities.

### `displacements`

Surface displacement calculation.

- **`DisplacementTable`** — computes displacements at a set of (x, y) site coordinates from one or more fault patches.
- **`DisplacementGrid`** — same, but on a regular grid defined by x/y ranges or from a bounds specification.

Both classes expose `greens_functions_array(rake)` to return the full Green's function matrix, and `write_displacements_csv` / `write_displacements_shp` for output.

### `utilities`

Standalone geometry utilities:

- `slipvec(strike, dip, rake)` — slip vector azimuth from fault parameters
- `slipdip2rake(strike, dip, slipvec)` — inverse of `slipvec`
- `geopandas_polygon_to_gmt`, `geopandas_linestring_to_gmt`, `geopandas_points_to_gmt` — export GeoPandas geometries to GMT format

### `coastlines`

NZ coastline plotting helpers using a bundled 1:150k polygon dataset (NZTM / EPSG:2193).

- `plot_coast(ax, clip_boundary)` — plot coastlines clipped to a bounding box
- `clip_coast(x1, y1, x2, y2)` — return a clipped `GeoSeries` of coastline polygons

### `practical`

Helpers for the dislocation modelling practical exercise. Includes parameter randomisation, pickle-based answer storage, and convenience plotting functions for 2D displacement profiles.

## Quick example

```python
from dislocations.faults import Patch
from dislocations.displacements import DisplacementGrid
import numpy as np

# Define a single fault patch from its top edge endpoints (NZTM metres)
patch = Patch.from_top_endpoints(
    x1=1_750_000, y1=5_450_000,
    x2=1_800_000, y2=5_400_000,
    dip=45., bottom_z=-15_000.
)

# Compute vertical displacements on a grid for 1 m of pure dip slip
grid = DisplacementGrid.from_bounds(
    patch,
    x_min=1_700_000, x_max=1_850_000, x_step=5_000,
    y_min=5_350_000, y_max=5_500_000,
)
ss_gf, ds_gf = grid.greens_functions_array()
vertical = ds_gf[:, 2].reshape(grid.x.shape)  # 1 m dip slip
```

## Coordinate conventions

- All coordinates are in **metres** (projected, typically NZTM EPSG:2193).
- Depths are **negative** (below sea level): a patch top at the surface is `z = 0`, a 15 km deep bottom is `z = -15000`.
- Strike follows the **right-hand rule**: dip direction = strike + 90°.
- Rake follows Aki & Richards: 0° = left-lateral, 90° = reverse, 180° = right-lateral, 270° = normal.
