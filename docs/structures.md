# Structures

DiffHydro builds on a small set of data structures that standardize shapes, metadata, and graph information across modules and pipelines.  
Understanding them upfront avoids subtle dimension mismatches later.

## DataTensor

All tensors flowing through DiffHydro are instances of `xtensor.DataTensor`.  
They carry:

- a multi-dimensional PyTorch tensor (`values`)
- ordered dimension names (`dims`)
- coordinate dictionaries per dimension (`coords`)

Hydrological tensors follow two canonical layouts:

| Symbol | Meaning | Expected dims |
| --- | --- | --- |
| `BST` | batch of spatial nodes through time | `("batch", "spatial", "time")` |
| `BSTV` | same as above with an extra feature axis | `("batch", "spatial", "time", "variable")` |

`diffhydro.structs.ensure_bst_dims` asserts that tensors entering routing/interpolation layers respect this convention so gradients and broadcasting stay consistent.

Example:

```python
from xtensor import DataTensor
import torch

values = torch.rand(8, 1200, 336)     # [batch, spatial, time]
coords = {
    "batch": range(8),
    "spatial": reach_ids,             # pandas Index, numpy array, or list
    "time": pd.date_range("2024-01-01", periods=336, freq="H"),
}
runoff = DataTensor(values, dims=("batch","spatial","time"), coords=coords)
```

Because coordinates persist through slicing and routing, you can always map tensor positions back to reach identifiers or timestamps.

## River Graphs

DiffHydro reuses DiffRoute’s graph objects:

- **`RivTree`** – a single tree-shaped river network. Stores adjacency, topological order, routing parameters, and metadata such as impulse response names.
- **`RivTreeCluster`** – a list-like container of `RivTree` sub-graphs accompanied by `node_ranges` describing how channels are partitioned into clusters. Used when full kernels do not fit into GPU memory; routing happens one cluster at a time with upstream discharges cached between stages.

Both structures keep node indices inside a `pandas.Index` (`nodes_idx`). DiffHydro modules expect the `spatial` coordinate of your tensors to match this index (or a subset).

Create graphs via:

```python
from diffhydro import io

g = io.read_rapid_graph("/path/to/rapid/cfg")          # RivTree
gs = io.read_multiple_rapid_graphs("/path/to/clustered/cfg")  # RivTreeCluster
```

## Catchment Interpolation Weights

The `CatchmentInterpolator` consumes a `weight_df` table indexed by reach id with at least:

- `pixel_idx` – integer index of the gridded forcing cell
- `area_sqm_total` – overlap area between the catchment and each pixel (used as weights)

Internally, the module records `dest_idxs`, `src_idxs`, and normalization buffers as torch tensors so each forward pass boils down to a weighted scatter.  
For huge basins (`>100k` reaches) use `StagedCatchmentInterpolator` which stores per-cluster subsets of the weights and pixel indices through the lightweight `BufferList` container.

## Dataset Helpers

While you can bring your own data pipeline, `diffhydro.pipelines.utils` defines ready-to-use datasets:

- `SimpleTimeSeriesDataset` – sliding windows over a single tensor, useful for toy problems or inference-only setups.
- `BaseDataset` – windows over paired `(x, y)` tensors plus accessors to static attributes (`statics` dict) and the graph (`g`). Pipelines rely on it to keep areas, channel distances, and extra features aligned with each reach.

Typical statics include:

```python
statics = {
    "cat_area": xt.DataTensor(cat_area_tensor, dims=("spatial",), coords={"spatial": reach_ids}),
    "channel_dist": xt.DataTensor(channel_len_tensor, dims=("spatial",), coords={"spatial": reach_ids}),
    "x_stat": xt.DataTensor(catchment_features, dims=("spatial","variable"), coords=...),
}
```

Providing statics unlocks helper functions like `mm_to_m3s` (unit conversion) and additional inputs to parameter networks.

## IO Utilities

The `diffhydro.io` namespace simply re-exports DiffRoute’s RAPID readers and xtensor helpers:

- `read_rapid_graph` / `read_multiple_rapid_graphs`
- `open_dataset`, `open_datatensor`, `read_pickle`, `read_feather`

Use them to load river graphs and multi-dimensional tensors while keeping metadata intact.
