# DiffHydro Quickstart

The `diffhydro` toolkit glues together runoff generation, pixel-to-catchment interpolation, and routing. The example below builds a synthetic workflow that you can adapt to real data sources.

## 1. Install and import

```bash
pip install torch pandas networkx
pip install ./DiffRoute ./DiffHydro
```

```python
import pandas as pd
import torch
import networkx as nx

from diffhydro.structs import TimeSeriesThDF
from diffhydro.interp import CatchmentInterpolator
from diffhydro.routing import LTIRouter as HydroRouter
from diffroute import RivTree
```

## 2. Describe the river network

Create a simple directed network with per-reach routing parameters. We reuse `diffroute`'s `RivTree` to encode sparse routing metadata that both DiffRoute and DiffHydro understand.

```python
g = nx.DiGraph()
g.add_node(101, tau=18.0)
g.add_node(102, tau=22.0)
g.add_edge(101, 102, delay=3)

riv_tree = RivTree(g, irf_fn="linear_storage", include_index_diag=False)
```

## 3. Create synthetic forcings

Assume you ingested gridded forcings for three pixels over a one-week horizon. Wrap them in a `TimeSeriesThDF` so metadata (pixel IDs, timeline) travels with the tensor.

```python
hours = pd.date_range("2024-01-01", periods=168, freq="H")
pixels = ["pixel_a", "pixel_b", "pixel_c"]
forcing_tensor = torch.rand(1, len(pixels), len(hours))  # [batch=1, pixels, time]
pixel_series = TimeSeriesThDF(forcing_tensor, columns=pixels, index=hours)
```

## 4. Map pixels to catchments

Provide area-weighted mappings from source pixels to catchments. Each row corresponds to a contributing pixel for a catchment.

```python
weight_df = pd.DataFrame(
    {
        "pixel_idx": ["pixel_a", "pixel_b", "pixel_b", "pixel_c"],
        "area_sqm_total": [0.6, 0.4, 0.7, 0.3],
    },
    index=pd.Index([101, 101, 102, 102], name="catchment_id"),
)

interp = CatchmentInterpolator(riv_tree, pixel_series, weight_df)
catchment_runoff = interp(pixel_series)
```

`catchment_runoff` is still a `TimeSeriesThDF`, now aligned to the catchment IDs of the routing graph.

## 5. Route the flows

Instantiate the hydrological router wrapper (which internally calls `diffroute.LTIRouter`) and pass the catchment runoffs along with the `RivTree`.

```python
router = HydroRouter(max_delay=48, dt=1)
discharge = router(catchment_runoff, riv_tree)
```

The result is a `TimeSeriesThDF` with routed discharge per catchment.

## Where to go next
- Replace the synthetic forcings with outputs from a neural runoff model (see `diffhydro.runoff.Runoff`).
- Use `diffhydro.routing.LTIStagedRouter` when the river network is clustered into sub-graphs.
- Chain the pipeline into a training loop and compute metrics such as Nash–Sutcliffe efficiency via `diffhydro.utils.nse_fn`.
