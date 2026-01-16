# Quick Start

Follow this guide to get a minimal DiffHydro pipeline running on your machine.  
The steps mirror the RAPID IO example notebook and highlight where to plug in your own data.

## 1. Install Dependencies

DiffHydro is published on PyPI and only depends on PyTorch, xtensor, and DiffRoute.

```bash
pip install diffhydro
```

To track the latest commits:

```bash
git clone https://github.com/TristHas/DiffHydro.git
cd DiffHydro
pip install -e .
```

You need a recent CUDA build of PyTorch to benefit from GPU routing; CPU-only runs do work but defeat the purpose of DiffRoute’s kernels.

## 2. Prepare Your Data

1. **River network** – convert your RAPID configuration to a `RivTree` or `RivTreeCluster` using `diffhydro.io.read_rapid_graph` or `read_multiple_rapid_graphs`.  
   The resulting object stores reach topology, river parameters, and clustering metadata for staged execution.
2. **Forcing / runoff inputs** – DiffHydro expects tensors as `xtensor.DataTensor` with dimensions `[batch, spatial, time]`.  
   Use `xtensor.open_dataset` / `open_datatensor` to read NetCDF, Parquet, or pickle files and to keep coordinates aligned.
3. **Catchment-to-pixel weights** – build a `weight_df` DataFrame indexed by reach id with at least `pixel_idx` and `area_sqm_total` columns. This drives the catchment interpolator.
4. **Static attributes** – store catchment areas, channel lengths/gradients, and other scalars in the dataset `statics` dict. Pipelines use them to convert mm/day to m³/s or to parameterize MLPs.

```python
from diffhydro import io
from xtensor import open_dataset

rapid_cfg = "/path/to/rapid/input"
g = io.read_rapid_graph(rapid_cfg).to("cuda")        # RivTree

pixel_runoff = open_dataset("runoff.nc").sel(variable="precip") \
                                         .to_datatensor(dims=("batch","spatial","time"))
```

## 3. Instantiate Core Modules

```python
import torch
import pandas as pd
from diffhydro import CatchmentInterpolator, Runoff, LTIRouter
from diffhydro.structs import DataTensor

weight_df = pd.read_parquet("pixel_to_reach_weights.parquet")

catch_interp = CatchmentInterpolator(g, pixel_runoff, weight_df).to("cuda")
runoff_model = Runoff(input_size=pixel_runoff.sizes["variable"], hidden_size=256)
router = LTIRouter(max_delay=48, dt=1.0)

# Move tensors to GPU for efficiency
pixel_runoff = pixel_runoff.to("cuda")
```

### Combine Operators

```python
# Interpolate gridded runoff onto reaches
reach_runoff = catch_interp(pixel_runoff)               # DataTensor [batch, catchments, time]

# Generate learnable runoff (optional, skip if you already have reach runoff)
simulated_runoff = runoff_model(reach_runoff)

# Route discharges through the river network
discharge = router(simulated_runoff, g)
print(discharge)                                        # DataTensor named "discharge"
```

At this stage you can visualize outlet hydrographs, compute NSE scores, or integrate the discharge tensor into downstream differentiable tasks.

## 4. Train with Pipelines

To avoid rewriting training loops, build a dataset and use one of the high-level pipeline modules.

```python
import xtensor as xt
from diffhydro.pipelines.utils import BaseDataset
from diffhydro.pipelines.runoff_routing import RunoffRoutingModel, RunoffRoutingModule
from diffhydro.pipelines.routing import LearnedRouter, RoutingModule
from diffhydro.pipelines.utils import MLP

# Example dataset, keeps coordinates + statics aligned
train_ds = BaseDataset(
    x=forcings_train,                      # DataTensor [batch, spatial, time, variable]
    y=observed_discharge_train,            # DataTensor [batch, spatial, time]
    init_len=7 * 24,                       # warm-up window
    pred_len=14 * 24,                      # prediction window
    g=g,
    statics={"cat_area": cat_area_tensor,
             "channel_dist": channel_distance_tensor,
             "x_stat": static_features}
)
val_ds = BaseDataset(... same structure ...)

# Example: joint runoff + routing learning
param_model = MLP(in_dim=2, out_dim=len(weight_df.columns)-1)
rr_model = RunoffRoutingModel(param_model, input_size=forcings_train.sizes["variable"])

runner = RunoffRoutingModule(rr_model, train_ds, val_ds, device="cuda")
train_history, val_history = runner.train(n_epoch=10, device="cuda")

# For routing-only calibration
router_model = LearnedRouter(
    irf_name=g.irf_fn,
    param_model=nn.Identity(),             # or a learnable MLP
    max_delay=48, dt=1.0, temp_res_h=1.0
)
routing_pipeline = RoutingModule(router_model, train_ds, val_ds)
routing_history, _ = routing_pipeline.train(n_epoch=5)
```

Key pieces handled for you:

- batching and collating `xtensor.DataTensor` windows
- coordinate-aware slicing so outputs match observed reaches
- variance-normalized loss (per reach NSE proxy)
- gradient clipping, schedulers, and evaluation hooks
- sequential cluster training for graphs that do not fit in memory

## 5. Explore the Notebooks

Use the `examples/` notebooks as both documentation and regression tests:

1. **RAPID IO** – load RAPID inputs, run a GPU-accelerated routing pass, and visualize outlets.
2. **Routing Memory Management** – learn staged routing, chunked time windows, and GPU/CPU data streaming strategies.
3. **Large Scale Routing Simulation** – rerun a basin-scale RAPID simulation (20 GB inputs, hundreds of thousands of reaches) and validate max discharge statistics.
4. **Routing Calibration** – optimize routing parameters sequentially when the full basin cannot be calibrated in one go.

Each notebook is self-contained: download helpers fetch RAPID data to a configurable directory, while comments explain how to customize parameters for your basin.
