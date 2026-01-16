# Pipelines

DiffHydro ships opinionated training/calibration pipelines so you can focus on modelling decisions rather than data loaders and optimizer wiring.  
Each pipeline is an `nn.Module` that wraps a model, datasets, and optimizer logic.

## BaseModule

`diffhydro.pipelines.base.BaseModule` is the foundation for most recent pipelines.

- Accepts a `model`, training/validation datasets, device string, batch size, and gradient clipping configuration.
- Constructs matching `DataLoader`s using the provided datasets and the `collate_fn` that understands `DataTensor` objects.
- Implements:
  - `train(n_epoch)` with optional iteration limits and LR schedulers
  - `train_epoch`, `valid_epoch` using a variance-normalized mean squared error (an NSE proxy)
  - `_extract_full_ts` helpers to reconstruct full-length hydrographs from sliding windows
  - sequential training APIs (`train_epoch_one_cluster`, `train_sequentially`) for staged graphs

To use it, subclass `BaseModule` and implement:

- `run_model(self, ds, x)` – how to run your model on a batch
- `seq_run_model(self, ds, x, cluster_idx)` – optional staged variant
- `init_optimizer(self)` – configure optimizers/schedulers

Most high-level pipelines included in the repo already implement these methods.

## Routing Pipelines

### LearnedRouter

`diffhydro.pipelines.routing.LearnedRouter` wraps `LTIRouter` with a learnable parameter model (often an MLP).  
It constrains parameters via sigmoid activations and empirically derived bounds (`PARAMS_BOUNDS`), ensuring outputs stay within hydrologically plausible ranges.

Features:

- works with both `RivTree` and `RivTreeCluster`
- exposes `read_params`, `route_one_cluster`, and `route_clusters_sequence`
- seamlessly handles multi-parameter IRFs (`muskingum`, `hayami`, etc.) including extra scalars such as wave celerity

### RoutingModule

Pair the model with `diffhydro.pipelines.routing.RoutingModule`, a `BaseModule` subclass that:

- reads graph/statics from the dataset, runs the router, and aligns outputs with observed reaches
- supports sequential training for large clusters via `train_sequentially`
- uses Adam with configurable LR and gradient clipping

Example:

```python
from diffhydro.pipelines.routing import LearnedRouter, RoutingModule
from diffhydro.pipelines.utils import MLP

param_model = MLP(in_dim=2, out_dim=2)
router_model = LearnedRouter(irf_name=g.irf_fn,
                             param_model=param_model,
                             max_delay=48, dt=1.0, temp_res_h=1.0)
pipeline = RoutingModule(router_model, train_ds, val_ds, device="cuda")
pipeline.train(n_epoch=10)
```

## Calibration Pipelines

`diffhydro.pipelines.routing_calibration` targets parameter estimation when you already have reach-level runoff.

- **`CalibrationRouter`** – holds per-node learnable tensors (one parameter vector per reach). Handles both full graphs and staged clusters, initializing upstream discharges when necessary.
- **`CalibrationModule`** – orchestrates stochastic NSE-driven calibration. Provides:
  - `train_one_epoch` / `callibrate` (sic) for simultaneous calibration
  - `train_one_iter_one_cluster` / `calibrate_staged_sequentially` for cases where only a sub-basin fits in memory
  - built-in NSE loss via `diffhydro.utils.nse_fn`

This module mirrors the “Routing Calibration” notebook, where the dataset supplies paired `(runoff, discharge)` tensors and the optimization focuses on routing parameters.

## Runoff + Routing Pipelines

When you need to learn runoff generation and routing jointly, use the pipelines defined in `diffhydro.pipelines.runoff_routing`.

- **`RunoffRoutingModel`** – composes a `Runoff` LSTM (mm output) with a `LearnedRouter`. It automatically converts mm to m³/s using catchment areas and the forcing temporal resolution.
- **`RunoffRoutingModule`** – extends `BaseModule` to:
  - read statics (`cat_area`, `channel_dist`, optional `x_stat`) from the dataset
  - optimize routing and runoff submodules with separate learning rates/weight decay
  - optionally apply LR schedulers independently on both parameter groups

Training looks like:

```python
from diffhydro.pipelines.runoff_routing import RunoffRoutingModel, RunoffRoutingModule
from diffhydro.pipelines.utils import MLP

param_model = MLP(in_dim=2, out_dim=2)
model = RunoffRoutingModel(param_model,
                           input_size=forcings.sizes["variable"],
                           irf_name=g.irf_fn,
                           temp_res_h=1.0)
pipeline = RunoffRoutingModule(model, train_ds, val_ds, device="cuda")
pipeline.train(n_epoch=20)
```

Internally, the module collates windows, applies the LSTM runoff generator, converts to discharge units, and routes through the graph before computing the NSE-style loss against observed discharge.

## Experimental / Legacy Pipelines

The `routing_runoff_learning.py` file contains earlier pipeline prototypes that still illustrate useful ideas:

- manual handling of `StridedStartSampler` to reduce overlap in validation windows
- utility functions for exporting predictions/targets as long-form `DataTensor`s
- combined routing calibration with additional per-reach parameters (`additional_params`)

Refer to this module if you need fine-grained control over evaluation streams or want to port legacy experiments.

## Unit Conversion & Losses

All pipelines rely on helpers defined in `diffhydro.pipelines.utils`:

- `mm_to_m3s` / `m3s_to_mm` – convert between depth and discharge using the cat area tensor and temporal resolution (hours).
- `format_param_bounds` – scales IRF parameter ranges when temporal resolutions differ from hourly data.
- `collate_fn` – concatenates `DataTensor` batches while preserving metadata. Use it whenever you build custom `DataLoader`s.
- `nse_fn` (`diffhydro.utils`) – computes a memory-efficient Nash–Sutcliffe Efficiency score directly on PyTorch tensors, used as either the training objective or a metric.

By composing these building blocks, you can cover workflows ranging from “stochastic Muskingum calibration” to “joint neural runoff and routing learning” with only a few dozen lines of glue code.
