# DiffHydro Overview

`diffhydro` assembles the core routing primitives from `diffroute` into a reusable pipeline for differentiable hydrology. It introduces higher-level building blocks that operate on labelled time-series data and orchestrate runoff generation, spatial interpolation, and routing.

## Core Modules
- `diffhydro.structs.DataTensor`: a thin alias around `xtensor.DataTensor` with dims `("batch", "spatial", "time")` for hydrometeorological signals.
- `diffhydro.interp.CatchmentInterpolator`: maps gridded forcings (e.g. remote-sensing pixels) into catchment space with area-weighted interpolation.
- `diffhydro.runoff.Runoff`: wraps neural runoff estimators (currently LSTM-based) and returns `DataTensor` objects.
- `diffhydro.routing.LTIRouter` and `diffhydro.routing.LTIStagedRouter`: thin adapters around the `diffroute` routers that accept and return structured time series.
- `diffhydro.utils`: evaluation helpers such as the Nash–Sutcliffe efficiency (`nse_fn`) and a GPU-friendly profiler.

## Design Highlights
- **Differentiable end-to-end**: every stage (runoff → interpolation → routing) is implemented in PyTorch, so gradients flow across the entire pipeline.
- **Batch-first API**: the `DataTensor` wrapper keeps time indices and catchment IDs alongside tensors, simplifying alignment with pandas data sources.
- **GPU aware**: optional CUDA interop via cuDF/CuPy enables zero-copy IO when available.
- **Composable**: mix-and-match runoff models, interpolation schemes, and routing strategies while reusing a consistent interface.

Continue to the quickstart to wire these pieces together for a minimal differentiable hydrological experiment.
