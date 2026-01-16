# DiffHydro Overview

DiffHydro is a research-first toolkit for assembling, training, and operating end-to-end differentiable hydrological pipelines in PyTorch.  
It wraps the low-level routing primitives from [DiffRoute](https://github.com/TristHas/DiffRoute) with higher-level data structures, neural operators, staging utilities, and ready-to-train pipelines so you can modernize hydrological workflows without rewriting core physics.

DiffHydro treats a hydrological system as a directed pipeline of differentiable operators (catchment interpolation, runoff generation, routing, dams, etc.).  
If each operator exposes PyTorch-compatible tensors and gradients, the entire cascade becomes trainable with standard optimizers, unlocking parameter calibration, hybrid ML/physics modelling, and uncertainty quantification on GPUs.

> **Status**: DiffHydro targets research prototyping. Surface APIs are stable enough to reproduce the examples below, but expect iteration as we add modules and datasets.

## Why DiffHydro?

- **Differentiable by design** – every core structure (runoff models, interpolation, routing) speaks PyTorch tensors and autograd so gradients flow from outlets back to forcings and parameters.
- **GPU-first performance** – DiffHydro reuses DiffRoute’s LTI routing kernels and staged execution to scale to river networks with millions of reaches while fitting tight GPU memory envelopes.
- **Composable operators** – mix classic hydrological components with neural surrogates. Replace isolated pieces without rewriting everything else.
- **Research-friendly** – reference pipelines show how to calibrate, pretrain, or fine-tune routing and runoff modules, while remaining hackable for bespoke studies.
- **IO convenience** – rely on `xtensor.DataTensor` containers, RAPID-compatible graph readers, and helper datasets to keep multi-dimensional data aligned and documented.

## Core Ideas

| Idea | Description |
| --- | --- |
| **Data tensors** | `xtensor.DataTensor` tracks `[batch, spatial, time]` semantics everywhere, preventing silent shape/order bugs and keeping metadata (coords, dims) with the tensors. |
| **Graph-backed routing** | River networks are stored as `diffroute.RivTree` or `RivTreeCluster`. Clustering lets you route massive basins cluster-by-cluster when kernels no longer fit in memory. |
| **Differentiable operators** | Modules such as `CatchmentInterpolator`, `Runoff`, and `LTIRouter` expose clean forward methods, can be combined like any PyTorch layer, and were written to remain differentiable and torchscript-friendly. |
| **Pipelines, not scripts** | Instead of rewriting training loops, you instantiate provided pipeline classes (routing-only, runoff+routing, calibration) that already implement batching, variance-normalized objectives, evaluation, and sequential cluster training. |
| **Examples first** | Four notebooks walk through RAPID IO, memory-aware routing, basin-scale simulation, and parameter calibration. They double as executable tests whenever you change the library. |

## Package Layout

```text
diffhydro/
├── structs/      # DataTensor helpers, graph shape utilities
├── modules/      # Differentiable operators (interpolation, routing, runoff)
├── pipelines/    # Training/calibration orchestration
├── io/           # RAPID/xtensor readers
examples/         # RAPID-based tutorials and large-scale experiments
```

The `diffhydro` top-level package re-exports the most frequently used pieces (`Runoff`, `LTIRouter`, `CatchmentInterpolator`, `DataTensor`, etc.) for convenience.

## Relationship to DiffRoute

DiffRoute focuses solely on efficient Linear Time Invariant (LTI) routing. DiffHydro imports those kernels and layers additional abstractions:

- `diffhydro.modules.routing.LTIRouter` wraps `diffroute.LTIStagedRouter` with helpers to loop over clusters, stream kernels, and compute impulse responses per path or per channel.
- Interpolation modules translate gridded forcings to river reaches using GPU-friendly sparse scatter logic.
- Pipelines keep track of upstream discharge buffers, sequential cluster passes, and per-cluster parameter vectors so you can route or calibrate networks that are far larger than a single GPU.

When you only need routing, DiffRoute suffices. The moment you want to add learnable runoff models, couple routing with calibration data, or perform experiment management, DiffHydro is the recommended entry point.

## When to Use DiffHydro

Use DiffHydro if you:

- Want to train routing parameters (Muskingum, Hayami, Nash cascade, etc.) using gradient descent rather than black-box calibration.
- Need to combine neural runoff generation (LSTM-based in the current release) with physics-based routing inside the same backpropagated objective.
- Care about GPU throughput on very large graphs and need ready-made staging and chunking utilities.
- Have RAPID input files and want to accelerate them with minimal glue code.

Skip DiffHydro if you only need deterministic routing or if you prefer black-box calibration frameworks; in those cases DiffRoute or traditional solvers may be simpler.

## Next Steps

1. Head to [Quick Start](quickstart.md) for installation, data preparation, and a minimal composite pipeline.
2. Review [Structures](structures.md) and [Modules](modules.md) to understand the building blocks.
3. Explore pre-built [Pipelines](pipelines.md) and the [Examples](examples.md) notebooks to see DiffHydro in action on RAPID datasets, memory-constrained routing, basin-scale simulation, and calibration tasks.
