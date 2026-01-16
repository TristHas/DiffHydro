# Examples

The `examples/` folder contains runnable notebooks that demonstrate DiffHydro on real RAPID inputs. They double as integration tests—if a notebook runs end-to-end, your environment is set up correctly.

## Notebook Overview

| Notebook | Focus | Highlights |
| --- | --- | --- |
| `1. RAPID IO.ipynb` | Basic routing workflow | Reads RAPID configuration files, creates a `RivTree`, loads runoff NetCDF files as `DataTensor`s, instantiates `LTIRouter`, performs a GPU routing pass, and plots outlet hydrographs. |
| `2. Routing Memory Management.ipynb` | GPU memory strategies | Shows how to stage routing kernels, chunk time windows, stream catchment interpolation on-the-fly, and transfer data between CPU/GPU when the full routing kernel cannot fit in memory. |
| `3. Large Scale Routing Simulation.ipynb` | Basin-scale simulation | Runs a large RAPID simulation (~20 GB forcings, hundreds of thousands of reaches) with DiffRoute’s kernels, computes maximum discharge statistics, and optionally validates against original RAPID outputs (~700 GB). |
| `4. Routing Calibration.ipynb` | Parameter calibration | Demonstrates the `CalibrationModule`, including sequential sub-basin calibration for extremely large graphs and reporting NSE scores through training. |

Each notebook shares the same structure:

1. **Parameters** – configure download paths, graph selections, cluster sizes, and GPU choices.
2. **Data download** – optional cells fetch RAPID forcing and graph files if they are not already present. Downloads are large; set the `root_path` to a volume with adequate free space.
3. **Pre-processing** – load NetCDF or Parquet files, reshape them to `[batch, spatial, time]`, and build the weight DataFrames used by interpolators.
4. **Execution** – run routing, staged routing, calibration, or validation loops using the modules described in this documentation.
5. **Analysis** – convert CUDA tensors back to pandas objects, compute metrics (NSE, max-Q), and display plots.

## Tips for Using the Notebooks

- **Start with Notebook 1** to confirm your RAPID inputs, CUDA runtime, and xtensor installation are correct.
- **Adopt the memory tricks from Notebook 2** whenever the routing kernel size or runoff tensor exceeds GPU memory. Techniques include sub-cluster routing, temporal tiling, and on-the-fly catchment interpolation.
- **Leverage Notebook 3 as a stress test** before launching long simulations on HPC clusters—it demonstrates how to keep only a subset of ground truth on disk while still computing validation metrics.
- **Follow Notebook 4 when calibrating giant networks**—it illustrates staged calibration where you sweep from upstream to downstream, caching intermediate discharges between clusters.

Feel free to copy-paste notebook sections into scripts or pipelines once you understand the workflow; all key functions are part of the public API described in the other sections of this documentation.
