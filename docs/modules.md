# Modules

DiffHydro modules are thin, differentiable wrappers around physical operators.  
Each module consumes/returns `xtensor.DataTensor` objects and plays nicely with autograd, so you can mix and match them inside custom `nn.Module` graphs.

## CatchmentInterpolator

`diffhydro.modules.interp.CatchmentInterpolator` maps gridded forcings (e.g. RAPID runoff pixels) to reach-aligned time series.

Key traits:

- accepts a `RivTree`, a `pixel_runoff` `DataTensor`, and a `weight_df` describing pixel→reach overlaps
- precomputes GPU buffers (`dest_idxs`, `src_idxs`, `weights`) so the forward pass is a single weighted scatter-add
- returns a `DataTensor` with the same batch/time axes but with `spatial` reordered to match `g.nodes_idx`

```python
ci = CatchmentInterpolator(g, pixel_runoff, weight_df).to("cuda")
reach_runoff = ci(pixel_runoff)    # DataTensor [batch, reach, time]
```

In addition to interpolating time series, the module supports kernel manipulation: feeding a `SparseKernel` lets you aggregate impulse responses per reach/pixel pair (useful for visualizing routing paths).

## StagedCatchmentInterpolator

When the basin contains more reaches than you can hold in memory, wrap multiple `CatchmentInterpolator` instances in `StagedCatchmentInterpolator`.

- Construct it with a `RivTreeCluster`, the raw pixel tensor, and the same `weight_df`.
- Internally stores one interpolator per cluster plus the list of pixel indices needed in that cluster.
- Provides helpers such as `read_pixels` and `yield_all_runoffs` so you can stream data cluster-by-cluster, mirroring staged routing.

Usage pattern:

```python
staged_ci = StagedCatchmentInterpolator(gs, pixel_runoff, weight_df)
for idx in range(len(gs)):
    reach_runoff = staged_ci.interpolate_runoff(pixel_runoff, idx)
    discharge_chunk = router(reach_runoff, gs, cluster_idx=idx)
```

## Runoff

`diffhydro.modules.runoff.Runoff` is a wrapper around a configurable LSTM (`LSTMModel`).  
It expects hydrometeorological forcings along the `variable` axis and (optionally) static descriptors, then emits reach-level runoff in mm.

Highlights:

- Input dims: `[batch, spatial, time, variable]` for dynamic forcings, plus `[batch, spatial, variable_static]` for statics.
- Static features are concatenated along the variable axis before feeding the LSTM.
- The internal LSTM runs on flattened `(batch * spatial, time, features)` so gradients can propagate through many time steps.
- The default output activation is `Softplus`, ensuring non-negative runoff, but you can disable it via `softplus=False`.

```python
runoff_model = Runoff(input_size=forcings.sizes["variable"], hidden_size=256)
runoff = runoff_model(inp_dyn=forcings, inp_stat=static_features)
```

## LTIRouter

`diffhydro.modules.routing.LTIRouter` wraps DiffRoute’s `LTIStagedRouter`, exposing routing utilities aligned with DiffHydro conventions.

Capabilities:

- Route entire graphs (`RivTree`) or staged clusters (`RivTreeCluster`) with the same call signature.
- Accepts either graph-native parameters (`g.params`) or user-provided tensors (useful for learning Muskingum/Hayami coefficients).
- Offers staged APIs: `route_clusters_sequence`, `route_one_cluster`, and `_init_upstream_discharges` help stream memory-bound runs.
- Provides diagnostics: `compute_path_irfs` returns sparse kernels per upstream→downstream path, `compute_channel_irfs` exports impulse responses per reach for visualization or regularization.

Typical usage:

```python
router = LTIRouter(max_delay=48, dt=1.0)
discharge = router(reach_runoff, g, params=learned_params)

kernel_coords, kernel_vals = router.compute_path_irfs(g)
```

The module ensures outputs are `DataTensor` instances with preserved dims/coords, so they integrate seamlessly into downstream losses.

## Helper Utilities

- **`BufferList`** – a minimal module that registers a list of tensor buffers and moves them with `.to(device)` calls. Staged interpolators rely on it to keep per-cluster pixel indices on the correct device.
- **Parameter bounds** – `diffhydro.pipelines.utils.PARAMS_BOUNDS` and `infer_param_bounds` describe reasonable ranges (min/max/init) for each supported IRF (`hayami`, `muskingum`, `nash_cascade`, `linear_storage`, `pure_lag`). These feed into sigmoidal parameterizations used by the routing pipelines.
- **Unit conversions** – `mm_to_m3s` and `m3s_to_mm` convert between depth-based runoff and volumetric discharge using reach areas and the forcing temporal resolution.

Together, these modules let you compose hydrological systems in idiomatic PyTorch while retaining domain knowledge (physical units, graph structure, constraints) directly in the layers.
