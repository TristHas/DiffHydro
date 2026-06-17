"""Stochastic LSTM runoff modules that emit an ensemble axis.

Both modules take an input *without* an ensemble dimension
(``[batch, spatial, time, variable]``) and return runoff *with* one
(``[batch, ensemble, spatial, time]``), so they slot into ``RRModel`` in place
of the deterministic :class:`~diffhydro.modules.runoff.runoff.Runoff` and let the
ensemble axis flow through the (ensemble-aware) router.

Two complementary sampling strategies:

* :class:`LSTM_inp_sampler` -- *input sampling*: a random seed channel is
  appended to the LSTM inputs (Karan's method). All ensemble members are run in
  a **single batched LSTM call** (the ensemble axis is folded into the LSTM
  batch), not a Python loop over seeds.
* :class:`LSTM_out_sampler` -- *output sampling*: one deterministic LSTM is run
  and a multiplicative distribution (log-normal, or Gaussian with std
  proportional to the mean) is placed on its output. Applying the multiplicative
  factor at the runoff level is, for the *joint* (shared-across-catchments) case,
  equivalent to applying it at the routed output -- routing is linear -- so the
  joint output-sampler reproduces the classic log-normal baseline.

For both, ``joint`` vs ``indep`` is controlled by ``shared_across_catchments``:
the per-member seed/factor is shared across all catchments (joint) or drawn
independently per catchment (indep). ``use_quantiles`` swaps the random draw for
the deterministic standard-normal quantile grid ``Phi^{-1}((s+0.5)/S)`` (which is
the natural choice for the joint case; with ``indep`` the grid is shared per
catchment and so coincides with the joint seeds). ``run_deterministic`` collapses
the module to a single member: a zero seed (input sampler) or the bare mean
(output sampler).
"""
import numpy as np
import torch
import torch.nn as nn
import xtensor as xt

from .lstm import LSTMModel

_OUT_DIMS = ("batch", "ensemble", "spatial", "time")


def _quantile_z(S, device, dtype):
    """Standard-normal quantiles at the S equal-probability bin midpoints."""
    q = (torch.arange(S, device=device, dtype=dtype) + 0.5) / S
    return torch.special.ndtri(q)


def _merge_input(inp_dyn, inp_stat):
    inp = inp_dyn if inp_stat is None else xt.concat([inp_dyn, inp_stat], "variable")
    if "ensemble" in inp.dims:
        raise ValueError("sampler runoff expects an input without an ensemble axis, "
                         f"got dims {inp.dims}")
    return inp


def _out_tensor(values, inp, S):
    coords = {"batch": inp.coords["batch"],
              "ensemble": np.arange(S),
              "spatial": inp.coords["spatial"],
              "time": inp.coords["time"]}
    return xt.DataTensor(values, dims=_OUT_DIMS, coords=coords, name="runoff")


class LSTM_inp_sampler(nn.Module):
    """Seeded-input ensemble runoff (input sampling), batched over members.

    A noise channel is appended to the LSTM inputs; ``num_ensembles`` members are
    produced in one LSTM call by folding the ensemble axis into the LSTM batch.

    Args:
        input_size: Number of base (dynamic + static) input features; the noise
            channel is added internally, so the LSTM sees ``input_size + 1``.
        num_ensembles: Number of ensemble members to generate.
        seed_std: Standard deviation of the sampled seed value.
        shared_across_catchments: If True (joint), every catchment of a given
            member shares the seed; if False (indep), the seed is drawn
            independently per catchment.
        use_quantiles: Use the deterministic standard-normal quantile grid for
            the seeds instead of random draws.
        run_deterministic: Generate a single member with the seed set to zero for
            all catchments (disables the stochastic run).
    """
    def __init__(self, input_size, num_ensembles=8, seed_std=2.0,
                 shared_across_catchments=True, use_quantiles=False,
                 run_deterministic=False, hidden_size=256, num_layers=1,
                 output_size=1, softplus=True):
        super().__init__()
        self.core = LSTMModel(input_size + 1, hidden_size=hidden_size,
                              output_size=output_size, num_layers=num_layers,
                              softplus=softplus)
        self.num_ensembles = num_ensembles
        self.seed_std = seed_std
        self.shared_across_catchments = shared_across_catchments
        self.use_quantiles = use_quantiles
        self.run_deterministic = run_deterministic

    def _seeds(self, batch, spatial, device, dtype):
        """Per-member seed values, shape ``[batch, S, spatial]``."""
        if self.run_deterministic:
            return torch.zeros(batch, 1, spatial, device=device, dtype=dtype)
        S = self.num_ensembles
        if self.shared_across_catchments or self.use_quantiles:
            z = (_quantile_z(S, device, dtype) if self.use_quantiles
                 else torch.randn(S, device=device, dtype=dtype))
            return (z * self.seed_std).view(1, S, 1).expand(batch, S, spatial)
        z = torch.randn(batch, S, spatial, device=device, dtype=dtype)
        return z * self.seed_std

    def forward(self, inp_dyn, inp_stat=None):
        inp = _merge_input(inp_dyn, inp_stat)
        batch, spatial, time, var = inp.shape
        S = 1 if self.run_deterministic else self.num_ensembles

        x = inp.values.unsqueeze(1).expand(batch, S, spatial, time, var)
        noise = self._seeds(batch, spatial, inp.values.device, inp.values.dtype)
        noise = noise[..., None, None].expand(batch, S, spatial, time, 1)
        x = torch.cat([x, noise], dim=-1)                      # [b, S, s, t, var+1]

        flat = x.reshape(-1, time, var + 1)
        y = self.core(flat).reshape(batch, S, spatial, time)
        return _out_tensor(y, inp, S)


class LSTM_out_sampler(nn.Module):
    """Output-distribution ensemble runoff (output sampling).

    One deterministic LSTM produces the mean runoff; a per-member multiplicative
    factor is then applied. ``run_deterministic`` returns the bare mean.

    Args:
        input_size: Number of base (dynamic + static) input features.
        num_ensembles: Number of ensemble members to generate.
        sigma: Spread parameter of the multiplicative distribution.
        distribution: ``"lognormal"`` -- factor ``exp(sigma * z)`` (the classic
            baseline); ``"gaussian_prop"`` -- factor ``1 + sigma * z`` (Gaussian
            whose std is proportional to the mean).
        shared_across_catchments: If True (joint), one factor per member shared
            across all catchments -- equivalent, after linear routing, to a
            multiplicative factor on the output; if False (indep), an
            independent factor per catchment.
        use_quantiles: Use the deterministic standard-normal quantile grid
            instead of random draws.
        run_deterministic: Generate a single member equal to the mean.
    """
    DISTRIBUTIONS = ("lognormal", "gaussian_prop")

    def __init__(self, input_size, num_ensembles=20, sigma=0.1,
                 distribution="lognormal", shared_across_catchments=True,
                 use_quantiles=True, run_deterministic=False,
                 hidden_size=256, num_layers=1, output_size=1, softplus=True):
        super().__init__()
        if distribution not in self.DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {self.DISTRIBUTIONS}, "
                             f"got {distribution!r}")
        self.core = LSTMModel(input_size, hidden_size=hidden_size,
                              output_size=output_size, num_layers=num_layers,
                              softplus=softplus)
        self.num_ensembles = num_ensembles
        self.sigma = sigma
        self.distribution = distribution
        self.shared_across_catchments = shared_across_catchments
        self.use_quantiles = use_quantiles
        self.run_deterministic = run_deterministic

    def _factor(self, batch, spatial, device, dtype):
        """Per-member multiplicative factor, shape ``[batch, S, spatial]``."""
        if self.run_deterministic:
            z = torch.zeros(batch, 1, spatial, device=device, dtype=dtype)
        else:
            S = self.num_ensembles
            if self.shared_across_catchments or self.use_quantiles:
                base = (_quantile_z(S, device, dtype) if self.use_quantiles
                        else torch.randn(S, device=device, dtype=dtype))
                z = base.view(1, S, 1).expand(batch, S, spatial)
            else:
                z = torch.randn(batch, S, spatial, device=device, dtype=dtype)
        if self.distribution == "lognormal":
            return torch.exp(self.sigma * z)
        return 1.0 + self.sigma * z          # gaussian_prop

    def forward(self, inp_dyn, inp_stat=None):
        inp = _merge_input(inp_dyn, inp_stat)
        batch, spatial, time, var = inp.shape

        flat = inp.values.reshape(batch * spatial, time, var)
        mean = self.core(flat).reshape(batch, spatial, time)

        factor = self._factor(batch, spatial, mean.device, mean.dtype)   # [b, S, s]
        members = mean.unsqueeze(1) * factor[..., None]                  # [b, S, s, t]
        S = members.shape[1]
        return _out_tensor(members, inp, S)
