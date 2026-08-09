"""D22: CatchmentInterpolator nan_policy="renormalize" (non-breaking flag)."""
import numpy as np
import pytest
import torch
import xtensor as xt

from diffhydro.modules.interp.cat_interp import CatchmentInterpolator


def _make(nan_policy):
    # 2 nodes: node0 <- pixels {0 (w .25), 1 (w .75)}; node1 <- pixel {2 (w 1)}
    return CatchmentInterpolator.from_tensors(
        dest_idxs=[0, 0, 1], src_idxs=[0, 1, 2], weights=[0.25, 0.75, 1.0],
        out_index=[101, 202], n_pix=3, nan_policy=nan_policy)


def _runoff(vals):
    v = torch.tensor(np.asarray(vals, dtype=np.float32)[None, :, None])  # (B=1, pix, T=1)
    return xt.DataTensor(v, coords={"batch": [0], "spatial": [0, 1, 2], "time": [0]},
                         dims=["batch", "spatial", "time"])


def test_default_propagates():
    out = _make("propagate").interpolate_runoff(_runoff([1.0, np.nan, 3.0]))
    assert np.isnan(out.values[0, 0, 0])
    assert out.values[0, 1, 0] == pytest.approx(3.0)


def test_renormalize_partial():
    out = _make("renormalize").interpolate_runoff(_runoff([1.0, np.nan, 3.0]))
    assert out.values[0, 0, 0] == pytest.approx(1.0)  # weight renorm over pixel 0 only
    assert out.values[0, 1, 0] == pytest.approx(3.0)


def test_renormalize_all_nan_stays_nan():
    out = _make("renormalize").interpolate_runoff(_runoff([np.nan, np.nan, 3.0]))
    assert np.isnan(out.values[0, 0, 0])
    assert out.values[0, 1, 0] == pytest.approx(3.0)


def test_mass_conservation_constant_field():
    for policy in ("propagate", "renormalize"):
        out = _make(policy).interpolate_runoff(_runoff([7.0, 7.0, 7.0]))
        assert np.allclose(out.values[0, :, 0], 7.0)


def test_invalid_policy_rejected():
    with pytest.raises(AssertionError):
        _make("clip")
