import torch.nn as nn
import xtensor as xt

from .lstm import LSTMModel

class Runoff(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, **kwargs):
        """
        """
        super().__init__()
        self.core = LSTMModel(**kwargs)

    def forward(self, inp_dyn: xt.DataTensor, inp_stat=None) -> xt.DataTensor:
        """Run the LSTM core over any leading (batch-like) dims.

        Every dimension before the trailing (time, variable) dims -- i.e.
        ``batch``, an optional ``ensemble`` axis and ``spatial`` -- is merged
        into the LSTM batch dimension, run, then restored on the output. The
        ``variable`` axis is consumed by the core.
        """
        if inp_stat is None:
            inp = inp_dyn
        else:
            inp = xt.concat([inp_dyn, inp_stat], "variable")
        *lead, time, var = inp.shape

        flat = inp.values.reshape(-1, time, var)

        y = self.core(flat).reshape(*lead, time)

        out_dims = inp.dims[:-1]  # drop the trailing "variable" axis
        return xt.DataTensor(y, dims=out_dims,
                             coords={d: inp.coords[d] for d in out_dims
                                     if d in inp.coords},
                             name="runoff")
