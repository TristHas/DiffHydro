import torch.nn as nn

from ..structs import DataTensor, ensure_bst_dims
from .lstm import LSTMModel

class Runoff(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.core = LSTMModel(**kwargs)

    def forward(self, dyn_inp: DataTensor, stat_inp=None) -> DataTensor:
        batch, spatial, time, var = dyn_inp.shape
        flattened = dyn_inp.values.reshape(batch * spatial, time, var)
        y = self.core(flattened)
        reshaped = y.view(batch, spatial, time, 1)
        reshaped = reshaped.squeeze(-1)

        out_dims = dyn_inp.dims[:3]
        return DataTensor(reshaped, dims=out_dims, 
                          coords={d:dyn_inp.coords[d] \
                                  for d in out_dims})
