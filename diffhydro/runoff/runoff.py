import torch.nn as nn

from ..structs import DataTensor
from ..structs.time_series import ensure_bst_dims, datatensor_like
from .lstm import LSTMModel

class Runoff(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.core = LSTMModel(**kwargs)

    def forward(self, inp_df: DataTensor) -> DataTensor:
        ensure_bst_dims(inp_df)
        batch, spatial, time = inp_df.shape
        flattened = inp_df.values.reshape(batch * spatial, time, 1)
        y = self.core(flattened)
        reshaped = y.view(batch, spatial, time, -1)
        if reshaped.shape[-1] != 1:
            raise ValueError("Runoff model is expected to output a single feature.")
        reshaped = reshaped.squeeze(-1)
        return datatensor_like(inp_df, reshaped)
