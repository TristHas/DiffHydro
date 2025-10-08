import torch
import torch.nn as nn
import itertools

from ..structs import TimeSeriesThDF
from .lstm import LSTMModel

class Runoff(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.core = LSTMModel(**kwargs)

    def forward(self, inp_df: TimeSeriesThDF) -> TimeSeriesThDF:
        y = self.core(inp_df.values) 
        return TimeSeriesThDF(y,
                              columns=inp_df.columns,
                              index=inp_df.index)
