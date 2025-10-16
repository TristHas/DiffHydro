import torch
import torch.nn as nn

from .. import TimeSeriesThDF

class RunoffConverter(nn.Module):
    def __init__(self, catchment_areas, runoff_std=1, discharge_std=1):
        """ """
        super().__init__(self)
        runoff_coefficient = torch.from_numpy(catchment_areas)*\
                             runoff_std / discharge_std
        self.register_buffer("runoff_conversion", runoff_coefficient.float()[None, :, None])

    def forward(self, runoff):
        """ """
        return TimeSeriesThDF(x.values * self.runoff_conversion, 
                              index=runoff._index, 
                              columns=runoff._columns)

class BaseE2EPipeline(nn.Module):
    def __init__(self, 
                 runoff_model,
                 runoff_converter,
                 routing_model
                ):
        """
        """
        super().__init__()
        self.runoff_model = runoff_model
        self.runoff_converter = runoff_converter
        self.routing_model  = routing_model

    def forward(self, x, g=None):
        """
        """
        runoffs = self.runoff_model(x)
        runoffs = self.runoff_coverter(runoffs)
        return self.routing_model(runoffs, g)
