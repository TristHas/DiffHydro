import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader
import xtensor as xt

from .utils import PARAMS_BOUNDS, format_param_bounds
from .base import BaseModule

from .. import LTIRouter, RivTreeCluster

class LearnedRouter(nn.Module):
    """
    """
    def __init__(
            self,
            irf_name,
            param_model,
            max_delay: int = 32,
            dt: float = 1.0,
            temp_res_h: float = 1.0,
            **routing_kwargs
        ) -> None:
        super().__init__()
        self._init_router(max_delay, dt, **routing_kwargs)
        self._init_buffers(irf_name, temp_res_h)
        self.param_model = param_model

    def _init_router(self, max_delay, dt, **routing_kwargs):
        self.router = LTIRouter(
                          max_delay=max_delay,
                          dt=dt, **routing_kwargs
                    )

    
    def _init_buffers(self, irf_name, temp_res_h):
        param_mins, param_maxs, param_inits = format_param_bounds(irf_name, temp_res_h)
        self.register_buffer("param_init", param_inits)
        self.register_buffer("offset", param_mins)
        self.register_buffer("range", param_maxs - param_mins)

    def _read_params(self, p):
        return torch.sigmoid(self.param_model(p)\
                             + self.param_init)\
                * self.range \
                + self.offset
        
    def read_params(self, g, additional_params=None):
        params = self._read_params(g.params)
        if g.irf_fn == "hayami":
            params = torch.cat([additional_params.unsqueeze(-1), params], dim=-1)
        return params
        
    def forward(self, x, g, 
                additional_params=None,
                cluster_idx=None):
        """
        """
        params = params=self.read_params(g, additional_params)
        return self.router(x, g, params, cluster_idx=cluster_idx)
        
class CalibrationParamModel(nn.Module):
    def __init__(self, g):
        super().__init__()
        n_params = len(PARAMS_BOUNDS[g.irf_fn].columns)
        n_nodes  = len(g.nodes_idx)
        self.params = nn.Parameter(torch.zeros((n_nodes, n_params)))

    def forward(self, *args, **kwargs):
        return self.params

class CalibrationRouter(LearnedRouter):
    """
    """
    def __init__(
            self, g,
            max_delay: int = 32,
            dt: float = 1.0,
            temp_res_h= 1,
            **routing_kwargs
        ) -> None:
        """
        """
        param_model = CalibrationParamModel(g)
        super().__init__(g.irf_fn, param_model,
                        max_delay=max_delay, dt=dt,
                        temp_res_h=temp_res_h)
        self.g = g

class RoutingModule(BaseModule):
    def __init__(self, model, 
                 tr_ds, val_ds,
                 device="cuda:0",
                 batch_size=256,
                 clip_grad_norm=1,
                 lr=10**-2):
        super().__init__(model, tr_ds, val_ds, device, batch_size, clip_grad_norm)
        self.init_optimizer(lr)

    def run_model(self, ds, x):
        return self.model(x, ds.g)

    def seq_run_model(self, ds, x, cluster_idx):
        return self.model(x, ds.g, cluster_idx=cluster_idx)
        
    def init_optimizer(self, lr):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
