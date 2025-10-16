import torch
import torch.nn as nn
from .. import LTIStagedRouter, LTIRouter, RivTreeCluster, RivTree

class CalibrationRouter(nn.Module):
    """
    """
    def __init__(
            self, riv_trees,
            max_delay: int = 32,
            dt: float = 1.0,
            param_mins = [.005, .0],
            param_maxs = [.25, 1.2],
            **routing_kwargs
        ) -> None:
        """
        """
        super().__init__()
        self.g = riv_trees # gs is fixed, so it is kept as attribute.
        self._init_router(max_delay, dt, **routing_kwargs)
        self._init_buffers(param_mins, param_maxs)

    def _init_router(self, max_delay, dt, **routing_kwargs):
        if isinstance(self.g, RivTreeCluster):
            self.staged = True
            self.router = LTIStagedRouter(
                              max_delay=max_delay,
                              dt=dt, **routing_kwargs
                        )
            self.params = nn.ParameterList([nn.Parameter(g.params) \
                                            for g in self.g])
            with torch.no_grad(): 
                for p in self.params: p[:]=0

        elif isinstance(self.g, RivTree):
            self.staged = False
            self.router = LTIRouter(
                  max_delay=max_delay,
                  dt=dt, **routing_kwargs
            )
            self.params = nn.Parameter(self.g.params)
            with torch.no_grad(): 
                self.params[:]=0
        else:
            raise NotImplementedError

    def _init_buffers(self, param_mins, param_maxs):
        self.register_buffer("offset", torch.tensor(param_mins)[None])
        self.register_buffer("range", (torch.tensor(param_maxs) -\
                                       torch.tensor(param_mins))[None])

    def _read_param(self, p):
        """
        """
        return torch.sigmoid(p) * self.range + self.offset
        
    def read_params(self):
        """
        """
        if self.staged:
            return [self._read_param(p) for p in self.params]
        else:
            return self._read_param(self.params)
            
    def forward(self, x, *args):
        """
        """
        return self.router(x, self.g, params=self.read_params())

    def init_upstream_discharges(self, x, cluster_idx,
                                        display_progress=False):
        assert self.staged, "Non-staged pipeline can not init_upstream_discharges"
        return self.router.init_upstream_discharges(
                                        x, self.g, cluster_idx,
                                        params=self.read_params(), 
                                        display_progress=display_progress
        )

    def process_one_cluster(self, x, cluster_idx, transfer_bucket=None, cat=None):
        assert self.staged, "Non-staged pipeline can not process_one_cluster"
        params = self.read_params()
        output = self.router.route_one_cluster(x,
                                               gs=self.g, 
                                               cluster_idx=cluster_idx, 
                                               params=params[cluster_idx], 
                                               transfer_bucket=transfer_bucket)
        return output

    def format_calibrated_params(self):
        raise NotImplementedError