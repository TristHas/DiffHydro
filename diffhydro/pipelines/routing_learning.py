import torch
import torch.nn as nn

from .utils import MLP
from .. import LTIStagedRouter, LTIRouter, RivTreeCluster, RivTree

class LearnedRouter(nn.Module):
    """
    """
    def __init__(
            self,
            max_delay: int = 32,
            dt: float = 1.0,
            param_mins = [.005, .0],
            param_maxs = [.25, 1.2],
            mlp = None,
            **routing_kwargs
        ) -> None:
        super().__init__()
        self._init_router(max_delay, dt, **routing_kwargs)
        self._init_buffers(param_mins, param_maxs)
        self.mlp = mlp or MLP(in_dim=3, out_dim=2)

    def _init_router(self, max_delay, dt, **routing_kwargs):
        self.staged_router = LTIStagedRouter(
                                  max_delay=max_delay,
                                  dt=dt, **routing_kwargs
                            )
        self.router = LTIRouter(
                          max_delay=max_delay,
                          dt=dt, **routing_kwargs
                    )

    
    def _init_buffers(self, param_mins, param_maxs):
        self.register_buffer("offset", torch.tensor(param_mins)[None])
        self.register_buffer("range", (torch.tensor(param_maxs) -\
                                       torch.tensor(param_mins))[None])

    def _read_params(self, p):
        return torch.sigmoid(self.mlp(p)) * self.range + self.offset
        
    def read_params(self, g):
        if isinstance(g, RivTree):
            return self._read_params(g.params)
        elif isinstance(g, RivTreeCluster):
            return [self._read_params(g_.params) for g_ in g]
        else:
            raise NotImplementedError
            
    def forward(self, x, g):
        """
        """
        router = self.router if isinstance(g, RivTree) else self.staged_router
        return router(x, g, params=self.read_params(g))

    def init_upstream_discharges(self, x, g, cluster_idx,
                                        display_progress=False):
        assert isinstance(g, RivTreeCluster)
        params = self.read_params(g)
        return self.staged_router.init_upstream_discharges(
                                        x, g, cluster_idx,
                                        params=params, 
                                        display_progress=display_progress
        )

    def route_one_cluster(self, x, g, cluster_idx, transfer_bucket=None):
        assert isinstance(g, RivTreeCluster)
        params = self.read_params(g)
        return self.staged_router.route_one_cluster(x, g, 
                                             cluster_idx=cluster_idx, 
                                             params=params[cluster_idx], 
                                             transfer_bucket=transfer_bucket)