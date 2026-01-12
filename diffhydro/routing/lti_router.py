import itertools
import torch.nn as nn
import xtensor as xt

from ..structs import ensure_bst_dims

from diffroute import (
    RivTree,
    LTIRouter as LTIRouterCore,
    LTIStagedRouter as LTIStagedRouterCore,
)

class LTIRouter(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.core = LTIRouterCore(**kwargs)
        self.staged_core = LTIStagedRouterCore(**kwargs)

    def forward(self, runoff: xt.DataTensor, g, params=None) -> xt.DataTensor:
        """
        """
        router = self.core if isinstance(g, RivTree) else self.staged_core
        discharge = router(runoff.values, g, params) 
        return xt.DataTensor(discharge, dims=runoff.dims, coords=runoff.coords)

    ###
    ### Iterative routing procedure helpers
    ###
    def route_clusters_sequence(self, runoff_iter, gs, params=None):
        """
        """
        meta_q = []
        
        def xs_tensor_gen(runoff_iter):
            for x in runoff_iter:
                meta_q.append((x.dims, x.coords))
                yield x.values
        
        core_iter = self.staged_core.route_all_clusters_yield(xs_tensor_gen(runoff_iter), 
                                                               gs, params=params)
        
        for discharge in core_iter:
            dims, coords = meta_q.pop(0)
            yield xt.DataTensor(discharge, dims=dims, coords=coords)
    
    def route_one_cluster(self,
                          runoff: xt.DataTensor,
                          gs,
                          cluster_idx: int,
                          params=None,
                          transfer_bucket=None):
        """
        """
        y_c, transfer_bucket = self.staged_core.route_one_cluster(
            runoff.values, gs, cluster_idx, 
            params, transfer_bucket
        )
        return xt.DataTensor(y_c, dims=runoff.dims, coords=runoff.coords)
    
    def init_upstream_discharges(self, runoff, gs, cluster_idx, 
                                 params=None):
        """
        """
        if isinstance(runoff, xt.DataTensor):
            xs_tensor_gen = (runoff.values[:, s:e] for s, e in gs.node_ranges)
        else:
            xs_tensor_gen = (x_df.values for x_df in runoff) 
        return self.staged_core.init_upstream_discharges(
             xs_tensor_gen, gs,
             cluster_idx,
             params=params,
        )

LTIStagedRouter = LTIRouter