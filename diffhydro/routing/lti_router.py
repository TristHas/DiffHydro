import itertools
import torch
import torch.nn as nn
import xtensor as xt

from ..structs import ensure_bst_dims

from diffroute import (
    RivTree, RivTreeCluster,
    LTIStagedRouter as LTIStagedRouterCore,
)
from diffroute.irfs import IRF_FN

class LTIRouter(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.core = LTIStagedRouterCore(**kwargs)

    def forward(self, 
                runoff: xt.DataTensor, 
                g, 
                params=None, 
                cluster_idx=None) -> xt.DataTensor:
        """
        """
        if cluster_idx is None:
            router = self.core.model if isinstance(g, RivTree) else \
                     self.core
            discharge = router(runoff.values, g, params) 
            return xt.DataTensor(discharge, dims=runoff.dims, 
                                 coords=runoff.coords,
                                 name="discharge")
        else:
            assert isinstance(g, RivTreeCluster), \
                   f"LTIRouter.forward only accepts cluster_idx \
                   for RivTreeCluster graphs, got {type(g)}"
            return self._forward_seq(runoff, g, 
                                     cluster_idx=cluster_idx, 
                                     params=params)
            
    def _forward_seq(self, runoff: xt.DataTensor, gs, cluster_idx, params=None):
        if isinstance(params, torch.Tensor):
            params = [params[s:e] for s, e in gs.node_ranges[:cluster_idx+1]]

        with torch.no_grad():
            q_init = self._init_upstream_discharges(runoff, gs, cluster_idx, params)
        
        start, end = gs.node_ranges[cluster_idx]
        runoff = runoff.isel(spatial=slice(start, end))
        discharge = self._route_one_cluster(runoff, gs, cluster_idx,
                                            params[cluster_idx], q_init)
        return discharge
        
    ###
    ### Iterative routing procedure helpers
    ###
    def route_clusters_sequence(self, runoff_iter, gs, params=None):
        """
        """
        meta_q = []
        if isinstance(params, torch.Tensor):
            params = [params[s:e] for s, e in gs.node_ranges]

        def xs_tensor_gen(runoff_iter):
            for x in runoff_iter:
                meta_q.append((x.dims, x.coords))
                yield x.values
        
        core_iter = self.core.route_all_clusters_yield(xs_tensor_gen(runoff_iter), 
                                                               gs, params=params)
        
        for discharge in core_iter:
            dims, coords = meta_q.pop(0)
            yield xt.DataTensor(discharge, dims=dims, 
                                coords=coords, name="discharge")
    
    def _route_one_cluster(self,
                          runoff: xt.DataTensor,
                          gs,
                          cluster_idx: int,
                          params=None,
                          transfer_bucket=None):
        """
        """
        q, upstream_q = self.core.route_one_cluster(
            runoff.values, gs, cluster_idx, 
            params, transfer_bucket
        )
        return xt.DataTensor(q, dims=runoff.dims, coords=runoff.coords)
    
    def _init_upstream_discharges(self, runoff, gs, cluster_idx, 
                                 params=None):
        """
        """
        if isinstance(runoff, xt.DataTensor):
            xs_tensor_gen = (runoff.values[:, s:e] for s, e in gs.node_ranges)
        else:
            xs_tensor_gen = (x_df.values for x_df in runoff) 
        return self.core.init_upstream_discharges(
             xs_tensor_gen, gs,
             cluster_idx,
             params=params,
        )

    ###
    ### Kernel helpers
    ###
    def compute_path_irfs(self, g, params=None, inp_channels=None, out_channels=None):
        kernel = self.core.aggregator(g, params)#.to(runoff.device)
        msk = torch.ones_like(kernel.coords[:,0], dtype=torch.bool)
        if out_channels is not None:
            msk *= torch.isin(kernel.coords[:,0], 
                              torch.tensor(g.nodes_idx[inp_channels].values,
                                           device=kernel.coords.device))
        if inp_channels is not None:
            msk *= torch.isin(kernel.coords[:,1], 
                              torch.tensor(g.nodes_idx[inp_channels].values,
                                           device=kernel.coords.device))
        return kernel.coords[msk], kernel.vals[msk]
        
    def compute_channel_irfs(self, g, params=None, channels=None):
        irf_fn = IRF_FN[g.irf_fn]
        if params is None: params = g.params
        
        irfs= irf_fn(params, 
                     time_window=self.max_delay, 
                     dt=self.dt).squeeze()
        
        if channels is not None:
            msk = torch.tensor(g.nodes_idx[channels].values,
                               device=irfs.device)
            irfs = irfs[msk]
        return irfs

    @property
    def dt(self):
        return self.core.model.aggregator.dt

    @property
    def max_delay(self):
        return self.core.model.aggregator.max_delay
    