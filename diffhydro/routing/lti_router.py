# routers_wrappers.py
import torch.nn as nn
import itertools

from ..structs import DataTensor
from ..structs.time_series import (
    ensure_bst_dims,
    datatensor_like,
    to_coord_sequence,
)
from diffroute import (
    LTIRouter as LTIRouterCore,
    LTIStagedRouter as LTIStagedRouterCore,
)

class LTIRouter(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.core = LTIRouterCore(**kwargs)

    def forward(self, runoff_df: DataTensor, g, params=None) -> DataTensor:
        ensure_bst_dims(runoff_df)
        y = self.core(runoff_df.values, g, params)
        return datatensor_like(runoff_df, y)

class LTIStagedRouter(nn.Module):
    """
        Wrapper class around diffroute.LTIStagedRouter
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.core = LTIStagedRouterCore(**kwargs)

    def forward(self, runoff_df: DataTensor, g, params=None) -> DataTensor:
        """
        """
        return self.route_all_clusters(runoff_df, g, params)
        
    def route_one_cluster(self,
                          x_df: DataTensor,
                          gs,
                          cluster_idx: int,
                          params=None,
                          transfer_bucket=None):
        """
        """
        y_c, transfer_bucket = self.core.route_one_cluster(
            x_df.values, gs, cluster_idx, params, transfer_bucket
        )
        return datatensor_like(x_df, y_c)
    
    def route_all_clusters(self, x_df: DataTensor, gs,
                           params=None, 
                           display_progress=False) -> DataTensor:
        """
        """
        y = self.core.route_all_clusters(x_df.values, gs, 
                                         params=params, 
                                         display_progress=display_progress)
        spatial_coords = to_coord_sequence(gs.nodes_idx)
        return datatensor_like(x_df, y, spatial_coords=spatial_coords)

    def route_all_clusters_yield(self, xs_df, gs, 
                                 params=None, 
                                 display_progress=False):
        """
        """
        xs_for_labels, xs_for_tensors = itertools.tee(xs_df)
        xs_tensor_gen = (x_df.values for x_df in xs_for_tensors)
        core_iter = self.core.route_all_clusters_yield(
            xs_tensor_gen, gs, params=params, 
            display_progress=display_progress
        )
        
        for x_df, y_tensor in zip(xs_for_labels, core_iter):
            yield datatensor_like(x_df, y_tensor)

    def init_upstream_discharges(self, xs_df, gs, cluster_idx, 
                                 params=None, 
                                 display_progress=False):
        """
        """
        if isinstance(xs_df, DataTensor):
            xs_tensor_gen = (xs_df.values[:, s:e] for s, e in gs.node_ranges)
        else:
            xs_tensor_gen = (x_df.values for x_df in xs_df) 
        return self.core.init_upstream_discharges(
             xs_tensor_gen, gs,
             cluster_idx,
             params=params,
             display_progress=display_progress
        )

    
