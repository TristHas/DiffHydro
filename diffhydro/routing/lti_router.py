# routers_wrappers.py
import torch
import torch.nn as nn
import itertools

from ..structs import TimeSeriesThDF
from diffroute import (LTIRouter as LTIRouterCore, 
                       LTIStagedRouter as LTIStagedRouterCore)

class LTIRouter(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.core = LTIRouterCore(**kwargs)

    def forward(self, runoff_df: TimeSeriesThDF, g, params=None) -> TimeSeriesThDF:
        y = self.core(runoff_df.values, g, params) 
        return TimeSeriesThDF(y,
                              columns=runoff_df.columns,
                              index=runoff_df.index)

class LTIStagedRouter(nn.Module):
    """
        Wrapper class around diffroute.LTIStagedRouter
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.core = LTIStagedRouterCore(**kwargs)

    def forward(self, runoff_df: TimeSeriesThDF, g, params=None) -> TimeSeriesThDF:
        """
        """
        return self.route_all_clusters(runoff_df, g, params)
        
    def route_one_cluster(self,
                          x_df: TimeSeriesThDF,
                          gs,
                          cluster_idx: int,
                          params=None,
                          transfer_bucket=None):
        """
        """
        y_c, transfer_bucket = self.core.route_one_cluster(
            x_df.values, gs, cluster_idx, params, transfer_bucket
        )
        out_df = TimeSeriesThDF(y_c, columns=x_df.columns, index=x_df.index)
        return out_df
    
    def route_all_clusters(self, x_df: TimeSeriesThDF, gs, 
                           params=None, 
                           display_progress=False) -> TimeSeriesThDF:
        """
        """
        y = self.core.route_all_clusters(x_df.values, gs, 
                                         params=params, 
                                         display_progress=display_progress)
        return TimeSeriesThDF(y,
                              columns=gs.nodes_idx,
                              index=x_df.index)

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
            yield TimeSeriesThDF(y_tensor, columns=x_df.columns, 
                                 index=x_df.index)

    def init_upstream_discharges(self, xs_df, gs, cluster_idx, 
                                 params=None, 
                                 display_progress=False):
        """
        """
        if isinstance(xs_df, TimeSeriesThDF):
            xs_tensor_gen = (xs_df.values[:,s:e] \
                             for s,e in gs.node_ranges) 
        else:
            xs_tensor_gen = (x_df.values for x_df in xs_df) 
        return self.core.init_upstream_discharges(
             xs_tensor_gen, gs,
             cluster_idx,
             params=params,
             display_progress=display_progress
        )

    