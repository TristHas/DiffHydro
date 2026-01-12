import numpy as np
import pandas as pd

import diffhydro as dh
import xtensor as xt

from diffroute.graph_utils import define_schedule
from diffroute.io import _read_rapid_graph, read_rapid_graph

def load_single_vpu(root, vpu, 
                    plength_thr=None, 
                    node_thr=None,
                    device="cpu"):
    """
    """
    rapid_dir = root / "geoglows" / 'configs' / vpu
    vpu_config_path = root / "geoglows" / 'configs'
    runoff_path = root / "geoglows" / "daily_sparse_runoff.feather"
    discharge_path = root / "geoglows" / "retro_feather" / f"{vpu}.feather"
    interp_weight_path = root / "geoglows" / "interp_weight.feather"
    
    g = read_rapid_graph(rapid_dir,
                         plength_thr=plength_thr,
                         node_thr=node_thr).to(device)
    
    cat_interp_df = pd.read_feather(interp_weight_path).set_index("river_id")
    pix_idxs = cat_interp_df.loc[g.nodes]["pixel_idx"].unique()
    
    pixel_runoff = xt.read_feather(runoff_path, dims=["time", "spatial"])\
                     .sel(time=slice(None, "2019"))\
                     .expand_dims("batch")\
                     .to(device)\
                     .transpose("batch", "spatial", "time") / (3600. * 24)
    
    cat = dh.CatchmentInterpolator(g, pixel_runoff, cat_interp_df).to(device) 
    cat_runoff = cat(pixel_runoff)

    
    discharge = xt.read_feather(discharge_path, dims=["time", "spatial"])\
                     .expand_dims("batch")\
                     .to(device)\
                     .transpose("batch", "spatial", "time")\
                     .sel(spatial=cat_runoff["spatial"])

    return cat_runoff, discharge, g
    
def load_rapid_graph_with_attributes(root, vpu, plength_thr=None, node_thr=None):
    g = _read_rapid_graph(root / 'configs' / vpu)[0]
    df = pd.read_feather(root / "tdxhydro_feather" / f"streams_{vpu}.feather")
    
    params = pd.DataFrame({ 
        "is_lake": df["musk_x"]==.01,
        "dist": df["LengthGeodesicMeters"],
        "upa": np.sqrt(df["DSContArea"])
    }).astype("float32")

    # Standardize river channel parameters
    params[["dist", "upa"]] = (params[["dist", "upa"]] \
                             - params[["dist", "upa"]].mean()) \
                             / params[["dist", "upa"]].std()
    
    if (plength_thr is not None) and (node_thr is not None):
        clusters_g, node_transfer = define_schedule(g, plength_thr=plength_thr, 
                                                    node_thr=node_thr)
        g = dh.RivTreeCluster(clusters_g, 
                              node_transfer,
                              irf_fn = "muskingum",
                              include_index_diag=True,
                              param_df=params,
                              param_names=["dist", "upa"])
    else:
        g = dh.RivTree(g, include_index_diag=True,
                       param_df=params,
                       param_names=["dist", "upa"],
                       irf_fn = "muskingum")
    return g

def load_vpu(root, vpu, 
             runoff=None, 
             interp_df=None, 
             device="cpu", 
             plength_thr=10**4, 
             node_thr=10**4):
    """
    """
    if interp_df is None:
        interp_df = pd.read_feather(root / "interp_weight.feather").set_index("river_id")
    if runoff is None:
        runoff_path = root / "daily_sparse_runoff.feather"
        runoff = xt.read_feather(runoff_path, dims=["time", "spatial"])\
                   .sel(time=slice(None, "2019"))\
                   .expand_dims("batch")\
                   .to(device)\
                   .transpose("batch", "spatial", "time") / (3600. * 24)
        
    g = load_rapid_graph_with_attributes(root, vpu, 
                                         plength_thr=plength_thr, 
                                         node_thr=node_thr).to(device)

    interp_df = interp_df.loc[g.nodes]
    pix_idxs  = interp_df["pixel_idx"].unique()
    
    runoff = runoff.sel(spatial=pix_idxs).to(device)
    ci = dh.CatchmentInterpolator(g, runoff, interp_df).to(device)
    runoff = ci.interpolate_runoff(runoff)


    discharge_path = root / "retro_feather" / f"{vpu}.feather"
    q = xt.read_feather(discharge_path, dims=["time", "spatial"])\
          .expand_dims("batch")\
          .to(device)\
          .transpose("batch", "spatial", "time")\
          .sel(spatial=runoff["spatial"])
    
    return g, q, runoff
