import numpy as np
import pandas as pd

from diffhydro import RivTree, RivTreeCluster, TimeSeriesThDF
from diffroute.graph_utils import define_schedule
from diffroute.io import _read_rapid_graph, read_rapid_graph

def load_single_vpu(root, vpu, 
                    plength_thr=None, 
                    node_thr=None,
                    device="cpu"):
    rapid_dir = root / "geoglows" / 'configs' / vpu
    vpu_config_path = root / "geoglows" / 'configs'
    runoff_path = root / "geoglows" / "daily_sparse_runoff.feather"
    interp_weight_path = root / "geoglows" / "interp_weight.feather"
    
    g = read_rapid_graph(rapid_dir,
                         plength_thr=plength_thr,
                         node_thr=node_thr).to(device)
    
    cat_interp_df = pd.read_feather(interp_weight_path).set_index("river_id")
    pix_idxs = cat_interp_df.loc[g.nodes_idx.index]["pixel_idx"].unique()
    
    pixel_runoff = pd.read_feather(runoff_path)
    pixel_runoff = pixel_runoff[pix_idxs].loc[:"2019"]  / (3600. * 24)
    
    pixel_runoff = TimeSeriesThDF.from_pandas(pixel_runoff).to(device)
    cat = CatchmentInterpolator(g, pixel_runoff, cat_interp_df).to(device) 
    cat_runoff = cat(pixel_runoff)
    
    discharge = pd.read_feather(root / "geoglows" / "retro_feather" / f"{vpu}.feather")
    discharge = TimeSeriesThDF.from_pandas(discharge).to(device)[cat_runoff.columns]
    return cat_runoff, discharge, g
    
def load_rapid_graph_with_attributes(root, vpu, plength_thr=None, node_thr=None):

    g = _read_rapid_graph(root / 'configs' / vpu)[0]
    df = pd.read_feather(root / "tdxhydro_feather" / f"streams_{vpu}.feather")
    
    params = pd.DataFrame({ 
        "is_lake": df["musk_x"]==.01,
        "dist": df["LengthGeodesicMeters"],
        "upa": np.sqrt(df["DSContArea"])
    }).astype("float32")

    # Standardize
    params[["dist", "upa"]] = (params[["dist", "upa"]] \
                             - params[["dist", "upa"]].mean()) \
                             / params[["dist", "upa"]].std()
    
    if (plength_thr is not None) and (node_thr is not None):
        clusters_g, node_transfer = define_schedule(g, plength_thr=plength_thr, 
                                                    node_thr=node_thr)
        g = RivTreeCluster(clusters_g, 
                           node_transfer,
                           include_index_diag=True,
                           param_df=params)
        for g_ in g: g_.irf_fn = "muskingum"
    else:
        g = RivTree(g,
                    include_index_diag=True,
                    param_df=params)
        g.irf_fn = "muskingum"
    return g

def load_vpu(root, vpu, 
             runoff=None, 
             interp_df=None, 
             device="cpu", 
             plength_thr=10**4, 
             node_thr=10**4):
    if interp_df is None:
        interp_df = pd.read_feather(root / "interp_weight.feather").set_index("river_id")
    if runoff is None:
        runoff = pd.read_feather(root / "daily_sparse_runoff.feather").loc[:"2019"] / (3600. * 24)
        runoff = TimeSeriesThDF.from_pandas(runoff)

    g = load_rapid_graph_with_attributes(root, vpu, 
                                         plength_thr=plength_thr, 
                                         node_thr=node_thr).to(device)

    interp_df = interp_df.loc[g.nodes]
    pix_idxs  = interp_df["pixel_idx"].unique()
    runoff = runoff[pix_idxs].to(device)
    ci = CatchmentInterpolator(g, runoff, interp_df).to(device)
    runoff = ci.interpolate_runoff(runoff)
    
    q = TimeSeriesThDF.from_pandas(pd.read_feather(root / "retro_feather" /f"{vpu}.feather")).to(device)[g.nodes]
    
    return g, q, runoff