import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xtensor as xt

from .. import RivTree

PARAMS_BOUNDS = {
     "hayami":pd.DataFrame({"c": [.5, 12.5, "mid"], 
                            "D" : [.1, .9, "low"]}, 
                            index=["min", "max", "init"]),
     "pure_lag":pd.DataFrame({"delay":[.1, 5, "low"]}, 
                             index=["min", "max", "init"]),
     "linear_storage":pd.DataFrame({"tau":[.1, 9.9, "mid"]}, 
                                   index=["min", "max", "init"]),
     "nash_cascade":pd.DataFrame({"tau":[.05, 3.25, "low"]}, 
                                 index=["min", "max", "init"]),
     "muskingum":pd.DataFrame({"k":[.12, 6, "low"], 
                               "x":[.0, 1.2, "low"]}, 
                               index=["min", "max", "init"])
}

def infer_param_bounds(irf_fn, runoff_temp_res_h):
    """
    """
    params = PARAMS_BOUNDS[irf_fn].copy()
    if irf_fn=="hayami":
        params.loc[["min", "max"]]*=runoff_temp_res_h
    if irf_fn=="muskingum":
        params.loc[["min", "max"], "k"]/=runoff_temp_res_h
    if irf_fn=="linear_storage":
        params.loc[["min", "max"]]/=runoff_temp_res_h
    if irf_fn=="nash_cascade":
        params.loc[["min", "max"]]/=runoff_temp_res_h
    return params

def format_param_bounds(irf_fn, runoff_temp_res_h):
    bounds_df = infer_param_bounds(irf_fn, runoff_temp_res_h)
    bounds_df = bounds_df.replace({"mid":"0", "low":"-3", "high":"3"}).astype("float32")
    
    return ( 
         torch.from_numpy(bounds_df.loc["min"].values),
         torch.from_numpy(bounds_df.loc["max"].values),
         torch.from_numpy(bounds_df.loc["init"].values)
       )

M3S_TO_MMKM2 = 10**12 / (3600 * 10**9)

def mm_to_m3s(runoff: xt.DataTensor, cat_area, temp_res_h=1): # TODO: handle other temporal resolution
    scale = cat_area.to(runoff.values.device).view(1, -1, 1)
    values = runoff.values * scale * M3S_TO_MMKM2 / temp_res_h
    return xt.DataTensor(values, dims=runoff.dims, coords=runoff.coords)

def m3s_to_mm(discharge: xt.DataTensor, basin_area, temp_res_h=1):
    scale = basin_area.to(discharge.values.device).view(1, -1, 1)
    values = discharge.values / (scale * M3S_TO_MMKM2 / temp_res_h)
    return xt.DataTensor(values, dims=discharge.dims, coords=discharge.coords)
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=3, hidden_dim=256):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def collate_fn(batch):
    xs, ys = zip(*batch) 
    return cat_xtensor(xs), cat_xtensor(ys)

def cat_xtensor(xs):
    x_coords = {k:v for k,v in xs[0].coords.items()}
    x_dims = xs[0].dims
    x_coords["time"] = np.arange(len(x_coords["time"]))
    x_coords["batch"] = np.arange(len(xs))
    values = torch.cat([x.values for x in xs])
    return xt.DataTensor(values, coords=x_coords, dims=x_dims)
    
class SimpleTimeSeriesDataset(Dataset):
    def __init__(self, x: xt.DataTensor, y: xt.DataTensor, init_len, pred_len):
        super().__init__()
        if x.coords["time"] != y.coords["time"]:
            raise ValueError("Index misalignment")
        self.x = x
        self.y = y
        self.y_var = y.values.var(-1)
        self.init_len = init_len
        self.pred_len = pred_len
        self.n_samples = self.x.shape[2] - self.init_len - self.pred_len
        
    def __getitem__(self, idx):
        start = idx
        middle = idx + self.init_len
        end = idx + self.init_len + self.pred_len
        x_slice = self.x.isel(time=slice(start, end))
        y_slice = self.y.isel(time=slice(start, end)) #slice(middle, end)
        return x_slice, y_slice
        
    def __len__(self):
        return self.n_samples

class BaseDataset(Dataset):
    def __init__(self, 
                 x: xt.DataTensor, 
                 y: xt.DataTensor, 
                 init_len, 
                 pred_len,
                 g = None,
                 statics=None #g: RivTree,
                 #cat_area,
                 #basin_area=None,
                 #channel_dist=None,
                 #x_stat = None, 
                 ):
        super().__init__()
        if (x.coords["time"] != y.coords["time"]).any():
            raise ValueError("Index misalignment")
        self.g = g
        self.x = x
        self.y = y
        self.statics = statics
        
        self.init_len = init_len
        self.pred_len = pred_len
        self.total_len = self.init_len + self.pred_len

        self.y_var = y.var("time")
        #self.cat_area = cat_area
        #self.basin_area = basin_area
        #self.channel_dist = channel_dist
        #var = torch.nan_to_num(y.values.var(-1), nan=1.0)
        #self.y_var = xt.DataTensor(var, coords=coords, dims=dims)
        #corrected victor and gpt
        #dims = y.dims[:-1]
        #coords = {d:y.coords[d] for d in dims}
        #mask = ~torch.isnan(y.values)
        #y0 = torch.where(mask, y.values, 0)
        #count = mask.sum(dim=-1,keepdim=True)
        #mean = y0.sum(dim=-1,keepdim=True) / count.clamp_min(1)
        #var = ((y0 - mean)**2 * mask).sum(dim=-1,keepdim=True) / count.clamp_min(1)
        #var = torch.nan_to_num(var, nan=1.0)
        #self.y_var =xt.DataTensor(var.squeeze(-1), coords=coords, dims=dims)
        
    def __getitem__(self, idx):
        y = self.y.isel(time=slice(idx, idx + self.total_len)) #slice(middle, end)
        x = self.x.isel(time=slice(idx, idx + self.total_len))
        return x, y

    def __len__(self):
        return len(self.x.coords["time"]) - self.total_len