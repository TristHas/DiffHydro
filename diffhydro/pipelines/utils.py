import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xtensor import DataTensor

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
    return DataTensor(values, coords=x_coords, dims=x_dims)
    
class SimpleTimeSeriesDataset(Dataset):
    def __init__(self, x: DataTensor, y: DataTensor, init_len, pred_len):
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

class JointRoutingRunoffDataset(Dataset):
    def __init__(self, 
                 x_dyn: xt.DataTensor, 
                 y: xt.DataTensor, 
                 g: dh.RivTree,
                 init_len, 
                 pred_len,
                 cat_area,
                 basin_area=None,
                 channel_dist=None,
                 x_stat = None, 
                 ):
        super().__init__()
        if x_dyn.coords["time"] != y.coords["time"]:
            raise ValueError("Index misalignment")
        self.g = g
        self.x_dyn  = x_dyn
        self.x_stat = x_stat
        self.y = y        

        self.init_len = init_len
        self.pred_len = pred_len
        self.total_len = self.init_len + self.pred_len
        
        self.cat_area = cat_area
        self.basin_area = basin_area
        self.channel_dist = channel_dist
        
        dims = y.dims[:-1]
        coords = {d:y.coords[d] for d in dims}
        self.y_var = DataTensor(y.values.var(-1), coords=coords, dims=dims)
        
    def __getitem__(self, idx):
        y = self.y.isel(time=slice(idx, idx + self.total_len)) #slice(middle, end)
        x = self.x_dyn.isel(time=slice(idx, idx + self.total_len))
        x = self.concat_x_dyn_stat(x)
        return x, y

    def concat_x_dyn_stat(self, x_slice):
        if self.x_stat is None:
            return x_slice
        x_stat = self.x_stat.values[None, :, None,:].expand(-1, -1, x_slice.shape[2], -1)
        x = torch.cat([x_stat, x_slice.values], dim=-1)
        
        x_coords = {k:v for k,v in x_slice.coords.items()}
        x_dims = x_slice.dims
        x_coords["variable"] = np.arange(x.shape[-1])
        
        return xt.DataTensor(x, coords=x_coords, dims=x_dims)

    def read_statics(self, device):
        lbl_var = self.y_var.to(device)
        cat_area = self.cat_area.to(device)
        channel_dist =self.channel_dist.to(device)
        g = self.g.to(device)
        init_window = self.init_len
        return g, lbl_var, cat_area, channel_dist, init_window

    def __len__(self):
        return len(self.x_dyn.coords["time"]) - self.total_len