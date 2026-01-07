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