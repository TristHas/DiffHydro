from tqdm.auto import tqdm

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader
from xtensor import DataTensor

from .utils import collate_fn
from .. import LTIRouter, Runoff

PARAMS_BOUNDS = {
    "muskingum":(torch.tensor([.1, 0.01])[None], # Parameter k and v minimum values
                 torch.tensor([5, .4])[None],    # Parameter k and v values range
                 torch.tensor([-3., -3.])[None]),# Parameter k and v initial values (before sigmoid)
    "pure_lag":(torch.tensor([.01])[None],       # Parameter lag minimum value
                torch.tensor([5.])[None],        # Parameter lag value range
                torch.tensor([-3.])[None]),      # Parameter initial values (before sigmoid)
    "linear_storage":(torch.tensor([.1])[None],  
                      torch.tensor([9.9])[None],
                      torch.tensor([0.])[None]),
    'nash_cascade':(torch.tensor([.05])[None], 
                    torch.tensor([3.25])[None],
                    torch.tensor([-3.])[None]),
    'hayami':(torch.tensor([.2,   .1])[None], 
              torch.tensor([.8,  12])[None],
              torch.tensor([-3,  0])[None]),
}

M3S_TO_MMKM2 = 10**12 / (3600 * 10**9)

def mm_to_m3s(runoff: DataTensor, cat_area, res_temp_h=1): # TODO: handle other temporal resolution
    scale = cat_area.to(runoff.values.device).view(1, -1, 1)
    values = runoff.values * scale * M3S_TO_MMKM2 / res_temp_h
    return DataTensor(values, dims=runoff.dims, coords=runoff.coords)

def m3s_to_mm(discharge: DataTensor, basin_area, res_temp_h=1):
    scale = basin_area.to(discharge.values.device).view(1, -1, 1)
    values = discharge.values / (scale * M3S_TO_MMKM2 / res_temp_h)
    return DataTensor(values, dims=discharge.dims, coords=discharge.coords)

class CalibrationModel(nn.Module):
    def __init__(self, g):
        super().__init__()
        n_params = PARAMS_BOUNDS[g.irf_fn][0].shape[-1]
        n_nodes = len(g.nodes_idx)
        self.params = nn.Parameter(torch.zeros((n_nodes, n_params)))

    def forward(self, *args, **kwargs):
        return self.params
        
class LearnedRouter(nn.Module):
    """
    """
    def __init__(
            self,
            irf_name,
            param_model,
            max_delay: int = 32,
            dt: float = 1.0,
            **routing_kwargs
        ) -> None:
        super().__init__()
        self._init_router(max_delay, dt, **routing_kwargs)
        self._init_buffers(irf_name)
        self.param_model = param_model

    def _init_router(self, max_delay, dt, **routing_kwargs):
        self.router = LTIRouter(
                          max_delay=max_delay,
                          dt=dt, **routing_kwargs
                    )

    
    def _init_buffers(self, irf_name):
        param_mins, param_maxs, param_inits = PARAMS_BOUNDS[irf_name]
        self.register_buffer("param_init", param_inits)
        self.register_buffer("offset", param_mins)
        self.register_buffer("range", param_maxs)# - param_mins

    def _read_params(self, p):
        return torch.sigmoid(self.param_model(p) + self.param_init) * self.range + self.offset
        
    def read_params(self, g, additional_params=None):
        params = self._read_params(g.params)
        if g.irf_fn == "hayami":
            params = torch.cat([additional_params.unsqueeze(-1), params], dim=-1)
        return params
        
    def forward(self, x, g, additional_params):
        """
        """
        return self.router(x, g, params=self.read_params(g, additional_params))

class RunoffRoutingModel(nn.Module):
    def __init__(self, 
                 param_model,
                 input_size = 1,
                 max_delay: int = 32,
                 dt: float = 1.0,
                 irf_name="hayami",
                 res_temp_h=1,
                 **routing_kwargs):
        """
        """
        super().__init__()
        self.res_temp_h = res_temp_h
        self.runoff_model  = Runoff(input_size=input_size, softplus=True)     
        self.routing_model = LearnedRouter( irf_name,
                                            max_delay=max_delay, dt=dt,
                                            param_model=param_model,
                                            **routing_kwargs
                                           )
        
    def forward(self, inp, g, cat_area, additional_params=None):
        """
        """
        runoff_mm  = self.runoff_model(inp)
        runoff_m3s = mm_to_m3s(runoff_mm, cat_area, self.res_temp_h)
        return self.routing_model(runoff_m3s, g, additional_params)


class StridedStartSampler(Sampler[int]):
    def __init__(self, dataset: Dataset):
        self.dataset_len = len(dataset)          # number of valid start indices
        self.stride = int(dataset.pred_len)

    def __iter__(self):
        yield from range(0, self.dataset_len, self.stride)

    def __len__(self):
        return (self.dataset_len + self.stride - 1) // self.stride
        
class RunoffRoutingModule(nn.Module):
    def __init__(self, model, 
                 tr_ds, val_ds,
                 batch_size=256,
                 clip_grad_norm=1,
                 routing_lr=10**-4, 
                 routing_wd=10**-3, 
                 runoff_lr=.005, 
                 runoff_wd=.001,
                 scheduler_step_size=None,
                 scheduler_gamma=.1,
                 **opt_kwargs):
        super().__init__()
        self.init_dataloaders(tr_ds, val_ds, batch_size)
        
        self.model = model
        self.init_optimizer( routing_lr=routing_lr, routing_wd=routing_wd,
                             runoff_lr=runoff_lr, runoff_wd=runoff_wd,
                             scheduler_step_size=scheduler_step_size,
                             scheduler_gamma=scheduler_gamma)
        self.clip_grad_norm = clip_grad_norm
                
    def init_dataloaders(self, tr_ds, val_ds, batch_size):
        self.tr_ds = tr_ds
        self.val_ds = val_ds
        self.tr_dl = DataLoader(self.tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        sampler = StridedStartSampler(self.val_ds)
        self.val_dl = DataLoader(self.val_ds, 
                                 batch_size=batch_size,
                                 drop_last=False,
                                 sampler=sampler, 
                                 collate_fn=collate_fn)
        
    def init_optimizer(self, 
                       routing_lr=10**-4, routing_wd=10**-3, 
                       runoff_lr=.005, runoff_wd=.001,
                       scheduler_step_size=None,
                       scheduler_gamma=.1):
        self.opt = torch.optim.Adam([
            {'params': self.model.runoff_model.parameters(), 'lr': runoff_lr, 'weight_decay': runoff_wd},
            {'params': self.model.routing_model.parameters(), 'lr': routing_wd, 'weight_decay': routing_wd}
        ])
        if (scheduler_step_size is not None):
            self.scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.3)
        else:
            self.scheduler = None

    def train(self, n_epoch, device):
        """
            Training loop with an optional learning rate scheduler.
        """
        tr_losses, val_losses = [],[]
        self.model = self.model.to(device)
        
        for epoch in range(n_epoch):
            tr_loss = self.train_epoch(device=device)
            tr_losses.append(tr_loss)
    
            val_loss = self.new_valid_epoch(device=device)
            val_losses.append(val_loss)
            
        return tr_losses, val_losses

    def train_epoch(self, device):
        losses = []
        (g, lbl_var, cat_area, 
         channel_dist, init_window) = self.tr_ds.read_statics(device)
        
        for x,y in tqdm(self.tr_dl):
            x = x.to(device)
            y = y.to(device)
    
            o = self.model(x, g, cat_area, channel_dist)
            o = o.sel(spatial=y.coords["spatial"])
            
            o = o.isel(time=slice(init_window, None))
            y = y.isel(time=slice(init_window, None))

            loss = self.loss_fn(o, y, lbl_var)
            
            self.opt.zero_grad()
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               max_norm=self.clip_grad_norm)
            self.opt.step()
            losses.append(loss.item())
    
        if self.scheduler is not None: self.scheduler.step()
        return losses

    def loss_fn(self, o, y, var):
        y = y.values
        o = o.values
        var = var.values.squeeze()
        
        valid_mask = ~y.isnan()            
        safe_o = torch.where(valid_mask, o, 0)
        safe_y   = torch.where(valid_mask, y,   0)
        
        mse = ((safe_o - safe_y) ** 2).mean((0, -1))  
        loss = (mse / var).mean()

        return loss
        
    def valid_epoch(self, device):
        yte, ote = self.extract_test(device)
        return 1-((yte-ote)**2).mean() / yte.var()
    
    def extract_test(self, device):
        (g, lbl_var, cat_area, 
         channel_dist, init_window) = self.val_ds.read_statics(device)

        with torch.no_grad(): 
            O  = self.model(self.val_ds.x.to(device), g,
                            cat_area, channel_dist).values[:,2] 
        YY = self.val_ds.y.squeeze().to_pandas()
        OO = pd.Series(O.squeeze().detach().cpu(), index=YY.index)
        return YY.iloc[init_window:], OO.iloc[init_window:]

    def new_valid_epoch(self, device):
        (g, lbl_var, cat_area, 
         channel_dist, init_window) = self.val_ds.read_statics(device)

        losses = []
        with torch.no_grad(): 
            for x,y in tqdm(self.val_dl):
                x = x.to(device)
                y = y.to(device)
                
                o = self.model(x, g, cat_area, channel_dist)
                o = o.sel(spatial=y.coords["spatial"])
                
                o = o.isel(time=slice(init_window,None))
                y = y.isel(time=slice(init_window,None))
    
                loss = self.loss_fn(o, y, lbl_var)
                losses.append(1-loss.item())
                
        return losses

    def new_extract_test(self, device):
        (g, lbl_var, cat_area, 
         channel_dist, init_window) = self.val_ds.read_statics(device)
        data = []
        with torch.no_grad(): 
            for x,y in tqdm(self.val_dl):
                x = x.to(device)
                y = y.to(device)
                
                o = self.model(x, g, cat_area, channel_dist)
                o = o.sel(spatial=y.coords["spatial"])
                
                o = o.isel(time=slice(init_window, None))
                y = y.isel(time=slice(init_window, None))
    
                data.append((o,y))
        o,y = zip(*data)
        return y,o