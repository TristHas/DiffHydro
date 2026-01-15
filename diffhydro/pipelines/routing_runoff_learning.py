from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader
import xtensor as xt

from .routing import LearnedRouter
from .utils import collate_fn, format_param_bounds, mm_to_m3s, PARAMS_BOUNDS
from .. import LTIRouter, Runoff

class LearnedRouterOld(nn.Module):
    """
    """
    def __init__(
            self,
            irf_name,
            param_model,
            max_delay: int = 32,
            dt: float = 1.0,
            temp_res_h: float = 1.0,
            **routing_kwargs
        ) -> None:
        super().__init__()
        self._init_router(max_delay, dt, **routing_kwargs)
        self._init_buffers(irf_name, temp_res_h)
        self.param_model = param_model

    def _init_router(self, max_delay, dt, **routing_kwargs):
        self.router = LTIRouter(
                          max_delay=max_delay,
                          dt=dt, **routing_kwargs
                    )

    
    def _init_buffers(self, irf_name, temp_res_h):
        param_mins, param_maxs, param_inits = format_param_bounds(irf_name, temp_res_h)
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
        
    def forward(self, x, g, additional_params=None):
        """
        """
        return self.router(x, g, params=self.read_params(g, additional_params))

    def init_upstream_discharges(self, x, g, cluster_idx, 
                                 additional_params=None):
        assert isinstance(g, RivTreeCluster)
        params = self.read_params(g)
        return self.router.init_upstream_discharges(
                                        x, g, cluster_idx,
                                        params=params
        )

    def route_one_cluster(self, x, g, cluster_idx, 
                          transfer_bucket=None, 
                          additional_params=None):
        assert isinstance(g, RivTreeCluster)
        params = self.read_params(g)
        return self.staged_router.route_one_cluster(x, g, 
                                             cluster_idx=cluster_idx, 
                                             params=params[cluster_idx], 
                                             transfer_bucket=transfer_bucket)


class RunoffRoutingModel(nn.Module):
    def __init__(self, 
                 param_model,
                 input_size = 1,
                 max_delay: int = 32,
                 dt: float = 1.0,
                 irf_name="hayami",
                 temp_res_h=1,
                 **routing_kwargs):
        """
        """
        super().__init__()
        self.temp_res_h = temp_res_h
        self.runoff_model  = Runoff(input_size=input_size, softplus=True)     
        self.routing_model = LearnedRouter( irf_name,
                                            max_delay=max_delay, dt=dt,
                                            param_model=param_model,
                                            temp_res_h=temp_res_h,
                                            **routing_kwargs
                                           )
        
    def forward(self, inp_dyn, inp_stat, g, cat_area, additional_params=None):
        """
        """
        runoff_mm  = self.runoff_model(inp_dyn, inp_stat)
        runoff_m3s = mm_to_m3s(runoff_mm, cat_area, self.temp_res_h)
        return self.routing_model(runoff_m3s, g, additional_params)

class StridedStartSampler(Sampler[int]):
    def __init__(self, dataset: Dataset):
        self.dataset_len = len(dataset)          # number of valid start indices
        self.stride = int(dataset.pred_len)

    def __iter__(self):
        yield from range(0, self.dataset_len, self.stride)

    def __len__(self):
        return (self.dataset_len + self.stride - 1) // self.stride

def format_batched_series(batches, ds):
    data = torch.cat([x.transpose("spatial", "batch", "time")\
                       .values.contiguous().view(x.sizes["spatial"], -1)\
                      for x in batches], -1)
    coords = {"spatial": ds.y.coords["spatial"],
              "time"   : ds.y.coords["time"][ds.init_len:data.shape[-1] + ds.init_len]}
    return xt.DataTensor(data, dims=("spatial", "time"), coords=coords)
        
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
        self.tr_dl = DataLoader(self.tr_ds, batch_size=batch_size, 
                                shuffle=True, collate_fn=collate_fn)
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

    def read_statics(self, ds, device):
        lbl_var = ds.y_var.to(device)
        g = ds.g.to(device)
        init_window = ds.init_len
        
        cat_area = ds.statics["cat_area"].to(device)
        channel_dist = ds.statics["channel_dist"].to(device)
        x_stat = ds.statics["x_stat"].to(device)
        return (g, x_stat, lbl_var, 
                cat_area, channel_dist, init_window)

    def train(self, n_epoch, device):
        """
            Training loop with an optional learning rate scheduler.
        """
        tr_losses, val_losses = [],[]
        self.model = self.model.to(device)
        
        for epoch in range(n_epoch):
            tr_loss = self.train_epoch(device=device)
            tr_losses.append(tr_loss)
    
            val_loss = self.valid_epoch(device=device)
            val_losses.append(val_loss)

        tr_losses  = pd.Series([np.mean(1-np.array(x)) for x in tr_losses])
        val_losses = pd.Series([np.mean(x) for x in val_losses])
        return tr_losses, val_losses

    def train_epoch(self, device):
        losses = []
        (g, x_stat, lbl_var, cat_area, 
         channel_dist, init_window) = self.read_statics(self.tr_ds, device)
        
        for x,y in tqdm(self.tr_dl):
            x, y = x.to(device), y.to(device)
            o = self.model(x, x_stat, g, cat_area, channel_dist)
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
        safe_y   = torch.where(valid_mask, y, 0)
    
        mse = ((safe_o - safe_y) ** 2).sum((0, -1)) /\
                            valid_mask.sum((0, -1)).clip(1) 
        loss = (mse / var).mean()
    
        return loss
        
    def valid_epoch(self, device):
        (g, x_stat, lbl_var, cat_area, 
         channel_dist, init_window) = self.read_statics(self.val_ds, device)

        losses = []
        with torch.no_grad(): 
            for x,y in tqdm(self.val_dl):
                x = x.to(device)
                y = y.to(device)
                
                o = self.model(x, x_stat, g, cat_area, channel_dist)
                o = o.sel(spatial=y.coords["spatial"])
                
                o = o.isel(time=slice(init_window,None))
                y = y.isel(time=slice(init_window,None))
    
                loss = self.loss_fn(o, y, lbl_var)
                losses.append(1-loss.item())
                
        return losses

    def _extract_full_ts(self, ds, batch_size, device):
        sampler = StridedStartSampler(ds)
        dl = DataLoader(ds, batch_size=batch_size,
                        drop_last=False,
                        sampler=sampler, 
                        collate_fn=collate_fn)

        (g, x_stat, lbl_var, cat_area, 
         channel_dist, init_window) = self.read_statics(ds, device)
        
        data = []
        with torch.no_grad(): 
            for x,y in tqdm(dl):
                x = x.to(device)
                y = y.to(device)
                
                o = self.model(x, x_stat, g, cat_area, channel_dist)
                o = o.sel(spatial=y.coords["spatial"])
                
                o = o.isel(time=slice(init_window, None))
                y = y.isel(time=slice(init_window, None))
    
                data.append((o,y))
                
        o,y = zip(*data)
        o = format_batched_series(o, ds)
        y = format_batched_series(y, ds)
        return y,o

    def extract_val(self, batch_size, device):
        return self._extract_full_ts(self.val_ds, batch_size, device)

    def extract_train(self, batch_size, device):
        return self._extract_full_ts(self.tr_ds, batch_size, device)

    def train_one_epoch_one_cluster(self, cluster_idx, n_iter=50):
        """
        """
        pbar = tqdm(range(n_iter), desc="Training")
        nses = []
        for i in pbar:
            nse = self.train_one_iter_one_cluster(cluster_idx)
            nses.append(nse)
            pbar.set_postfix({f"Tr NSE cluster {cluster_idx}:": nse})
        test_nse = self.test_one_epoch_one_cluster(cluster_idx)
        return test_nse, np.mean(nses)

    def test_one_epoch_one_cluster(self, cluster_idx):
        """
        """
        inp = self.te_data.x
        lbl = self.te_data.y
        with torch.no_grad():
            q_init = self.model.init_upstream_discharges(inp, cluster_idx)
            idxs   = self.model.g[cluster_idx].nodes
            inp    = inp.sel(spatial=idxs)
            lbl    = lbl.sel(spatial=idxs)
            out    = self.model.route_one_cluster(inp, cluster_idx, q_init)
        
            nse = nse_fn(out.values[...,self.tr_data.init_len:],
                         lbl.values[...,self.tr_data.init_len:]).mean()
        return nse.item()
        
    def calibrate_staged_sequentially(self, n_iter=50, n_epoch=20):
        """
        """
        results = []
        for cluster_idx,_ in enumerate(self.model.g):
            results = [self.train_one_epoch_one_cluster(cluster_idx, n_iter) \
                       for _ in range(n_epoch)]
            te_nse, tr_nse = map(pd.Series, zip(*results))
        results.append([te_nse, tr_nse])
        te_nse, tr_nse = zip(*results)
        return te_nse, tr_nse #pd.concat(te_nse), pd.Series(tr_nse)
