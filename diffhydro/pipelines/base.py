from itertools import islice
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader
import xtensor as xt

from .utils import collate_fn, format_param_bounds, mm_to_m3s, PARAMS_BOUNDS

def init_optim_dl(ds, batch_size):
    return DataLoader(ds, batch_size=batch_size, 
                      shuffle=True, collate_fn=collate_fn)
    
def init_inference_dl(ds, batch_size):
    return DataLoader(
        ds, batch_size=batch_size,
        drop_last=False,
        sampler=StridedStartSampler(ds), 
        collate_fn=collate_fn
    )

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

class BaseModule(nn.Module):
    def __init__(self, model, 
                 tr_ds, val_ds,
                 device,
                 batch_size=256,
                 clip_grad_norm=1):
        super().__init__()
        self.model = model
        self.init_data(tr_ds, val_ds, batch_size)
        
        self.clip_grad_norm = clip_grad_norm
        self.scheduler = None
        self.batch_size = batch_size
        self.device = device

    ###
    ### Functions to be overridden
    ###
    def run_model(self, ds, x, cluster_idx):
        raise NotImplementedError

    def seq_run_model(self, ds, x, cluster_idx):
        raise NotImplementedError
                
    def init_optimizer(self):
        raise NotImplementedError
        
    ###
    ### These functions should be used as is most for most pipelines
    ###
    def init_data(self, tr_ds, val_ds, batch_size):
        self.tr_ds  = tr_ds
        self.tr_dl  = init_optim_dl(tr_ds, batch_size)
        self.val_ds = val_ds
        self.val_dl = init_inference_dl(val_ds, batch_size)

    def train(self, n_epoch, n_iter=None, device=None):
        """
            Training loop with an optional learning rate scheduler.
        """
        device = device or self.device
        tr_losses, val_losses = [],[]
        self.model = self.model.to(device)
        
        for epoch in range(n_epoch):
            tr_loss = self.train_epoch(device=device, n_iter=n_iter)
            tr_losses.append(tr_loss)
    
            val_loss = self.valid_epoch(device=device)
            val_losses.append(val_loss)

        tr_losses  = pd.Series([np.mean(x) for x in tr_losses])
        val_losses = pd.Series([np.mean(x) for x in val_losses])
        return tr_losses, val_losses

    def train_epoch(self, n_iter=None, device=None):
        device = device or self.device
        losses = []
        init_window = self.tr_ds.init_len
        lbl_var = self.tr_ds.y_var.to(device)
        
        for x, y in tqdm(islice(self.tr_dl, n_iter)):
            x, y = x.to(device), y.to(device)
            o = self.run_model(self.tr_ds, x)
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
        return 1-np.array(losses)

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

    def valid_epoch(self, device=None):
        device = device or self.device
        init_window = self.val_ds.init_len
        lbl_var = self.val_ds.y_var.to(device)
        
        losses = []
        with torch.no_grad(): 
            for x,y in tqdm(self.val_dl):
                x = x.to(device)
                y = y.to(device)
                
                o = self.run_model(self.val_ds, x)
                o = o.sel(spatial=y.coords["spatial"])
                
                o = o.isel(time=slice(init_window,None))
                y = y.isel(time=slice(init_window,None))
    
                loss = self.loss_fn(o, y, lbl_var)
                losses.append(1-loss.item())
                
        return losses

    def _extract_full_ts(self, ds, batch_size, device=None):
        device = device or self.device
        batch_size = batch_size or self.batch_size
        
        dl = init_inference_dl(ds, batch_size)
        init_window = ds.init_len

        data = []
        with torch.no_grad(): 
            for x,y in tqdm(dl):
                x = x.to(device)
                y = y.to(device)
                
                o = self.run_model(ds, x)
                o = o.sel(spatial=y.coords["spatial"])
                
                o = o.isel(time=slice(init_window, None))
                y = y.isel(time=slice(init_window, None))
    
                data.append((o,y))
                
        o,y = zip(*data)
        o = format_batched_series(o, ds)
        y = format_batched_series(y, ds)
        return y,o

    def extract_val(self, batch_size=None, device=None):
        return self._extract_full_ts(self.val_ds, batch_size, device)

    def extract_train(self, batch_size=None, device=None):
        return self._extract_full_ts(self.tr_ds, batch_size, device)

    def train_epoch_one_cluster(self, cluster_idx, n_iter=None, device=None):
        losses = []
        device = device or self.device
        init_window = self.tr_ds.init_len
        lbl_var = self.tr_ds.y_var.to(device)
        
        for x, y in tqdm(islice(self.tr_dl, n_iter)):
            x, y = x.to(device), y.to(device)
            o = self.seq_run_model(self.tr_ds, x, cluster_idx)
            # TODO: mask system
            y = y.sel(spatial=o.coords["spatial"])
            lbl_var = lbl_var.sel(spatial=o.coords["spatial"])
            
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
        return 1-np.array(losses)
        
    def train_sequentially(self, n_epoch, n_iter=50, device=None):
        """
        """
        results = []
        for cluster_idx,_ in enumerate(self.tr_ds.g):
            result = [self.train_epoch_one_cluster(cluster_idx, n_iter, device) 
                       for _ in range(n_epoch)]
            #te_nse, tr_nse = map(pd.Series, zip(*results))
            results.append(result)
        return results