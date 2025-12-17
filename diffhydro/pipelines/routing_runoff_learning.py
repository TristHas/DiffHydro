import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .. import DataTensor, Runoff, nse_fn, SPATIAL_DIM, TIME_DIM
from ..structs.time_series import ensure_bst_dims, datatensor_like
from .routing_learning import LearnedRouter, MLP 
from tqdm.auto import tqdm


PARAMS_BOUNDS = {
    "muskingum":([.1, 0.01], [5, .4], [-3., -3.]),
    "pure_lag" :([.01], [5.], [-3.]),       
    "linear_storage":([.1], [9.9], [0.]),
    'nash_cascade':([.05], [3.25], [-3.]),
    'hayami':([.2,   .1], [.8,  5.9], [-3,  -3.])
}

MS_TO_MMKM2 = 10**12 / (24 * 3600 * 10**9)
RUNOFF_STD  = 12

def mm_to_m3s(runoff: DataTensor, cat_area):
    ensure_bst_dims(runoff)
    scale = cat_area.to(runoff.values.device).view(1, -1, 1)
    values = runoff.values * scale * MS_TO_MMKM2 * RUNOFF_STD
    return datatensor_like(runoff, values)

class RunoffRoutingModel(nn.Module):
    def __init__(self, 
                 input_size = 1,
                 mlp_inp = 2,
                 max_delay: int = 32,
                 dt: float = 1.0,
                 irf_name="hayami",
                 **routing_kwargs):
        """
        """
        super().__init__()
        param_mins, param_maxs, params_init = PARAMS_BOUNDS[irf_name]
        self.runoff_model  = Runoff(input_size=input_size, softplus=True)     
        mlp = MLP(in_dim=mlp_inp, out_dim=len(param_mins))
        self.routing_model = LearnedRouter(max_delay=max_delay, dt=dt,
                                           param_mins=param_mins,
                                           param_maxs=param_maxs,
                                           mlp=mlp,
                                           **routing_kwargs
                                           )
        
    def forward(self, inp, g, cat_area):
        """
        """
        inp_format = self.format_runoff_input(inp)
        runoff_mm  = self.runoff_model(inp_format)
        runoff_mm = self.format_runoff_output(runoff_mm, inp)
        runoff_m3s = mm_to_m3s(runoff_mm, cat_area)
        return self.routing_model(runoff_m3s, g)

    def format_runoff_input(self, inp):
        ensure_bst_dims(inp)
        return inp

    def format_runoff_output(self, out, inp):
        return out

class RunoffRoutingModule(nn.Module):
    def __init__(self, model, 
                 tr_data, te_data, 
                 cat_areas, 
                 scheduler=None,
                 clip_grad_norm=1,
                 **opt_kwargs):
        """
        """
        super().__init__()
        self.model = model
        self.tr_data = tr_data
        self.te_data = te_data
        self.init_optimizer(scheduler=scheduler, **opt_kwargs)
        self.clip_grad_norm = clip_grad_norm
        

    def train_one_epoch(self, g, cat_areas, n_iter=50):
        """
        """
        pbar = tqdm(range(n_iter), desc="Training")
        nses = []
        for i in pbar:
            nse = self.train_one_iter(g, cat_areas)
            nses.append(nse)
            pbar.set_postfix({"Tr NSE:": nse})
        test_nse = self.test_one_epoch()
        return test_nse, np.mean(nses)
            
    def train_one_iter(self, g, cat_areas):
        """
        """
        inp, lbl = self.tr_data.sample()
        out = self.model(inp, g, cat_areas)
        selected = out.sel(spatial=lbl.coords[SPATIAL_DIM])
        pred = selected.values[..., self.tr_data.init_len:]
        nse = nse_fn(pred, lbl.values, var_y=torch.ones(1, device=pred.device)).mean()
        self.opt.zero_grad()
        loss = 1 - nse
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                       max_norm=self.clip_grad_norm)

        self.opt.step()
        return nse.item()

    def test_one_epoch(self):
        """
        """
        inp = self.te_data.x
        lbl = self.te_data.y
        with torch.no_grad():
            out = self.model(inp, g, cat_areas).sel(spatial=lbl.coords[SPATIAL_DIM])
            nse = nse_fn(
                out.values[..., self.tr_data.init_len:],
                lbl.values[..., self.tr_data.init_len:],
            ).mean()
        if self.scheduler is not None: self.scheduler.step()
        print(nse.item())
        return nse.item()

    def learn(self, g, cat_areas, n_iter=50, n_epoch=20):
        """
        """
        results = [self.train_one_epoch(g, cat_areas, n_iter) for _ in range(n_epoch)]
        te_nse, tr_nse = zip(*results)
        return pd.Series(te_nse), pd.Series(tr_nse)

    def init_optimizer(self, 
                       routing_lr=.01, routing_wd=0, 
                       runoff_lr=.005, runoff_wd=.001,
                       scheduler=None):
        self.opt = torch.optim.Adam([
            {'params': self.model.runoff_model.parameters(), 'lr': runoff_lr, 'weight_decay': runoff_wd},
            {'params': self.model.routing_model.parameters(), 'lr': routing_lr, 'weight_decay': routing_wd}
        ])
        if scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=100, gamma=0.3)
        else:
            self.scheduler = None
