import torch
import torch.nn as nn

from .utils import mm_to_m3s
from .base import BaseModule
from .routing import LearnedRouter
from .. import Runoff

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

class RunoffRoutingModule(BaseModule):
    def __init__(self, model, 
                 tr_ds, val_ds,
                 device="cuda:0",
                 batch_size=256,
                 clip_grad_norm=1,
                 routing_lr=10**-4, 
                 routing_wd=10**-3, 
                 runoff_lr=.005, 
                 runoff_wd=.001,
                 scheduler_step_size=None,
                 scheduler_gamma=.1,
                 **opt_kwargs):
        super().__init__(model, tr_ds, val_ds, device, 
                         batch_size, clip_grad_norm)
        self.init_optimizer( routing_lr=routing_lr, 
                             routing_wd=routing_wd,
                             runoff_lr=runoff_lr, 
                             runoff_wd=runoff_wd,
                             scheduler_step_size=scheduler_step_size,
                             scheduler_gamma=scheduler_gamma)
                        
    def init_optimizer(self, 
                       routing_lr=10**-4, 
                       routing_wd=10**-3, 
                       runoff_lr=.005, 
                       runoff_wd=.001,
                       scheduler_step_size=None,
                       scheduler_gamma=.1):
        self.opt = torch.optim.Adam([
            {'params': self.model.runoff_model.parameters(), 
             'lr': runoff_lr, 'weight_decay': runoff_wd},
            {'params': self.model.routing_model.parameters(), 
             'lr': routing_wd, 'weight_decay': routing_wd}
        ])
        if (scheduler_step_size is not None):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                                opt, step_size=100, gamma=0.3
            )
        else:
            self.scheduler = None
            
    def run_model(self, ds, x):
        g, x_stat, cat_area, channel_dist = self.read_statics(ds, x.device)
        return self.model(x, x_stat, g, cat_area, channel_dist)
    
    def read_statics(self, ds, device):
        lbl_var = ds.y_var.to(device)
        g = ds.g.to(device)
        
        cat_area = ds.statics["cat_area"].to(device)
        channel_dist = ds.statics["channel_dist"].to(device)
        x_stat = ds.statics.get("x_stat", None)
        if x_stat is not None: x_stat=x_stat.to(device)
        
        return (g, x_stat, cat_area, channel_dist)