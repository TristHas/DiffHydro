from tqdm.auto import tqdm 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from .. import LTIRouter, RivTreeCluster, RivTree, nse_fn

class CalibrationRouter(nn.Module):
    """
    """
    def __init__(
            self, riv_trees,
            max_delay: int = 32,
            dt: float = 1.0,
            param_mins = [.005, .0],
            param_maxs = [.25, 1.2],
            **routing_kwargs
        ) -> None:
        """
        """
        super().__init__()
        self.g = riv_trees # gs is fixed, so it is kept as attribute.
        self._init_router(max_delay, dt, **routing_kwargs)
        self._init_buffers(param_mins, param_maxs)

    def _init_router(self, max_delay, dt, **routing_kwargs):
        if isinstance(self.g, RivTreeCluster):
            self.staged = True
            self.router = LTIRouter(
                              max_delay=max_delay,
                              dt=dt, **routing_kwargs
                        )
            self.params = nn.ParameterList([nn.Parameter(g.params) \
                                            for g in self.g])
            with torch.no_grad(): 
                for p in self.params: p[:]=0

        elif isinstance(self.g, RivTree):
            self.staged = False
            self.router = LTIRouter(
                  max_delay=max_delay,
                  dt=dt, **routing_kwargs
            )
            self.params = nn.Parameter(self.g.params)
            with torch.no_grad(): 
                self.params[:]=0
        else:
            raise NotImplementedError

    def _init_buffers(self, param_mins, param_maxs):
        self.register_buffer("offset", torch.tensor(param_mins)[None])
        self.register_buffer("range", (torch.tensor(param_maxs) -\
                                       torch.tensor(param_mins))[None])

    def _read_param(self, p):
        """
        """
        return torch.sigmoid(p) * self.range + self.offset
        
    def read_params(self):
        """
        """
        if self.staged:
            return [self._read_param(p) for p in self.params]
        else:
            return self._read_param(self.params)
            
    def forward(self, x, *args):
        """
        """
        return self.router(x, self.g, params=self.read_params())

    def init_upstream_discharges(self, x, cluster_idx):
        assert self.staged, "Non-staged pipeline can not init_upstream_discharges"
        return self.router.init_upstream_discharges(
                            x, self.g, cluster_idx,
                            params=self.read_params()
        )

    def route_one_cluster(self, x, cluster_idx, transfer_bucket=None, cat=None):
        assert self.staged, "Non-staged pipeline can not route_one_cluster"
        params = self.read_params()
        output = self.router.route_one_cluster(x,
                                               gs=self.g, 
                                               cluster_idx=cluster_idx, 
                                               params=params[cluster_idx], 
                                               transfer_bucket=transfer_bucket)
        return output

    def format_calibrated_params(self):
        raise NotImplementedError

class CalibrationModule(nn.Module):
    def __init__(self, model, tr_data, te_data):
        """
        """
        super().__init__()
        self.model = model
        self.tr_data = tr_data
        self.te_data = te_data
        self.opt = self.init_opt()
        
    def init_opt(self, lr=.1):
        """
        """
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_one_epoch(self, n_iter=50):
        """
        """
        pbar = tqdm(range(n_iter), desc="Training")
        nses = []
        for i in pbar:
            nse = self.train_one_iter()
            nses.append(nse)
            pbar.set_postfix({"Tr NSE:": nse})
        test_nse = self.test_one_epoch()
        return test_nse, np.mean(nses)
            
    def train_one_iter(self):
        """
        """
        inp, lbl = self.tr_data.sample()
        out = self.model(inp)
        out = out.values[...,self.tr_data.init_len:]
        nse = nse_fn(out, lbl.values).mean()
        self.opt.zero_grad()
        loss = 1-nse
        loss.backward()
        self.opt.step()
        return nse.item()

    def test_one_epoch(self):
        """
        """
        inp = self.te_data.x
        lbl = self.te_data.y
        with torch.no_grad():
            out = self.model(inp)
            nse = nse_fn(out.values[...,self.tr_data.init_len:],
                         lbl.values[...,self.tr_data.init_len:]).mean()
        return nse.item()

    def callibrate(self, n_iter=50, n_epoch=20):
        """
        """
        results = [self.train_one_epoch(n_iter) for _ in range(n_epoch)]
        te_nse, tr_nse = zip(*results)
        return pd.Series(te_nse), pd.Series(tr_nse)

    def train_one_iter_one_cluster(self, cluster_idx):
        """
        """
        inp, lbl = self.tr_data.sample()
        with torch.no_grad():
            q_init = self.model.init_upstream_discharges(inp, cluster_idx)

        idxs = self.model.g[cluster_idx].nodes
        inp = inp.sel(spatial=idxs)
        lbl = lbl.sel(spatial=idxs)
        out = self.model.route_one_cluster(inp, cluster_idx, q_init)
        out = out.values[...,self.tr_data.init_len:]
        
        nse = nse_fn(out, lbl.values).mean()
        
        self.opt.zero_grad()
        loss = 1-nse
        loss.backward()
        self.opt.step()
        
        return nse.item()

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
            results = [self.train_one_epoch_one_cluster(cluster_idx, n_iter) for _ in range(n_epoch)]
            te_nse, tr_nse = map(pd.Series, zip(*results))
        results.append([te_nse, tr_nse])
        te_nse, tr_nse = zip(*results)
        return te_nse, tr_nse #pd.concat(te_nse), pd.Series(tr_nse)
