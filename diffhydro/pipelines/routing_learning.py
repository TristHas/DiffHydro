from tqdm.auto import tqdm 

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .utils import MLP
from .. import LTIStagedRouter, LTIRouter, RivTreeCluster, RivTree, nse_fn

class LearnedRouter(nn.Module):
    """
    """
    def __init__(
            self,
            max_delay: int = 32,
            dt: float = 1.0,
            param_mins = [.005, .0],
            param_maxs = [.25, 1.2],
            mlp = None,
            **routing_kwargs
        ) -> None:
        super().__init__()
        self._init_router(max_delay, dt, **routing_kwargs)
        self._init_buffers(param_mins, param_maxs)
        self.mlp = mlp or MLP(in_dim=3, out_dim=2)

    def _init_router(self, max_delay, dt, **routing_kwargs):
        self.staged_router = LTIStagedRouter(
                                  max_delay=max_delay,
                                  dt=dt, **routing_kwargs
                            )
        self.router = LTIRouter(
                          max_delay=max_delay,
                          dt=dt, **routing_kwargs
                    )

    
    def _init_buffers(self, param_mins, param_maxs):
        self.register_buffer("offset", torch.tensor(param_mins)[None])
        self.register_buffer("range", (torch.tensor(param_maxs) -\
                                       torch.tensor(param_mins))[None])

    def _read_params(self, p):
        return torch.sigmoid(self.mlp(p)) * self.range + self.offset
        
    def read_params(self, g):
        if isinstance(g, RivTree):
            return self._read_params(g.params)
        elif isinstance(g, RivTreeCluster):
            return [self._read_params(g_.params) for g_ in g]
        else:
            raise NotImplementedError
            
    def forward(self, x, g):
        """
        """
        router = self.router if isinstance(g, RivTree) else self.staged_router
        return router(x, g, params=self.read_params(g))

    def init_upstream_discharges(self, x, g, cluster_idx):
        assert isinstance(g, RivTreeCluster)
        params = self.read_params(g)
        return self.staged_router.init_upstream_discharges(
                                        x, g, cluster_idx,
                                        params=params
        )

    def route_one_cluster(self, x, g, cluster_idx, transfer_bucket=None):
        assert isinstance(g, RivTreeCluster)
        params = self.read_params(g)
        return self.staged_router.route_one_cluster(x, g, 
                                             cluster_idx=cluster_idx, 
                                             params=params[cluster_idx], 
                                             transfer_bucket=transfer_bucket)

class LearningModule(nn.Module):
    def __init__(self, model, tr_data, te_data, tr_g, te_g):
        """
        """
        super().__init__()
        self.model = model
        self.tr_data = tr_data
        self.te_data = te_data
        self.tr_g = tr_g
        self.te_g = te_g
        self.opt = self.init_opt()
        
    def init_opt(self, lr=.001):
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
        out = self.model(inp, self.tr_g)
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
            out = self.model(inp, self.te_g)
            nse = nse_fn(out.values[...,self.tr_data.init_len:],
                         lbl.values[...,self.tr_data.init_len:]).mean()
        return nse.item()

    def learn(self, n_iter=50, n_epoch=20):
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
            q_init = self.model.init_upstream_discharges(inp, self.tr_g, cluster_idx)

        idxs = self.tr_g[cluster_idx].nodes
        inp = inp[idxs]
        lbl = lbl[idxs]
        out = self.model.route_one_cluster(inp, self.tr_g, cluster_idx, q_init)
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
            q_init = self.model.init_upstream_discharges(inp, self.te_g, cluster_idx)
            idxs   = self.te_g[cluster_idx].nodes
            inp    = inp[idxs]
            lbl    = lbl[idxs]
            out    = self.model.route_one_cluster(inp, self.te_g, cluster_idx, q_init)
        
            nse = nse_fn(out.values[...,self.tr_data.init_len:],
                         lbl.values[...,self.tr_data.init_len:]).mean()
        return nse.item()
        
    def learn_staged_sequentially(self, n_iter=50, n_epoch=20):
        """
        """
        results = []
        for cluster_idx,_ in enumerate(self.tr_g):
            results = [self.train_one_epoch_one_cluster(cluster_idx, n_iter) for _ in range(n_epoch)]
            te_nse, tr_nse = map(pd.Series, zip(*results))
        results.append([te_nse, tr_nse])
        return results