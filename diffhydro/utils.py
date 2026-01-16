import time
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def nse_fn(o: torch.Tensor, lbl: torch.Tensor, 
           var_y = None,
           eps: float = 1e-12, dim: int = -1):
    """
        Memory efficient NSE function using the formula:
        MSE(o,lbl)=⟨o^2⟩ + ⟨lbl^2⟩ − 2⟨o·lbl⟩
    """
    T = o.size(dim)
    ss_oo = torch.einsum('...t,...t->...', o,   o)
    ss_ll = torch.einsum('...t,...t->...', lbl, lbl)
    ss_ol = torch.einsum('...t,...t->...', o,   lbl)
    mse   = (ss_oo + ss_ll - 2*ss_ol) / T
    if var_y is None:
        var_y = torch.var(lbl, dim=dim, unbiased=False)
    return 1.0 - mse / (var_y.clamp_min(eps))
        
class Timer:
    def __init__(self, device="cuda"):
        self.device = device
        self.step_time = defaultdict(float)

    def __call__(self, key: str):
        """Use as `with timer("name"):`"""
        return self._Context(self, key)

    def reset(self):
        for k in self.step_time:
            self.step_time[k] = 0.0

    def summary(self):
        total = sum(self.step_time.values()) or 1e-12
        df = pd.DataFrame(
            [{"step": k, "sec": v, "pct": 100.0 * v / total}
             for k, v in self.step_time.items()]
        ).sort_values("sec", ascending=False)
        return df, total

    class _Context:
        def __init__(self, outer, key):
            self.outer = outer
            self.key = key

        def __enter__(self):
            torch.cuda.synchronize(self.outer.device)
            self.t0 = time.perf_counter()

        def __exit__(self, exc_type, exc, tb):
            torch.cuda.synchronize(self.outer.device)
            dt = time.perf_counter() - self.t0
            self.outer.step_time[self.key] += dt
            return False 
