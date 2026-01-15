import torch.nn as nn
import xtensor as xt

from .lstm import LSTMModel

class Runoff(nn.Module):
    """
        Wrapper class around diffroute.LTIRouter.
    """
    def __init__(self, **kwargs):
        """
        """
        super().__init__()
        self.core = LSTMModel(**kwargs)

    def forward(self, inp_dyn: xt.DataTensor, inp_stat=None) -> xt.DataTensor:
        """
        """
        if inp_stat is None:
            inp = inp_dyn
        else:
            inp = xt.concat([inp_dyn, inp_stat], "variable")
        batch, spatial, time, var = inp.shape
        inp = inp.values.reshape(batch * spatial, time, var)
        
        y = self.core(inp)
        
        reshaped = y.view(batch, spatial, time, 1)
        reshaped = reshaped.squeeze(-1)
        out_dims = inp_dyn.dims[:3]
        return xt.DataTensor(reshaped, dims=out_dims, 
                          coords={d:inp_dyn.coords[d] \
                                  for d in out_dims})
