from xtensor import DataTensor
from diffroute import RivTree, RivTreeCluster

BST_DIMS  = ("batch", "spatial", "time")
BSTV_DIMS = ("batch", "spatial", "time", "variable")

def ensure_bst_dims(tensor: DataTensor) -> None:
    if tensor.dims != BST_DIMS:
        raise ValueError(f"Expected dims {BST_DIMS}, received {tensor.dims}")

__all__ = [
    "DataTensor",
    "ensure_bst_dims",
    "BufferList",
    "RivTree",
    "RivTreeCluster",
]
