from xtensor import DataTensor

#from .time_series import (
#    BATCH_DIM,
#    SPATIAL_DIM,
#    TIME_DIM,
#    BST_DIMS,
#    ensure_bst_dims,
#    coords_index,
#    coords_lookup,
#    datatensor_from_components,
#    datatensor_like,
#    to_coord_sequence,
#)
from .utils import BufferList
from diffroute import RivTree, RivTreeCluster

BST_DIMS = ("batch", "spatial", "time")


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
