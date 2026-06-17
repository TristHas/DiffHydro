from xtensor import DataTensor
from diffroute import RivTree, RivTreeCluster

BST_DIMS  = ("batch", "spatial", "time")
BSTV_DIMS = ("batch", "spatial", "time", "variable")
# Canonical order with the optional ensemble axis: ordered after batch but
# before spatial/time.
BEST_DIMS = ("batch", "ensemble", "spatial", "time")

def ensure_bst_dims(tensor: DataTensor) -> None:
    if tensor.dims != BST_DIMS:
        raise ValueError(f"Expected dims {BST_DIMS}, received {tensor.dims}")

def ensure_bste_dims(tensor: DataTensor) -> None:
    """Validate routing dims: batch, spatial, time and an optional ensemble axis.

    The ensemble axis is optional but, when present, must be ordered after batch
    and before spatial/time. No other dims are allowed.
    """
    dims = tensor.dims
    expected = BEST_DIMS if "ensemble" in dims else BST_DIMS
    if dims != expected:
        raise ValueError(
            f"Expected dims {expected} (ensemble axis optional, ordered after "
            f"batch and before spatial/time), received {dims}")

__all__ = [
    "DataTensor",
    "ensure_bst_dims",
    "ensure_bste_dims",
    "BufferList",
    "RivTree",
    "RivTreeCluster",
]
