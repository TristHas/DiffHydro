from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from xtensor import DataTensor

BATCH_DIM = "batch"
SPATIAL_DIM = "spatial"
TIME_DIM = "time"
BST_DIMS = (BATCH_DIM, SPATIAL_DIM, TIME_DIM)


def ensure_bst_dims(tensor: DataTensor) -> None:
    if tensor.dims != BST_DIMS:
        raise ValueError(f"Expected dims {BST_DIMS}, received {tensor.dims}")


def coords_index(tensor: DataTensor, dim: str) -> pd.Index:
    return pd.Index(tensor.coords[dim], name=dim)


def coords_lookup(tensor: DataTensor, dim: str) -> pd.Series:
    idx = coords_index(tensor, dim)
    return pd.Series(np.arange(len(idx), dtype=np.int64), index=idx)


def to_coord_sequence(values: Sequence | pd.Series | pd.Index) -> tuple:
    if isinstance(values, pd.Series):
        return tuple(values.index.tolist())
    if isinstance(values, pd.Index):
        return tuple(values.tolist())
    return tuple(values)


def datatensor_from_components(
    values: torch.Tensor,
    *,
    batch_coords: Iterable,
    spatial_coords: Iterable,
    time_coords: Iterable,
) -> DataTensor:
    coords = {
        BATCH_DIM: tuple(batch_coords),
        SPATIAL_DIM: tuple(spatial_coords),
        TIME_DIM: tuple(time_coords),
    }
    return DataTensor(values, coords, BST_DIMS)


def datatensor_like(
    tensor: DataTensor,
    values: torch.Tensor,
    *,
    batch_coords: Iterable | None = None,
    spatial_coords: Iterable | None = None,
    time_coords: Iterable | None = None,
) -> DataTensor:
    ensure_bst_dims(tensor)
    coords = {
        BATCH_DIM: tuple(batch_coords) if batch_coords is not None else tensor.coords[BATCH_DIM],
        SPATIAL_DIM: tuple(spatial_coords) if spatial_coords is not None else tensor.coords[SPATIAL_DIM],
        TIME_DIM: tuple(time_coords) if time_coords is not None else tensor.coords[TIME_DIM],
    }
    return DataTensor(values, coords, BST_DIMS)
