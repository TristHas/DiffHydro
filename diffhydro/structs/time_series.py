import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import cupy as cp
    import cudf
    from torch.utils.dlpack import to_dlpack
except Exception:
    cp = cudf = to_dlpack = None


class TimeSeriesThDF(nn.Module):
    """
    """
    def __init__(
        self,
        values: torch.Tensor,  # accepts [C,T] or [B,C,T]; stored as [B,C,T]
        columns,
        index,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        make_contiguous: bool = True,
    ):
        super().__init__()
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if values.ndim not in (2, 3):
            raise ValueError("values must be 2D [C,T] or 3D [B,C,T]")

        if dtype is not None and values.dtype != dtype:
            values = values.to(dtype)
        if device is not None:
            values = values.to(device)

        # normalize to [B,C,T]
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if make_contiguous:
            values = values.contiguous()

        B, C, T = values.shape

        if len(columns) != C:
            raise ValueError(f"len(columns)={len(columns)} != n_cols={C}")
        if len(index) != T:
            raise ValueError(f"len(index)={len(index)} != n_rows={T}")

        if isinstance(columns, pd.Series):
            self._columns = columns
        else:
            self._columns = pd.Series(
                np.arange(C, dtype=np.int64),
                index=pd.Index(columns, name="columns"),
            )

        if isinstance(index, pd.Series):
            self._index = index
        else:
            self._index = pd.Series(
                np.arange(T, dtype=np.int64),
                index=pd.Index(index, name="index"),
            )

        self.register_buffer("values", values)  # [B,C,T]

    # ---- properties ----
    @property
    def columns(self) -> np.ndarray:
        return self._columns.index.values

    @property
    def index(self) -> np.ndarray:
        return self._index.index.values

    @property
    def device(self):
        return self.values.device

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self):
        return self.values.shape  # [B,C,T]

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    # ---- constructors ----
    @staticmethod
    def from_pandas(
        df: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame] | dict,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Build from pandas.
        - Single DataFrame -> [1,C,T]
        - List/Tuple/Dict of DataFrames -> [B,C,T]
          (all DFs must share identical index & columns and order)
        """
        def _df_to_CT(dfi: pd.DataFrame) -> torch.Tensor:
            arr = dfi.to_numpy(copy=False)
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)
            return torch.from_numpy(arr).t()  # [C,T]

        if isinstance(df, pd.DataFrame):
            t = _df_to_CT(df).unsqueeze(0)  # [1,C,T]
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            if device is not None:
                t = t.to(device)
            return TimeSeriesThDF(t, df.columns, df.index)

        frames = list(df.values()) if isinstance(df, dict) else list(df)
        if len(frames) == 0:
            raise ValueError("from_pandas: empty collection")

        base_cols = frames[0].columns
        base_idx  = frames[0].index
        for k, dfi in enumerate(frames[1:], start=1):
            if not base_cols.equals(dfi.columns):
                raise ValueError(f"from_pandas: columns mismatch at batch {k}")
            if not base_idx.equals(dfi.index):
                raise ValueError(f"from_pandas: index mismatch at batch {k}")

        tensors = [_df_to_CT(dfi) for dfi in frames]  # [C,T] each
        t3 = torch.stack(tensors, dim=0)              # [B,C,T]
        if dtype is not None and t3.dtype != dtype:
            t3 = t3.to(dtype)
        if device is not None:
            t3 = t3.to(device)
        return TimeSeriesThDF(t3, base_cols, base_idx)

    # ---- indexing ----
    def __getitem__(self, key):
        """
        key can be:
          - cols
          - (cols, time)
        Applies to every batch.
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Expected key=(cols, time)")
            col_key, time_key = key
        else:
            col_key, time_key = key, slice(None)

        cols  = self._columns[col_key]
        times = self._index[time_key]

        V = self.values[:, cols.values][:, :, times.values]  # [B,C',T']
        return TimeSeriesThDF(V, columns=cols.index, index=times.index)

    # ---- conversions ----
    def to_pandas(self, batch: int | slice | list[int] | None = None):
        """
        If batch is None:
          - If B==1 -> return a single DataFrame.
          - Else     -> return a list[DataFrame] of length B.
        If batch is provided:
          - int      -> return a single DataFrame for that batch.
          - slice/list/array -> return a list[DataFrame] for the selection.
        """
        B, C, T = self.values.shape
        V = self.values.detach().cpu()

        def _one_df(mat_CT: torch.Tensor):
            v = mat_CT.T.numpy()  # [T,C]
            return pd.DataFrame(v, index=self.index.copy(), columns=self.columns.copy())

        if batch is None:
            if B == 1:
                return _one_df(V[0])
            return [_one_df(V[b]) for b in range(B)]

        if isinstance(batch, (list, tuple, np.ndarray, torch.Tensor, slice)):
            idxs = range(B) if isinstance(batch, slice) else batch
            return [_one_df(V[int(b)]) for b in idxs]
        else:
            return _one_df(V[int(batch)])

    def to_cudf(self, device_index: int | None = None, *, batch: int | None = None):
        """
        cuDF export for a single batch.
        If batch is None and B==1 -> export that single batch.
        If batch is None and B>1  -> require batch=int.
        """
        if cudf is None or cp is None or to_dlpack is None:
            raise RuntimeError("to_cudf requires cupy, cudf, and torch.utils.dlpack")

        B = self.batch_size
        b = 0 if (batch is None and B == 1) else batch
        if b is None:
            raise ValueError("to_cudf: please specify batch=<int> when batch_size>1")

        V = self.values[int(b)]  # [C,T]

        if device_index is None:
            if V.is_cuda:
                device_index = int(V.get_device())
            else:
                raise ValueError("values is on CPU. Please pass device_index to choose a CUDA device.")

        V_row_major = V.T.contiguous()  # [T,C]

        with cp.cuda.Device(device_index):
            cp_mat = cp.fromDlpack(to_dlpack(V_row_major))  # zero-copy
            gdf = cudf.DataFrame(cp_mat)
            gdf.columns = list(self.columns)
            return gdf

    # ---- misc ----
    def clone(self) -> "TimeSeriesThDF":
        """Clone values and reuse labels."""
        return TimeSeriesThDF(
            self.values.clone(),  # [B,C,T]
            self.columns.copy(),
            self.index.copy(),
            device=self.device,
            dtype=self.values.dtype,
            make_contiguous=True,
        )

    def __repr__(self) -> str:
        B, C, T = self.values.shape
        idx = self.index

        period_str = "—"
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
            first, last = idx[0], idx[-1]
            try:
                first_s, last_s = first.isoformat(), last.isoformat()
            except Exception:
                first_s, last_s = str(first), str(last)
            try:
                freq = idx.freqstr or pd.infer_freq(idx)
            except Exception:
                freq = None
            period_str = f"{first_s} → {last_s}"
            if freq:
                period_str += f" (freq={freq})"
        elif len(idx) == 0:
            period_str = "empty index"

        return (f"TimeSeriesThDF(B={B}, n_series={C}, length={T}, "
                f"period={period_str}, dtype={self.values.dtype}, device={self.values.device})")
