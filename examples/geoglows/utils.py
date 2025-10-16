from __future__ import annotations

from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from diffhydro import RivTree, RivTreeCluster, TimeSeriesThDF
from diffroute.graph_utils import define_schedule
from diffroute.io import _read_rapid_graph, read_rapid_graph

from examples.utils.download import (
    download_input_runoff,
    download_interp_weights,
    download_rapid_config,
    download_tdxhydro_attributes,
)

_CALIB_END_DATE = "2019-12-31"
_DISCHARGE_BUCKET = "s3://geoglows-v2/retrospective/daily.zarr"


def _ensure_path(root: Path | str) -> Path:
    return root if isinstance(root, Path) else Path(root)


def _ensure_rapid_config(root: Path, vpu: str) -> Path:
    target = root / "geoglows" / "rapid_config" / vpu
    if not target.exists() or not any(target.iterdir()):
        download_rapid_config(root, vpu)
    return target


def _ensure_tdxhydro_streams(root: Path, vpu: str) -> Path:
    tdx_dir = root / "geoglows" / "tdxhydro" / vpu
    gpkg = tdx_dir / f"streams_{vpu}.gpkg"
    if not gpkg.exists():
        download_tdxhydro_attributes(root, vpu)
    if not gpkg.exists():
        raise FileNotFoundError(f"TDXHydro streams file not found at {gpkg}")
    return gpkg


def _load_interp_weights(root: Path, vpu: str, river_ids: Sequence[int]) -> pd.DataFrame:
    input_dir = root / "geoglows" / "input"
    candidates = [
        input_dir / f"{vpu}_interp_weight.pkl",
        input_dir / "interp_weight.pkl",
    ]
    interp_df = None
    for path in candidates:
        if path.exists():
            interp_df = pd.read_pickle(path)
            break
    if interp_df is None:
        download_interp_weights(root)
        path = input_dir / "interp_weight.pkl"
        if not path.exists():
            raise FileNotFoundError("Interpolation weights not available locally or via download.")
        interp_df = pd.read_pickle(path)

    if interp_df.index.name != "river_id":
        interp_df = interp_df.set_index("river_id")

    missing = [rid for rid in river_ids if rid not in interp_df.index]
    if missing:
        raise KeyError(f"Interpolation weights missing for river_ids: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return interp_df.loc[list(river_ids)].copy()


def _load_runoff(root: Path, vpu: str, pixel_ids: Sequence[int]) -> pd.DataFrame:
    input_dir = root / "geoglows" / "input"
    candidates = [
        input_dir / f"{vpu}_daily_sparse_runoff.pkl",
        input_dir / "daily_sparse_runoff.feather",
    ]
    runoff_df = None
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".pkl":
            runoff_df = pd.read_pickle(path)
        elif path.suffix == ".feather":
            runoff_df = pd.read_feather(path)
            if "time" in runoff_df.columns:
                runoff_df = runoff_df.set_index("time")
        if runoff_df is not None:
            break
    if runoff_df is None:
        download_input_runoff(root)
        feather_path = input_dir / "daily_sparse_runoff.feather"
        if not feather_path.exists():
            raise FileNotFoundError("Runoff inputs not available locally or via download.")
        runoff_df = pd.read_feather(feather_path)
        if "time" in runoff_df.columns:
            runoff_df = runoff_df.set_index("time")

    runoff_df = runoff_df.loc[:_CALIB_END_DATE]

    missing = [pid for pid in pixel_ids if pid not in runoff_df.columns]
    if missing:
        raise KeyError(f"Runoff data missing for pixel_idx: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return runoff_df.loc[:, list(pixel_ids)].copy()


def _download_discharge(river_ids: Sequence[int]) -> pd.DataFrame:
    if len(river_ids) == 0:
        return pd.DataFrame()
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
    store = s3fs.S3Map(root=_DISCHARGE_BUCKET, s3=s3, check=False)
    ds = xr.open_zarr(store)
    try:
        q_da = ds["Q"].sel(river_id=list(map(int, river_ids)))
        q_df = q_da.to_pandas()
    finally:
        ds.close()
    return q_df


def _load_discharge(root: Path, vpu: str, river_ids: Sequence[int]) -> pd.DataFrame:
    input_dir = root / "geoglows" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    discharge_path = input_dir / f"{vpu}_daily_discharge.parquet"

    if discharge_path.exists():
        discharge_df = pd.read_parquet(discharge_path)
    else:
        discharge_df = _download_discharge(river_ids)
        discharge_df.to_parquet(discharge_path)

    missing = [rid for rid in river_ids if rid not in discharge_df.columns]
    if missing:
        refreshed = _download_discharge(missing)
        discharge_df = discharge_df.join(refreshed, how="left")
        discharge_df.to_parquet(discharge_path)

    discharge_df = discharge_df.loc[:_CALIB_END_DATE]
    return discharge_df.loc[:, list(river_ids)].copy()


def _split_timeseries(ts: TimeSeriesThDF, split_year: str) -> tuple[TimeSeriesThDF, TimeSeriesThDF]:
    train_slice = slice(None, split_year)
    test_slice = slice(split_year, None)
    return ts[:, train_slice], ts[:, test_slice]


def init_calib_exp(root, vpu, split_year="1980",
                   plength_thr=10**4,
                   node_thr=10**4,
                   device="cuda:6"):
    """
    Initialize calibration experiment tensors for a single VPU.
    """
    root_path = _ensure_path(root)
    vpu_str = str(vpu)

    rapid_dir = _ensure_rapid_config(root_path, vpu_str)
    g = read_rapid_graph(rapid_dir,
                         plength_thr=plength_thr,
                         node_thr=node_thr).to(device)

    interp_df = _load_interp_weights(root_path, vpu_str, g.nodes)
    pix_idxs = np.unique(interp_df["pixel_idx"].to_numpy())

    runoff_df = _load_runoff(root_path, vpu_str, pix_idxs)
    runoff_ts = TimeSeriesThDF.from_pandas(runoff_df).to(device)
    tr_runoff_pix, te_runoff_pix = _split_timeseries(runoff_ts, split_year)

    discharge_df = _load_discharge(root_path, vpu_str, g.nodes)
    discharge_ts = TimeSeriesThDF.from_pandas(discharge_df).to(device)
    tr_discharge, te_discharge = _split_timeseries(discharge_ts, split_year)

    return (g, interp_df,
            tr_runoff_pix, te_runoff_pix,
            tr_discharge, te_discharge)


def load_rapid_graph_with_attributes(root, vpu, plength_thr=None, node_thr=None):
    root_path = _ensure_path(root)
    vpu_str = str(vpu)
    rapid_dir = _ensure_rapid_config(root_path, vpu_str)

    g_nx, _ = _read_rapid_graph(rapid_dir)
    streams_path = _ensure_tdxhydro_streams(root_path, vpu_str)
    df = gpd.read_file(streams_path).set_index("LINKNO")

    params = pd.DataFrame({
        "is_lake": (df["musk_x"] == 0.01),
        "dist": df["LengthGeodesicMeters"],
        "upa": np.sqrt(df["DSContArea"]),
    }).astype("float32")
    params[["dist", "upa"]] = (
        params[["dist", "upa"]] - params[["dist", "upa"]].mean()
    ) / params[["dist", "upa"]].std()

    if (plength_thr is not None) and (node_thr is not None):
        clusters_g, node_transfer = define_schedule(g_nx,
                                                    plength_thr=plength_thr,
                                                    node_thr=node_thr)
        g = RivTreeCluster(clusters_g,
                           node_transfer,
                           include_index_diag=True,
                           param_df=params)
        for tree in g:
            tree.irf_fn = "muskingum"
    else:
        g = RivTree(g_nx,
                    include_index_diag=True,
                    param_df=params)
        g.irf_fn = "muskingum"
    return g
