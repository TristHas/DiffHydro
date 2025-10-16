import zipfile
import requests
from pathlib import Path

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import xarray as xr

import s3fs

# --------- utils ---------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# --------- s3 helpers ---------
def _strip_s3_scheme(path: str) -> str:
    """Return the key without the leading s3:// scheme if present."""
    return path.split("://", 1)[1] if "://" in path else path


def _download_s3_prefix(fs: s3fs.S3FileSystem, s3_prefix: str, local_dir: Path) -> Path:
    """
    Recursively download all files under an S3 prefix into a local directory.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        An initialized s3fs filesystem (pass once to reuse connections).
    s3_prefix : str
        e.g., 's3://geoglows-v2/routing-configs/vpu=305/'
    local_dir : Path
        Local directory to place files under.
    """
    local_dir = _ensure_dir(local_dir)
    normalized_prefix = _strip_s3_scheme(s3_prefix).rstrip("/") + "/"

    # List all remote files under the prefix
    objects = fs.find(s3_prefix)
    if not objects:
        return local_dir

    for remote_path in tqdm(objects, desc=f"Downloading {s3_prefix}", unit="file"):
        remote_norm = _strip_s3_scheme(remote_path)
        if not remote_norm.startswith(normalized_prefix):
            rel = Path(remote_norm).name
        else:
            rel = remote_norm[len(normalized_prefix):]
        if not rel:
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_path, str(dest))
    return local_dir

def download_rapid_config(root: Path, vpu: str | int) -> Path:
    """
        Download routing configuration for a given VPU using s3fs.
    """
    vpu = str(vpu)
    output_dir = _ensure_dir(root / "geoglows" / "rapid_config" / vpu)
    input_url = f"s3://geoglows-v2/routing-configs/vpu={vpu}/"
    fs = s3fs.S3FileSystem(anon=True)
    _download_s3_prefix(fs, input_url, output_dir)
    return output_dir

def download_tdxhydro_attributes(root: Path, vpu: str | int) -> Path:
    """
        Download TDXHydro attributes for a given VPU using s3fs.
    """
    vpu = str(vpu)
    output_dir = _ensure_dir(root / "geoglows" / "tdxhydro" / vpu)
    input_url = f"s3://geoglows-v2/hydrography/vpu={vpu}/"
    fs = s3fs.S3FileSystem(anon=True)
    _download_s3_prefix(fs, input_url, output_dir)
    return output_dir

def read_remote_restrospective_q(river_id):
    """
        Read the retrospective daily discharge zarr remotely via s3fs.
    """
    bucket_uri = 's3://geoglows-v2/retrospective/daily.zarr'
    region_name = 'us-west-2'
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
    s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)
    ds = xr.open_zarr(s3store)
    return ds["Q"].sel(river_id=river_id).load().to_pandas()

def list_vpus() -> list[str]:
    """
    Return a sorted list of VPU IDs (as strings) available under
    s3://geoglows-v2/routing-configs/
    e.g., ["101", "102", "103", ...]
    """
    fs = s3fs.S3FileSystem(anon=True)
    entries = fs.ls("s3://geoglows-v2/routing-configs/")
    vpus: list[str] = []
    for entry in entries:
        last = entry.rstrip("/").split("/")[-1]  # "vpu=305"
        if last.startswith("vpu="):
            v = last.split("=", 1)[1]
            if v.isdigit():
                vpus.append(v)
    vpus = [str(x) for x in sorted(map(int, vpus))]
    return vpus

# --------- Zenodo helpers  ---------
def download_zenodo_file(url: str, dest_path: str):
    """
    Download a file from a URL (e.g., Zenodo) to a specific location.

    Parameters
    ----------
    url : str
        Direct download URL to the file (e.g. Zenodo file link).
    dest_path : str
        Path to save the downloaded file. Can include the desired filename.
        Example: "/data/interp/interp_weight_305.pkl"
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as f, tqdm(
            desc=f"Downloading {dest.name}", total=total, unit="B", unit_scale=True
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest

def download_single_vpu_data(root: Path):
    dest_path = root / "geoglows" / "input" / "305_daily_sparse_runoff.pkl"
    download_zenodo_file(
        "https://zenodo.org/records/17346036/files/runoff_pixel_305.pkl",
        str(dest_path),
    )
    dest_path = root / "geoglows" / "input" / "305_interp_weight.pkl"
    download_zenodo_file(
        "https://zenodo.org/records/17346036/files/interp_weight_305.pkl",
        str(dest_path),
    )
    download_rapid_config(root, 305)

def download_input_runoff(root: Path):
    dest_path = root / "geoglows" / "input" / "daily_sparse_runoff.feather"
    download_zenodo_file(
        "https://zenodo.org/records/17346036/files/daily_sparse_runoff.feather",
        str(dest_path),
    )

def download_interp_weights(root: Path):
    dest_path = root / "geoglows" / "input" / "interp_weight.pkl"
    download_zenodo_file(
        "https://zenodo.org/records/17346036/files/interp_weight.pkl",
        str(dest_path),
    )

def download_full_geoglows_data(root: Path):
    """
    Download:
      - large input runoff (Zenodo, ~16GB)
      - catchment interpolation weights (~500MB)
      - RAPID configuration dirs for all available VPUs from the public S3 bucket
    """
    print("Downloading large input runoff (~16GB), it may take a while...")
    download_input_runoff(root)

    print("Downloading catchment interpolation DataFrame (~500MB)...")
    download_interp_weights(root)

    print("Downloading RAPID configuration files from S3...")
    vpus = list_vpus()
    for vpu in tqdm(vpus, desc="VPUs"):
        download_rapid_config(root, vpu)

def download_ono(root: Path) -> Path:
    """
    Download the 'ono.zip' archive from Zenodo and extract it to root / 'ono'.

    Parameters
    ----------
    root : Path
        Root directory where the 'ono' folder will be created.

    Returns
    -------
    Path
        Path to the extracted 'ono' directory.
    """
    # Define destination paths
    ono_dir = _ensure_dir(root / "ono")
    zip_path = ono_dir / "ono.zip"

    # Download the zip file
    download_zenodo_file(
        "https://zenodo.org/records/17346036/files/ono.zip",
        str(zip_path),
    )

    # Extract all files from the zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ono_dir)

    # Optionally, remove the zip file after extraction
    zip_path.unlink(missing_ok=True)

    return ono_dir
