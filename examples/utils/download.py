import zipfile
import requests
from pathlib import Path

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import xarray as xr

ZENODO_RECORD = 17873745

# --------- utils ---------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# --------- s3 helpers ---------
def _strip_s3_scheme(path: str) -> str:
    """Return the key without the leading s3:// scheme if present."""
    return path.split("://", 1)[1] if "://" in path else path


def download_full_geoglows_data(root, exclude_discharge=True):
    try:
        from huggingface_hub import snapshot_download
    except:
        Exception("Dataset download requires huggingface_hub. Please install it first with pip install huggingface_hub")
    ignore_patterns="retro_feather/**" if exclude_discharge else None
    local_dir = snapshot_download(
        repo_id="prediction2/diffroute_exp",  # or any other dataset id
        repo_type="dataset",
        local_dir=root / "geoglows",      # where to put the files
        ignore_patterns=ignore_patterns
    )
    print("Downloaded to:", local_dir)

def download_rapid_config(root, idx=305):
    import os
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except:
        Exception("Download requires huggingface_hub. Please install it first with pip install huggingface_hub")
    
    REPO_ID = "prediction2/diffroute_exp"
    REPO_TYPE = "dataset"
    folder = f"configs/{idx}" 
    
    api = HfApi()
    local_dir = root / "geoglows" / "rapid_config" / str(idx)

    os.makedirs(local_dir, exist_ok=True)

    for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE):
        if f.startswith(folder + "/"):
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=f,
                local_dir=local_dir,
            )

def download_geoglows_without_discharge(root):
    try:
        from huggingface_hub import snapshot_download
    except:
        Exception("Dataset download requires huggingface_hub. Please install it first with pip install huggingface_hub")
    local_dir = snapshot_download(
        repo_id="prediction2/diffroute_exp",  # or any other dataset id
        repo_type="dataset",
        local_dir=root,      # where to put the files
    )
    print("Downloaded to:", local_dir)

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
        Example: "/data/interp/interp_weight_305.feather"
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

def process_single_vpu_data(root, device="cpu"):
    from diffhydro import TimeSeriesThDF, CatchmentInterpolator
    from diffhydro.io import read_rapid_graph

    runoff_path = root / 'geoglows' / 'input' / '305_daily_sparse_runoff.feather'
    interp_weight_path = root / 'geoglows' / 'input' / '305_interp_weight.feather'
    rapid_path = root / "geoglows" / "rapid_config" / "305"
    
    # Load input pixel-wise runoff.
    pixel_runoff = pd.read_feather(runoff_path) # Convert values in m3 / s
    pixel_runoff = TimeSeriesThDF.from_pandas(pixel_runoff).to(device) # Convert pandas DataFrame to TimeSeriesThDF
    # Interpolate the pixel-wise runoffs onto the graph catchments 
    interp_df = pd.read_feather(interp_weight_path)
    g = read_rapid_graph(rapid_path)
    cat = CatchmentInterpolator(g, pixel_runoff, interp_df).to(device)
    runoff = cat(pixel_runoff)
    # Dump as netcdf
    runoff = xr.DataArray(runoff.to_pandas(), dims=["time", "river_id"])
    runoff.to_netcdf(rapid_path / "runoff.nc")

    
def download_single_vpu_data(root: Path):
    dest_path = root / "geoglows" / "input" / "305_daily_sparse_runoff.feather"
    download_zenodo_file(
        f"https://zenodo.org/records/{ZENODO_RECORD}/files/305_daily_sparse_runoff.feather",
        str(dest_path),
    )
    dest_path = root / "geoglows" / "input" / "305_interp_weight.feather"
    download_zenodo_file(
        f"https://zenodo.org/records/{ZENODO_RECORD}/files/305_interp_weight.feather",
        str(dest_path),
    )
    download_rapid_config(root, 305)
    process_single_vpu_data(root)

def download_input_runoff(root: Path):
    dest_path = root / "geoglows" / "input" / "daily_sparse_runoff.feather"
    download_zenodo_file(
        f"https://zenodo.org/records/{ZENODO_RECORD}/files/daily_sparse_runoff.feather",
        str(dest_path),
    )

def download_interp_weights(root: Path):
    dest_path = root / "geoglows" / "input" / "interp_weight.feather"
    download_zenodo_file(
        f"https://zenodo.org/records/{ZENODO_RECORD}/files/interp_weight.feather",
        str(dest_path),
    )

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
        f"https://zenodo.org/records/{ZENODO_RECORD}/files/ono.zip",
        str(zip_path),
    )

    # Extract all files from the zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ono_dir)

    # Optionally, remove the zip file after extraction
    zip_path.unlink(missing_ok=True)

    return ono_dir
