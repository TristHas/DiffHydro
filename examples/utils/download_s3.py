import s3fs

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
    return ds#ds["Q"].sel(river_id=river_id).load().to_pandas()

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
