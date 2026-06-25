import numpy as np
import yaml
from notebook_utils import revert_sweep_name


def to_numpy(x):
    # zarr/dask/xarray support
    if hasattr(x, "compute"):
        x = x.compute()
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x)


def parse_model_meta_from_cfg_key(cfg_key, sweep):
    """
    Parse meta data: for empty sweep dict, use fixed c,F and None noise params.
    Otherwise parse from cfg_key via revert_sweep_name(cfg_key, sweep).
    """
    if sweep == {}:
        return dict(noise_type=None, ar_order=None, delta_t=None)

    parsed = revert_sweep_name(cfg_key, sweep)
    noise_type = parsed.get("noise_type")
    ar_order = parsed.get("ar_order")
    ar_order = int(ar_order) if ar_order is not None else None
    delta_t = parsed.get("delta_t")
    delta_t = int(delta_t) if delta_t is not None else None

    return dict(noise_type=noise_type, ar_order=ar_order, delta_t=delta_t)


def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
