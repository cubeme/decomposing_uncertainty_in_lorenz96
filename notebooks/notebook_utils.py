from itertools import product
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def generate_sweep_dict_list(sweep):
    keys = sweep.keys()
    values = sweep.values()
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _bundle_spec_values(sweep: Dict[str, Any], k: str) -> List[Any]:
    bundle = sweep.get("__noise_bundle__")
    if not isinstance(bundle, list):
        return []
    vals: List[Any] = []
    for item in bundle:
        if isinstance(item, dict) and k in item:
            vals.append(item[k])
    return vals


def _spec_for_key(sweep: Dict[str, Any], k: str) -> Any:
    """Return a 'spec-like' object for k, either sweep[k] or values from __noise_bundle__."""
    if k in sweep:
        return sweep[k]
    vals = _bundle_spec_values(sweep, k)
    return vals if vals else None


def _maybe_number(s: str) -> Any:
    """Parse numeric strings to int/float; otherwise return original string."""
    try:
        f = float(s)
    except ValueError:
        return s
    # keep ints as int (e.g. "5" -> 5), otherwise float
    return int(f) if f.is_integer() else f


def _parse_hidden_dims(value_str: str) -> list[int]:
    """Parse '8_8' -> [8, 8]."""
    return [int(x) for x in value_str.split("_")]


def _parse_f_schedule(value_str: str) -> dict:
    """
    Parse f_schedule token produced by get_sweep_name() back into a dict.

    Supports:
      - const:  'const_<F>'
      - linear: 'linear_F0_<...>_F1_<...>_t0_<...>_t1_<...>'
      - osc:    'osc_Fmean_<...>_amp_<...>_freq_<...>'
    """
    parts = value_str.split("_")
    if not parts:
        return {"type": ""}

    kind = parts[0].lower()

    if kind == "const":
        return {"type": "constant", "F": _maybe_number(parts[1])}

    if kind == "linear":
        # parts = ["linear","F0","18","F1","23","t0","0","t1","10000"]
        d = {"type": "linear"}
        for k, v in zip(parts[1::2], parts[2::2]):
            d[k] = _maybe_number(v)
        return d

    if kind == "osc":
        # parts = ["osc","Fmean","20","amp","2","freq","100"]
        d = {"type": "oscillating"}
        for k, v in zip(parts[1::2], parts[2::2]):
            d[k] = _maybe_number(v)
        return d

    # fallback: keep raw
    return {"type": kind, "raw": value_str}


def revert_sweep_name(sweep_name: str, sweep: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sweep name back to typed parameters (float/str/dict/list)."""
    components = sweep_name.split("-")

    # include nested keys from __noise_bundle__
    bundle_keys: List[str] = []
    bundle = sweep.get("__noise_bundle__")
    if isinstance(bundle, list):
        ks = set()
        for item in bundle:
            if isinstance(item, dict):
                ks.update(item.keys())
        bundle_keys = sorted(ks)

    # match longer keys first (prefix-safe)
    all_keys = list(sweep.keys()) + bundle_keys
    sorted_keys_by_length = sorted(set(all_keys), key=len, reverse=True)

    parsed: Dict[str, str] = {}
    for component in components:
        for key in sorted_keys_by_length:
            key_prefix = f"{key}_"
            if component.startswith(key_prefix):
                parsed[key] = component[len(key_prefix) :]
                break

    # typed post-process
    out: Dict[str, Any] = {}
    for k, v_str in sorted(parsed.items()):
        if k == "f_schedule":
            out[k] = _parse_f_schedule(v_str)
        elif k == "hidden_dims":
            out[k] = _parse_hidden_dims(v_str)
        else:
            # decide numeric vs string based on sweep spec if possible
            spec = sweep.get(k)
            # if the sweep spec contains numbers, interpret as numeric
            if (
                isinstance(spec, list)
                and spec
                and all(isinstance(x, (int, float)) for x in spec)
            ):
                out[k] = _maybe_number(v_str)
            else:
                # noise_type etc stays string
                out[k] = v_str

    return out


def print_param_dict(param_dict: Dict[str, Any]) -> None:
    """Print parameters in a readable format."""
    param_strings = [f"{key}={param_dict[key]}" for key in param_dict]
    return ", ".join(param_strings)


def save_plot(figure: plt.Figure, path: str, format="pdf") -> None:
    """Store a figure in a given location on disk."""
    if path is not None:
        figure.savefig(path, bbox_inches="tight", format=format)
        plt.close(figure)
