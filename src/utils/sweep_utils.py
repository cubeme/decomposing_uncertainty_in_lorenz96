"""Expand configuration values into parameter sweeps."""

import re
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Sequence, Tuple


def _format_value(v: Any) -> str:
    if isinstance(v, list):
        # e.g. [8, 8] -> 8_8
        return "_".join(str(x) for x in v)
    if isinstance(v, dict):
        if "type" in v:  # forcing schedule dict
            if v["type"] == "constant":
                return f"const_{v['F']}"
            if v["type"] == "linear":
                return f"linear_F0_{v['F0']}_F1_{v['F1']}_t0_{v['t0']}_t1_{v['t1']}"
            if v["type"] == "oscillating":
                return f"osc_Fmean_{v['Fmean']}_amp_{v['amp']}_freq_{v['freq']}"
            raise ValueError(f"Unknown f_schedule type: {v['type']}")
        else:
            raise ValueError(f"Unknown dict format for sweep value: {v}")
    return str(v)


def get_sweep_name(value_dict: Dict[str, Any]) -> str:
    """Build output directory name from dictionary values."""
    parts = []
    for k, v in value_dict.items():
        if k == "__noise_bundle__":
            parts.append(f"noise_type_{v['noise_type']}-ar_order_{v['ar_order']}")
        else:
            parts.append(f"{k}_{_format_value(v)}")
    return "-".join(parts)


def _is_list_of_lists(x) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(isinstance(el, list) for el in x)


def _should_sweep_key(k: str, v, exclude) -> bool:
    if k in exclude:
        return False
    if k == "hidden_dims" or k == "init_rho":
        return _is_list_of_lists(v)  # only sweep if it's like [[64,64], [128,128]]
    return isinstance(v, list)


def expand_forcing_schedule_sweep(base: Dict[str, Any]) -> Dict[str, Any]:
    """Return sweep entries for forcing.

    Sweep rules (as requested):
    - Any list-valued forcing spec counts as a sweep (even len==1).
      This includes:
        * top-level F: [...]
        * f_schedule params like F1: [...], t0: [...]
    - Additionally, a list of multiple schedule dicts counts as a sweep over
      schedule *options*, even if all params are scalars:
        f_schedule: [ {type: linear, ...}, {type: oscillating, ...} ]

    Backwards compatible mapping:
    - Constant forcing sweeps via {"F": [...]}
    - Otherwise sweeps via {"f_schedule": [schedule_dicts...]}
    """
    if base.get("f_schedule") is not None and base.get("F") is not None:
        raise ValueError("Specify either f_schedule or F in config, not both.")

    schedule = base.get("f_schedule")

    # ---- Top-level F ----
    if schedule is None:
        F = base.get("F")
        return {"F": F} if isinstance(F, list) else {}

    # Normalize to list of dicts; remember if user provided a list
    schedule_was_list = isinstance(schedule, list)
    schedule_list = schedule if schedule_was_list else [schedule]

    # validate entries
    for fs in schedule_list:
        if not isinstance(fs, dict):
            raise TypeError(f"f_schedule entries must be dicts; got {type(fs)}")

    # ---- If all constant: map to F sweep when appropriate ----
    all_constant = all(
        str(fs.get("type", "")).lower() == "constant" for fs in schedule_list
    )
    if all_constant:
        F_vals: list[Any] = []
        any_list_param = False

        for fs in schedule_list:
            v = fs.get("F")
            if v is None:
                raise ValueError(
                    'Missing required key "F" for f_schedule type "constant".'
                )
            if isinstance(v, list):
                any_list_param = True
                F_vals.extend(v)
            else:
                F_vals.append(v)

        # sweep if user used list-valued F OR provided multiple constant schedules
        if any_list_param or (schedule_was_list and len(schedule_list) >= 1):
            return {"F": F_vals}
        return {}

    # ---- Non-constant / mixed schedules ----
    expanded: list[Dict[str, Any]] = []
    any_list_param = False

    for fs in schedule_list:
        t = str(fs.get("type", "")).lower()

        if t == "constant":
            keys = ["F"]
        elif t == "linear":
            keys = ["F0", "F1", "t0", "t1"]
        elif t == "oscillating":
            keys = ["Fmean", "amp", "freq"]
        else:
            raise ValueError(f"Unknown f_schedule type: {t}")

        value_lists = []
        for k in keys:
            v = fs.get(k)
            if v is None:
                raise ValueError(
                    f'Missing required key "{k}" for f_schedule type "{t}".'
                )
            if isinstance(v, list):
                any_list_param = True
                value_lists.append(v)
            else:
                value_lists.append([v])

        for combo in product(*value_lists):
            sched = {"type": t}
            for k, v in zip(keys, combo):
                sched[k] = v
            expanded.append(sched)

    # sweep if:
    #   - user provided multiple schedule dicts (choice sweep), OR
    #   - any list-valued param existed (param sweep, even len==1)
    if (schedule_was_list and len(schedule_list) >= 1) or any_list_param:
        return {"f_schedule": expanded}

    return {}


def expand_noise_ar_sweep(base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Couple noise_type and ar_order so we don't take a cartesian product.

    Rules:
      - "white" always implies ar_order = 0
      - "ar_p" implies ar_order in provided list (or scalar)
    """
    noise = base.get("noise_type", None)
    ar = base.get("ar_order", None)

    # If noise_type not specified, don't do anything
    if noise is None:
        # Note: this means that ar_order by itself is not swept, which is intentional since it either only applies to
        # ar_p noise (flow models) or the baseline parameterization. In the latter case and if no noise_type was
        # specified for flow models, we want to have a list of AR orders to fit multiple parameters.
        return {}

    noise_list = noise if isinstance(noise, list) else [noise]

    # normalize ar_order to list if present
    if isinstance(ar, list):
        ar_list = ar
    elif ar is None:
        ar_list = []
    else:
        ar_list = [ar]

    combos: list[dict[str, Any]] = []
    for nt in noise_list:
        if nt == "white":
            combos.append({"noise_type": "white", "ar_order": 0})
        elif nt == "ar_p":
            if not ar_list:
                # default to ar_order = 1 if not specified
                combos.append({"noise_type": "ar_p", "ar_order": 1})
            for p in ar_list:
                if p == 0:
                    raise ValueError('ar_order must be >= 1 when noise_type is "ar_p".')
                combos.append({"noise_type": "ar_p", "ar_order": p})
        else:
            raise ValueError(f"Unknown noise_type: {nt}")

    # Catch inconsistent configs
    if (
        "white" in noise_list
        and "ar_p" not in noise_list
        and (ar is not None and ar != 0)
    ):
        raise ValueError(
            'ar_order is only used with noise_type "ar_p". For "white", ar_order is forced to 0.'
        )

    sweep_needed = isinstance(noise, list) or isinstance(ar, list)
    return {"__noise_bundle__": combos} if sweep_needed else {}


def generate_run_configs(
    base: Dict[str, Any],
) -> Tuple[list[Dict[str, Any]], dict[str, Sequence[Any]]]:
    """Generate run configurations from the given config file."""

    exclude = [
        "params_to_fit",
        "load_sweep",
        "conditional_params",
        "f_schedule",
        "F",
        "noise_type",
        "ar_order",
    ]  # Keys to exclude from sweep even if they are lists
    sweep = {k: v for k, v in base.items() if _should_sweep_key(k, v, exclude)}

    # Handle forcing schedule sweeps
    sweep.update(expand_forcing_schedule_sweep(base))

    # Handle coupled noise/ar_order sweep
    sweep.update(expand_noise_ar_sweep(base))

    # If we are running a GCM baseline with AR(p) noise, we still need
    # to sweep multiple ar_orders
    if base.get("parameterization_type") == "baseline_ar_p":
        ar_orders = base.get("ar_order")
        if isinstance(ar_orders, list):
            sweep["ar_order"] = ar_orders

    sweep = dict(sorted(sweep.items(), key=lambda item: item[0].lower()))

    # Load sweep should be a subset of the main sweep that are used to load precomputed data
    # e.g., flow GCM will sweep noise type but init states do not consider noise type
    load_sweep = base.get("load_sweep", {})
    load_sweep = dict(sorted(load_sweep.items(), key=lambda item: item[0].lower()))
    sweep.update(load_sweep)

    if not sweep:
        # No sweep parameters
        config = deepcopy(base)
        config["sweep_name"] = ""
        return [config], {}

    # Separate sweep keys and fixed keys
    sweep_keys = list(sweep.keys())
    sweep_values = [sweep[k] for k in sweep_keys]

    # Cartesian product over all sweep lists
    configs = []
    for values in product(*sweep_values):
        cfg = deepcopy(base)

        # Remember which arguments were used for the sweep
        sweep_args = dict()
        for k, val in zip(sweep_keys, values):
            if isinstance(val, dict) and k == "__noise_bundle__":
                # bundled assignment for noise_type
                for kk, vv in val.items():
                    cfg[kk] = vv
                    sweep_args[kk] = vv
            else:
                # If the original config contained a constant f_schedule, it was swept via F
                if k == "F" and base.get("f_schedule") is not None:
                    del cfg["f_schedule"]
                cfg[k] = val
                sweep_args[k] = val
        # Remove temporary keys
        cfg.pop("__noise_bundle__", None)

        # Create an output name based on the sweep arguments
        cfg["sweep_name"] = get_sweep_name(sweep_args)
        # Add load_sweep info to config
        cfg["load_sweep"] = load_sweep

        # Handle conditional parameters
        conditional_params = base.get("conditional_params", {})
        for cond_param, spec in conditional_params.items():
            source_key = spec["depends_on"]
            rules = spec["values"]  # list of {"when": ..., "set": ...}

            if source_key not in cfg:
                raise ValueError(
                    f'Conditional parameter "{cond_param}" depends on missing key "{source_key}".'
                )

            source_val = cfg[source_key]

            for rule in rules:
                cond = rule["when"]
                if isinstance(cond, dict):
                    match = isinstance(source_val, dict) and all(
                        source_val.get(k) == v for k, v in cond.items()
                    )
                else:
                    match = source_val == cond

                if match:
                    cfg[cond_param] = rule["set"]
                    break
        cfg.pop("conditional_params", None)

        configs.append(cfg)

    return configs, sweep


def keep_only_load_sweep(sweep_name: str, load_sweep: dict) -> str:
    """
    Filter sweep identifier to retain only key-value tokens present in ``load_sweep``.

    The input ``sweep_name`` is expected to be a hyphen-separated string of
    ``<key>_<value>`` tokens. For each key listed in ``load_sweep``, the
    corresponding token is kept only if its value matches one of the allowed`
    values in ``load_sweep`` (compared numerically when possible).

    The result is a hyphen-joined string of the retained tokens, in original
    order. If no tokens match, an empty string is returned.

    Example
    -------
    >>> keep_only_load_sweep(
    ...     "hidden_dims_8_8-c_10.0-F_20.0",
    ...     {"F": [18.0, 20.0, 22.0], "c": [5.0, 10.0, 15.0]}
    ... )
    'c_10.0-F_20.0'

    """
    # keys we want to keep
    keys = list(load_sweep.keys())
    key_alt = "|".join(map(re.escape, keys))

    # matches "-c_10.0" or start "c_10.0" (value is anything up to next '-')
    pattern = re.compile(rf"(?:^|-)({key_alt})_([^-]+)")

    kept = []
    for m in pattern.finditer(sweep_name):
        k, v_str = m.group(1), m.group(2)

        # membership test, robust to floats vs strings
        allowed = load_sweep[k]
        ok = False
        # case 1: list-encoded token like "8_8"
        if "_" in v_str:
            ok = v_str in {_format_value(a) for a in allowed}

        # case 2: numeric token
        else:
            try:
                v = float(v_str)
                ok = any(float(a) == v for a in allowed)
            except (TypeError, ValueError):
                # case 3: plain string token
                ok = v_str in {str(a) for a in allowed}

        if ok:
            kept.append(f"{k}_{v_str}")

    return "-".join(kept)
