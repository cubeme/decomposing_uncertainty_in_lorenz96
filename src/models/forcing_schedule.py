"""Define constant and time-varying forcing schedules."""

import numpy as np
from numba import njit


class ConstantForcingSchedule:
    """Constant forcing schedule: F is constant over time.

    Config specification:
        f_schedule:
            type: constant
            F: 20.0
    """

    def __init__(self, F):
        self.F = float(F)


class LinearForcingSchedule:
    """Linear forcing schedule:
    F varies linearly from F0 at time t0 to F1 at time t1.

    Config specification:
        f_schedule:
            type: linear
            F0: 18.0
            F1: 22.0
            t0: 0.0
            t1: 5000.0

    Note: Linear schedule will only activate after spin up.
    During spin up, F is constant at F0.
    """

    def __init__(self, F0, F1, t0, t1):
        self.F0, self.F1, self.t0, self.t1 = map(float, (F0, F1, t0, t1))


class OscillatingForcingSchedule:
    """Oscillating forcing schedule:
    F varies sinusoidally around Fmean with given amplitude and frequency.

    Config specification:
        f_schedule:
            type: oscillating
            Fmean: 20.0
            amp: 2.0
            freq: 0.5

    Note: Oscillating schedule will only activate after spin up.
    During spin up, F is constant at Fmean.
    """

    def __init__(self, Fmean, amp, freq):
        self.Fmean, self.amp, self.freq = map(float, (Fmean, amp, freq))


def parse_forcing_schedule(schedule_data):
    if isinstance(
        schedule_data,
        (ConstantForcingSchedule, LinearForcingSchedule, OscillatingForcingSchedule),
    ):
        return schedule_data

    if isinstance(schedule_data, (int, float)):
        return ConstantForcingSchedule(schedule_data)

    if not isinstance(schedule_data, dict):
        raise TypeError(
            "f_schedule must be a dict or a ForcingSchedule instance; "
            f"got {type(schedule_data)}"
        )

    schedule_type = schedule_data.get("type")
    if schedule_type is None:
        raise ValueError("f_schedule must include a 'type' field.")

    schedule_type = str(schedule_type).lower()
    if schedule_type == "constant":
        required = ("F",)
        missing = [k for k in required if k not in schedule_data]
        if missing:
            raise ValueError(
                f"Constant f_schedule missing required keys: {sorted(missing)}"
            )
        return ConstantForcingSchedule(schedule_data["F"])

    if schedule_type == "linear":
        required = ("F0", "F1", "t0", "t1")
        missing = [k for k in required if k not in schedule_data]
        if missing:
            raise ValueError(
                f"Linear f_schedule missing required keys: {sorted(missing)}"
            )
        return LinearForcingSchedule(
            schedule_data["F0"],
            schedule_data["F1"],
            schedule_data["t0"],
            schedule_data["t1"],
        )

    if schedule_type == "oscillating":
        required = ("Fmean", "amp", "freq")
        missing = [k for k in required if k not in schedule_data]
        if missing:
            raise ValueError(
                f"Oscillating f_schedule missing required keys: {sorted(missing)}"
            )
        return OscillatingForcingSchedule(
            schedule_data["Fmean"],
            schedule_data["amp"],
            schedule_data["freq"],
        )

    raise ValueError(
        "Invalid f_schedule type: "
        f"{schedule_type}. Must be 'constant', 'linear', or 'oscillating'."
    )


# todo: add test for this function
def forcing_schedule_to_dict(schedule):
    """
    Convert a ForcingSchedule object into a dictionary representation
    compatible with parse_forcing_schedule().
    """
    if isinstance(schedule, ConstantForcingSchedule):
        return {
            "type": "constant",
            "F": schedule.F,
        }

    if isinstance(schedule, LinearForcingSchedule):
        return {
            "type": "linear",
            "F0": schedule.F0,
            "F1": schedule.F1,
            "t0": schedule.t0,
            "t1": schedule.t1,
        }

    if isinstance(schedule, OscillatingForcingSchedule):
        return {
            "type": "oscillating",
            "Fmean": schedule.Fmean,
            "amp": schedule.amp,
            "freq": schedule.freq,
        }

    raise TypeError(
        f"f_schedule must be a ForcingSchedule instance; got {type(schedule)}"
    )


def forcing_at(schedule, t: float) -> float:
    if isinstance(schedule, ConstantForcingSchedule):
        return F_const(t, schedule.F)
    if isinstance(schedule, LinearForcingSchedule):
        return F_linear(t, schedule.F0, schedule.F1, schedule.t0, schedule.t1)
    if isinstance(schedule, OscillatingForcingSchedule):
        return F_osc(t, schedule.Fmean, schedule.amp, schedule.freq)
    raise TypeError(f"Unknown forcing schedule type: {type(schedule)}")


def forcing_at_array(schedule, t_values):
    t_values = np.asarray(t_values, dtype=float)
    return np.array([forcing_at(schedule, t) for t in t_values], dtype=float)


@njit
def F_const(t, F):
    return F


@njit
def F_linear(t, F0, F1, t0, t1):
    if t <= t0:
        return F0
    elif t >= t1:
        return F1
    else:
        return F0 + (F1 - F0) * (t - t0) / (t1 - t0)


@njit
def F_osc(t, Fmean, amp, freq):
    return Fmean + amp * np.sin(2.0 * np.pi * freq * t)
