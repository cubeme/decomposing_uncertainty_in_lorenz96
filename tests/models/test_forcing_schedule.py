import numpy as np
import pytest

from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
    forcing_at,
    forcing_at_array,
    parse_forcing_schedule,
)


def test_parse_forcing_schedule_constant_number():
    schedule = parse_forcing_schedule(12.5)
    assert isinstance(schedule, ConstantForcingSchedule)
    assert schedule.F == 12.5


def test_parse_forcing_schedule_constant_dict():
    schedule = parse_forcing_schedule({"type": "constant", "F": 18.0})
    assert isinstance(schedule, ConstantForcingSchedule)
    assert schedule.F == 18.0


def test_parse_forcing_schedule_linear_dict():
    schedule = parse_forcing_schedule(
        {"type": "linear", "F0": 10.0, "F1": 20.0, "t0": 0.0, "t1": 2.0}
    )
    assert isinstance(schedule, LinearForcingSchedule)
    assert schedule.F0 == 10.0
    assert schedule.F1 == 20.0
    assert schedule.t0 == 0.0
    assert schedule.t1 == 2.0


def test_parse_forcing_schedule_oscillating_dict():
    schedule = parse_forcing_schedule(
        {"type": "oscillating", "Fmean": 10.0, "amp": 2.0, "freq": 0.5}
    )
    assert isinstance(schedule, OscillatingForcingSchedule)
    assert schedule.Fmean == 10.0
    assert schedule.amp == 2.0
    assert schedule.freq == 0.5


def test_parse_forcing_schedule_invalid_type():
    with pytest.raises(ValueError, match="Invalid f_schedule type"):
        parse_forcing_schedule({"type": "unknown", "F": 20.0})


def test_parse_forcing_schedule_missing_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        parse_forcing_schedule({"type": "linear", "F0": 10.0, "F1": 20.0})


def test_forcing_at_constant():
    schedule = ConstantForcingSchedule(20.0)
    assert forcing_at(schedule, 0.0) == 20.0
    assert forcing_at(schedule, 1.5) == 20.0


def test_forcing_at_linear():
    schedule = LinearForcingSchedule(F0=10.0, F1=20.0, t0=0.0, t1=2.0)
    assert forcing_at(schedule, -1.0) == 10.0
    assert forcing_at(schedule, 0.0) == 10.0
    assert forcing_at(schedule, 1.0) == 15.0
    assert forcing_at(schedule, 2.0) == 20.0
    assert forcing_at(schedule, 3.0) == 20.0


def test_forcing_at_oscillating():
    schedule = OscillatingForcingSchedule(Fmean=10.0, amp=2.0, freq=0.5)
    assert np.isclose(forcing_at(schedule, 0.0), 10.0)
    assert np.isclose(forcing_at(schedule, 1.0), 10.0)


def test_forcing_at_array_matches_scalar():
    schedule = LinearForcingSchedule(F0=10.0, F1=20.0, t0=0.0, t1=2.0)
    t_vals = np.array([0.0, 0.5, 1.0, 2.0])
    expected = np.array([forcing_at(schedule, t) for t in t_vals])
    assert np.allclose(forcing_at_array(schedule, t_vals), expected)
