import numpy as np
import pytest

from models.forcing_schedule import forcing_at
from models.GCM.time_stepping import RK2, RK4, euler_forward
from models.helpers import (
    L96_2t_x_dot_y_dot,
    L96_eq1_x_dot,
    parse_time_stepping_func,
)


def test_l96_2t_x_dot_y_dot_shapes(sample_x, sample_y, l96_config):
    """Test that L96_2t_x_dot_y_dot returns correct shapes."""
    F = forcing_at(l96_config.f_schedule, t=0.0)
    x_dot, y_dot = L96_2t_x_dot_y_dot(
        sample_x,
        sample_y,
        F,
        l96_config.h,
        l96_config.b,
        l96_config.c,
    )

    assert x_dot.shape == sample_x.shape
    assert y_dot.shape == sample_y.shape
    assert np.all(np.isfinite(x_dot))
    assert np.all(np.isfinite(y_dot))


def test_l96_eq1_x_dot_shapes(sample_x, l96_config):
    """Test that L96_eq1_x_dot returns correct shape."""
    F = forcing_at(l96_config.f_schedule, t=0.0)
    x_dot = L96_eq1_x_dot(sample_x, F)

    assert x_dot.shape == sample_x.shape
    assert np.all(np.isfinite(x_dot))


def test_parse_time_stepping_func_known_values():
    """Ensure known time stepping strings resolve to the correct functions."""
    assert parse_time_stepping_func("euler_forward") is euler_forward
    assert parse_time_stepping_func("RK2") is RK2
    assert parse_time_stepping_func("RK4") is RK4


def test_parse_time_stepping_func_unknown_value():
    """Unknown time stepping strings should raise a ValueError."""
    with pytest.raises(ValueError):
        parse_time_stepping_func("not_a_method")
