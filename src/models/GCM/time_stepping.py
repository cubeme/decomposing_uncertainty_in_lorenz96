"""Provide time-stepping schemes for coarse-grained models."""

def euler_forward(fn, dt, x, t):
    """
    Perform a single Euler forward time-stepping step for d/dt X = fn(X, t, ...).

    Args:
        fn (callable): The function returning the time rate of change of the variables X.
        dt (float): The time step size.
        x (np.ndarray): Current state of the variables at time t.

    Returns:
        np.ndarray: Updated state of the variables at time t + dt.
    """

    return x + dt * fn(x, t)


def RK2(fn, dt, x, t):
    """
    Perform a single second-order Runge-Kutta (RK2) time-stepping step for d/dt X = fn(X, t, ...).

    Args:
        fn (callable): The function returning the time rate of change of the variables X.
        dt (float): The time step size.
        x (np.ndarray): Current state of the variables at time t.

    Returns:
        np.ndarray: Updated state of the variables at time t + dt.
    """
    x_dot1 = fn(x, t)
    x_dot2 = fn(x + 0.5 * dt * x_dot1, t + 0.5 * dt)
    return x + dt * x_dot2


def RK4(fn, dt, x, t):
    """
    Perform a single second-order Runge-Kutta (RK2) time-stepping step for d/dt X = fn(X, t, ...).

    Args:
        fn (callable): The function returning the time rate of change of the variables X.
        dt (float): The time step size.
        x (np.ndarray): Current state of the variables at time t.

    Returns:
        np.ndarray: Updated state of the variables at time t + dt.
    """

    x_dot1 = fn(x, t)
    x_dot2 = fn(x + 0.5 * dt * x_dot1, t + 0.5 * dt)
    x_dot3 = fn(x + 0.5 * dt * x_dot2, t + 0.5 * dt)
    x_dot4 = fn(x + dt * x_dot3, t + dt)
    return x + (dt / 6.0) * ((x_dot1 + x_dot4) + 2.0 * (x_dot2 + x_dot3))
