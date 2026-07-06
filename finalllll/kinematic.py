import numpy as np
import warnings


def get_acc_jerk(time_buf: np.ndarray, vel_buf: np.ndarray, window_size: int, poly_order: int, pos=None):
    """Estimate acceleration and jerk using a local polynomial fit.

    This is in the spirit of a Savitzky–Golay filter, where we fit a polynomial of
    degree ``poly_order`` to a sliding window of velocity samples and then compute
    derivatives at a specified sample index.

    Parameters
    ----------
    time_buf : np.ndarray
        1D array of timestamps (seconds) for the velocity samples.
    vel_buf : np.ndarray
        1D array of velocity samples (deg/s).
    window_size : int
        Number of samples in the sliding window (must equal len(time_buf)).
    poly_order : int
        Polynomial order for the local fit. Must be < window_size.
    pos : int, optional
        Index inside the window to evaluate derivatives at (default center).

    Returns
    -------
    acc : float
        Estimated acceleration (deg/s^2) at time_buf[pos].
    jerk : float
        Estimated jerk (deg/s^3) at time_buf[pos].
    """

    time_buf = np.asarray(time_buf, dtype=float)
    vel_buf = np.asarray(vel_buf, dtype=float)

    if window_size != len(time_buf) or window_size != len(vel_buf):
        raise ValueError("window_size must match length of time_buf and vel_buf")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if poly_order >= window_size:
        raise ValueError("poly_order must be less than window_size")

    if pos is None:
        pos = window_size // 2
    if pos < 0 or pos >= window_size:
        raise ValueError("'pos' parameter must be in range(0, window_size)")

    # Shift time to reduce numerical conditioning issues.
    # This makes polynomial fitting more stable when time values are large (e.g., epoch timestamps).

    t_shift = time_buf - time_buf[0]

    # Fit a local polynomial to the velocity samples (in shifted time)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        coeffs = np.polyfit(t_shift, vel_buf, poly_order)

    d0 = np.polyder(coeffs, 0)
    # Acceleration is the first derivative of velocity
    d1 = np.polyder(coeffs, 1)
    # Jerk is the second derivative of velocity
    d2 = np.polyder(coeffs, 2) if poly_order >= 2 else np.array([0.0])

    # Evaluate at t=0 in the shifted coordinate system (center sample)
    t0 = t_shift[pos]
    vel = np.polyval(d0, t0)
    acc = np.polyval(d1, t0)
    jerk = np.polyval(d2, t0)

    return float(vel), float(acc), float(jerk)

