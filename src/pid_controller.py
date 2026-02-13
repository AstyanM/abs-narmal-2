"""
PID Controller with filtered derivative for ABS slip regulation.

Implements the transfer function from the reference paper:
    U(s) = Kp * ((1 + Td*s) / (1 + alpha*Td*s)) * ((1 + Ti*s) / (Ti*s)) * E(s)
"""

import numpy as np


class PIDController:
    """PID controller with filtered derivative component.

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ti : float
        Integral time constant.
    Td : float
        Derivative time constant.
    alpha : float
        Lag factor in the derivative component.
    dt : float
        Sampling time in seconds.
    """

    def __init__(self, Kp, Ti, Td, alpha, dt):
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.alpha = alpha
        self.dt = dt

        self.integral = 0.0
        self.e_prev = 0.0
        self.d_filtered = 0.0

    def compute(self, error):
        """Compute the control output for the given error signal.

        Parameters
        ----------
        error : float
            Difference between reference and measured slip ratio.

        Returns
        -------
        float
            Control output (brake pressure), clipped to [0, 1800] Pa.
        """
        self.integral += error * self.dt

        raw_derivative = (error - self.e_prev) / self.dt
        self.d_filtered = (
            (self.alpha * self.Td * self.d_filtered + raw_derivative * self.dt)
            / (self.alpha * self.Td + self.dt)
        )

        u = self.Kp * (error + (self.integral / self.Ti) + self.Td * self.d_filtered)

        self.e_prev = error

        return np.clip(u, 0, 1800)

    def reset(self):
        """Reset the internal state of the controller."""
        self.integral = 0.0
        self.e_prev = 0.0
        self.d_filtered = 0.0
