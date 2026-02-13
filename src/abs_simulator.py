"""
Quarter-car ABS system simulator.

Implements the mathematical model from the reference paper:
    - Vehicle dynamics:          M * dv/dt = -mu(lambda) * Fz - Cx * v^2
    - Wheel rotational dynamics: I * domega/dt = mu(lambda) * Fz * r - B * omega - tau_b
    - Electro-mechanical brake:  dtau_b/dt = (-tau_b + Kb * vb) / tau

References equations (1)-(7) of:
    J. O Pedro, O. T. C Nyandoro, S John,
    "Neural Network Based Feedback Linearisation Slip Control of an Anti-Lock Braking System",
    7th Asian Control Conference, Hong Kong, China, August 27-29, 2009.
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from .pid_controller import PIDController


class ABSSystemSimulator:
    """ABS system simulator based on a quarter-car model.

    Generates synthetic braking data for neural network training
    using the mathematical model from the reference paper.

    Parameters
    ----------
    mu0 : float, optional
        Maximum friction coefficient (default: 0.9, dry asphalt).
    lambda0 : float, optional
        Optimal slip ratio (default: 0.25).
    """

    def __init__(self, mu0=0.9, lambda0=0.25):
        # Model parameters from Table I of the paper
        self.M = 440        # Total mass (kg)
        self.I = 1.6        # Wheel moment of inertia (kg.m^2)
        self.r = 0.3        # Wheel radius (m)
        self.B = 0.08       # Rolling friction coefficient (kg.m^2/s)
        self.Cx = 0.856     # Aerodynamic friction coefficient (kg/m)
        self.Kb = 0.8       # Braking gain
        self.tau = 0.003    # Electro-mechanical system time constant (s)
        self.g = 9.81       # Gravitational acceleration (m/s^2)
        self.Fz = self.M * self.g  # Normal force (N)

        # Road surface parameters
        self.mu0 = mu0
        self.lambda0 = lambda0

    def friction_model(self, slip_ratio):
        """Compute the tire-road friction coefficient.

        Implements the mu-lambda friction model (equation 6):
            mu(lambda) = (2 * mu0 * lambda0 * lambda) / (lambda0^2 + lambda^2)

        Parameters
        ----------
        slip_ratio : float
            Current wheel slip ratio.

        Returns
        -------
        float
            Friction coefficient.
        """
        lambda_val = np.abs(slip_ratio)
        mu = (2 * self.mu0 * self.lambda0 * lambda_val) / (
            self.lambda0**2 + lambda_val**2
        )
        return mu

    def system_dynamics(self, state, t, control_input):
        """Compute the state derivatives for the quarter-car model.

        Parameters
        ----------
        state : list of float
            Current state [omega, v, tau_b].
        t : float
            Current time (required by odeint, unused).
        control_input : float
            Brake voltage command.

        Returns
        -------
        list of float
            State derivatives [domega/dt, dv/dt, dtau_b/dt].
        """
        omega, v, tau_b = state

        if v > 0.5:
            slip_ratio = (v - self.r * omega) / v
        else:
            slip_ratio = 0

        mu = self.friction_model(slip_ratio)

        domega_dt = (mu * self.Fz * self.r / self.I) - (self.B * omega / self.I) - (tau_b / self.I)
        dv_dt = -(mu * self.Fz / self.M) - (self.Cx * v**2 / self.M)
        dtau_b_dt = (-tau_b + self.Kb * control_input) / self.tau

        return [domega_dt, dv_dt, dtau_b_dt]

    def generate_training_data(self, n_scenarios=50, duration=5.0, dt=0.01, controlled=True):
        """Generate synthetic ABS training data.

        Simulates multiple braking scenarios with randomized initial conditions
        using a PID controller to regulate the slip ratio.

        Parameters
        ----------
        n_scenarios : int
            Number of braking scenarios to simulate.
        duration : float
            Maximum simulation duration in seconds.
        dt : float
            Time step in seconds.
        controlled : bool
            If True, use a PID controller to regulate slip.

        Returns
        -------
        pd.DataFrame
            Training data with columns: time, scenario, omega, v, tau_b,
            control_input, slip_ratio, mu, slip_k_1, slip_k_2,
            control_k_1, control_k_2.
        """
        time_points = np.arange(0, duration, dt)
        all_data = []

        for scenario in range(n_scenarios):
            v0 = np.random.uniform(20, 40)
            initial_slip = np.random.uniform(0.1, 0.4)
            omega0 = v0 * (1 - initial_slip) / self.r
            tau_b0 = np.random.uniform(0, 20)
            state = [omega0, v0, tau_b0]
            trajectory = []

            if controlled:
                pid = PIDController(Kp=3000, Ti=0.1, Td=0.01, alpha=0.1, dt=dt)

            for i, t in enumerate(time_points):
                omega, v, tau_b = state
                if v > 0.1:
                    slip_ratio = (v - self.r * omega) / v
                else:
                    slip_ratio = 0

                mu = self.friction_model(slip_ratio)

                if controlled:
                    error = self.lambda0 - slip_ratio
                    vb_control = pid.compute(error)

                if i > 0:
                    state = odeint(
                        self.system_dynamics, state,
                        [time_points[i - 1], t], args=(vb_control,)
                    )[1]

                omega, v, tau_b = state

                if v > 0.1:
                    slip_ratio = (v - self.r * omega) / v
                else:
                    slip_ratio = 0

                data_point = {
                    "time": t,
                    "scenario": scenario,
                    "omega": omega,
                    "v": v,
                    "tau_b": tau_b,
                    "control_input": vb_control,
                    "slip_ratio": slip_ratio,
                    "mu": mu,
                    "slip_k_1": 0 if i == 0 else trajectory[i - 1]["slip_ratio"],
                    "slip_k_2": 0 if i < 2 else trajectory[i - 2]["slip_ratio"],
                    "control_k_1": 0 if i == 0 else trajectory[i - 1]["control_input"],
                    "control_k_2": 0 if i < 2 else trajectory[i - 2]["control_input"],
                }

                trajectory.append(data_point)
                all_data.append(data_point)

                if v < 0.5:
                    break

        return pd.DataFrame(all_data)
