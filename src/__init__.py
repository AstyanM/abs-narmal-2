"""
NARMA-L2 Neural Network Controller for Anti-Lock Braking System (ABS)

A quarter-car model simulation with NARMA-L2 feedback linearization
controller for optimal slip ratio regulation.
"""

from .abs_simulator import ABSSystemSimulator
from .pid_controller import PIDController
from .narmal2_controller import NARMAL2Controller

__all__ = ["ABSSystemSimulator", "PIDController", "NARMAL2Controller"]
