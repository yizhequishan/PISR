"""Modular, dimensionless 2-D incompressible Navier--Stokes PINN baselines."""

from .config import Config
from .model import FlowMLP
from .physics import navier_stokes

__all__ = ["Config", "FlowMLP", "navier_stokes"]
