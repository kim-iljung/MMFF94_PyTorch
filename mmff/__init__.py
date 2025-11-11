"""MMFF94 implementation using PyTorch.

The package exposes a :class:`MMFFForceField` class that can be used to
compute molecular mechanics energies with PyTorch tensors.  The module is
 designed to follow the same spirit as the ``UFF_PyTorch`` project while
staying lightweight and dependency free.
"""

from .forcefield import MMFFForceField
from . import parameters
from .builder import build_forcefield, forcefield_from_file

__all__ = [
    "MMFFForceField",
    "parameters",
    "build_forcefield",
    "forcefield_from_file",
]
