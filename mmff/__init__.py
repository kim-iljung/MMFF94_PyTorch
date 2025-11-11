"""MMFF94 implementation using PyTorch."""

from .forcefield import MMFFForceField
from . import parameters

__all__ = ["MMFFForceField", "parameters"]

try:  # pragma: no cover - optional RDKit dependency
    from .builder import build_mmff_forcefield, forcefield_from_file
    from . import atom_typing
except ImportError:  # RDKit not available
    build_mmff_forcefield = None  # type: ignore[assignment]
    forcefield_from_file = None  # type: ignore[assignment]
    atom_typing = None  # type: ignore[assignment]
else:
    __all__ += ["build_mmff_forcefield", "forcefield_from_file", "atom_typing"]
