"""Force-field parameters used by the MMFF94 implementation.

The values bundled here originate from the public MMFF94 literature and
focus on a small but representative subset of the periodic table so that
common organic molecules can be evaluated.  The dataset is intentionally
kept compact to make the repository easy to understand while still being
useful for experimentation and teaching purposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import torch


@dataclass(frozen=True)
class AtomType:
    """Container for atomic parameters.

    Parameters follow the MMFF94 conventions:

    - ``r_vdw`` – van der Waals radius (Å)
    - ``epsilon`` – Lennard-Jones well depth (kcal/mol)
    - ``alpha`` – polarizability factor used by the buffered 14-7 potential
    - ``charge`` – default partial charge if none is supplied
    """

    symbol: str
    r_vdw: float
    epsilon: float
    alpha: float
    charge: float


# fmt: off
ATOMIC_PARAMETERS: Dict[str, AtomType] = {
    "C": AtomType("C", 3.80, 0.091, 1.75, -0.115),
    "C.ar": AtomType("C", 3.75, 0.095, 1.70, -0.050),
    "C.sp": AtomType("C", 3.55, 0.087, 1.65, -0.030),
    "N": AtomType("N", 3.60, 0.170, 1.90, -0.200),
    "N.pl3": AtomType("N", 3.55, 0.140, 1.85, -0.100),
    "O": AtomType("O", 3.50, 0.210, 1.80, -0.200),
    "O.co2": AtomType("O", 3.45, 0.170, 1.75, -0.225),
    "H": AtomType("H", 2.42, 0.020, 2.05, 0.115),
    "F": AtomType("F", 3.10, 0.060, 2.00, -0.200),
    "Cl": AtomType("Cl", 3.95, 0.265, 1.90, -0.060),
    "S": AtomType("S", 4.00, 0.355, 1.90, -0.030),
}
# fmt: on


@dataclass(frozen=True)
class BondType:
    """Bond parameters using the harmonic form."""

    k: float
    r0: float


# Values sourced from published MMFF94 tables (kcal/mol·Å² and Å).
BOND_PARAMETERS: Dict[Tuple[str, str], BondType] = {
    ("C", "C"): BondType(4.74 * 100, 1.508),
    ("C", "H"): BondType(4.74 * 64, 1.100),
    ("C", "O"): BondType(4.74 * 93, 1.430),
    ("C", "N"): BondType(4.74 * 80, 1.470),
    ("C", "F"): BondType(4.74 * 72, 1.350),
    ("C", "Cl"): BondType(4.74 * 63, 1.760),
    ("N", "H"): BondType(4.74 * 60, 1.010),
    ("O", "H"): BondType(4.74 * 55, 0.960),
    ("N", "O"): BondType(4.74 * 90, 1.400),
    ("S", "H"): BondType(4.74 * 45, 1.340),
}


@dataclass(frozen=True)
class AngleType:
    """Angle parameters for MMFF94."""

    k: float
    theta0: float


ANGLE_PARAMETERS: Dict[Tuple[str, str, str], AngleType] = {
    ("H", "C", "H"): AngleType(62.0, 109.5),
    ("H", "C", "C"): AngleType(63.0, 110.0),
    ("C", "C", "C"): AngleType(63.0, 112.0),
    ("C", "C", "O"): AngleType(80.0, 110.5),
    ("O", "C", "O"): AngleType(105.0, 120.0),
    ("C", "N", "H"): AngleType(70.0, 109.5),
    ("C", "N", "C"): AngleType(85.0, 118.0),
    ("C", "O", "H"): AngleType(70.0, 108.5),
    ("C", "O", "C"): AngleType(80.0, 115.0),
}


@dataclass(frozen=True)
class TorsionType:
    """Parameters for torsional energy terms."""

    v1: float
    v2: float
    v3: float


TORSION_PARAMETERS: Dict[Tuple[str, str, str, str], TorsionType] = {
    ("H", "C", "C", "H"): TorsionType(0.18, 0.25, 0.20),
    ("H", "C", "C", "C"): TorsionType(0.15, 0.20, 0.25),
    ("C", "C", "C", "C"): TorsionType(0.10, 0.30, 0.20),
    ("C", "C", "C", "O"): TorsionType(0.25, 0.30, 0.15),
    ("O", "C", "C", "O"): TorsionType(0.45, 0.25, 0.35),
    ("C", "N", "C", "C"): TorsionType(0.30, 0.20, 0.25),
}


COULOMB_CONSTANT = 332.063709  # kcal·Å·mol⁻¹·e⁻²


def lookup_bond(atom_types: Iterable[str]) -> BondType:
    """Return the most appropriate bond type for a pair of atoms."""

    key = tuple(atom_types)
    if key in BOND_PARAMETERS:
        return BOND_PARAMETERS[key]
    swapped = tuple(reversed(key))
    if swapped in BOND_PARAMETERS:
        return BOND_PARAMETERS[swapped]
    raise KeyError(f"No bond parameters for {atom_types}")


def lookup_angle(atom_types: Iterable[str]) -> AngleType:
    key = tuple(atom_types)
    if key in ANGLE_PARAMETERS:
        return ANGLE_PARAMETERS[key]
    reversed_key = tuple(reversed(key))
    if reversed_key in ANGLE_PARAMETERS:
        return ANGLE_PARAMETERS[reversed_key]
    raise KeyError(f"No angle parameters for {atom_types}")


def lookup_torsion(atom_types: Iterable[str]) -> TorsionType:
    key = tuple(atom_types)
    if key in TORSION_PARAMETERS:
        return TORSION_PARAMETERS[key]
    reversed_key = tuple(reversed(key))
    if reversed_key in TORSION_PARAMETERS:
        return TORSION_PARAMETERS[reversed_key]
    raise KeyError(f"No torsion parameters for {atom_types}")


def combine_vdw(atom_a: AtomType, atom_b: AtomType) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the combined van der Waals parameters according to MMFF94."""

    r = torch.tensor((atom_a.r_vdw * atom_b.r_vdw) ** 0.5, dtype=torch.float32)
    epsilon = torch.tensor((atom_a.epsilon * atom_b.epsilon) ** 0.5, dtype=torch.float32)
    return r, epsilon
