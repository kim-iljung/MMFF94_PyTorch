"""Helpers to create MMFF force fields directly from RDKit molecules."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - RDKit is optional at runtime
    raise ImportError(
        "RDKit is required for mmff.builder; install rdkit-pypi or rdkit"
    ) from exc

from .atom_typing import (
    assign_atom_types,
    compute_gasteiger_charges,
    load_molecule,
    prepare_molecule,
)
from .forcefield import MMFFForceField


def _collect_bonds(mol: "Chem.Mol") -> List[Tuple[int, int]]:
    bonds: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds.append((i, j))
    return bonds


def _collect_angles(mol: "Chem.Mol") -> List[Tuple[int, int, int]]:
    angles: List[Tuple[int, int, int]] = []
    for atom in mol.GetAtoms():
        center = atom.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        for i, j in combinations(neighbors, 2):
            angles.append((i, center, j))
    return angles


def _collect_torsions(mol: "Chem.Mol") -> List[Tuple[int, int, int, int]]:
    torsions: List[Tuple[int, int, int, int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        left = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(i).GetNeighbors() if nbr.GetIdx() != j]
        right = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(j).GetNeighbors() if nbr.GetIdx() != i]
        for a in left:
            for b in right:
                torsions.append((a, i, j, b))
    return torsions


def _collect_nonbonded_pairs(
    mol: "Chem.Mol",
    *,
    exclude_bonds: Iterable[Tuple[int, int]],
    exclude_angles: Iterable[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    excluded = {tuple(sorted(pair)) for pair in exclude_bonds}
    excluded.update(tuple(sorted(pair)) for pair in exclude_angles)

    pairs: List[Tuple[int, int]] = []
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            pair = (i, j)
            if pair in excluded:
                continue
            pairs.append(pair)
    return pairs


def _angle_exclusions(angles: Iterable[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    exclusions: List[Tuple[int, int]] = []
    for i, _, k in angles:
        exclusions.append(tuple(sorted((i, k))))
    return exclusions


def build_mmff_forcefield(
    mol: "Chem.Mol",
    *,
    compute_charges: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> MMFFForceField:
    """Construct an :class:`MMFFForceField` from *mol*."""

    prepared = prepare_molecule(mol)
    atom_types = assign_atom_types(prepared)
    bonds = _collect_bonds(prepared)
    angles = _collect_angles(prepared)
    torsions = _collect_torsions(prepared)
    angle_exclusions = _angle_exclusions(angles)
    pairs = _collect_nonbonded_pairs(prepared, exclude_bonds=bonds, exclude_angles=angle_exclusions)

    charges: Optional[Sequence[float]] = None
    if compute_charges:
        charges = compute_gasteiger_charges(prepared)

    return MMFFForceField(
        atom_types=atom_types,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        pairs=pairs,
        charges=charges,
        dtype=dtype,
        device=device,
    )


def forcefield_from_file(
    path: str,
    *,
    sanitize: bool = True,
    add_hs: bool = True,
    compute_charges: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> MMFFForceField:
    """Convenience wrapper that loads a molecule from *path* and builds the force field."""

    mol = load_molecule(path, sanitize=sanitize, remove_hs=not add_hs)
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)
    return build_mmff_forcefield(
        mol,
        compute_charges=compute_charges,
        dtype=dtype,
        device=device,
    )
