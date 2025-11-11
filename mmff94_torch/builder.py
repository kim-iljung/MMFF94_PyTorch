"""Utilities to build :class:`mmff.forcefield.MMFFForceField` instances."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import rdmolops

from .atom_typing import assign_atom_types_and_charges
from .forcefield import MMFFForceField


@dataclass(frozen=True)
class Topology:
    """Connectivity information extracted from an RDKit molecule."""

    bonds: List[Tuple[int, int]]
    angles: List[Tuple[int, int, int]]
    torsions: List[Tuple[int, int, int, int]]
    pairs: List[Tuple[int, int]]


def _list_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    bonds: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((begin, end))
    return bonds


def _list_angles(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
    angles: List[Tuple[int, int, int]] = []
    for atom in mol.GetAtoms():
        center = atom.GetIdx()
        neighbours = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        for first, second in combinations(neighbours, 2):
            angles.append((first, center, second))
    return angles


def _list_torsions(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    torsions: List[Tuple[int, int, int, int]] = []
    seen = set()
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        neighbours_j = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(j).GetNeighbors() if nbr.GetIdx() != k]
        neighbours_k = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(k).GetNeighbors() if nbr.GetIdx() != j]
        for i in neighbours_j:
            for l in neighbours_k:
                torsion = (i, j, k, l)
                reverse = (l, k, j, i)
                if reverse in seen:
                    continue
                seen.add(torsion)
                torsions.append(torsion)
    return torsions


def _list_nonbonded_pairs(mol: Chem.Mol) -> List[Tuple[int, int]]:
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    pairs: List[Tuple[int, int]] = []
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if distance_matrix[i, j] >= 3:  # exclude 1-2 and 1-3 interactions
                pairs.append((i, j))
    return pairs


def build_topology(mol: Chem.Mol) -> Topology:
    """Extract bond, angle, torsion and non-bonded pair indices from ``mol``."""

    return Topology(
        bonds=_list_bonds(mol),
        angles=_list_angles(mol),
        torsions=_list_torsions(mol),
        pairs=_list_nonbonded_pairs(mol),
    )


def build_forcefield(
    mol: Chem.Mol,
    *,
    mmff_variant: str = "MMFF94",
    sanitize: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> MMFFForceField:
    """Construct an :class:`MMFFForceField` from an RDKit molecule."""

    atom_types, charges = assign_atom_types_and_charges(
        mol,
        mmff_variant=mmff_variant,
        sanitize=sanitize,
    )
    topology = build_topology(mol)
    return MMFFForceField(
        atom_types,
        bonds=topology.bonds,
        angles=topology.angles,
        torsions=topology.torsions,
        pairs=topology.pairs,
        charges=charges,
        device=device,
        dtype=dtype,
    )


def _load_molecule(path: Path, sanitize: bool = True) -> Chem.Mol:
    ext = path.suffix.lower()
    if ext in {".mol", ".sdf"}:
        mol = Chem.MolFromMolFile(str(path), sanitize=sanitize, removeHs=False)
    elif ext in {".smi", ".smiles"}:
        text = path.read_text(encoding="utf8").strip()
        mol = Chem.MolFromSmiles(text, sanitize=sanitize)
    else:
        raise ValueError(f"Unsupported molecule format '{path.suffix}'")
    if mol is None:
        raise ValueError(f"RDKit failed to parse molecule file '{path}'")
    return mol


def forcefield_from_file(
    path: Path | str,
    *,
    mmff_variant: str = "MMFF94",
    sanitize: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> MMFFForceField:
    """Load a molecule from ``path`` and build the corresponding force field."""

    mol = _load_molecule(Path(path), sanitize=sanitize)
    # The molecule has already been sanitised.  Avoid repeating the work when
    # determining the MMFF parameters.
    return build_forcefield(
        mol,
        mmff_variant=mmff_variant,
        sanitize=False,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "Topology",
    "build_forcefield",
    "build_topology",
    "forcefield_from_file",
]
