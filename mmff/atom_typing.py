"""Utilities for assigning MMFF atom types from RDKit molecules."""
from __future__ import annotations

from typing import List

try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType
except ImportError as exc:  # pragma: no cover - RDKit is optional at runtime
    raise ImportError(
        "RDKit is required for mmff.atom_typing; install rdkit-pypi or rdkit"
    ) from exc


def _carbon_type(atom: "Chem.Atom") -> str:
    if atom.GetIsAromatic():
        return "C.ar"
    if atom.GetHybridization() == HybridizationType.SP:
        return "C.sp"
    return "C"


def _nitrogen_type(atom: "Chem.Atom") -> str:
    if atom.GetIsAromatic() or atom.GetHybridization() in {
        HybridizationType.SP2,
        HybridizationType.SP,
    }:
        return "N.pl3"
    return "N"


def _oxygen_type(atom: "Chem.Atom") -> str:
    if atom.GetFormalCharge() == -1 or atom.GetTotalValence() == 1:
        return "O.co2"
    return "O"


def _sulfur_type(atom: "Chem.Atom") -> str:
    return "S"


def _halogen_type(symbol: str) -> str:
    return symbol


def infer_atom_type(atom: "Chem.Atom") -> str:
    """Infer an MMFF94 atom type for an RDKit atom."""

    symbol = atom.GetSymbol()
    if symbol == "C":
        return _carbon_type(atom)
    if symbol == "N":
        return _nitrogen_type(atom)
    if symbol == "O":
        return _oxygen_type(atom)
    if symbol == "S":
        return _sulfur_type(atom)
    if symbol in {"F", "Cl"}:
        return _halogen_type(symbol)
    if symbol == "H":
        return "H"
    raise KeyError(f"Unsupported atom symbol '{symbol}' for MMFF typing")


def assign_atom_types(mol: "Chem.Mol") -> List[str]:
    """Return a list of MMFF atom types for *mol*."""

    return [infer_atom_type(atom) for atom in mol.GetAtoms()]


def compute_gasteiger_charges(mol: "Chem.Mol", *, max_iter: int = 12) -> List[float]:
    """Compute Gasteiger partial charges for use with the force field."""

    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, maxIters=max_iter)
    charges: List[float] = []
    for atom in mol.GetAtoms():
        value = atom.GetDoubleProp("_GasteigerCharge")
        # RDKit returns very small positive numbers instead of zero for some atoms
        if abs(value) < 1e-6:
            value = 0.0
        charges.append(float(value))
    return charges


def ensure_molecule_has_hs(mol: "Chem.Mol") -> "Chem.Mol":
    """Return a copy of *mol* that explicitly contains hydrogens."""

    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        return mol
    return Chem.AddHs(mol)


def sanitize_molecule(mol: "Chem.Mol", sanitize: bool = True) -> "Chem.Mol":
    """Optionally sanitize the RDKit molecule before typing."""

    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def load_molecule(path: str, *, sanitize: bool = True, remove_hs: bool = False) -> "Chem.Mol":
    """Load an RDKit molecule from *path* (SDF/MOL supported)."""

    mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=remove_hs)
    if mol is None:
        raise ValueError(f"Failed to read molecule from '{path}'")
    return sanitize_molecule(mol, sanitize=sanitize)


def prepare_molecule(
    mol: "Chem.Mol",
    *,
    sanitize: bool = True,
    add_hs: bool = True,
) -> "Chem.Mol":
    """Return a molecule ready for MMFF typing."""

    mol = Chem.Mol(mol)
    mol = sanitize_molecule(mol, sanitize=sanitize)
    if add_hs:
        mol = ensure_molecule_has_hs(mol)
    return mol
