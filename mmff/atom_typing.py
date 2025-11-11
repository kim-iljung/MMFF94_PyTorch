"""MMFF atom typing utilities leveraging RDKit data structures.

This module mirrors the spirit of the :mod:`UFF_PyTorch.atom_typing` helpers
by exposing light-weight wrappers that translate a :class:`rdkit.Chem.Mol`
object into the symbolic atom types and partial charges required by the
``MMFFForceField`` implementation.  The functions are intentionally thin and
avoid any PyTorch specific logic so they can be reused in different contexts
(such as data preprocessing pipelines).
"""
from __future__ import annotations

from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


def _ensure_mmff_properties(
    mol: Chem.Mol,
    mmff_variant: str = "MMFF94",
    sanitize: bool = True,
) -> AllChem.MMFFMolProperties:
    """Return the RDKit MMFF property container for ``mol``.

    Parameters
    ----------
    mol:
        Molecule for which the MMFF properties should be computed.  The
        molecule is optionally sanitised to match RDKit's expectations.
    mmff_variant:
        Either ``"MMFF94"`` or ``"MMFF94s"``.  The value is forwarded to
        :func:`rdkit.Chem.AllChem.MMFFGetMoleculeProperties`.
    sanitize:
        When ``True`` (the default) the molecule is sanitised before the
        parameters are generated.  The option mirrors the behaviour in the
        reference UFF builder implementation.
    """

    if sanitize:
        Chem.SanitizeMol(mol)
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
    if props is None:
        raise ValueError("RDKit failed to create MMFF properties for the molecule")
    return props


def assign_atom_types(
    mol: Chem.Mol,
    *,
    mmff_variant: str = "MMFF94",
    sanitize: bool = True,
) -> List[str]:
    """Return the symbolic MMFF atom types for ``mol``.

    The returned strings are compatible with the keys used in
    :mod:`mmff.parameters`.
    """

    atom_types, _ = assign_atom_types_and_charges(
        mol,
        mmff_variant=mmff_variant,
        sanitize=sanitize,
    )
    return atom_types


def assign_charges(
    mol: Chem.Mol,
    *,
    mmff_variant: str = "MMFF94",
    sanitize: bool = True,
) -> List[float]:
    """Return the MMFF partial charges associated with ``mol``."""

    _, charges = assign_atom_types_and_charges(
        mol,
        mmff_variant=mmff_variant,
        sanitize=sanitize,
    )
    return charges


def assign_atom_types_and_charges(
    mol: Chem.Mol,
    *,
    mmff_variant: str = "MMFF94",
    sanitize: bool = True,
) -> Tuple[List[str], List[float]]:
    """Convenience wrapper returning both atom types and charges.

    The helper mirrors :func:`UFF_PyTorch.atom_typing.assign_atom_types_and_charges`
    to streamline client code that requires both pieces of information.
    """

    props = _ensure_mmff_properties(mol, mmff_variant=mmff_variant, sanitize=sanitize)
    atom_types: List[str] = []
    charges: List[float] = []
    for atom_idx in range(mol.GetNumAtoms()):
        if hasattr(props, "GetMMFFAtomType"):
            atom_type = props.GetMMFFAtomType(atom_idx)
        elif hasattr(props, "GetMMFF94Type"):
            atom_type = props.GetMMFF94Type(atom_idx)
        else:
            raise AttributeError("RDKit MMFF properties object lacks atom type accessors")
        atom_types.append(str(atom_type))
        charges.append(float(props.GetMMFFPartialCharge(atom_idx)))
    return atom_types, charges
