"""
MMFF94 atom typing and partial charges with RDKit (fast path).

- Accepts RDKit Chem.Mol and returns NumPy arrays (optionally Torch tensors).
- Uses RDKit's validated MMFF implementation for correctness and speed.

Requirements:
  - RDKit
  - NumPy
  - (optional) PyTorch for tensor outputs

Notes:
  * Atom types are returned as numeric MMFF atom types (unsigned ints) as
    exposed by RDKit's MMFFMolProperties.GetMMFFAtomType(i).
  * Charges are the MMFF partial charges from MMFFMolProperties.GetMMFFPartialCharge(i).
  * You do NOT need 3D coordinates to assign atom types or charges.
  * RDKit may adjust aromaticity flags when computing MMFF properties. This is expected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal

import numpy as np

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from rdkit import Chem
from rdkit.Chem import AllChem

MMFFVariant = Literal["MMFF94", "MMFF94s"]

@dataclass(frozen=True)
class MMFFTypingResult:
    atom_types: np.ndarray          # shape (N,), dtype=int32
    partial_charges: np.ndarray     # shape (N,), dtype=float32
    has_all_params: bool
    used_hs: bool                   # True if explicit Hs were present/added
    variant: MMFFVariant

    def as_torch(self, device: Optional[Union[str, "torch.device"]] = None):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not available. Install torch or use NumPy arrays.")
        return (
            torch.from_numpy(self.atom_types.astype(np.int64)).to(device),
            torch.from_numpy(self.partial_charges.astype(np.float32)).to(device),
            self.has_all_params,
            self.used_hs,
            self.variant,
        )


def _ensure_explicit_hs(mol: Chem.Mol, include_hs: bool) -> Tuple[Chem.Mol, bool]:
    """Return a molecule with explicit Hs if requested.

    We avoid copying heavy data structures when unnecessary. If include_hs=True
    and the molecule lacks explicit Hs, we add them *without* coordinates (fast).
    """
    if include_hs:
        has_explicit_h = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
        if not has_explicit_h:
            mol_h = Chem.AddHs(mol, addCoords=False)
            return mol_h, True
        return mol, True
    else:
        return mol, False


def _get_mmff_props(mol: Chem.Mol, variant: MMFFVariant) -> "AllChem.MMFFMolProperties":
    """Create and return MMFFMolProperties (handles old/new RDKit)."""
    try:
        return AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    except Exception:
        # 구버전 RDKit은 mmffVariant 인자를 지원하지 않을 수 있음
        return AllChem.MMFFGetMoleculeProperties(mol)


def has_all_mmff_params(mol: Chem.Mol, *, variant: MMFFVariant = "MMFF94") -> bool:
    """
    RDKit 버전별 시그니처 차이를 흡수:
    - 신버전: MMFFHasAllMoleculeParams(mol, mmffVariant=...)
    - 구버전: MMFFHasAllMoleculeParams(mol)
    """
    try:
        # 신버전 경로
        return bool(AllChem.MMFFHasAllMoleculeParams(mol, mmffVariant=variant))
    except Exception:
        # 구버전: variant 인자를 받지 않음
        return bool(AllChem.MMFFHasAllMoleculeParams(mol))


def assign_mmff_types_and_charges(
    mol: Chem.Mol,
    *,
    include_hs: bool = True,
    variant: MMFFVariant = "MMFF94",
    dtype_float: Union[np.dtype, str] = np.float32,
    dtype_int: Union[np.dtype, str] = np.int32,
) -> MMFFTypingResult:
    """Assign MMFF atom types and partial charges using RDKit and return arrays.

    Parameters
    ----------
    mol
        RDKit molecule (sanitized). Conformers are NOT required.
    include_hs
        If True, ensure that explicit hydrogens are present during typing.
        This often yields more complete parameterization, though RDKit can
        also work with implicit Hs. The returned arrays align with the molecule
        actually used (with or without Hs).
    variant
        'MMFF94' (default) or 'MMFF94s'.
    dtype_float, dtype_int
        Output dtypes for charges and types.

    Returns
    -------
    MMFFTypingResult
        atom_types: numeric MMFF types for each atom (0-based indexing)
        partial_charges: per-atom MMFF partial charges
        has_all_params: boolean from MMFFHasAllMoleculeParams
        used_hs: whether explicit Hs were used
        variant: the MMFF variant
    """
    # Prepare molecule (optionally add explicit Hs). Avoid copying if possible.
    mol_in = mol
    mol_work, used_hs = _ensure_explicit_hs(mol_in, include_hs=include_hs)

    # Obtain properties once, reusing for all lookups (minimize Python↔C calls).
    props = _get_mmff_props(mol_work, variant)

    n = mol_work.GetNumAtoms()
    # Vectorized-ish extraction using Python generator and NumPy fromiter.
    types = np.fromiter((props.GetMMFFAtomType(i) for i in range(n)), dtype=np.int64, count=n)
    # RDKit returns doubles; cast to desired dtype.
    charges = np.fromiter((props.GetMMFFPartialCharge(i) for i in range(n)), dtype=np.float64, count=n)

    # Cast to requested dtypes for downstream frameworks.
    types = types.astype(dtype_int, copy=False)
    charges = charges.astype(dtype_float, copy=False)

    all_params = has_all_mmff_params(mol_work, variant=variant)
    return MMFFTypingResult(types, charges, all_params, used_hs, variant)


# Convenience wrappers ---------------------------------------------------------

def assign_mmff_atom_types(
    mol: Chem.Mol, *, include_hs: bool = True, variant: MMFFVariant = "MMFF94", dtype: Union[np.dtype, str] = np.int32
) -> np.ndarray:
    """Return only the numeric MMFF atom types as a NumPy array."""
    return assign_mmff_types_and_charges(mol, include_hs=include_hs, variant=variant, dtype_int=dtype).atom_types


def assign_mmff_partial_charges(
    mol: Chem.Mol, *, include_hs: bool = True, variant: MMFFVariant = "MMFF94", dtype: Union[np.dtype, str] = np.float32
) -> np.ndarray:
    """Return only the MMFF partial charges as a NumPy array."""
    return assign_mmff_types_and_charges(mol, include_hs=include_hs, variant=variant, dtype_float=dtype).partial_charges


# Optional: simple "one-hot" expansion utility for atom types ------------------
def one_hot_atom_types(types: np.ndarray, *, num_types: Optional[int] = None, dtype: Union[np.dtype, str] = np.float32) -> np.ndarray:
    """One-hot encode integer atom types → (N, num_types) float array.

    If num_types is omitted, it is inferred as (types.max()+1).
    """
    if types.ndim != 1:
        raise ValueError("`types` must be a 1D array of integer MMFF types")
    tmax = int(types.max()) if num_types is None else int(num_types - 1)
    out = np.zeros((types.shape[0], tmax + 1), dtype=dtype)
    out[np.arange(types.shape[0]), types.astype(int)] = 1.0
    return out


# Basic smoke test -------------------------------------------------------------
if __name__ == "__main__":
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    mol = Chem.MolFromSmiles(smi)
    # No conformer required.
    res = assign_mmff_types_and_charges(mol, include_hs=True, variant="MMFF94")
    print("N atoms:", len(res.atom_types))
    print("First 10 types:", res.atom_types[:10])
    print("First 10 charges:", np.round(res.partial_charges[:10], 4))
    print("Has all params:", res.has_all_params)
    try:
        at_t, q, *_ = res.as_torch()
        print("Torch tensors:", at_t.dtype, q.dtype)
    except Exception as e:
        print("Torch unavailable:", e)
