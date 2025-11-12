"""
MMFF94 topology/builder for RDKit molecules (fast, no heavy Python overhead).

Produces indices for bonds, angles, torsions, impropers and, optionally,
retrieves the corresponding MMFF parameter tuples via RDKit's MMFF APIs.

Requirements:
  - RDKit
  - NumPy
  - (optional) PyTorch for tensor outputs

Design goals:
  * Accept an RDKit `Chem.Mol` directly (no file I/O).
  * Minimize Python↔C overhead: get MMFFMolProperties once and reuse.
  * Keep outputs simple (NumPy arrays and Python lists of tuples).
  * Be tolerant to missing parameters: mask or drop on request.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from rdkit import Chem
from rdkit.Chem import AllChem

from mmff94_torch.atom_typing import MMFFVariant, assign_mmff_types_and_charges  # local module


@dataclass
class MMFFSystem:
    # Core
    n_atoms: int
    atom_types: np.ndarray        # (N,), int
    charges: np.ndarray           # (N,), float
    # Topology
    bonds: np.ndarray             # (Nb, 2), int
    angles: np.ndarray            # (Na, 3), int
    torsions: np.ndarray          # (Nt, 4), int
    impropers: np.ndarray         # (Ni, 4), int
    # Parameter masks (True where params exist)
    bond_mask: Optional[np.ndarray] = None    # (Nb,)
    angle_mask: Optional[np.ndarray] = None   # (Na,)
    torsion_mask: Optional[np.ndarray] = None # (Nt,)
    improper_mask: Optional[np.ndarray] = None# (Ni,)
    # Raw parameter tuples (lists to accommodate heterogeneous shapes)
    bond_params: Optional[List[Tuple[float, float]]] = None           # (kb, r0)
    angle_params: Optional[List[Tuple[float, float]]] = None          # (ka, theta0 in degrees)
    torsion_params: Optional[List[Tuple[float, ...]]] = None          # variable length (V1..Vn, periodicities...)
    improper_params: Optional[List[Tuple[float, ...]]] = None         # typically (koop,)
    # Meta
    variant: MMFFVariant = "MMFF94"
    used_hs: bool = True
    has_all_params: bool = True

    # Optional conversion to torch tensors for ML pipelines
    def as_torch(self, device: Optional[Union[str, "torch.device"]] = None):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not available. Install torch or use NumPy arrays.")
        to_i = lambda x: torch.from_numpy(np.ascontiguousarray(x.astype(np.int64))).to(device)
        to_f = lambda x: torch.from_numpy(np.ascontiguousarray(x.astype(np.float32))).to(device)
        out = dict(
            n_atoms=self.n_atoms,
            atom_types=to_i(self.atom_types),
            charges=to_f(self.charges),
            bonds=to_i(self.bonds),
            angles=to_i(self.angles),
            torsions=to_i(self.torsions),
            impropers=to_i(self.impropers),
            variant=self.variant,
            used_hs=self.used_hs,
            has_all_params=self.has_all_params,
        )
        # Masks
        if self.bond_mask is not None: out["bond_mask"] = torch.from_numpy(self.bond_mask.astype(np.bool_)).to(device)
        if self.angle_mask is not None: out["angle_mask"] = torch.from_numpy(self.angle_mask.astype(np.bool_)).to(device)
        if self.torsion_mask is not None: out["torsion_mask"] = torch.from_numpy(self.torsion_mask.astype(np.bool_)).to(device)
        if self.improper_mask is not None: out["improper_mask"] = torch.from_numpy(self.improper_mask.astype(np.bool_)).to(device)
        return out


# ---- Topology enumeration -------------------------------------------------

def _ensure_explicit_hs(mol: Chem.Mol, include_hs: bool) -> Tuple[Chem.Mol, bool]:
    if include_hs:
        has_explicit_h = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
        if not has_explicit_h:
            return Chem.AddHs(mol, addCoords=False), True
        return mol, True
    else:
        return mol, False


def _enumerate_bonds(mol: Chem.Mol) -> np.ndarray:
    pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    # Store canonical orientation (i<j) for consistent downstream processing.
    pairs = [(i, j) if i < j else (j, i) for (i, j) in pairs]
    return np.array(pairs, dtype=np.int64)


def _neighbors(mol: Chem.Mol, idx: int) -> List[int]:
    return [nb.GetIdx() for nb in mol.GetAtomWithIdx(idx).GetNeighbors()]


def _enumerate_angles(mol: Chem.Mol) -> np.ndarray:
    triples: List[Tuple[int, int, int]] = []
    for j in range(mol.GetNumAtoms()):
        nbs = _neighbors(mol, j)
        if len(nbs) < 2:
            continue
        for i, k in combinations(nbs, 2):
            # canonical orientation (i<k) to avoid duplicates
            if i < k:
                triples.append((i, j, k))
            else:
                triples.append((k, j, i))
    if not triples:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(triples, dtype=np.int64)


def _enumerate_torsions(mol: Chem.Mol) -> np.ndarray:
    quads: set[Tuple[int, int, int, int]] = set()
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        j_nbs = [x for x in _neighbors(mol, j) if x != k]
        k_nbs = [x for x in _neighbors(mol, k) if x != j]
        if not j_nbs or not k_nbs:
            continue
        for i in j_nbs:
            for l in k_nbs:
                if i == l:
                    continue
                # canonicalize orientation to suppress duplicates:
                # we fix the central bond (j-k) order with j<k.
                jj, kk = (j, k) if j < k else (k, j)
                ii, ll = (i, l) if j < k else (l, i)
                quads.add((ii, jj, kk, ll))
    if not quads:
        return np.zeros((0, 4), dtype=np.int64)
    return np.array(sorted(quads), dtype=np.int64)


def _enumerate_impropers(mol: Chem.Mol) -> np.ndarray:
    """Generate central-atom impropers as (i, j, k, l) with j the central atom.
    We do not attempt chemical heuristics here; RDKit/MMFF will filter by parameter availability.
    """
    quads: List[Tuple[int, int, int, int]] = []
    for j in range(mol.GetNumAtoms()):
        nbs = _neighbors(mol, j)
        if len(nbs) < 3:
            continue
        from itertools import combinations as comb3
        for i, k, l in comb3(nbs, 3):
            quads.append((i, j, k, l))
    if not quads:
        return np.zeros((0, 4), dtype=np.int64)
    return np.array(quads, dtype=np.int64)


# ---- Parameter retrieval --------------------------------------------------

def _get_param_masks_and_values(
    mol: Chem.Mol,
    props: "AllChem.MMFFMolProperties",
    bonds: np.ndarray,
    angles: np.ndarray,
    torsions: np.ndarray,
    impropers: np.ndarray,
    *,
    drop_missing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, ...]], List[Tuple[float, ...]]]:
    Nb, Na, Nt, Ni = len(bonds), len(angles), len(torsions), len(impropers)
    bond_mask = np.zeros(Nb, dtype=np.bool_)
    angle_mask = np.zeros(Na, dtype=np.bool_)
    torsion_mask = np.zeros(Nt, dtype=np.bool_)
    improper_mask = np.zeros(Ni, dtype=np.bool_)
    bond_params: List[Tuple[float, float]] = []
    angle_params: List[Tuple[float, float]] = []
    torsion_params: List[Tuple[float, ...]] = []
    improper_params: List[Tuple[float, ...]] = []

    # Bonds
    for n, (i, j) in enumerate(bonds.tolist()):
        p = props.GetMMFFBondStretchParams(mol, int(i), int(j))
        if p is not None:
            bond_mask[n] = True
            # Expect (kb, r0)
            bond_params.append(tuple(map(float, p)))
        else:
            bond_params.append(tuple())

    # Angles
    for n, (i, j, k) in enumerate(angles.tolist()):
        p = props.GetMMFFAngleBendParams(mol, int(i), int(j), int(k))
        if p is not None:
            angle_mask[n] = True
            # Expect (ka, theta0_deg)
            angle_params.append(tuple(map(float, p)))
        else:
            angle_params.append(tuple())

    # Torsions
    for n, (i, j, k, l) in enumerate(torsions.tolist()):
        p = props.GetMMFFTorsionParams(mol, int(i), int(j), int(k), int(l))
        if p is not None:
            torsion_mask[n] = True
            # tuple of floats (length depends on parameterization)
            torsion_params.append(tuple(map(float, p)))
        else:
            torsion_params.append(tuple())

    # Impropers (out-of-plane)
    for n, (i, j, k, l) in enumerate(impropers.tolist()):
        p = props.GetMMFFOopBendParams(mol, int(i), int(j), int(k), int(l))
        if p is not None:
            improper_mask[n] = True
            improper_params.append(tuple(map(float, p)))
        else:
            improper_params.append(tuple())

    if drop_missing:
        # Filter out entries with missing parameters to streamline downstream math.
        if Nb:
            keep = bond_mask.nonzero()[0]
            bonds = bonds[keep]
            bond_mask = bond_mask[keep]
            bond_params = [bond_params[i] for i in keep]
        if Na:
            keep = angle_mask.nonzero()[0]
            angles = angles[keep]
            angle_mask = angle_mask[keep]
            angle_params = [angle_params[i] for i in keep]
        if Nt:
            keep = torsion_mask.nonzero()[0]
            torsions = torsions[keep]
            torsion_mask = torsion_mask[keep]
            torsion_params = [torsion_params[i] for i in keep]
        if Ni:
            keep = improper_mask.nonzero()[0]
            impropers = impropers[keep]
            improper_mask = improper_mask[keep]
            improper_params = [improper_params[i] for i in keep]

    return bond_mask, angle_mask, torsion_mask, improper_mask, bond_params, angle_params, torsion_params, improper_params


# ---- Public builder -------------------------------------------------------

def build_mmff_inputs(
    mol: Chem.Mol,
    *,
    include_hs: bool = True,
    variant: MMFFVariant = "MMFF94",
    with_params: bool = True,
    drop_missing: bool = True,
) -> MMFFSystem:
    """Build MMFF topology and (optionally) parameter tables directly from an RDKit Mol.

    Returns an MMFFSystem with NumPy arrays. Use `.as_torch()` to convert.
    """
    # Prepare molecule for typing and parameterization
    mol_work, used_hs = _ensure_explicit_hs(mol, include_hs=include_hs)

    # Atom types + charges (single MMFF properties handle reused throughout)
    typing = assign_mmff_types_and_charges(mol_work, include_hs=False, variant=variant)
    props = AllChem.MMFFGetMoleculeProperties(mol_work, mmffVariant=variant)

    # Topology
    bonds = _enumerate_bonds(mol_work)
    angles = _enumerate_angles(mol_work)
    torsions = _enumerate_torsions(mol_work)
    impropers = _enumerate_impropers(mol_work)

    # Parameter retrieval
    bond_mask = angle_mask = torsion_mask = improper_mask = None
    bond_params = angle_params = torsion_params = improper_params = None
    if with_params:
        (bond_mask, angle_mask, torsion_mask, improper_mask,
         bond_params, angle_params, torsion_params, improper_params
        ) = _get_param_masks_and_values(
            mol_work, props, bonds, angles, torsions, impropers, drop_missing=drop_missing
        )

    sys = MMFFSystem(
        n_atoms=mol_work.GetNumAtoms(),
        atom_types=typing.atom_types,
        charges=typing.partial_charges,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        impropers=impropers,
        bond_mask=bond_mask,
        angle_mask=angle_mask,
        torsion_mask=torsion_mask,
        improper_mask=improper_mask,
        bond_params=bond_params,
        angle_params=angle_params,
        torsion_params=torsion_params,
        improper_params=improper_params,
        variant=variant,
        used_hs=used_hs,
        has_all_params=typing.has_all_params,
    )
    return sys


# Convenience: build from SMILES (sanitized), no conformer required
def build_from_smiles(
    smi: str,
    *,
    include_hs: bool = True,
    variant: MMFFVariant = "MMFF94",
    with_params: bool = True,
    drop_missing: bool = True,
) -> MMFFSystem:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi!r}")
    return build_mmff_inputs(mol, include_hs=include_hs, variant=variant, with_params=with_params, drop_missing=drop_missing)



def build_forcefield(mol, *, variant: str = "MMFF94", include_hs: bool = True):
    # 지연 임포트로 순환의존성 회피
    from .model import build_forcefield as _bf
    return _bf(mol, variant=variant, include_hs=include_hs)


def forcefield_from_file(path: str, *, variant: str = "MMFF94", include_hs: bool = True):
    from .model import forcefield_from_file as _fff
    return _fff(path, variant=variant, include_hs=include_hs)


# Basic smoke test ---------------------------------------------------------
if __name__ == "__main__":
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    sys = build_from_smiles(smi, include_hs=True, variant="MMFF94", with_params=True, drop_missing=True)
    print("n_atoms:", sys.n_atoms)
    print("n_bonds/angles/torsions/impropers:", len(sys.bonds), len(sys.angles), len(sys.torsions), len(sys.impropers))
    print("first 5 bonds:", sys.bonds[:5].tolist())
    if sys.bond_params:
        print("first 5 bond params:", sys.bond_params[:5])
