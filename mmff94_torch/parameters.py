"""
Open Babel MMFF94/94s parameter access (RDKit 불필요, RDKit 타입ID도 호환).

- mmff94.ff(또는 mmff94s.ff)에서 실제 mmff*.par 경로를 읽어들임
- mmffvdw.par / mmffbond.par / mmffang.par / mmfftor.par 파싱
- 심볼 기반 테이블 + 숫자 타입ID 기반 테이블을 **동시에** 구축
  (빌더가 '1','2' 같은 RDKit 타입ID를 내보내도 바로 동작)

노출 API (model.py가 기대하는 것):
  - ATOMIC_PARAMETERS: Dict[str, AtomType]  # 심볼 및 "숫자문자열" 키 모두 지원
  - lookup_bond/angle/torsion((t1, t2[, t3[, t4]]))  # 심볼/숫자 섞여도 OK
  - combine_vdw(atom_a, atom_b)
  - COULOMB_CONSTANT
"""

from __future__ import annotations

import os
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class AtomType:
    """
    MMFF94 atom-type container.

    r_vdw : R*_{II} (A_i * alpha_i^pexp)
    epsilon : placeholder(0.0). ε is pairwise (computed on demand).
    alpha : α_i (Å^3)
    n     : N_i (effective electrons)
    a     : A_i (for R*)
    g     : G_i (for ε)
    da    : Donor/Acceptor flag ('D','A','-')
    charge: placeholder(0.0). MMFF charges are not per-type constants.
    mmff_id: integer type ID
    symbol : Open Babel symbol (e.g., 'CR','HC','C=O',...)
    """
    symbol: str
    r_vdw: float
    epsilon: float
    alpha: float
    charge: float
    n: float
    a: float
    g: float
    da: str
    mmff_id: int


@dataclass(frozen=True)
class BondType:
    k: float
    r0: float


@dataclass(frozen=True)
class AngleType:
    k: float
    theta0: float  # degrees


@dataclass(frozen=True)
class TorsionType:
    v1: float
    v2: float
    v3: float


@dataclass
class MMFFParams:
    """Container for MMFF force-field parameters extracted from RDKit."""

    # bonds
    bonds: np.ndarray
    kb: np.ndarray
    r0: np.ndarray
    # angles
    angles: np.ndarray
    ka: np.ndarray
    theta0_deg: np.ndarray
    angle_type: np.ndarray
    # stretch–bend
    stbn: np.ndarray
    kba_ijk: np.ndarray
    kba_kji: np.ndarray
    sb_r0_ij: np.ndarray
    sb_r0_kj: np.ndarray
    sb_theta0_deg: np.ndarray
    # out-of-plane
    impropers: np.ndarray
    koop: np.ndarray
    # torsions
    torsions: np.ndarray
    V1: np.ndarray
    V2: np.ndarray
    V3: np.ndarray
    # nonbonded
    nb_pairs: np.ndarray
    is14: np.ndarray
    Rstar: np.ndarray
    eps: np.ndarray
    # charges
    qi: np.ndarray


# =============================================================================
# RDKit parameter extraction helpers
# =============================================================================

def _neighbors(mol: Chem.Mol, idx: int) -> List[int]:
    return [nb.GetIdx() for nb in mol.GetAtomWithIdx(idx).GetNeighbors()]


def _enumerate_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < j:
            pairs.append((i, j))
        else:
            pairs.append((j, i))
    return sorted(set(pairs))


def _enumerate_angles(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    for j in range(mol.GetNumAtoms()):
        neighbors = _neighbors(mol, j)
        for a in range(len(neighbors) - 1):
            for b in range(a + 1, len(neighbors)):
                i = neighbors[a]
                k = neighbors[b]
                if i < k:
                    triples.append((i, j, k))
                else:
                    triples.append((k, j, i))
    return triples


def _enumerate_torsions(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    quads: set[Tuple[int, int, int, int]] = set()
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        j_neighbors = [x for x in _neighbors(mol, j) if x != k]
        k_neighbors = [x for x in _neighbors(mol, k) if x != j]
        for i in j_neighbors:
            for l in k_neighbors:
                if i == l:
                    continue
                jj, kk = (j, k) if j < k else (k, j)
                ii, ll = (i, l) if j < k else (l, i)
                quads.add((ii, jj, kk, ll))
    return sorted(quads)


def _enumerate_impropers(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    quads: List[Tuple[int, int, int, int]] = []
    for j in range(mol.GetNumAtoms()):
        neighbors = _neighbors(mol, j)
        if len(neighbors) < 3:
            continue
        for a in range(len(neighbors) - 2):
            for b in range(a + 1, len(neighbors) - 1):
                for c in range(b + 1, len(neighbors)):
                    i = neighbors[a]
                    k = neighbors[b]
                    l = neighbors[c]
                    quads.append((i, j, k, l))
    return quads


def _shortest_path_matrix(mol: Chem.Mol) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    adjacency: List[List[int]] = [[] for _ in range(n_atoms)]
    for i, j in _enumerate_bonds(mol):
        adjacency[i].append(j)
        adjacency[j].append(i)
    dist = np.full((n_atoms, n_atoms), np.inf, dtype=float)
    for source in range(n_atoms):
        dist[source, source] = 0.0
        queue: deque[int] = deque([source])
        seen = {source}
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v in seen:
                    continue
                dist[source, v] = dist[source, u] + 1.0
                seen.add(v)
                queue.append(v)
    return dist


def collect_mmff_params(mol: Chem.Mol, variant: str = "MMFF94") -> MMFFParams:
    """Collect RDKit MMFF parameters for the supplied molecule."""

    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    except Exception:
        props = AllChem.MMFFGetMoleculeProperties(mol)

    n_atoms = mol.GetNumAtoms()
    qi = np.array([props.GetMMFFPartialCharge(i) for i in range(n_atoms)], dtype=np.float64)

    bonds = np.array(_enumerate_bonds(mol), dtype=np.int64)
    kb: List[float] = []
    r0: List[float] = []
    bond_idx: Dict[Tuple[int, int], int] = {}
    for idx, (i, j) in enumerate(bonds.tolist()):
        params = props.GetMMFFBondStretchParams(mol, i, j)
        if params is None:
            kb.append(0.0)
            r0.append(0.0)
        else:
            if len(params) == 3:
                _, kb_ij, r0_ij = params
            else:
                kb_ij, r0_ij = params
            kb.append(float(kb_ij))
            r0.append(float(r0_ij))
        bond_idx[(i, j)] = idx
    kb_arr = np.asarray(kb, dtype=np.float64)
    r0_arr = np.asarray(r0, dtype=np.float64)

    angles = np.array(_enumerate_angles(mol), dtype=np.int64)
    ka: List[float] = []
    theta0_deg: List[float] = []
    angle_type: List[int] = []
    for (i, j, k) in angles.tolist():
        params = props.GetMMFFAngleBendParams(mol, i, j, k)
        if params is None:
            angle_type.append(0)
            theta0_deg.append(0.0)
            ka.append(0.0)
            continue
        at, x, y = params
        x_is_angle = isinstance(x, (int, float)) and 30.0 <= float(x) <= 210.0
        y_is_angle = isinstance(y, (int, float)) and 30.0 <= float(y) <= 210.0
        if x_is_angle and not y_is_angle:
            theta0, k_force = float(x), float(y)
        elif y_is_angle and not x_is_angle:
            theta0, k_force = float(y), float(x)
        else:
            if float(x) >= float(y):
                theta0, k_force = float(x), float(y)
            else:
                theta0, k_force = float(y), float(x)
        angle_type.append(int(at))
        theta0_deg.append(theta0)
        ka.append(k_force)
    ka_arr = np.asarray(ka, dtype=np.float64)
    theta0_arr = np.asarray(theta0_deg, dtype=np.float64)
    angle_type_arr = np.asarray(angle_type, dtype=np.int64)

    stbn: List[Tuple[int, int, int]] = []
    kba_ijk: List[float] = []
    kba_kji: List[float] = []
    sb_r0_ij: List[float] = []
    sb_r0_kj: List[float] = []
    sb_theta0: List[float] = []
    angle_lookup: Dict[Tuple[int, int, int], float] = {
        tuple(tri): th for tri, th in zip(angles.tolist(), theta0_deg)
    }
    angle_lookup.update({(k, j, i): th for (i, j, k), th in angle_lookup.items()})
    for (i, j, k) in angles.tolist():
        params = props.GetMMFFStretchBendParams(mol, i, j, k)
        if params is None:
            continue
        _, k1, k2 = params
        key_ij = (min(i, j), max(i, j))
        key_kj = (min(k, j), max(k, j))
        r0_ij = r0_arr[bond_idx[key_ij]] if key_ij in bond_idx else 0.0
        r0_kj = r0_arr[bond_idx[key_kj]] if key_kj in bond_idx else 0.0
        stbn.append((i, j, k))
        kba_ijk.append(float(k1))
        kba_kji.append(float(k2))
        sb_r0_ij.append(float(r0_ij))
        sb_r0_kj.append(float(r0_kj))
        sb_theta0.append(float(angle_lookup.get((i, j, k), 0.0)))
    stbn_arr = np.array(stbn, dtype=np.int64) if stbn else np.zeros((0, 3), dtype=np.int64)
    kba_ijk_arr = np.asarray(kba_ijk, dtype=np.float64)
    kba_kji_arr = np.asarray(kba_kji, dtype=np.float64)
    sb_r0_ij_arr = np.asarray(sb_r0_ij, dtype=np.float64)
    sb_r0_kj_arr = np.asarray(sb_r0_kj, dtype=np.float64)
    sb_theta0_arr = np.asarray(sb_theta0, dtype=np.float64)

    torsions = np.array(_enumerate_torsions(mol), dtype=np.int64)
    V1: List[float] = []
    V2: List[float] = []
    V3: List[float] = []
    for (i, j, k, l) in torsions.tolist():
        params = props.GetMMFFTorsionParams(mol, i, j, k, l)
        if params is None:
            V1.append(0.0)
            V2.append(0.0)
            V3.append(0.0)
        else:
            _, v1, v2, v3 = params
            V1.append(float(v1))
            V2.append(float(v2))
            V3.append(float(v3))
    V1_arr = np.asarray(V1, dtype=np.float64)
    V2_arr = np.asarray(V2, dtype=np.float64)
    V3_arr = np.asarray(V3, dtype=np.float64)

    impropers = np.array(_enumerate_impropers(mol), dtype=np.int64)
    koop: List[float] = []
    for (i, j, k, l) in impropers.tolist():
        params = props.GetMMFFOopBendParams(mol, i, j, k, l)
        koop.append(0.0 if params is None else float(params))
    koop_arr = np.asarray(koop, dtype=np.float64)

    dist = _shortest_path_matrix(mol)
    nb_pairs: List[Tuple[int, int]] = []
    is14: List[bool] = []
    Rstar: List[float] = []
    eps: List[float] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dij = dist[i, j]
            if dij < 1.5:
                continue
            if dij < 2.5:
                continue
            params = props.GetMMFFVdWParams(i, j)
            if params is None:
                continue
            _, _, Rij, eij = params
            nb_pairs.append((i, j))
            is14.append(bool(abs(dij - 3.0) < 1e-6))
            Rstar.append(float(Rij))
            eps.append(float(eij))
    nb_pairs_arr = np.array(nb_pairs, dtype=np.int64) if nb_pairs else np.zeros((0, 2), dtype=np.int64)
    is14_arr = np.asarray(is14, dtype=np.bool_)
    Rstar_arr = np.asarray(Rstar, dtype=np.float64)
    eps_arr = np.asarray(eps, dtype=np.float64)

    return MMFFParams(
        bonds=bonds,
        kb=kb_arr,
        r0=r0_arr,
        angles=angles,
        ka=ka_arr,
        theta0_deg=theta0_arr,
        angle_type=angle_type_arr,
        stbn=stbn_arr,
        kba_ijk=kba_ijk_arr,
        kba_kji=kba_kji_arr,
        sb_r0_ij=sb_r0_ij_arr,
        sb_r0_kj=sb_r0_kj_arr,
        sb_theta0_deg=sb_theta0_arr,
        impropers=impropers,
        koop=koop_arr,
        torsions=torsions,
        V1=V1_arr,
        V2=V2_arr,
        V3=V3_arr,
        nb_pairs=nb_pairs_arr,
        is14=is14_arr,
        Rstar=Rstar_arr,
        eps=eps_arr,
        qi=qi,
    )


# =============================================================================
# 파일 검색
# =============================================================================

def _candidate_search_roots() -> List[Path]:
    roots: List[Path] = []

    # 1) 환경변수
    for env in ("BABEL_DATADIR", "OPENBABEL_DATADIR"):
        val = os.environ.get(env)
        if val:
            roots.append(Path(val))

    # 2) conda
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        base = Path(cp) / "share" / "openbabel"
        roots += [
            base / "3.1.2",
            base / "3.1.1",
            base / "3.1.0",
            base,
            Path(cp) / "share",
        ]

    # 3) 시스템 기본
    roots += [
        Path("/usr/local/share/openbabel/3.1.2"),
        Path("/usr/local/share/openbabel/3.1.1"),
        Path("/usr/local/share/openbabel/3.1.0"),
        Path("/usr/local/share/openbabel"),
        Path("/usr/share/openbabel/3.1.2"),
        Path("/usr/share/openbabel/3.1.1"),
        Path("/usr/share/openbabel/3.1.0"),
        Path("/usr/share/openbabel"),
        Path("/opt/homebrew/share/openbabel"),
        Path("/usr/local/share"),
        Path("/usr/share"),
        Path("/usr/share/openbabel/data"),
        Path("/usr/local/share/openbabel/data"),
    ]
    return roots


def _find_mmff_ff_file(mmff_variant: str = "MMFF94") -> Path:
    """Find mmff94.ff or mmff94s.ff."""
    fname = "mmff94s.ff" if mmff_variant.upper() == "MMFF94S" else "mmff94.ff"
    patterns = (fname, f"*/{fname}", f"**/{fname}")

    for root in _candidate_search_roots():
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.rglob(pat):
                if not p.is_file():
                    continue
                try:
                    txt = p.read_text(encoding="utf8", errors="ignore")
                except Exception:
                    continue
                if any(k in txt for k in ("vdw", "bond", "ang", "tor")):
                    return p

    searched = ", ".join(str(p) for p in _candidate_search_roots()) or "no known locations"
    raise FileNotFoundError(f"Cannot find {fname}. Searched: {searched}")


def _parse_ff_index(mmff_ff_path: Path) -> Dict[str, Path]:
    """Map section keys in mmff94.ff/94s.ff to actual .par paths."""
    mapping: Dict[str, Path] = {}
    base = mmff_ff_path.parent
    text = mmff_ff_path.read_text(encoding="utf8", errors="ignore")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        key, rel = parts
        if key in {"vdw", "bond", "ang", "tor"}:
            mapping[key] = (base / rel).resolve()
    for req in ("vdw", "bond", "ang", "tor"):
        if req not in mapping or not mapping[req].exists():
            raise FileNotFoundError(f"Parameter file missing for '{req}' from {mmff_ff_path}")
    return mapping


# =============================================================================
# 파서 (관대한/내성적 파싱)
# =============================================================================

class _VDWControl:
    __slots__ = ("pexp", "afact", "bfact", "darad", "daeps")
    def __init__(self, pexp: float, afact: float, bfact: float, darad: float, daeps: float):
        self.pexp = pexp
        self.afact = afact
        self.bfact = bfact
        self.darad = darad
        self.daeps = daeps


def _parse_mmffvdw(path: Path) -> Tuple[_VDWControl, Dict[int, AtomType], Dict[int, str]]:
    """
    Header: read 5 floats from any mixed label+numeric line.
    Body rows: "ID alpha N A G [D/A] Symb ..." (D/A may be omitted in some builds).
    """
    ctrl: Optional[_VDWControl] = None
    id2atom: Dict[int, AtomType] = {}
    id2sym: Dict[int, str] = {}

    with path.open("r", encoding="utf8", errors="ignore") as f:
        # Header scan (tolerant to labels)
        for raw in f:
            s = raw.strip()
            if not s or s[0] in "*#;":
                continue
            nums: List[float] = []
            for tok in s.replace(",", " ").split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    continue
            if len(nums) >= 5:
                ctrl = _VDWControl(*nums[:5])
                break

        if ctrl is None:
            raise ValueError(f"Failed to read VDW header constants from {path}")

        # Body
        for raw in f:
            s = raw.strip()
            if not s or s[0] in "*#;":
                continue
            cols = s.split()
            if len(cols) < 6:
                continue
            try:
                type_id = int(cols[0])
                alpha = float(cols[1])
                nval  = float(cols[2])
                aval  = float(cols[3])
                gval  = float(cols[4])
            except Exception:
                continue

            # D/A flag + symbol (handle missing D/A)
            da = "-"
            if len(cols) >= 7 and cols[5] in {"D", "A", "-"}:
                da = cols[5]
                symb = cols[6]
            else:
                symb = cols[5] if len(cols) > 5 else f"T{type_id}"

            r_star = aval * (alpha ** ctrl.pexp)
            atom = AtomType(
                symbol=symb, r_vdw=r_star, epsilon=0.0,
                alpha=alpha, charge=0.0,
                n=nval, a=aval, g=gval, da=da, mmff_id=type_id
            )
            id2atom[type_id] = atom
            id2sym[type_id]  = symb

    return ctrl, id2atom, id2sym


def _best_keep(existing: Optional[Tuple[int, object]], new_bt: int, new_obj: object) -> Tuple[int, object]:
    """Keep the entry preferring BT/AT/TT == 0 if duplicates exist."""
    if existing is None:
        return (new_bt, new_obj)
    old_bt, _ = existing
    if old_bt != 0 and new_bt == 0:
        return (new_bt, new_obj)
    return existing


def _parse_mmffbond(path: Path, id2sym: Mapping[int, str]) -> Tuple[
    Dict[Tuple[str, str], BondType],
    Dict[Tuple[int, int], BondType],
]:
    """
    Format: BTIJ I J kb r0 ...
    Keep both symbol-keyed and id-keyed tables.
    """
    chosen_sym: Dict[Tuple[str, str], Tuple[int, BondType]] = {}
    chosen_id:  Dict[Tuple[int, int], Tuple[int, BondType]] = {}

    with path.open("r", encoding="utf8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s[0] in "*#;":
                continue
            cols = s.split()
            if len(cols) < 5:
                continue
            try:
                bt = int(cols[0]); i = int(cols[1]); j = int(cols[2])
                kb = float(cols[3]); r0 = float(cols[4])
            except Exception:
                continue
            bt_entry = BondType(k=kb, r0=r0)

            # ID keyed (both orders)
            for key in ((i, j), (j, i)):
                chosen_id[key] = _best_keep(chosen_id.get(key), bt, bt_entry)

            # Symbol keyed (both orders)
            si, sj = id2sym.get(i), id2sym.get(j)
            if si and sj:
                for key in ((si, sj), (sj, si)):
                    chosen_sym[key] = _best_keep(chosen_sym.get(key), bt, bt_entry)

    return (
        {k: v[1] for k, v in chosen_sym.items()},
        {k: v[1] for k, v in chosen_id.items()},
    )


def _parse_mmffang(path: Path, id2sym: Mapping[int, str]) -> Tuple[
    Dict[Tuple[str, str, str], AngleType],
    Dict[Tuple[int, int, int], AngleType],
]:
    """
    Format: ATIJK I J K k theta0 ...
    Keep both symbol-keyed and id-keyed tables.
    """
    chosen_sym: Dict[Tuple[str, str, str], Tuple[int, AngleType]] = {}
    chosen_id:  Dict[Tuple[int, int, int], Tuple[int, AngleType]] = {}

    with path.open("r", encoding="utf8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s[0] in "*#;":
                continue
            cols = s.split()
            if len(cols) < 6:
                continue
            try:
                at = int(cols[0]); i = int(cols[1]); j = int(cols[2]); k = int(cols[3])
                kval = float(cols[4]); theta = float(cols[5])
            except Exception:
                continue
            at_entry = AngleType(k=kval, theta0=theta)

            # ID keyed (both orders)
            for key in ((i, j, k), (k, j, i)):
                chosen_id[key] = _best_keep(chosen_id.get(key), at, at_entry)

            # Symbol keyed (both orders)
            si, sj, sk = id2sym.get(i), id2sym.get(j), id2sym.get(k)
            if si and sj and sk:
                for key in ((si, sj, sk), (sk, sj, si)):
                    chosen_sym[key] = _best_keep(chosen_sym.get(key), at, at_entry)

    return (
        {k: v[1] for k, v in chosen_sym.items()},
        {k: v[1] for k, v in chosen_id.items()},
    )


def _parse_mmfftor(path: Path, id2sym: Mapping[int, str]) -> Tuple[
    Dict[Tuple[str, str, str, str], TorsionType],
    Dict[Tuple[int, int, int, int], TorsionType],
]:
    """
    Format: TTIJKL I J K L V1 V2 V3 ...
    Keep both symbol-keyed and id-keyed tables.
    """
    chosen_sym: Dict[Tuple[str, str, str, str], Tuple[int, TorsionType]] = {}
    chosen_id:  Dict[Tuple[int, int, int, int], Tuple[int, TorsionType]] = {}

    with path.open("r", encoding="utf8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s[0] in "*#;":
                continue
            cols = s.split()
            if len(cols) < 8:
                continue
            try:
                tt = int(cols[0]); i = int(cols[1]); j = int(cols[2]); k = int(cols[3]); l = int(cols[4])
                v1 = float(cols[5]); v2 = float(cols[6]); v3 = float(cols[7])
            except Exception:
                continue
            tt_entry = TorsionType(v1=v1, v2=v2, v3=v3)

            # ID keyed (both directions)
            for key in ((i, j, k, l), (l, k, j, i)):
                chosen_id[key] = _best_keep(chosen_id.get(key), tt, tt_entry)

            # Symbol keyed (both directions)
            si, sj, sk, sl = id2sym.get(i), id2sym.get(j), id2sym.get(k), id2sym.get(l)
            if si and sj and sk and sl:
                for key in ((si, sj, sk, sl), (sl, sk, sj, si)):
                    chosen_sym[key] = _best_keep(chosen_sym.get(key), tt, tt_entry)

    return (
        {k: v[1] for k, v in chosen_sym.items()},
        {k: v[1] for k, v in chosen_id.items()},
    )


# =============================================================================
# 초기화
# =============================================================================

# Expose ID<->symbol map for callers that want to convert in the builder layer.
ID_TO_SYMBOL: Dict[int, str] = {}
SYMBOL_TO_ID: Dict[str, int] = {}

def _initialise_parameter_tables(
    mmff_variant: str = "MMFF94",
) -> Tuple[
    "_VDWControl",
    Dict[str, AtomType],  # ATOMIC_PARAMETERS  (symbol & numeric-string keys)
    Dict[Tuple[str, str], BondType],           # symbol-keyed bonds
    Dict[Tuple[int, int], BondType],           # id-keyed bonds
    Dict[Tuple[str, str, str], AngleType],     # symbol-keyed angles
    Dict[Tuple[int, int, int], AngleType],     # id-keyed angles
    Dict[Tuple[str, str, str, str], TorsionType],   # symbol-keyed torsions
    Dict[Tuple[int, int, int, int], TorsionType],   # id-keyed torsions
]:
    ff_path = _find_mmff_ff_file(mmff_variant)
    files = _parse_ff_index(ff_path)

    vdw_ctrl, id2atom, id2sym = _parse_mmffvdw(files["vdw"])

    # Publish global maps
    ID_TO_SYMBOL.clear()
    ID_TO_SYMBOL.update(id2sym)
    SYMBOL_TO_ID.clear()
    SYMBOL_TO_ID.update({sym: i for i, sym in id2sym.items()})

    # Bonds / Angles / Torsions (both symbol & id keyed)
    bonds_sym, bonds_id = _parse_mmffbond(files["bond"], id2sym)
    ang_sym, ang_id = _parse_mmffang(files["ang"], id2sym)
    tors_sym, tors_id = _parse_mmfftor(files["tor"], id2sym)

    # ATOMIC_PARAMETERS: allow lookup by symbol **and** by numeric-string id
    sym2atom: Dict[str, AtomType] = {}
    for i, at in id2atom.items():
        sym2atom[at.symbol] = at
        sym2atom[str(i)] = at  # numeric-string alias (e.g., "1")

    return vdw_ctrl, sym2atom, bonds_sym, bonds_id, ang_sym, ang_id, tors_sym, tors_id


(
    _VDW_CTRL,
    ATOMIC_PARAMETERS,               # Dict[str, AtomType]  (symbol & "id-string")
    _BOND_SYM, _BOND_ID,             # internal caches
    _ANG_SYM, _ANG_ID,
    _TOR_SYM, _TOR_ID,
) = _initialise_parameter_tables(os.environ.get("MMFF_VARIANT", "MMFF94"))

# Common Coulomb constant (kcal·Å·mol⁻¹·e⁻²)
COULOMB_CONSTANT = 332.063709


# =============================================================================
# 유틸: 타입 키 정규화
# =============================================================================

_TypeKey = Union[str, int]

def _as_int_or_none(x: _TypeKey) -> Optional[int]:
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        try:
            return int(x)
        except Exception:
            return None
    return None

def _as_symbol_or_none(x: _TypeKey) -> Optional[str]:
    if isinstance(x, str) and not x.isdigit():
        return x
    if isinstance(x, int):
        return ID_TO_SYMBOL.get(x)
    if isinstance(x, str) and x.isdigit():
        return ID_TO_SYMBOL.get(int(x))
    return None


# =============================================================================
# 룩업 API (심볼/숫자 모두 지원)
# =============================================================================

def lookup_bond(atom_types: Iterable[_TypeKey]) -> BondType:
    a, b = tuple(atom_types)
    # 1) direct symbol
    sa, sb = _as_symbol_or_none(a), _as_symbol_or_none(b)
    if sa is not None and sb is not None:
        key = (sa, sb)
        if key in _BOND_SYM:  # exact
            return _BOND_SYM[key]
        rkey = (sb, sa)
        if rkey in _BOND_SYM:
            return _BOND_SYM[rkey]
    # 2) id-based
    ia, ib = _as_int_or_none(a), _as_int_or_none(b)
    if ia is not None and ib is not None:
        keyi = (ia, ib)
        if keyi in _BOND_ID:
            return _BOND_ID[keyi]
        rkeyi = (ib, ia)
        if rkeyi in _BOND_ID:
            return _BOND_ID[rkeyi]
    # 3) mixed: convert numeric to symbol and retry
    if sa is None:
        sa = _as_symbol_or_none(a)
    if sb is None:
        sb = _as_symbol_or_none(b)
    if sa and sb:
        key = (sa, sb)
        if key in _BOND_SYM:
            return _BOND_SYM[key]
        rkey = (sb, sa)
        if rkey in _BOND_SYM:
            return _BOND_SYM[rkey]
    raise KeyError(f"No bond parameters for {tuple(str(x) for x in (a,b))}")


def lookup_angle(atom_types: Iterable[_TypeKey]) -> AngleType:
    a, b, c = tuple(atom_types)  # (left, center, right)

    # 1) symbol 우선
    sa = _as_symbol_or_none(a)
    sb = _as_symbol_or_none(b)
    sc = _as_symbol_or_none(c)
    if sa and sb and sc:
        key = (sa, sb, sc)
        if key in _ANG_SYM:
            return _ANG_SYM[key]
        rkey = (sc, sb, sa)
        if rkey in _ANG_SYM:
            return _ANG_SYM[rkey]

    # 2) id로 폴백
    ia = _as_int_or_none(a)
    ib = _as_int_or_none(b)
    ic = _as_int_or_none(c)
    if ia is not None and ib is not None and ic is not None:
        keyi = (ia, ib, ic)
        if keyi in _ANG_ID:
            return _ANG_ID[keyi]
        rkeyi = (ic, ib, ia)
        if rkeyi in _ANG_ID:
            return _ANG_ID[rkeyi]

    # 못 찾으면 에러
    raise KeyError(f"No angle parameters for {(a, b, c)}")


def lookup_torsion(atom_types: Iterable[_TypeKey]) -> TorsionType:
    a, b, c, d = tuple(atom_types)
    # 1) symbol
    sa, sb, sc, sd = (_as_symbol_or_none(a), _as_symbol_or_none(b),
                      _as_symbol_or_none(c), _as_symbol_or_none(d))
    if sa and sb and sc and sd:
        key = (sa, sb, sc, sd)
        if key in _TOR_SYM:
            return _TOR_SYM[key]
        rkey = (sd, sc, sb, sa)
        if rkey in _TOR_SYM:
            return _TOR_SYM[rkey]
    # 2) id
    ia, ib, ic, id_ = (_as_int_or_none(a), _as_int_or_none(b),
                       _as_int_or_none(c), _as_int_or_none(d))
    if ia is not None and ib is not None and ic is not None and id_ is not None:
        keyi = (ia, ib, ic, id_)
        if keyi in _TOR_ID:
            return _TOR_ID[keyi]
        rkeyi = (id_, ic, ib, ia)
        if rkeyi in _TOR_ID:
            return _TOR_ID[rkeyi]
    # 3) mixed → cast to symbols
    sa = sa or _as_symbol_or_none(a)
    sb = sb or _as_symbol_or_none(b)
    sc = sc or _as_symbol_or_none(c)
    sd = sd or _as_symbol_or_none(d)
    if sa and sb and sc and sd:
        key = (sa, sb, sc, sd)
        if key in _TOR_SYM:
            return _TOR_SYM[key]
        rkey = (sd, sc, sb, sa)
        if rkey in _TOR_SYM:
            return _TOR_SYM[rkey]
    raise KeyError(f"No torsion parameters for {tuple(str(x) for x in (a,b,c,d))}")


# =============================================================================
# VDW (Halgren buffered 14-7)
# =============================================================================

_EPS_PREF = 181.16  # literature constant

def _pair_rstar_and_epsilon(a: AtomType, b: AtomType) -> Tuple[float, float]:
    r_i = a.r_vdw
    r_j = b.r_vdw

    # Donor present: arithmetic mean for R*
    if a.da == "D" or b.da == "D":
        r_ij = 0.5 * (r_i + r_j)
    else:
        if r_i <= 0 or r_j <= 0:
            r_ij = 0.5 * (r_i + r_j)
        else:
            gij = (r_i - r_j) / (r_i + r_j)
            rmean = 0.5 * (r_i + r_j)
            r_ij = rmean * (1.0 + _VDW_CTRL.afact * (1.0 - math.exp(-_VDW_CTRL.bfact * gij * gij)))

    denom = math.sqrt(a.alpha / a.n) + math.sqrt(b.alpha / b.n)
    if denom <= 0 or r_ij <= 0:
        eps_ij = 0.0
    else:
        eps_ij = _EPS_PREF * (a.g * b.g * a.alpha * b.alpha) / denom * (r_ij ** -6)

    # Donor–Acceptor pair scaling
    if (a.da == "D" and b.da == "A") or (a.da == "A" and b.da == "D"):
        r_ij *= _VDW_CTRL.darad
        eps_ij *= _VDW_CTRL.daeps

    return r_ij, eps_ij


def combine_vdw(atom_a: AtomType, atom_b: AtomType) -> Tuple[torch.Tensor, torch.Tensor]:
    r, eps = _pair_rstar_and_epsilon(atom_a, atom_b)
    return torch.tensor(r, dtype=torch.float32), torch.tensor(eps, dtype=torch.float32)
