# forcefield_full_torch.py  (fixed)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict
import math
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem

# ---- MMFF constants ----
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
# Bond stretch scale (mdyne/Å -> kcal/mol)
FC_BOND = 143.9325
# Angle/oop scale
FC_ANGLE = 0.043844
# Stretch–bend scale
FC_STBN = 2.51210
# Cubic corrections
CS = -2.0      # Å^-1 (bond)
CB = -0.007    # deg^-1 (angle)
# Electrostatics
KE = 332.0716
COUL_BUF = 0.05
SCALE14_ELEC = 0.75
# vdW Buffered 14–7 (Halgren)
# Evdw = eps * ( (1.07 R*/(r + 0.07 R*))^7 ) * ( (1.12 R*^7/(r^7 + 0.12 R*^7)) - 2 )

# ------------ dataclasses ------------
@dataclass
class MMFFParams:
    # bonds
    bonds: np.ndarray
    kb: np.ndarray
    r0: np.ndarray
    # angles
    angles: np.ndarray
    ka: np.ndarray
    theta0_deg: np.ndarray
    angle_type: np.ndarray          # 0: normal, 1: linear (RDKit)
    # stretch–bend
    stbn: np.ndarray
    kba_ijk: np.ndarray
    kba_kji: np.ndarray
    sb_r0_ij: np.ndarray
    sb_r0_kj: np.ndarray
    sb_theta0_deg: np.ndarray       # *** NEW: per triplet reference angle ***
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

def _as(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=ref.dtype, device=ref.device)

# ------------ topology enumeration ------------
def _neighbors(mol: Chem.Mol, idx: int):
    return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]

def enumerate_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    ps = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        if i < j: ps.append((i, j))
        else:     ps.append((j, i))
    return sorted(set(ps))

def enumerate_angles(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
    triples = []
    for j in range(mol.GetNumAtoms()):
        nbs = _neighbors(mol, j)
        for a in range(len(nbs)-1):
            for b in range(a+1, len(nbs)):
                i, k = nbs[a], nbs[b]
                if i < k: triples.append((i, j, k))
                else:     triples.append((k, j, i))
    return triples

def enumerate_torsions(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    quads = set()
    for b in mol.GetBonds():
        j = b.GetBeginAtomIdx(); k = b.GetEndAtomIdx()
        jn = [x for x in _neighbors(mol, j) if x != k]
        kn = [x for x in _neighbors(mol, k) if x != j]
        for i in jn:
            for l in kn:
                if i == l: continue
                jj, kk = (j, k) if j < k else (k, j)
                ii, ll = (i, l) if j < k else (l, i)
                quads.add((ii, jj, kk, ll))
    return sorted(quads)

def enumerate_impropers(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    quads = []
    for j in range(mol.GetNumAtoms()):
        nbs = _neighbors(mol, j)
        if len(nbs) < 3: continue
        for a in range(len(nbs)-2):
            for b in range(a+1, len(nbs)-1):
                for c in range(b+1, len(nbs)):
                    i, k, l = nbs[a], nbs[b], nbs[c]
                    quads.append((i, j, k, l))
    return quads

def shortest_path_matrix(mol: Chem.Mol) -> np.ndarray:
    N = mol.GetNumAtoms()
    adj = [[] for _ in range(N)]
    for i, j in enumerate_bonds(mol):
        adj[i].append(j); adj[j].append(i)
    dist = np.full((N, N), np.inf, dtype=float)
    from collections import deque
    for s in range(N):
        dist[s, s] = 0.0
        dq = deque([s]); seen = {s}
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in seen:
                    dist[s, v] = dist[s, u] + 1.0
                    seen.add(v); dq.append(v)
    return dist

# ------------ parameter collection (RDKit) ------------
def collect_mmff_params(mol: Chem.Mol, variant: str = "MMFF94") -> MMFFParams:
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    except Exception:
        props = AllChem.MMFFGetMoleculeProperties(mol)

    N = mol.GetNumAtoms()
    # charges
    qi = np.array([props.GetMMFFPartialCharge(i) for i in range(N)], dtype=np.float64)

    # bonds
    bonds = np.array(enumerate_bonds(mol), dtype=np.int64)
    kb, r0 = [], []
    bond_idx = {}
    for n, (i, j) in enumerate(bonds.tolist()):
        p = props.GetMMFFBondStretchParams(mol, i, j)
        if p is None:
            kb.append(0.0); r0.append(0.0)
        else:
            if len(p) == 3:
                _, kb_ij, r0_ij = p
            else:
                kb_ij, r0_ij = p
            kb.append(float(kb_ij)); r0.append(float(r0_ij))
        bond_idx[(i, j)] = n
    kb = np.asarray(kb, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)

    # angles + angleType + θ0
    angles = np.array(enumerate_angles(mol), dtype=np.int64)
    ka = []
    theta0_deg = []
    angle_type = []

    for (i, j, k) in angles.tolist():
        p = props.GetMMFFAngleBendParams(mol, i, j, k)
        if p is None:
            angle_type.append(0); theta0_deg.append(0.0); ka.append(0.0)
            continue

        # RDKit 버전 따라 (atype, ka, theta0) 또는 (atype, theta0, ka) 로 나올 수 있어
        # → 값 범위로 안전하게 판별한다.
        at, x, y = p  # x,y 중 하나가 theta0(보통 60~180 deg), 다른 하나가 ka(보통 0~20)
        x_is_angle = isinstance(x, (int, float)) and 30.0 <= float(x) <= 210.0
        y_is_angle = isinstance(y, (int, float)) and 30.0 <= float(y) <= 210.0

        if x_is_angle and not y_is_angle:
            th0, kf = float(x), float(y)
        elif y_is_angle and not x_is_angle:
            th0, kf = float(y), float(x)
        else:
            # 모호하면 큰 값을 θ0, 작은 값을 ka로 둔다 (일반적 패턴)
            if float(x) >= float(y):
                th0, kf = float(x), float(y)
            else:
                th0, kf = float(y), float(x)

        angle_type.append(int(at))
        theta0_deg.append(th0)
        ka.append(kf)

    ka = np.asarray(ka, dtype=np.float64)
    theta0_deg = np.asarray(theta0_deg, dtype=np.float64)
    angle_type = np.asarray(angle_type, dtype=np.int64)

    # stretch–bend: 같은 (i,j,k) 순서로 kba와 r0(ij), r0(kj), 그리고 θ0를 보관
    stbn, kba_ijk, kba_kji, sb_r0_ij, sb_r0_kj, sb_theta0_deg = [], [], [], [], [], []
    # angle lookup: (i,j,k) -> θ0
    ang2th = {tuple(tri): th for tri, th in zip(angles.tolist(), theta0_deg.tolist())}
    ang2th.update({(k,j,i): th for (i,j,k), th in ang2th.items()})  # 양방향

    for (i, j, k) in angles.tolist():
        p = props.GetMMFFStretchBendParams(mol, i, j, k)  # (type, k1, k2)
        if p is None: 
            continue
        _, k1, k2 = p
        # bond r0s
        key_ij = (min(i,j), max(i,j))
        key_kj = (min(k,j), max(k,j))
        r0ij = r0[bond_idx[key_ij]] if key_ij in bond_idx else 0.0
        r0kj = r0[bond_idx[key_kj]] if key_kj in bond_idx else 0.0

        stbn.append((i, j, k))
        kba_ijk.append(float(k1)); kba_kji.append(float(k2))
        sb_r0_ij.append(float(r0ij)); sb_r0_kj.append(float(r0kj))
        sb_theta0_deg.append(float(ang2th.get((i, j, k), 0.0)))

    stbn = np.array(stbn, dtype=np.int64) if stbn else np.zeros((0,3), dtype=np.int64)
    kba_ijk = np.asarray(kba_ijk, dtype=np.float64)
    kba_kji = np.asarray(kba_kji, dtype=np.float64)
    sb_r0_ij = np.asarray(sb_r0_ij, dtype=np.float64)
    sb_r0_kj = np.asarray(sb_r0_kj, dtype=np.float64)
    sb_theta0_deg = np.asarray(sb_theta0_deg, dtype=np.float64)

    # torsions
    torsions = np.array(enumerate_torsions(mol), dtype=np.int64)
    V1, V2, V3 = [], [], []
    for (i, j, k, l) in torsions.tolist():
        p = props.GetMMFFTorsionParams(mol, i, j, k, l)  # (type, V1,V2,V3)
        if p is None:
            V1.append(0.0); V2.append(0.0); V3.append(0.0)
        else:
            _, v1, v2, v3 = p
            V1.append(float(v1)); V2.append(float(v2)); V3.append(float(v3))
    V1 = np.asarray(V1, dtype=np.float64); V2 = np.asarray(V2, dtype=np.float64); V3 = np.asarray(V3, dtype=np.float64)

    # out-of-plane
    impropers = np.array(enumerate_impropers(mol), dtype=np.int64)
    koop = []
    for (i, j, k, l) in impropers.tolist():
        p = props.GetMMFFOopBendParams(mol, i, j, k, l)
        koop.append(0.0 if p is None else float(p))
    koop = np.asarray(koop, dtype=np.float64)

    # nonbonded pairs & vdw
    D = shortest_path_matrix(mol)
    nb_pairs, is14, Rstar, eps = [], [], [], []
    for i in range(N):
        for j in range(i+1, N):
            dij = D[i, j]
            if dij < 1.5:   # 1-2 제외
                continue
            if dij < 2.5:   # 1-3 제외
                continue
            p = props.GetMMFFVdWParams(i, j)  # (R*unsc, eps_unsc, R*, eps)
            if p is None:
                continue
            _, _, Rij, eij = p
            nb_pairs.append((i, j))
            is14.append(bool(abs(dij - 3.0) < 1e-6))
            Rstar.append(float(Rij))
            eps.append(float(eij))

    nb_pairs = np.array(nb_pairs, dtype=np.int64) if nb_pairs else np.zeros((0,2), dtype=np.int64)
    is14 = np.asarray(is14, dtype=np.bool_)
    Rstar = np.asarray(Rstar, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)

    return MMFFParams(
        bonds=bonds, kb=kb, r0=r0,
        angles=angles, ka=ka, theta0_deg=theta0_deg, angle_type=angle_type,
        stbn=stbn, kba_ijk=kba_ijk, kba_kji=kba_kji,
        sb_r0_ij=sb_r0_ij, sb_r0_kj=sb_r0_kj, sb_theta0_deg=sb_theta0_deg,
        impropers=impropers, koop=koop,
        torsions=torsions, V1=V1, V2=V2, V3=V3,
        nb_pairs=nb_pairs, is14=is14, Rstar=Rstar, eps=eps,
        qi=qi
    )

# ------------ Torch geometry helpers ------------
def _norm(v: torch.Tensor, eps=1e-12): return torch.clamp(torch.linalg.norm(v, dim=-1), min=eps)

def _pair_r(x: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0: return x.new_zeros((0,))
    return _norm(x[pairs[:,0]] - x[pairs[:,1]])

def _angle(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    if ang.numel() == 0: return x.new_zeros((0,))
    i,j,k = ang[:,0], ang[:,1], ang[:,2]
    v1 = x[i] - x[j]; v2 = x[k] - x[j]
    c = torch.clamp((v1*v2).sum(-1) / (_norm(v1)*_norm(v2)), -1+1e-12, 1-1e-12)
    return torch.acos(c)  # rad

def _dihedral(x: torch.Tensor, tors: torch.Tensor) -> torch.Tensor:
    if tors.numel() == 0: return x.new_zeros((0,))
    i,j,k,l = tors[:,0], tors[:,1], tors[:,2], tors[:,3]
    r1,r2,r3,r4 = x[i], x[j], x[k], x[l]
    b0 = r1 - r2
    b1 = r3 - r2
    b2 = r4 - r3
    b1u = b1 / (torch.linalg.norm(b1, dim=1, keepdim=True) + 1e-15)
    v = b0 - (b0*b1u).sum(-1, keepdim=True)*b1u
    w = b2 - (b2*b1u).sum(-1, keepdim=True)*b1u
    vu = v / (torch.linalg.norm(v, dim=1, keepdim=True) + 1e-15)
    wu = w / (torch.linalg.norm(w, dim=1, keepdim=True) + 1e-15)
    xcomp = (vu*wu).sum(-1)
    ycomp = (torch.cross(b1u, vu, dim=1)*wu).sum(-1)
    return torch.atan2(ycomp, xcomp)

def _wilson_oop_deg(x: torch.Tensor, imp: torch.Tensor) -> torch.Tensor:
    if imp.numel() == 0: return x.new_zeros((0,))
    i,j,k,l = imp[:,0], imp[:,1], imp[:,2], imp[:,3]
    ri, rj, rk, rl = x[i], x[j], x[k], x[l]
    n = torch.cross(ri-rj, rk-rj, dim=1)
    nh = n / (torch.linalg.norm(n, dim=1, keepdim=True) + 1e-15)
    v = rl - rj
    vh = v / (torch.linalg.norm(v, dim=1, keepdim=True) + 1e-15)
    chi = torch.asin(torch.clamp((nh*vh).sum(-1), -1+1e-12, 1-1e-12))
    return chi * RAD2DEG

def _angle_mmff_deg(x: torch.Tensor,
                    angles: torch.Tensor,
                    ip_mask: torch.Tensor,
                    ip_m: torch.Tensor) -> torch.Tensor:
    """
    MMFF 각도(도). 기본은 ∠i-j-k, 단 center j가 trigonal(heavy 3)인 경우
    j를 (i,k,third) 평면으로 투영한 in-plane ∠i-X-k를 사용.
    """
    if angles.numel() == 0:
        return x.new_zeros((0,))

    i, j, k = angles[:,0], angles[:,1], angles[:,2]
    # 기본 각(라디안)
    v1 = x[i] - x[j]
    v2 = x[k] - x[j]
    cosT = torch.clamp((v1 * v2).sum(-1) / (_norm(v1) * _norm(v2)), -1.0 + 1e-12, 1.0 - 1e-12)
    theta = torch.acos(cosT)  # rad

    # in-plane 보정: ip_mask가 True인 항만 교체
    if ip_mask.any():
        idx = torch.nonzero(ip_mask, as_tuple=False).squeeze(-1)
        ii, jj, kk, mm = i[idx], j[idx], k[idx], ip_m[idx]
        ri, rj, rk, rm = x[ii], x[jj], x[kk], x[mm]

        # 평면 (ri, rk, rm)의 법선
        n = torch.cross(ri - rm, rk - rm, dim=1)
        n = n / (torch.linalg.norm(n, dim=1, keepdim=True) + 1e-15)

        # j를 평면으로 직교투영해 X 좌표 계산
        t = ((rj - rm) * n).sum(-1, keepdim=True)
        rx = rj - t * n

        a = ri - rx
        b = rk - rx
        cos_ip = torch.clamp((a * b).sum(-1) / (_norm(a) * _norm(b)), -1.0 + 1e-12, 1.0 - 1e-12)
        theta_ip = torch.acos(cos_ip)  # rad

        theta = theta.clone()
        theta[idx] = theta_ip

    return theta * RAD2DEG

# ------------ Energy kernels (Torch) ------------
def e_bond(x, pairs, kb, r0):
    if pairs.numel() == 0:
        return x.new_zeros(())
    r = _pair_r(x, pairs)          # (Nb,)
    dr = r - r0                     # r0는 이미 buffer 텐서
    t = 1.0 + CS*dr + (7.0/12.0)*(CS**2)*(dr**2)
    return (FC_BOND * 0.5 * kb * dr*dr * t).sum()

RAD2DEG  = 180.0 / math.pi
FC_ANGLE = 0.043844     # MMFF 표준 상수
CB       = -0.007       # 1/deg
LIN_THRESH = 175.0      # 선형 취급 임계값 (필요시 170~177 사이에서 조정)

def _angle_plain_rad(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    if ang.numel() == 0:
        return x.new_zeros((0,))
    i, j, k = ang[:,0], ang[:,1], ang[:,2]
    v1 = x[i] - x[j]
    v2 = x[k] - x[j]
    n1 = torch.clamp(torch.linalg.norm(v1, dim=1), min=1e-12)
    n2 = torch.clamp(torch.linalg.norm(v2, dim=1), min=1e-12)
    cosT = torch.clamp((v1 * v2).sum(-1) / (n1 * n2), -1.0 + 1e-12, 1.0 - 1e-12)
    return torch.acos(cosT)  # radians

def e_angle(x: torch.Tensor,
            ang: torch.Tensor,
            ka: torch.Tensor,
            theta0_deg: torch.Tensor,
            angle_type: torch.Tensor) -> torch.Tensor:
    """
    MMFF Angle bending:
      - '선형'은 오직 θ0가 near‑linear일 때만 적용 (예: θ0 >= LIN_THRESH).
      - 그 외는 비선형 3차 보정식.
    주의: RDKit의 angleType 코드는 '선형 공식을 쓰라'는 뜻이 아니다.
    """
    if ang.numel() == 0:
        return x.new_zeros(())

    theta = _angle_plain_rad(x, ang)         # rad
    theta_deg = theta * RAD2DEG

    # 선형 분기: theta0 기준으로만 결정
    lin_mask   = (theta0_deg >= LIN_THRESH)
    nonlin_mask = ~lin_mask

    E = x.new_zeros(())

    # 선형: E = 143.9325 * ka * (1 + cos(theta))
    if lin_mask.any():
        E = E + (143.9325 * ka[lin_mask] * (1.0 + torch.cos(theta[lin_mask]))).sum()

    # 비선형: E = 0.5 * 0.043844 * ka * dθ^2 * (1 + CB * dθ), dθ in deg
    if nonlin_mask.any():
        dth = theta_deg[nonlin_mask] - theta0_deg[nonlin_mask]
        E = E + (0.5 * FC_ANGLE * ka[nonlin_mask] * dth * dth * (1.0 + CB * dth)).sum()

    return E



def e_stretch_bend(x, stbn, kba_ijk, kba_kji, sb_r0_ij, sb_r0_kj, sb_theta0_deg):
    if stbn.numel() == 0:
        return x.new_zeros(())
    i, j, k = stbn[:,0], stbn[:,1], stbn[:,2]
    rij = _pair_r(x, torch.stack([i, j], dim=1))
    rkj = _pair_r(x, torch.stack([k, j], dim=1))
    Drij = rij - sb_r0_ij
    Drkj = rkj - sb_r0_kj
    th_deg = _angle(x, stbn) * RAD2DEG
    DT = th_deg - sb_theta0_deg
    # E_SB = 2.51210 * (kba_ijk * Δr_ij + kba_kji * Δr_kj) * Δθ
    return (FC_STBN * (kba_ijk * Drij + kba_kji * Drkj) * DT).sum()


def e_oop(x, imp, koop):
    if imp.numel()==0: return x.new_zeros(())
    chi = _wilson_oop_deg(x, imp)
    return (0.5 * FC_ANGLE * koop * chi * chi).sum()

def e_tors(x, tors, V1, V2, V3):
    if tors.numel()==0: return x.new_zeros(())
    w = _dihedral(x, tors)
    return (0.5*(V1*(1.0+torch.cos(w)) + V2*(1.0 - torch.cos(2*w)) + V3*(1.0 + torch.cos(3*w)))).sum()

def e_vdw(x, pairs, Rstar, eps, is14, scale14_vdw: float = 1.0):
    if pairs.numel()==0: return x.new_zeros(())
    r = _pair_r(x, pairs)
    R = Rstar
    t1 = ((1.07*R) / (r + 0.07*R)) ** 7
    R7 = R ** 7
    t2 = (1.12*R7) / (r**7 + 0.12*R7) - 2.0
    e = eps * t1 * t2
    if scale14_vdw != 1.0:
        scale = torch.where(is14, torch.tensor(scale14_vdw, dtype=e.dtype, device=e.device),
                            torch.tensor(1.0, dtype=e.dtype, device=e.device))
        e = e * scale
    return e.sum()

def e_elec(x, pairs, qi, is14, dielectric: float = 1.0, scale14_elec: float = SCALE14_ELEC):
    if pairs.numel()==0: return x.new_zeros(())
    r = _pair_r(x, pairs)
    qij = qi[pairs[:,0]] * qi[pairs[:,1]]
    base = KE * qij / (dielectric * (r + COUL_BUF))
    scale = torch.where(is14, torch.tensor(scale14_elec, dtype=base.dtype, device=base.device),
                        torch.tensor(1.0, dtype=base.dtype, device=base.device))
    return (scale * base).sum()

# ------------ Main module ------------
class MMFFForceFieldTorch(nn.Module):
    def __init__(self, mol: Chem.Mol, *, variant="MMFF94", include_hs=True,
                 enable_coulomb=True, enable_vdw=True,
                 dielectric: float = 1.0, scale14_vdw: float = 1.0):
        super().__init__()
        if include_hs and not any(a.GetAtomicNum()==1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=False)
        self.mol = mol
        self.variant = variant
        self.enable_coulomb = enable_coulomb
        self.enable_vdw = enable_vdw
        self.dielectric = dielectric
        self.scale14_vdw = scale14_vdw

        p = collect_mmff_params(mol, variant=variant)
        # register buffers (torch tensors)
        def ti(x): return torch.from_numpy(x).long()
        def tf(x): return torch.from_numpy(x).float()
        self.register_buffer("bonds", ti(p.bonds))
        self.register_buffer("kb", tf(p.kb))
        self.register_buffer("r0", tf(p.r0))
        self.register_buffer("angles", ti(p.angles))
        # --- in-plane angle metadata (trigonal centers only: heavy neighbors == 3) ---
        angle_ip_mask = []
        angle_ip_m = []  # third heavy neighbor index for the (i,j,k) angle, else -1

        for (i, j, k) in self.angles.cpu().numpy().tolist():
            # heavy neighbors of j (exclude hydrogens)
            nbs = [
                nb.GetIdx()
                for nb in self.mol.GetAtomWithIdx(int(j)).GetNeighbors()
                if self.mol.GetAtomWithIdx(nb.GetIdx()).GetAtomicNum() != 1
            ]
            rest = [t for t in nbs if t not in (i, k)]
            if len(rest) == 1:   # trigonal center: exactly 3 heavy neighbors
                angle_ip_mask.append(True)
                angle_ip_m.append(rest[0])
            else:
                angle_ip_mask.append(False)
                angle_ip_m.append(-1)

        self.register_buffer("angle_ip_mask", torch.tensor(angle_ip_mask, dtype=torch.bool))
        self.register_buffer("angle_ip_m",    torch.tensor(angle_ip_m,    dtype=torch.long))

        self.register_buffer("ka", tf(p.ka))
        self.register_buffer("theta0_deg", tf(p.theta0_deg))
        self.register_buffer("angle_type", torch.from_numpy(p.angle_type).long())
        self.register_buffer("stbn", ti(p.stbn))
        self.register_buffer("kba_ijk", tf(p.kba_ijk))
        self.register_buffer("kba_kji", tf(p.kba_kji))
        self.register_buffer("sb_r0_ij", tf(p.sb_r0_ij))
        self.register_buffer("sb_r0_kj", tf(p.sb_r0_kj))
        self.register_buffer("sb_theta0_deg", tf(p.sb_theta0_deg))
        self.register_buffer("impropers", ti(p.impropers))
        self.register_buffer("koop", tf(p.koop))
        self.register_buffer("torsions", ti(p.torsions))
        self.register_buffer("V1", tf(p.V1))
        self.register_buffer("V2", tf(p.V2))
        self.register_buffer("V3", tf(p.V3))
        self.register_buffer("nb_pairs", ti(p.nb_pairs))
        self.register_buffer("is14", torch.from_numpy(p.is14).bool())
        self.register_buffer("Rstar", tf(p.Rstar))
        self.register_buffer("eps", tf(p.eps))
        self.register_buffer("qi", tf(p.qi))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.energy(positions)

    def energy(self, positions: torch.Tensor) -> torch.Tensor:
        x = positions
        E  = e_bond(x, self.bonds, self.kb, self.r0)
        E += e_angle(x, self.angles, self.ka, self.theta0_deg, self.angle_type)
        E += e_stretch_bend(x, self.stbn, self.kba_ijk, self.kba_kji,
                            self.sb_r0_ij, self.sb_r0_kj, self.sb_theta0_deg)
        E += e_oop(x, self.impropers, self.koop)
        E += e_tors(x, self.torsions, self.V1, self.V2, self.V3)
        if self.enable_vdw:
            E += e_vdw(x, self.nb_pairs, self.Rstar, self.eps, self.is14, self.scale14_vdw)
        if self.enable_coulomb:
            E += e_elec(x, self.nb_pairs, self.qi, self.is14, dielectric=self.dielectric)
        return E

    def forces(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.requires_grad_(True)
        E = self.energy(positions)
        (grad,) = torch.autograd.grad(E, positions, create_graph=False)
        return -grad

def build_forcefield(mol: Chem.Mol, **kwargs) -> MMFFForceFieldTorch:
    return MMFFForceFieldTorch(mol, **kwargs)

def forcefield_from_file(path: str, **kwargs) -> MMFFForceFieldTorch:
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext in (".smi", ".smiles", ".ism"):
        from rdkit.Chem import SmilesMolSupplier
        suppl = SmilesMolSupplier(path, titleLine=False, sanitize=True)
        mol = next((m for m in suppl if m is not None), None)
    elif ext in (".sdf", ".sd"):
        from rdkit.Chem import SDMolSupplier
        suppl = SDMolSupplier(path, removeHs=False, sanitize=True)
        mol = next((m for m in suppl if m is not None), None)
    elif ext in (".mol", ".mdl"):
        mol = Chem.MolFromMolFile(path, sanitize=True, removeHs=False)
    else:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        mol = Chem.MolFromSmiles(first)
    if mol is None:
        raise ValueError(f"Could not read molecule from {path!r}")
    return build_forcefield(mol, **kwargs)
