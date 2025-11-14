# forcefield_full_torch.py  (fixed)
from __future__ import annotations

import math
import torch
from torch import nn
from rdkit import Chem

from .parameters import collect_mmff_params

# ---- MMFF constants ----
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
# Bond stretch scale (mdyne/Å -> kcal/mol)
FC_BOND = 143.9325
# Angle/oop scale
FC_ANGLE = 0.043844
CB = -0.007        # 1/deg (angle cubic correction)
LIN_THRESH = 175.0 # treat as linear when theta0 is above this threshold (deg)
# Stretch–bend scale
FC_STBN = 2.51210
# Cubic corrections
CS = -2.0      # Å^-1 (bond)
# Electrostatics
KE = 332.0716
COUL_BUF = 0.05
SCALE14_ELEC = 0.75
# vdW Buffered 14–7 (Halgren)
# Evdw = eps * ( (1.07 R*/(r + 0.07 R*))^7 ) * ( (1.12 R*^7/(r^7 + 0.12 R*^7)) - 2 )


# ------------ Torch geometry helpers ------------
def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Ensure ``x`` has a batch dimension."""
    if x.dim() == 2:
        return x.unsqueeze(0), True
    if x.dim() == 3:
        return x, False
    raise ValueError("positions tensor must have shape (N, 3) or (B, N, 3)")


def _norm(v: torch.Tensor, eps=1e-12):
    return torch.clamp(torch.linalg.norm(v, dim=-1), min=eps)


def _pair_r(x: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return x.new_zeros((x.shape[0], 0))
    xi = x[:, pairs[:, 0], :]
    xj = x[:, pairs[:, 1], :]
    return _norm(xi - xj)


def _angle(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    if ang.numel() == 0:
        return x.new_zeros((x.shape[0], 0))
    i, j, k = ang[:, 0], ang[:, 1], ang[:, 2]
    v1 = x[:, i, :] - x[:, j, :]
    v2 = x[:, k, :] - x[:, j, :]
    c = torch.clamp((v1 * v2).sum(-1) / (_norm(v1) * _norm(v2)), -1 + 1e-12, 1 - 1e-12)
    return torch.acos(c)  # rad

def _dihedral(x: torch.Tensor, tors: torch.Tensor) -> torch.Tensor:
    if tors.numel() == 0:
        return x.new_zeros((x.shape[0], 0))
    i, j, k, l = tors[:, 0], tors[:, 1], tors[:, 2], tors[:, 3]
    r1, r2, r3, r4 = x[:, i, :], x[:, j, :], x[:, k, :], x[:, l, :]
    b0 = r1 - r2
    b1 = r3 - r2
    b2 = r4 - r3
    b1u = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + 1e-15)
    v = b0 - (b0 * b1u).sum(-1, keepdim=True) * b1u
    w = b2 - (b2 * b1u).sum(-1, keepdim=True) * b1u
    vu = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-15)
    wu = w / (torch.linalg.norm(w, dim=-1, keepdim=True) + 1e-15)
    xcomp = (vu * wu).sum(-1)
    ycomp = (torch.cross(b1u, vu, dim=-1) * wu).sum(-1)
    return torch.atan2(ycomp, xcomp)

def _wilson_oop_deg(x: torch.Tensor, imp: torch.Tensor) -> torch.Tensor:
    if imp.numel() == 0:
        return x.new_zeros((x.shape[0], 0))
    i, j, k, l = imp[:, 0], imp[:, 1], imp[:, 2], imp[:, 3]
    ri, rj, rk, rl = x[:, i, :], x[:, j, :], x[:, k, :], x[:, l, :]
    n = torch.cross(ri - rj, rk - rj, dim=-1)
    nh = n / (torch.linalg.norm(n, dim=-1, keepdim=True) + 1e-15)
    v = rl - rj
    vh = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-15)
    chi = torch.asin(torch.clamp((nh * vh).sum(-1), -1 + 1e-12, 1 - 1e-12))
    return chi * RAD2DEG

# ------------ Energy kernels (Torch) ------------
def e_bond(x, pairs, kb, r0):
    if pairs.numel() == 0:
        return x.new_zeros((x.shape[0],))
    r = _pair_r(x, pairs)          # (B, Nb)
    dr = r - r0.unsqueeze(0)       # r0 is already a tensor buffer
    t = 1.0 + CS * dr + (7.0 / 12.0) * (CS ** 2) * (dr ** 2)
    return (FC_BOND * 0.5 * kb.unsqueeze(0) * dr * dr * t).sum(dim=1)

def _angle_plain_rad(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    if ang.numel() == 0:
        return x.new_zeros((x.shape[0], 0))
    i, j, k = ang[:, 0], ang[:, 1], ang[:, 2]
    v1 = x[:, i, :] - x[:, j, :]
    v2 = x[:, k, :] - x[:, j, :]
    n1 = torch.clamp(torch.linalg.norm(v1, dim=-1), min=1e-12)
    n2 = torch.clamp(torch.linalg.norm(v2, dim=-1), min=1e-12)
    cosT = torch.clamp((v1 * v2).sum(-1) / (n1 * n2), -1.0 + 1e-12, 1.0 - 1e-12)
    return torch.acos(cosT)  # radians

def e_angle(x: torch.Tensor,
            ang: torch.Tensor,
            ka: torch.Tensor,
            theta0_deg: torch.Tensor,
            angle_type: torch.Tensor) -> torch.Tensor:
    """
    MMFF angle bending:
      - Use the linear expression only when θ0 is near-linear (θ0 >= LIN_THRESH).
      - Otherwise use the non-linear cubic correction.
    Note: the RDKit angleType flag does not request the linear model explicitly; we
    select it based solely on θ0 as in the reference implementation.
    """
    if ang.numel() == 0:
        return x.new_zeros((x.shape[0],))

    theta = _angle_plain_rad(x, ang)         # rad
    theta_deg = theta * RAD2DEG

    # Split solely based on the reference angle θ0.
    lin_mask   = (theta0_deg >= LIN_THRESH)
    nonlin_mask = ~lin_mask

    E = x.new_zeros((x.shape[0],))

    # Linear term: E = 143.9325 * ka * (1 + cos(theta))
    if lin_mask.any():
        th_lin = theta[:, lin_mask]
        ka_lin = ka[lin_mask]
        E = E + (143.9325 * ka_lin * (1.0 + torch.cos(th_lin))).sum(dim=1)

    # Non-linear term: E = 0.5 * 0.043844 * ka * dθ^2 * (1 + CB * dθ), with dθ in degrees
    if nonlin_mask.any():
        th_nonlin = theta_deg[:, nonlin_mask]
        theta0_nonlin = theta0_deg[nonlin_mask]
        ka_nonlin = ka[nonlin_mask]
        dth = th_nonlin - theta0_nonlin
        E = E + (0.5 * FC_ANGLE * ka_nonlin * dth * dth * (1.0 + CB * dth)).sum(dim=1)

    return E



def e_stretch_bend(x, stbn, kba_ijk, kba_kji, sb_r0_ij, sb_r0_kj, sb_theta0_deg):
    if stbn.numel() == 0:
        return x.new_zeros((x.shape[0],))
    i, j, k = stbn[:,0], stbn[:,1], stbn[:,2]
    rij = _pair_r(x, torch.stack([i, j], dim=1))
    rkj = _pair_r(x, torch.stack([k, j], dim=1))
    Drij = rij - sb_r0_ij.unsqueeze(0)
    Drkj = rkj - sb_r0_kj.unsqueeze(0)
    th_deg = _angle(x, stbn) * RAD2DEG
    DT = th_deg - sb_theta0_deg.unsqueeze(0)
    # E_SB = 2.51210 * (kba_ijk * Δr_ij + kba_kji * Δr_kj) * Δθ
    return (FC_STBN * (kba_ijk.unsqueeze(0) * Drij + kba_kji.unsqueeze(0) * Drkj) * DT).sum(dim=1)


def e_oop(x, imp, koop):
    if imp.numel()==0:
        return x.new_zeros((x.shape[0],))
    chi = _wilson_oop_deg(x, imp)
    return (0.5 * FC_ANGLE * koop.unsqueeze(0) * chi * chi).sum(dim=1)

def e_tors(x, tors, V1, V2, V3):
    if tors.numel()==0: return x.new_zeros((x.shape[0],))
    w = _dihedral(x, tors)
    term = (0.5 * (V1.unsqueeze(0) * (1.0+torch.cos(w)) +
                   V2.unsqueeze(0) * (1.0 - torch.cos(2*w)) +
                   V3.unsqueeze(0) * (1.0 + torch.cos(3*w))))
    return term.sum(dim=1)

def e_vdw(x, pairs, Rstar, eps, is14, scale14_vdw: float = 1.0):
    if pairs.numel()==0: return x.new_zeros((x.shape[0],))
    r = _pair_r(x, pairs)
    R = Rstar.unsqueeze(0)
    t1 = ((1.07*R) / (r + 0.07*R)) ** 7
    R7 = R ** 7
    t2 = (1.12*R7) / (r**7 + 0.12*R7) - 2.0
    e = eps.unsqueeze(0) * t1 * t2
    if scale14_vdw != 1.0:
        scale = torch.ones_like(e)
        scale[:, is14] = scale14_vdw
        e = e * scale
    return e.sum(dim=1)

def e_elec(x, pairs, qi, is14, dielectric: float = 1.0, scale14_elec: float = SCALE14_ELEC):
    if pairs.numel()==0: return x.new_zeros((x.shape[0],))
    r = _pair_r(x, pairs)
    qij = qi[pairs[:,0]] * qi[pairs[:,1]]
    base = KE * qij.unsqueeze(0) / (dielectric * (r + COUL_BUF))
    if scale14_elec != 1.0 or is14.any():
        scale = torch.ones_like(base)
        scale[:, is14] = scale14_elec
    else:
        scale = 1.0
    return (scale * base).sum(dim=1)

# ------------ Main module ------------
class MMFFTorch(nn.Module):
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
        x, squeezed = _ensure_batch(positions)
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
        return E.squeeze(0) if squeezed else E

    def forces(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.requires_grad_(True)
        E = self.energy(positions)
        if E.dim() == 0:
            grad, = torch.autograd.grad(E, positions, create_graph=False)
        else:
            grad, = torch.autograd.grad(E.sum(), positions, create_graph=False)
        return -grad

def build_forcefield(mol: Chem.Mol, **kwargs) -> MMFFTorch:
    return MMFFTorch(mol, **kwargs)

def forcefield_from_file(path: str, **kwargs) -> MMFFTorch:
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
