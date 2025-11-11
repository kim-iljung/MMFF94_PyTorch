"""PyTorch implementation of the MMFF94 force field.

The implementation is intentionally compact but aims to stay faithful to the
functional forms of the MMFF94 potential.  The class is designed to work on
batched coordinates and supports autograd so that it can be used for geometry
optimisation or machine-learning applications.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from . import parameters


def _ensure_tensor(index_data: Optional[Iterable[Sequence[int]]], device: torch.device) -> Optional[Tensor]:
    if index_data is None:
        return None
    tensor = torch.as_tensor(index_data, dtype=torch.long, device=device)
    if tensor.numel() == 0:
        return None
    return tensor


def _vector_norm(v: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.sqrt(torch.sum(v * v, dim=-1) + eps)


def _angle(v1: Tensor, v2: Tensor) -> Tensor:
    dot = torch.sum(v1 * v2, dim=-1)
    denom = _vector_norm(v1) * _vector_norm(v2)
    cos_theta = torch.clamp(dot / denom, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _dihedral(r1: Tensor, r2: Tensor, r3: Tensor, r4: Tensor) -> Tensor:
    b0 = r2 - r1
    b1 = r3 - r2
    b2 = r4 - r3

    b1_norm = torch.norm(b1, dim=-1).unsqueeze(-1)
    b1_unit = b1 / b1_norm

    v = b0 - torch.sum(b0 * b1_unit, dim=-1, keepdim=True) * b1_unit
    w = b2 - torch.sum(b2 * b1_unit, dim=-1, keepdim=True) * b1_unit

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1_unit, v, dim=-1) * w, dim=-1)

    return torch.rad2deg(torch.atan2(y, x))


@dataclass
class EnergyBreakdown:
    bond: Tensor
    angle: Tensor
    torsion: Tensor
    vdw: Tensor
    electrostatic: Tensor

    def total(self) -> Tensor:
        return self.bond + self.angle + self.torsion + self.vdw + self.electrostatic


class MMFFForceField(torch.nn.Module):
    """Evaluate MMFF94 energies using PyTorch tensors."""

    def __init__(
        self,
        atom_types: Sequence[str],
        bonds: Optional[Iterable[Sequence[int]]] = None,
        angles: Optional[Iterable[Sequence[int]]] = None,
        torsions: Optional[Iterable[Sequence[int]]] = None,
        pairs: Optional[Iterable[Sequence[int]]] = None,
        charges: Optional[Sequence[float]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        self.atom_types = list(atom_types)
        self.register_buffer(
            "charges",
            self._init_charges(charges).to(self.device, dtype=self.dtype),
        )

        self.bonds = _ensure_tensor(bonds, self.device)
        self.angles = _ensure_tensor(angles, self.device)
        self.torsions = _ensure_tensor(torsions, self.device)
        self.pairs = _ensure_tensor(pairs, self.device)

        self._bond_parameters = self._prepare_bond_parameters()
        self._angle_parameters = self._prepare_angle_parameters()
        self._torsion_parameters = self._prepare_torsion_parameters()
        self._vdw_parameters = self._prepare_vdw_parameters()

    # ------------------------------------------------------------------
    def _init_charges(self, charges: Optional[Sequence[float]]) -> Tensor:
        if charges is not None:
            if len(charges) != len(self.atom_types):
                raise ValueError("Charges must match number of atoms")
            return torch.tensor(charges, dtype=torch.float32)

        values = []
        for atom_type in self.atom_types:
            params = self._get_atom_parameters(atom_type)
            values.append(params.charge)
        return torch.tensor(values, dtype=torch.float32)

    def _get_atom_parameters(self, atom_type: str) -> parameters.AtomType:
        if atom_type not in parameters.ATOMIC_PARAMETERS:
            raise KeyError(f"Unknown atom type '{atom_type}'")
        return parameters.ATOMIC_PARAMETERS[atom_type]

    # ------------------------------------------------------------------
    def _prepare_bond_parameters(self) -> Optional[Tuple[Tensor, Tensor]]:
        if self.bonds is None:
            return None
        k_list = []
        r0_list = []
        for idx0, idx1 in self.bonds.tolist():
            atom_a = self.atom_types[idx0]
            atom_b = self.atom_types[idx1]
            bond = parameters.lookup_bond((atom_a, atom_b))
            k_list.append(bond.k)
            r0_list.append(bond.r0)
        k = torch.tensor(k_list, device=self.device, dtype=self.dtype)
        r0 = torch.tensor(r0_list, device=self.device, dtype=self.dtype)
        return k, r0

    def _prepare_angle_parameters(self) -> Optional[Tuple[Tensor, Tensor]]:
        if self.angles is None:
            return None
        k_list = []
        theta0_list = []
        for idx0, idx1, idx2 in self.angles.tolist():
            types = (self.atom_types[idx0], self.atom_types[idx1], self.atom_types[idx2])
            angle = parameters.lookup_angle(types)
            k_list.append(angle.k)
            theta0_list.append(angle.theta0)
        k = torch.tensor(k_list, device=self.device, dtype=self.dtype)
        theta0 = torch.tensor(theta0_list, device=self.device, dtype=self.dtype)
        return k, theta0

    def _prepare_torsion_parameters(self) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        if self.torsions is None:
            return None
        v1_list, v2_list, v3_list = [], [], []
        for idx0, idx1, idx2, idx3 in self.torsions.tolist():
            types = (
                self.atom_types[idx0],
                self.atom_types[idx1],
                self.atom_types[idx2],
                self.atom_types[idx3],
            )
            torsion = parameters.lookup_torsion(types)
            v1_list.append(torsion.v1)
            v2_list.append(torsion.v2)
            v3_list.append(torsion.v3)
        v1 = torch.tensor(v1_list, device=self.device, dtype=self.dtype)
        v2 = torch.tensor(v2_list, device=self.device, dtype=self.dtype)
        v3 = torch.tensor(v3_list, device=self.device, dtype=self.dtype)
        return v1, v2, v3

    def _prepare_vdw_parameters(self) -> Tensor:
        epsilons = torch.zeros((len(self.atom_types), len(self.atom_types)), device=self.device, dtype=self.dtype)
        radii = torch.zeros_like(epsilons)
        for i, type_i in enumerate(self.atom_types):
            params_i = self._get_atom_parameters(type_i)
            for j, type_j in enumerate(self.atom_types):
                params_j = self._get_atom_parameters(type_j)
                r, epsilon = parameters.combine_vdw(params_i, params_j)
                radii[i, j] = r.to(self.device, dtype=self.dtype)
                epsilons[i, j] = epsilon.to(self.device, dtype=self.dtype)
        return torch.stack((radii, epsilons), dim=0)

    # ------------------------------------------------------------------
    def forward(self, coords: Tensor) -> EnergyBreakdown:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Coordinates must be a (n_atoms, 3) tensor")
        coords = coords.to(self.device, dtype=self.dtype)

        bond_energy = self._bond_energy(coords)
        angle_energy = self._angle_energy(coords)
        torsion_energy = self._torsion_energy(coords)
        vdw_energy = self._vdw_energy(coords)
        electrostatic_energy = self._electrostatic_energy(coords)

        return EnergyBreakdown(
            bond=bond_energy,
            angle=angle_energy,
            torsion=torsion_energy,
            vdw=vdw_energy,
            electrostatic=electrostatic_energy,
        )

    # ------------------------------------------------------------------
    def _bond_energy(self, coords: Tensor) -> Tensor:
        if self.bonds is None or self._bond_parameters is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        idx_i = self.bonds[:, 0]
        idx_j = self.bonds[:, 1]
        diffs = coords[idx_i] - coords[idx_j]
        dist = _vector_norm(diffs)
        k, r0 = self._bond_parameters
        stretch = dist - r0
        return 0.5 * torch.sum(k * stretch * stretch)

    def _angle_energy(self, coords: Tensor) -> Tensor:
        if self.angles is None or self._angle_parameters is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        idx_i = self.angles[:, 0]
        idx_j = self.angles[:, 1]
        idx_k = self.angles[:, 2]

        vec_ij = coords[idx_i] - coords[idx_j]
        vec_kj = coords[idx_k] - coords[idx_j]
        theta = _angle(vec_ij, vec_kj)
        k, theta0 = self._angle_parameters
        delta = theta - theta0
        return 0.5 * torch.sum(k * delta * delta)

    def _torsion_energy(self, coords: Tensor) -> Tensor:
        if self.torsions is None or self._torsion_parameters is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        idx = self.torsions
        phi = _dihedral(coords[idx[:, 0]], coords[idx[:, 1]], coords[idx[:, 2]], coords[idx[:, 3]])
        v1, v2, v3 = self._torsion_parameters
        phi_rad = torch.deg2rad(phi)
        energy = (
            v1 * (1 + torch.cos(phi_rad))
            + v2 * (1 - torch.cos(2 * phi_rad))
            + v3 * (1 + torch.cos(3 * phi_rad))
        )
        return torch.sum(energy)

    def _vdw_energy(self, coords: Tensor) -> Tensor:
        if self.pairs is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        radii = self._vdw_parameters[0]
        epsilons = self._vdw_parameters[1]
        idx_i = self.pairs[:, 0]
        idx_j = self.pairs[:, 1]
        rij = coords[idx_i] - coords[idx_j]
        r = _vector_norm(rij)
        r0 = radii[idx_i, idx_j]
        epsilon = epsilons[idx_i, idx_j]
        ratio = r0 / r
        ratio6 = ratio ** 6
        energy = epsilon * (ratio6 ** 2 - 2 * ratio6)
        return torch.sum(energy)

    def _electrostatic_energy(self, coords: Tensor) -> Tensor:
        if self.pairs is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        idx_i = self.pairs[:, 0]
        idx_j = self.pairs[:, 1]
        rij = coords[idx_i] - coords[idx_j]
        r = _vector_norm(rij)
        qi = self.charges[idx_i]
        qj = self.charges[idx_j]
        energy = parameters.COULOMB_CONSTANT * torch.sum(qi * qj / r)
        return energy

    # ------------------------------------------------------------------
    def energy(self, coords: Tensor) -> Tensor:
        """Return the total energy for convenience."""
        return self.forward(coords).total()

    def to(self, device: torch.device) -> "MMFFForceField":  # type: ignore[override]
        return super().to(device)
