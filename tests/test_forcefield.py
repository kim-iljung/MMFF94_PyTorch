import pytest

torch = pytest.importorskip("torch")

from mmff.forcefield import MMFFForceField


def methane_example():
    atom_types = ["C", "H", "H", "H", "H"]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    angles = [
        (1, 0, 2), (1, 0, 3), (1, 0, 4),
        (2, 0, 3), (2, 0, 4), (3, 0, 4)
    ]
    torsions = [
        (1, 0, 2, 3), (1, 0, 2, 4), (1, 0, 3, 4), (2, 0, 3, 4)
    ]
    pairs = [
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4), (3, 4)
    ]
    coords = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
        dtype=torch.float32,
    )
    return atom_types, bonds, angles, torsions, pairs, coords


def test_methane_equilibrium_energy_is_small():
    atom_types, bonds, angles, torsions, pairs, coords = methane_example()
    ff = MMFFForceField(
        atom_types,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        pairs=pairs,
    )
    energy = ff.energy(coords)
    assert torch.isfinite(energy)
    assert energy.item() < 5e-2


def test_gradients_backpropagate():
    atom_types, bonds, angles, torsions, pairs, coords = methane_example()
    coords.requires_grad_(True)
    ff = MMFFForceField(
        atom_types,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        pairs=pairs,
    )
    total_energy = ff.energy(coords)
    total_energy.backward()
    assert coords.grad is not None
    assert torch.all(torch.isfinite(coords.grad))
