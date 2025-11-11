# MMFF94 PyTorch

A lightweight PyTorch implementation of the Merck Molecular Force Field (MMFF94).
The project is heavily inspired by [kim-iljung/UFF_PyTorch](https://github.com/kim-iljung/UFF_PyTorch)
and follows a similar API so that switching between force fields is
straightforward.  The primary goal is to provide differentiable molecular
mechanics primitives that run efficiently on both CPUs and GPUs.

## Features

- Harmonic bond and angle potentials using MMFF94 parameters
- Proper torsion terms with cosine expansions
- Lennard-Jones style van der Waals interactions and Coulomb electrostatics
- Fully vectorised PyTorch implementation with autograd support

## Installation

The package is self-contained and only depends on PyTorch.  Install the
repository in editable mode after cloning:

```bash
pip install -e .
```

## Usage

```python
import torch
from mmff.forcefield import MMFFForceField

# Atom types follow the short-hand names commonly used in MMFF94 tables.
atom_types = ["C", "H", "H", "H", "H"]

bonds = [
    (0, 1), (0, 2), (0, 3), (0, 4)
]
angles = [
    (1, 0, 2), (1, 0, 3), (1, 0, 4),
    (2, 0, 3), (2, 0, 4), (3, 0, 4)
]
torsions = [
    (1, 0, 2, 3), (1, 0, 2, 4), (1, 0, 3, 4), (2, 0, 3, 4)
]

# Exclude bonded pairs from the non-bonded list in your own workflow.
pairs = [
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4), (3, 4)
]

forcefield = MMFFForceField(atom_types, bonds=bonds, angles=angles, torsions=torsions, pairs=pairs)

# Coordinates are given in Å.
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

energy = forcefield.energy(coords)
print(f"Total energy: {energy.item():.6f} kcal/mol")
```

## Testing

Run the unit tests with

```bash
pytest
```
