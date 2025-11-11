# MMFF94 PyTorch

A PyTorch implementation of the Merck Molecular Force Field (MMFF94) with utilities to build differentiable force fields directly from RDKit molecules.

## Installation

Install the package and its dependencies with pip. RDKit provides the MMFF94 parameter file bundled inside the wheel.

If your RDKit installation does not ship the ``MMFF94.ff`` data file (some
minimal builds omit the ``share/RDKit`` directory), either set the
``MMFF_PARAMETER_PATH`` environment variable to point to the missing file,
drop a copy into ``mmff/data`` inside this repository, or rely on the automatic
fallback that downloads the BSD-licensed parameter set from the RDKit GitHub
repository. The loader stores the downloaded files in ``mmff/data`` so they are
reused on subsequent runs.

```bash
pip install rdkit-pypi torch
```

Clone this repository and install it in editable mode if you plan to hack on the codebase:

```bash
git clone https://github.com/kim-iljung/MMFF94_PyTorch.git
cd MMFF94_PyTorch
pip install -e .
```

## Quick start

The example below demonstrates how to construct an `MMFFForceField` object from an RDKit molecule and evaluate the energy and forces.

```python
from rdkit import Chem
import torch

from mmff.builder import build_forcefield

# Create an RDKit molecule with 3D coordinates.  You can also load from SDF/MOL files.
mol = Chem.MolFromSmiles("CCO")
mol = Chem.AddHs(mol)
Chem.EmbedMolecule(mol)
Chem.UFFOptimizeMolecule(mol)  # Generates a reasonable starting geometry

# Convert the RDKit conformer to a tensor with shape (num_atoms, 3)
conformer = mol.GetConformer()
positions = torch.tensor(conformer.GetPositions(), dtype=torch.float32)

# Build the force field and compute the total energy and forces
forcefield = build_forcefield(mol)
energy = forcefield.energy(positions)
forces = forcefield.forces(positions)

print(f"Energy: {energy.item():.4f} kcal/mol")
print(f"Forces: {forces}")
```

To load molecules from files (`.mol`, `.sdf`, `.smi`, `.smiles`) use `forcefield_from_file`:

```python
from mmff.builder import forcefield_from_file
forcefield = forcefield_from_file("ethanol.sdf")
```

Refer to `mmff/atom_typing.py` and `mmff/builder.py` for more advanced usage patterns such as custom MMFF variants or batched evaluation on GPUs.

## Development

The project uses `pytest` for testing and `ruff` for linting. After cloning the repository, install the development dependencies:

```bash
pip install -e .[dev]
```

Run the tests and the linter:

```bash
pytest
ruff check .
```

> **Note**: RDKit is required for both the runtime and the tests. Some CI environments do not ship RDKit wheels; in that case you can install it from conda-forge or build from source.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
