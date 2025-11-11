"""MMFF94 parameter access backed by RDKit or Open Babel installations.

The original version of this project bundled a very small, hand curated
selection of MMFF94 parameters.  That made the repository easy to understand
but it limited the type of systems that could be modelled.  The new
implementation mirrors the canonical parameter set that ships with RDKit (and
is also distributed with Open Babel) by parsing the ``MMFF94.ff`` data file
from the locally installed chemistry toolkit.  A light-weight cache is
constructed at import time so that the rest of the code base can keep using the
simple dictionary based lookups from the previous revision.

Only a subset of the data present in ``MMFF94.ff`` is required by the current
PyTorch force-field implementation.  The parser therefore focuses on the
following sections:

``@ATOMTYPES``
    Defines the van der Waals parameters and default partial charges.

``@BONDSTRETCH``
    Provides harmonic bond force constants and equilibrium lengths.

``@ANGLEBEND``
    Contains the harmonic angle bending parameters.

``@TORSION``
    Holds the Fourier torsion parameters used in the torsional energy term.

The code deliberately keeps the parsing logic simple and permissive so that it
continues to work if the RDKit project slightly adjusts whitespace or includes
additional, currently unused columns in the future.  The loader will raise a
clear error when RDKit cannot be found or when the expected parameter file is
missing which makes debugging configuration issues significantly easier for
downstream users.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import torch


try:  # pragma: no cover - exercised indirectly through unit tests
    from rdkit import RDConfig
except ImportError:  # pragma: no cover - executed when RDKit is missing
    RDConfig = None


# ---------------------------------------------------------------------------
# Dataclasses used throughout the project


@dataclass(frozen=True)
class AtomType:
    """Container for atomic parameters."""

    symbol: str
    r_vdw: float
    epsilon: float
    alpha: float
    charge: float


@dataclass(frozen=True)
class BondType:
    k: float
    r0: float


@dataclass(frozen=True)
class AngleType:
    k: float
    theta0: float


@dataclass(frozen=True)
class TorsionType:
    v1: float
    v2: float
    v3: float


# ---------------------------------------------------------------------------
# File discovery helpers


def _candidate_search_roots() -> List[Path]:
    """Return directories that may contain the ``MMFF94.ff`` file."""

    roots: List[Path] = []

    if RDConfig is not None:
        roots.append(Path(RDConfig.RDDataDir))

    babel_data = os.environ.get("BABEL_DATADIR")
    if babel_data:
        roots.append(Path(babel_data))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        roots.append(Path(conda_prefix) / "share" / "openbabel")

    return roots


def _find_mmff_parameter_file(mmff_variant: str = "MMFF94") -> Path:
    """Return the path to the chemistry toolkit ``.ff`` file for the variant."""

    pattern = f"{mmff_variant.upper()}.ff"
    search_roots = _candidate_search_roots()

    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.rglob(pattern):
            if candidate.is_file():
                return candidate

    search_paths = ", ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        f"Unable to locate the MMFF parameter file '{pattern}'. Searched: {search_paths or 'no known locations'}"
    )


def _tokenize_parameter_file(path: Path) -> Mapping[str, List[List[str]]]:
    """Tokenise the RDKit parameter file into sections.

    The file is organised into sections starting with a line containing
    ``@SECTION_NAME``.  Each subsequent non-empty line belongs to that section
    until a new ``@SECTION`` line is encountered.  Lines beginning with ``#``
    are comments and ignored.
    """

    sections: MutableMapping[str, List[List[str]]] = {}
    current: Optional[str] = None
    with path.open("r", encoding="utf8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                current = line[1:].strip().upper()
                sections.setdefault(current, [])
                continue
            if current is None:
                raise ValueError(
                    f"Encountered data before section header in '{path}'."
                )
            sections[current].append(line.split())
    return sections


# ---------------------------------------------------------------------------
# Conversion helpers transforming the tokenised data into the dataclasses used
# by the force-field implementation.


def _build_atom_types(sections: Mapping[str, List[List[str]]]) -> Tuple[
    Dict[int, str],
    Dict[str, AtomType],
]:
    try:
        atom_records = sections["ATOMTYPES"]
    except KeyError as exc:
        raise KeyError("The MMFF parameter file does not define @ATOMTYPES") from exc

    id_to_symbol: Dict[int, str] = {}
    symbol_to_params: Dict[str, AtomType] = {}

    for fields in atom_records:
        if len(fields) < 6:
            raise ValueError(
                "ATOMTYPES entries must provide an identifier and at least "
                "the van der Waals parameters."
            )
        type_id = int(fields[0])
        symbol = fields[1]
        try:
            r_vdw, epsilon, alpha, charge = map(float, fields[-4:])
        except ValueError as exc:
            raise ValueError(f"Failed to parse ATOMTYPES record: {' '.join(fields)}") from exc

        id_to_symbol[type_id] = symbol
        symbol_to_params[symbol] = AtomType(symbol, r_vdw, epsilon, alpha, charge)

    return id_to_symbol, symbol_to_params


def _translate_keys(
    raw_fields: List[str],
    id_to_symbol: Mapping[int, str],
) -> Tuple[str, ...]:
    """Translate integer identifiers to symbolic atom type names."""

    key: List[str] = []
    for token in raw_fields:
        try:
            type_id = int(token)
        except ValueError:
            key.append(token)
            continue
        try:
            key.append(id_to_symbol[type_id])
        except KeyError as exc:
            raise KeyError(f"Unknown atom type id {type_id}") from exc
    return tuple(key)


def _build_pairwise_parameters(
    section_name: str,
    sections: Mapping[str, List[List[str]]],
    id_to_symbol: Mapping[int, str],
    value_width: int,
) -> Dict[Tuple[str, ...], Tuple[float, ...]]:
    try:
        records = sections[section_name]
    except KeyError as exc:
        raise KeyError(f"The MMFF parameter file lacks @{section_name}") from exc

    parameters: Dict[Tuple[str, ...], Tuple[float, ...]] = {}
    for fields in records:
        if len(fields) < value_width:
            raise ValueError(
                f"Expected at least {value_width} columns in @{section_name} record"
            )
        key_fields = fields[:-value_width]
        values = tuple(float(v) for v in fields[-value_width:])
        key = _translate_keys(key_fields, id_to_symbol)
        parameters[key] = values
    return parameters


def _initialise_parameter_tables(
    mmff_variant: str = "MMFF94",
) -> Tuple[
    Dict[str, AtomType],
    Dict[Tuple[str, str], BondType],
    Dict[Tuple[str, str, str], AngleType],
    Dict[Tuple[str, str, str, str], TorsionType],
]:
    path = _find_mmff_parameter_file(mmff_variant)
    sections = _tokenize_parameter_file(path)

    id_to_symbol, atom_parameters = _build_atom_types(sections)

    bond_raw = _build_pairwise_parameters("BONDSTRETCH", sections, id_to_symbol, 2)
    angle_raw = _build_pairwise_parameters("ANGLEBEND", sections, id_to_symbol, 2)
    torsion_raw = _build_pairwise_parameters("TORSION", sections, id_to_symbol, 3)

    bond_parameters: Dict[Tuple[str, str], BondType] = {}
    for key, (k, r0) in bond_raw.items():
        if len(key) != 2:
            raise ValueError(f"Bond record {key} does not contain two atom types")
        bond_parameters[(key[0], key[1])] = BondType(k, r0)

    angle_parameters: Dict[Tuple[str, str, str], AngleType] = {}
    for key, (k, theta0) in angle_raw.items():
        if len(key) != 3:
            raise ValueError(f"Angle record {key} does not contain three atom types")
        angle_parameters[(key[0], key[1], key[2])] = AngleType(k, theta0)

    torsion_parameters: Dict[Tuple[str, str, str, str], TorsionType] = {}
    for key, (v1, v2, v3) in torsion_raw.items():
        if len(key) != 4:
            raise ValueError(f"Torsion record {key} does not contain four atom types")
        torsion_parameters[(key[0], key[1], key[2], key[3])] = TorsionType(v1, v2, v3)

    return atom_parameters, bond_parameters, angle_parameters, torsion_parameters


# Initialise caches on module import.  RDKit ships with around 700 atom types
# and a few thousand interaction parameters which is compact enough to keep in
# memory without any noticeable overhead.


ATOMIC_PARAMETERS, BOND_PARAMETERS, ANGLE_PARAMETERS, TORSION_PARAMETERS = (
    _initialise_parameter_tables()
)


COULOMB_CONSTANT = 332.063709  # kcal·Å·mol⁻¹·e⁻²


# ---------------------------------------------------------------------------
# Lookup helpers mirroring the API of the previous revision


def lookup_bond(atom_types: Iterable[str]) -> BondType:
    key = tuple(atom_types)
    if key in BOND_PARAMETERS:
        return BOND_PARAMETERS[key]
    reversed_key = tuple(reversed(key))
    if reversed_key in BOND_PARAMETERS:
        return BOND_PARAMETERS[reversed_key]
    raise KeyError(f"No bond parameters for {atom_types}")


def lookup_angle(atom_types: Iterable[str]) -> AngleType:
    key = tuple(atom_types)
    if key in ANGLE_PARAMETERS:
        return ANGLE_PARAMETERS[key]
    reversed_key = tuple(reversed(key))
    if reversed_key in ANGLE_PARAMETERS:
        return ANGLE_PARAMETERS[reversed_key]
    raise KeyError(f"No angle parameters for {atom_types}")


def lookup_torsion(atom_types: Iterable[str]) -> TorsionType:
    key = tuple(atom_types)
    if key in TORSION_PARAMETERS:
        return TORSION_PARAMETERS[key]
    reversed_key = tuple(reversed(key))
    if reversed_key in TORSION_PARAMETERS:
        return TORSION_PARAMETERS[reversed_key]
    raise KeyError(f"No torsion parameters for {atom_types}")


def combine_vdw(atom_a: AtomType, atom_b: AtomType) -> Tuple[torch.Tensor, torch.Tensor]:
    r = torch.tensor((atom_a.r_vdw * atom_b.r_vdw) ** 0.5, dtype=torch.float32)
    epsilon = torch.tensor((atom_a.epsilon * atom_b.epsilon) ** 0.5, dtype=torch.float32)
    return r, epsilon

