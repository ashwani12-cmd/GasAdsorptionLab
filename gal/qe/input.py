"""Quantum ESPRESSO pw.x input serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.data import atomic_masses

from ..config import Config
from .pseudopotentials import pseudopotential_map


class QEInput:
    """Serialize a Quantum ESPRESSO pw.x input deck from configuration data."""

    def __init__(self, config: Config | dict[str, Any], atoms: Atoms | None = None) -> None:
        self.config = config if isinstance(config, Config) else Config.from_dict(config)
        self.atoms = atoms
        self._validate()

    @classmethod
    def from_config(cls, config: Config | dict[str, Any], atoms: Atoms | None = None, prefix: str | None = None) -> "QEInput":
        cfg = config.data if isinstance(config, Config) else config
        if prefix is not None:
            cfg = {**cfg, "qe": {**cfg.get("qe", {}), "prefix": prefix}}
        return cls(cfg, atoms=atoms)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QEInput":
        return cls(data)

    def _data(self) -> dict[str, Any]:
        return self.config.data

    def _validate(self) -> None:
        qe = self._data().get("qe")
        if qe is None:
            raise ValueError("Missing required configuration section: qe")
        for field in ["prefix", "ecutwfc", "ecutrho", "occupations", "conv_thr"]:
            if field not in qe:
                raise ValueError(f"Missing required QE field: {field}")
        if not isinstance(qe["prefix"], str) or not qe["prefix"].strip():
            raise ValueError("QE prefix must be a non-empty string")

    def _control_block(self) -> str:
        qe = self._data()["qe"]
        lines = ["&CONTROL", f"  calculation = '{qe.get('calculation', 'scf')}'", f"  prefix = '{qe['prefix']}'"]
        if qe.get("pseudo_dir"):
            lines.append(f"  pseudo_dir = '{qe['pseudo_dir']}'")
        lines.extend(["/", ""])
        return "\n".join(lines)

    def _system_block(self) -> str:
        qe = self._data()["qe"]
        nat = len(self.atoms) if self.atoms is not None else 0
        ntyp = len(set(self.atoms.get_chemical_symbols())) if self.atoms is not None else 0
        lines = ["&SYSTEM", "  ibrav = 0", f"  nat = {nat}", f"  ntyp = {ntyp}", f"  ecutwfc = {qe['ecutwfc']}", f"  ecutrho = {qe['ecutrho']}", f"  occupations = '{qe['occupations']}'"]
        if qe.get("smearing"):
            lines.append(f"  smearing = '{qe['smearing']}'")
        if qe.get("degauss") is not None:
            lines.append(f"  degauss = {qe['degauss']}")
        lines.extend(["/", ""])
        return "\n".join(lines)

    def _electrons_block(self) -> str:
        qe = self._data()["qe"]
        return "\n".join(["&ELECTRONS", f"  conv_thr = {qe['conv_thr']}", f"  electron_maxstep = {qe.get('electron_maxstep', 200)}", f"  mixing_beta = {qe.get('mixing_beta', 0.3)}", "/", ""])

    def _atomic_species_block(self) -> str:
        if self.atoms is None:
            return "ATOMIC_SPECIES\n"
        overrides = self._data().get("pseudopotentials")
        mapping = pseudopotential_map(self.atoms.get_chemical_symbols(), overrides)
        lines = ["ATOMIC_SPECIES"]
        for symbol in sorted(mapping):
            number = self.atoms[ self.atoms.get_chemical_symbols().index(symbol) ].number
            lines.append(f"  {symbol}  {atomic_masses[number]:.4f}  {mapping[symbol]}")
        return "\n".join([*lines, ""])

    def _atomic_positions_block(self) -> str:
        if self.atoms is None:
            return "ATOMIC_POSITIONS (angstrom)\n"
        lines = ["ATOMIC_POSITIONS (angstrom)"]
        lines.extend(f"  {atom.symbol}  {atom.position[0]:.8f}  {atom.position[1]:.8f}  {atom.position[2]:.8f}" for atom in self.atoms)
        return "\n".join([*lines, ""])

    def _cell_parameters_block(self) -> str:
        if self.atoms is None:
            return "CELL_PARAMETERS (angstrom)\n"
        lines = ["CELL_PARAMETERS (angstrom)"]
        lines.extend(f"  {vector[0]:.8f}  {vector[1]:.8f}  {vector[2]:.8f}" for vector in self.atoms.cell.array)
        return "\n".join([*lines, ""])

    def _k_points_block(self) -> str:
        mesh = self._data().get("kpoints", {}).get("scf", [6, 6, 1])
        return f"K_POINTS (automatic)\n  {' '.join(str(value) for value in mesh)}  0 0 0\n"

    def render(self) -> str:
        return "\n".join([self._control_block(), self._system_block(), self._electrons_block(), self._atomic_species_block(), self._atomic_positions_block(), self._cell_parameters_block(), self._k_points_block()])

    def write(self, filename: str | Path) -> Path:
        path = Path(filename)
        path.write_text(self.render())
        return path
