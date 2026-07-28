"""Helpers for generating Quantum ESPRESSO input files from configuration data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms

from .config import Config


class QEInput:
    """Serialize a minimal Quantum ESPRESSO input deck from configuration data."""

    def __init__(self, config: Config | dict[str, Any], atoms: Atoms | None = None) -> None:
        self.config = config if isinstance(config, Config) else Config.from_dict(config)
        self.atoms = atoms
        self._validate()

    @classmethod
    def from_config(cls, config: Config | dict[str, Any], atoms: Atoms | None = None, prefix: str | None = None) -> "QEInput":
        """Create a QE input object from a configuration object or mapping."""
        if isinstance(config, Config):
            cfg = config.data
        else:
            cfg = config

        if prefix is not None:
            cfg = dict(cfg)
            cfg.setdefault("qe", {})
            cfg["qe"]["prefix"] = prefix

        return cls(cfg, atoms=atoms)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QEInput":
        """Create a QE input object from a plain mapping."""
        return cls(data)

    def _validate(self) -> None:
        """Validate required sections and values before writing."""
        if isinstance(self.config, Config):
            data = self.config.data
        else:
            data = self.config

        required_sections = ["qe"]
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Missing required configuration section: {section}")

        qe = data["qe"]
        required_fields = ["prefix", "ecutwfc", "ecutrho", "occupations", "conv_thr"]
        for field in required_fields:
            if field not in qe:
                raise ValueError(f"Missing required QE field: {field}")

        if not isinstance(qe["prefix"], str) or not qe["prefix"].strip():
            raise ValueError("QE prefix must be a non-empty string")

    def _control_block(self) -> str:
        qe = self._data()["qe"]
        lines = ["&CONTROL", f"  prefix = '{qe.get('prefix', 'gas')}'"]
        if qe.get("calculation", {}).get("scf"):
            lines.append("  calculation = 'scf'")
        if qe.get("calculation", {}).get("relax"):
            lines.append("  calculation = 'relax'")
        lines.extend(["/", ""])
        return "\n".join(lines)

    def _system_block(self) -> str:
        qe = self._data()["qe"]
        lines = [
            "&SYSTEM",
            f"  ibrav = 0",
            f"  nat = {len(self.atoms) if self.atoms is not None else 0}",
            f"  ntyp = {len({atom.symbol for atom in self.atoms}) if self.atoms is not None else 0}",
            f"  ecutwfc = {qe.get('ecutwfc', 80)}",
            f"  ecutrho = {qe.get('ecutrho', 640)}",
            f"  occupations = '{qe.get('occupations', 'smearing')}'",
            f"  smearing = '{qe.get('smearing', 'mv')}'",
            f"  degauss = {qe.get('degauss', 0.02)}",
            "/",
            "",
        ]
        return "\n".join(lines)

    def _electrons_block(self) -> str:
        qe = self._data()["qe"]
        lines = [
            "&ELECTRONS",
            f"  conv_thr = {qe.get('conv_thr', 1.0e-8)}",
            f"  electron_maxstep = {qe.get('electron_maxstep', 200)}",
            f"  mixing_beta = {qe.get('mixing_beta', 0.3)}",
            "/",
            "",
        ]
        return "\n".join(lines)

    def _atomic_species_block(self) -> str:
        if self.atoms is None:
            return "ATOMIC_SPECIES\n\n"

        symbols = sorted({atom.symbol for atom in self.atoms})
        lines = ["ATOMIC_SPECIES"]
        for symbol in symbols:
            lines.append(f"  {symbol} 1.00  {symbol}.upf")
        lines.append("")
        return "\n".join(lines)

    def _atomic_positions_block(self) -> str:
        if self.atoms is None:
            return "ATOMIC_POSITIONS (angstrom)\n\n"

        lines = ["ATOMIC_POSITIONS (angstrom)"]
        for atom in self.atoms:
            lines.append(f"  {atom.symbol}  {atom.position[0]:.6f}  {atom.position[1]:.6f}  {atom.position[2]:.6f}")
        lines.append("")
        return "\n".join(lines)

    def _k_points_block(self) -> str:
        data = self._data()
        kpoints = data.get("kpoints", {})
        mesh = kpoints.get("scf", [6, 6, 1])
        lines = ["K_POINTS (automatic)", f"  {' '.join(str(value) for value in mesh)}  0 0 0", ""]
        return "\n".join(lines)

    def _data(self) -> dict[str, Any]:
        if isinstance(self.config, Config):
            return self.config.data
        return self.config

    def render(self) -> str:
        """Render the full input deck as a string."""
        return "\n".join(
            [
                self._control_block(),
                self._system_block(),
                self._electrons_block(),
                self._atomic_species_block(),
                self._atomic_positions_block(),
                self._k_points_block(),
            ]
        ).strip() + "\n"

    def write(self, filename: str | Path) -> Path:
        """Write the rendered input deck to disk."""
        path = Path(filename)
        path.write_text(self.render())
        return path
