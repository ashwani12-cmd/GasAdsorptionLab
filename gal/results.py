"""Extensible analysis of completed electronic-structure calculation campaigns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write


RY_TO_EV = 13.605693122994


@dataclass
class CalculationResult:
    """Normalized result fields returned by an output parser."""

    energy: float | None = None
    final_scf_energy: float | None = None
    converged: bool = False
    scf_iterations: int | None = None
    wall_time: float | None = None
    cpu_time: float | None = None
    cell: np.ndarray | None = None
    atoms: Atoms | None = None
    pressure: float | None = None
    fermi_energy: float | None = None
    magnetization: float | None = None


class OutputParser(Protocol):
    """Protocol implemented by code-specific output parsers."""

    def parse(self, output_file: str | Path) -> CalculationResult:
        """Parse one completed or incomplete calculation output."""


def _duration_seconds(value: str) -> float:
    """Convert QE's ``1h2m3.4s`` duration notation to seconds."""
    match = re.fullmatch(r"\s*(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s\s*", value)
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)


class QuantumEspressoParser:
    """Parser for Quantum ESPRESSO ``pw.x`` output files."""

    _energy = re.compile(r"!\s+total energy\s+=\s+([-+\d.Ee]+)\s+Ry")
    _iteration = re.compile(r"iteration\s+#\s*(\d+)", re.IGNORECASE)
    _time = re.compile(r"PWSCF\s*:\s*([\d.]+h?[\d.]*m?[\d.]*s)\s+CPU\s+([\d.]+h?[\d.]*m?[\d.]*s)\s+WALL")
    _pressure = re.compile(r"P\s*=\s*([-+\d.Ee]+)", re.IGNORECASE)
    _fermi = re.compile(r"the Fermi energy is\s+([-+\d.Ee]+)\s+ev", re.IGNORECASE)
    _magnetization = re.compile(r"total magnetization\s*=\s*([-+\d.Ee]+)", re.IGNORECASE)

    @staticmethod
    def _last_block(lines: list[str], header: str) -> list[str]:
        indices = [index for index, line in enumerate(lines) if line.strip().upper().startswith(header)]
        if not indices:
            return []
        block: list[str] = []
        for line in lines[indices[-1] + 1 :]:
            if not line.strip():
                if block:
                    break
                continue
            block.append(line)
        return block

    def parse(self, output_file: str | Path) -> CalculationResult:
        path = Path(output_file)
        if not path.exists():
            return CalculationResult()
        text = path.read_text(errors="ignore")
        lines = text.splitlines()
        energies = [float(value) * RY_TO_EV for value in self._energy.findall(text)]
        iterations = [int(value) for value in self._iteration.findall(text)]
        timing = self._time.findall(text)
        cell_block = self._last_block(lines, "CELL_PARAMETERS")
        position_block = self._last_block(lines, "ATOMIC_POSITIONS")
        cell = None
        if len(cell_block) >= 3:
            try:
                cell = np.array([[float(value) for value in line.split()[:3]] for line in cell_block[:3]])
            except ValueError:
                cell = None
        symbols: list[str] = []
        positions: list[list[float]] = []
        for line in position_block:
            fields = line.split()
            if len(fields) < 4:
                break
            try:
                positions.append([float(value) for value in fields[1:4]])
                symbols.append(fields[0])
            except ValueError:
                break
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell) if symbols else None
        pressure = self._pressure.findall(text)
        fermi = self._fermi.findall(text)
        magnetization = self._magnetization.findall(text)
        return CalculationResult(
            energy=energies[-1] if energies else None,
            final_scf_energy=energies[-1] if energies else None,
            converged="convergence has been achieved" in text.lower() or "job done" in text.lower(),
            scf_iterations=max(iterations) if iterations else None,
            cpu_time=_duration_seconds(timing[-1][0]) if timing else None,
            wall_time=_duration_seconds(timing[-1][1]) if timing else None,
            cell=cell,
            atoms=atoms,
            pressure=float(pressure[-1]) if pressure else None,
            fermi_energy=float(fermi[-1]) if fermi else None,
            magnetization=float(magnetization[-1]) if magnetization else None,
        )


class CampaignResults:
    """Discover, parse, rank, and export calculations from a campaign root."""

    parsers: dict[str, OutputParser] = {"qe": QuantumEspressoParser()}

    def __init__(self, directory: str | Path, dataframe: pd.DataFrame, structures: dict[str, Atoms]) -> None:
        self.directory = Path(directory)
        self.dataframe = dataframe
        self._structures = structures

    @classmethod
    def from_directory(cls, directory: str | Path, code: str = "qe") -> "CampaignResults":
        """Discover job metadata and parse outputs below a campaign directory."""
        root = Path(directory)
        if code not in cls.parsers:
            raise ValueError(f"Unsupported output code {code!r}. Available: {', '.join(cls.parsers)}")
        parser = cls.parsers[code]
        records: list[dict[str, Any]] = []
        structures: dict[str, Atoms] = {}
        for metadata_file in sorted(root.rglob("metadata.json")):
            metadata = json.loads(metadata_file.read_text())
            job_dir = metadata_file.parent
            result = parser.parse(job_dir / "pw.out")
            unique_id = metadata.get("unique_site_id", metadata.get("UniqueSiteID", job_dir.name))
            record = {
                "Directory": str(job_dir),
                "Site": metadata.get("site"),
                "UniqueSiteID": unique_id,
                "Adsorbate": metadata.get("adsorbate"),
                "Supercell": metadata.get("supercell"),
                "Energy": result.energy,
                "FinalSCFEnergy": result.final_scf_energy,
                "Converged": result.converged,
                "SCFIterations": result.scf_iterations,
                "CPUTime": result.cpu_time,
                "WallTime": result.wall_time,
                "Pressure": result.pressure,
                "FermiEnergy": result.fermi_energy,
                "Magnetization": result.magnetization,
                "Orientation": metadata.get("orientation"),
                "Height": metadata.get("height"),
            }
            records.append(record)
            if result.atoms is not None:
                structures[str(unique_id)] = result.atoms
        columns = ["Directory", "Site", "UniqueSiteID", "Adsorbate", "Supercell", "Energy", "FinalSCFEnergy", "Converged", "SCFIterations", "CPUTime", "WallTime", "Pressure", "FermiEnergy", "Magnetization", "Orientation", "Height"]
        return cls(root, pd.DataFrame(records, columns=columns), structures)

    def compute_adsorption_energy(self, clean_surface_energy: float, gas_phase_energy: float) -> pd.DataFrame:
        """Compute and store adsorption energies in eV from total energies."""
        self.dataframe["AdsorptionEnergy"] = self.dataframe["Energy"] - clean_surface_energy - gas_phase_energy
        return self.dataframe

    def rank_by_adsorption_energy(self) -> pd.DataFrame:
        """Return converged structures ordered from most to least stable."""
        if "AdsorptionEnergy" not in self.dataframe:
            raise ValueError("Compute adsorption energies before ranking")
        return self.dataframe.sort_values("AdsorptionEnergy", na_position="last").reset_index(drop=True)

    def to_csv(self, filename: str | Path | None = None) -> Path:
        """Export the analysis dataframe as CSV."""
        path = Path(filename or self.directory / "results.csv")
        self.dataframe.to_csv(path, index=False)
        return path

    def to_excel(self, filename: str | Path | None = None) -> Path:
        """Export the analysis dataframe as an Excel workbook."""
        path = Path(filename or self.directory / "results.xlsx")
        self.dataframe.to_excel(path, index=False)
        return path

    def to_json(self, filename: str | Path | None = None) -> Path:
        """Export the analysis dataframe as JSON records."""
        path = Path(filename or self.directory / "results.json")
        self.dataframe.to_json(path, orient="records", indent=2)
        return path

    def plot_adsorption_energy(self, filename: str | Path | None = None) -> Path:
        """Create an adsorption-energy-versus-site PNG plot."""
        if "AdsorptionEnergy" not in self.dataframe:
            raise ValueError("Compute adsorption energies before plotting")
        import matplotlib.pyplot as plt

        path = Path(filename or self.directory / "AdsorptionEnergy_vs_Site.png")
        data = self.rank_by_adsorption_energy().dropna(subset=["AdsorptionEnergy"])
        figure, axis = plt.subplots(figsize=(max(6, len(data) * 0.8), 4))
        labels = [f"{site}\n{identifier}" for site, identifier in zip(data["Site"], data["UniqueSiteID"])]
        axis.bar(labels, data["AdsorptionEnergy"])
        axis.set_ylabel("Adsorption energy (eV)")
        axis.set_xlabel("Adsorption site")
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)
        return path

    def export_optimized_structures(self, directory: str | Path | None = None) -> list[Path]:
        """Export parsed final structures as XYZ files for visualization."""
        root = Path(directory or self.directory / "optimized_structures")
        root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for identifier, atoms in self._structures.items():
            site = self.dataframe.loc[self.dataframe["UniqueSiteID"] == identifier, "Site"].iloc[0]
            path = root / f"{site}_{identifier}.xyz"
            write(path, atoms)
            paths.append(path)
        return paths
