"""Extensible analysis of completed electronic-structure calculation campaigns."""

from __future__ import annotations

import json
import re
import warnings
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
        text_lower = text.lower()
        job_done = "job done." in text_lower
        is_relax = "begin final coordinates" in text_lower or "bfgs" in text_lower
        if is_relax:
            converged = job_done and "bfgs converged" in text_lower
        else:
            converged = job_done and "convergence has been achieved" in text_lower
        return CalculationResult(
            energy=energies[-1] if energies else None,
            final_scf_energy=energies[-1] if energies else None,
            converged=converged,
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
                structures[str(job_dir)] = result.atoms
        columns = ["Directory", "Site", "UniqueSiteID", "Adsorbate", "Supercell", "Energy", "FinalSCFEnergy", "Converged", "SCFIterations", "CPUTime", "WallTime", "Pressure", "FermiEnergy", "Magnetization", "Orientation", "Height"]
        return cls(root, pd.DataFrame(records, columns=columns), structures)

    @classmethod
    def reference_energy(cls, output_file: str | Path, code: str = "qe") -> float | None:
        """Parse a clean-surface or gas-phase output and return its energy in eV.

        Use this to build the ``clean_surface_energy``/``gas_phase_energy``
        arguments for :meth:`compute_adsorption_energy` from separately
        converged calculations, so all three energies share the same (eV)
        units as the campaign's parsed ``Energy`` column.
        """
        if code not in cls.parsers:
            raise ValueError(f"Unsupported output code {code!r}. Available: {', '.join(cls.parsers)}")
        return cls.parsers[code].parse(output_file).energy

    def compute_adsorption_energy(self, clean_surface_energy: float, gas_phase_energy: float) -> pd.DataFrame:
        """Compute and store adsorption energies in eV from total energies.

        ``clean_surface_energy`` and ``gas_phase_energy`` must already be in
        eV, the same units as the parsed ``Energy`` column (Quantum ESPRESSO
        outputs are converted from Ry to eV during parsing). Use
        :meth:`reference_energy` to parse reference pw.out files so all three
        energies share units.
        """
        energies = self.dataframe["Energy"].dropna()
        if not energies.empty:
            median_energy = energies.abs().median()
            reference_magnitude = max(abs(clean_surface_energy), abs(gas_phase_energy))
            if median_energy > 0 and reference_magnitude > 0 and median_energy / reference_magnitude > RY_TO_EV / 2:
                warnings.warn(
                    "clean_surface_energy/gas_phase_energy are much smaller in magnitude than the "
                    "campaign's parsed energies (eV); they may still be in Ry. Use "
                    "CampaignResults.reference_energy() to parse references in eV.",
                    stacklevel=2,
                )
        self.dataframe["AdsorptionEnergy"] = self.dataframe["Energy"] - clean_surface_energy - gas_phase_energy
        return self.dataframe

    def rank_by_adsorption_energy(self, only_converged: bool = True) -> pd.DataFrame:
        """Return structures ordered from most to least stable adsorption energy.

        By default, rows whose relaxation did not fully converge are excluded
        so a half-optimized energy cannot rank as most stable.
        """
        if "AdsorptionEnergy" not in self.dataframe:
            raise ValueError("Compute adsorption energies before ranking")
        data = self.dataframe[self.dataframe["Converged"]] if only_converged else self.dataframe
        return data.sort_values("AdsorptionEnergy", na_position="last").reset_index(drop=True)

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

    def plot_adsorption_energy(self, filename: str | Path | None = None, only_converged: bool = True) -> Path:
        """Create an adsorption-energy-versus-site PNG plot.

        By default, unconverged rows are excluded (see
        :meth:`rank_by_adsorption_energy`).
        """
        if "AdsorptionEnergy" not in self.dataframe:
            raise ValueError("Compute adsorption energies before plotting")
        import matplotlib.pyplot as plt

        path = Path(filename or self.directory / "AdsorptionEnergy_vs_Site.png")
        data = self.rank_by_adsorption_energy(only_converged=only_converged).dropna(subset=["AdsorptionEnergy"])
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
        for job_dir, atoms in self._structures.items():
            rows = self.dataframe.loc[self.dataframe["Directory"] == job_dir]
            if rows.empty:
                site = Path(job_dir).name
                identifier = site
            else:
                row = rows.iloc[0]
                site = row["Site"]
                identifier = row["UniqueSiteID"]
            path = root / f"{site}_{identifier}_{Path(job_dir).name}.xyz"
            write(path, atoms)
            paths.append(path)
        return paths
