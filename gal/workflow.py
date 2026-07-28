"""Modular workflow orchestration for adsorption studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms

from .adsorption import place_adsorbate
from .config import Config
from .gas import Gas
from .qe import QEInput, QEInputBuilder, QEJobWriter
from .sites import Site, SiteFinder
from .surface import Surface


class AdsorptionWorkflow:
    """Coordinate loading, site detection, configuration generation, and QE output."""

    def __init__(self, config: Config | dict[str, Any] | None = None, output_dir: str | Path | None = None) -> None:
        self.config = config if config is not None else Config.from_dict({})
        self.output_dir = Path(output_dir or "output")
        self.surface: Surface | None = None
        self.sites: list[Site] = []
        self.configurations: list[tuple[Site, Atoms]] = []
        self.logs: list[str] = []
        self.summary: dict[str, Any] = {}

    def log(self, message: str) -> None:
        """Append a human-readable step to the workflow log."""
        self.logs.append(message)

    def load_structure(self, atoms: Atoms) -> Surface:
        """Load a structure into the workflow and create a surface object."""
        self.surface = Surface(atoms)
        self.log("Loaded structure")
        return self.surface

    def detect_sites(self) -> list[Site]:
        """Detect adsorption sites from the current structure."""
        if self.surface is None:
            raise ValueError("No structure loaded")

        finder = SiteFinder(self.surface.atoms)
        self.sites = finder.find_top_w()[:1]

        if not self.sites:
            self.sites = [
                Site(
                    name="Top",
                    position=self.surface.atoms.positions[0].copy(),
                    neighbors=(0,),
                    metadata={"kind": "top"},
                )
            ]

        self.log(f"Detected {len(self.sites)} adsorption sites")
        return self.sites

    def generate_adsorption_configurations(self) -> list[tuple[Site, Atoms]]:
        """Generate one adsorbate configuration per detected site."""
        if self.surface is None:
            raise ValueError("No structure loaded")
        if not self.sites:
            self.detect_sites()

        gas = Gas("H2")
        self.configurations = []
        for site in self.sites:
            structure = place_adsorbate(self.surface, gas, site, adsorption_height=2.0, rotation=0.0)
            self.configurations.append((site, structure))
            self.log(f"Generated configuration for {site.name}")

        return self.configurations

    def prepare_output_directories(self) -> dict[str, Path]:
        """Create the directory structure used by the workflow."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        directories = {
            "root": self.output_dir,
            "qe": self.output_dir / "qe",
            "structures": self.output_dir / "structures",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.log("Prepared output directories")
        return directories

    def write_qe_inputs(self) -> list[Path]:
        """Write Quantum ESPRESSO input files for each generated configuration."""
        if not self.configurations:
            raise ValueError("No configurations generated")

        directories = self.prepare_output_directories()
        qe_dir = directories["qe"]

        written_files: list[Path] = []
        for index, (_, structure) in enumerate(self.configurations):
            config = self.config if isinstance(self.config, Config) else Config.from_dict(self.config)
            if isinstance(config, Config):
                qe_data = config.data.get("qe", {})
            else:
                qe_data = config.get("qe", {})
            prefix = qe_data.get("prefix", "gas")
            qe_input = QEInput.from_config(config, atoms=structure, prefix=prefix if len(self.configurations) == 1 else f"ads_{index}")
            output_path = qe_dir / f"{qe_input._data()['qe']['prefix']}.in"
            qe_input.write(output_path)
            written_files.append(output_path)
            self.log(f"Wrote QE input: {output_path}")

        return written_files

    def build_qe_jobs(
        self,
        builder: QEInputBuilder | None = None,
        jobs_dir: str | Path | None = None,
        slurm: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Create ready-to-submit QE job directories for generated structures."""
        if not self.configurations:
            raise ValueError("No configurations generated")
        config_data = self.config.data if isinstance(self.config, Config) else self.config
        qe_data = config_data.get("qe", {})
        builder = builder or QEInputBuilder(
            pseudo_dir=qe_data.get("pseudo_dir", "./pseudo"),
            ecutwfc=qe_data.get("ecutwfc", 60),
            ecutrho=qe_data.get("ecutrho", 480),
            kpts=tuple(config_data.get("kpoints", {}).get("scf", [6, 6, 1])),
        )
        writer = QEJobWriter(builder, jobs_dir or self.output_dir / "jobs", slurm=slurm)
        surface_formula = self.surface.atoms.get_chemical_formula() if self.surface else None
        directories: list[Path] = []
        for index, (site, structure) in enumerate(self.configurations):
            adsorbate = structure.info.get("adsorbate", "adsorption")
            name = f"{str(site.name).replace(' ', '_')}_{adsorbate}"
            metadata = {
                "surface": surface_formula,
                "adsorbate": adsorbate,
                "site": str(site.name),
                "orientation": structure.info.get("orientation"),
                "height": structure.info.get("adsorption_height"),
            }
            directories.append(writer.write_job(structure, name if index == 0 else f"{name}_{index}", metadata))
        self.log(f"Generated {len(directories)} QE job directories")
        return directories

    def _build_summary(self) -> dict[str, Any]:
        """Assemble a compact workflow summary."""
        return {
            "status": "completed",
            "sites_detected": len(self.sites),
            "configurations_generated": len(self.configurations),
            "qe_inputs_written": len(self._qe_output_files()),
            "output_dir": str(self.output_dir),
            "logs": list(self.logs),
        }

    def _qe_output_files(self) -> list[Path]:
        qe_dir = self.output_dir / "qe"
        return sorted(qe_dir.glob("*.in")) if qe_dir.exists() else []

    def run(self) -> dict[str, Any]:
        """Run the full workflow and return a summary dictionary."""
        if self.surface is None:
            raise ValueError("No structure loaded")

        self.detect_sites()
        self.generate_adsorption_configurations()
        self.write_qe_inputs()
        self.summary = self._build_summary()
        return self.summary
