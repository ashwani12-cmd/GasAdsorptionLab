"""High-level orchestration for complete adsorption calculation campaigns."""

from __future__ import annotations

import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ase import Atoms
from ase.io import read, write

from .adsorbate import Adsorbate
from .placement import generate_orientations, place_adsorbate
from .qe import QEInputBuilder, QEJobWriter
from .sites import SiteFinder
from .surface import Surface


def _campaign_surface(source: str | Path | Surface | Atoms) -> tuple[Surface, str]:
    """Load a campaign surface without mutating caller-owned atoms."""
    if isinstance(source, Surface):
        return Surface(source.atoms.copy()), source.atoms.get_chemical_formula()
    if isinstance(source, Atoms):
        return Surface(source.copy()), source.get_chemical_formula()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"ase\.io\.cif")
        atoms = read(source)
    return Surface(atoms), Path(source).stem


def generate_campaign(
    surface: str | Path | Surface | Atoms,
    adsorbates: str | list[str],
    supercell: tuple[int, int, int] = (3, 3, 1),
    qe: bool = True,
    *,
    output_dir: str | Path = "campaigns",
    qe_builder: QEInputBuilder | None = None,
    slurm: dict[str, Any] | None = None,
    height: float | str = "auto",
    orientation: str = "auto",
    site_id_prefix: str | None = None,
) -> list[Path]:
    """Generate adsorption structures and optional QE jobs for any surface.

    ``surface`` may be an ASE :class:`~ase.Atoms`, a :class:`Surface`, or a
    structure filename.  A single adsorbate string and a list are both
    accepted.  Site detection, periodic deduplication, metadata, and CSV
    summaries are shared with the legacy WSe2 convenience workflow.
    """
    if len(supercell) != 3 or any(not isinstance(value, int) or value < 1 for value in supercell):
        raise ValueError("supercell must contain three positive integers")
    formulas = [adsorbates] if isinstance(adsorbates, str) else list(adsorbates)
    if not formulas:
        raise ValueError("adsorbates must contain at least one formula")
    campaign_surface, surface_name = _campaign_surface(surface)
    campaign_surface.atoms = campaign_surface.atoms.repeat(supercell)
    finder = SiteFinder(campaign_surface)
    all_sites = finder.find_all()
    sites = finder.find_unique_sites(all_sites)
    report = finder.site_deduplication_report(all_sites)
    print(f"Total sites: {report.total_sites}")
    print(f"Unique sites: {report.unique_sites}")
    print(f"Removed duplicates: {report.removed_duplicates}")
    root = Path(output_dir)
    builder = qe_builder or QEInputBuilder()
    rows: list[dict[str, str]] = []
    directories: list[Path] = []
    for formula in formulas:
        adsorbate_root = root if len(formulas) == 1 else root / formula
        writer = QEJobWriter(builder, adsorbate_root, slurm=slurm) if qe else None
        for index, site in enumerate(sites):
            structure = place_adsorbate(campaign_surface, site, Adsorbate(formula), height=height, orientation=orientation)
            name = f"{index:02d}_{str(site.name).replace(' ', '_')}"
            prefix = site_id_prefix or (surface_name if len(formulas) == 1 else f"{surface_name}-{formula}")
            unique_site_id = f"{prefix}-{index:03d}"
            metadata = {
                "unique_site_id": unique_site_id,
                "surface": surface_name,
                "site": str(site.name),
                "adsorption_position": site.position.tolist(),
                "adsorbate": formula,
                "orientation": structure.info["orientation"],
                "height": structure.info["adsorption_height"],
                "supercell": list(supercell),
                "creation_time": datetime.now(timezone.utc).isoformat(),
            }
            if writer is not None:
                directory = writer.write_job(structure, name, metadata)
                builder.build(structure, prefix=directory.name, calculation="relax").write(directory / "pw.in")
            else:
                directory = adsorbate_root / name
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            write(directory / "structure.xyz", structure, format="extxyz")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module=r"ase\.io\.cif")
                write(directory / "structure.cif", structure)
            directories.append(directory)
            rows.append({"Directory": str(directory), "UniqueSiteID": unique_site_id, "Site": str(site.name), "Adsorbate": formula, "Supercell": "x".join(map(str, supercell)), "Structure": "structure.xyz", "Input": "pw.in" if qe else ""})

    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", newline="") as handle:
        summary = csv.DictWriter(handle, fieldnames=["Directory", "UniqueSiteID", "Site", "Adsorbate", "Supercell", "Structure", "Input"])
        summary.writeheader()
        summary.writerows(rows)
    return directories


def generate_wse2_co_campaign(
    cif_path: str | Path = "examples/WSe2.cif",
    output_dir: str | Path = "campaigns/WSe2_CO",
    qe_builder: QEInputBuilder | None = None,
    slurm: dict[str, Any] | None = None,
    height: float | str = "auto",
    orientation: str = "c-down",
    supercell: tuple[int, int, int] = (3, 3, 1),
) -> list[Path]:
    """Backward-compatible WSe2 + CO wrapper around :func:`generate_campaign`."""
    return generate_campaign(
        surface=cif_path,
        adsorbates="CO",
        supercell=supercell,
        output_dir=output_dir,
        qe_builder=qe_builder,
        slurm=slurm,
        height=height,
        orientation=orientation,
        site_id_prefix="WSe2",
    )


class AdsorptionCampaign:
    """Generate placements and QE jobs for one or more surfaces and adsorbates."""

    def __init__(
        self,
        surface: str | Path | Surface | Atoms | Iterable[str | Path | Surface | Atoms],
        adsorbates: list[str],
        output_dir: str | Path = "campaign",
        heights: float | str | Iterable[float | str] = "auto",
        orientations: str | tuple[str, ...] = "auto",
        qe_builder: QEInputBuilder | None = None,
        slurm: dict[str, Any] | None = None,
    ) -> None:
        self.surfaces = list(surface) if isinstance(surface, (list, tuple)) else [surface]
        self.adsorbates = list(adsorbates)
        self.output_dir = Path(output_dir)
        self.heights = list(heights) if isinstance(heights, (list, tuple)) else [heights]
        self.orientations = orientations
        self.qe_builder = qe_builder or QEInputBuilder()
        self.slurm = slurm or {}
        self.jobs: list[Path] = []

    @staticmethod
    def _surface(source: str | Path | Surface | Atoms) -> Surface:
        if isinstance(source, Surface):
            return source
        if isinstance(source, Atoms):
            return Surface(source)
        return Surface(str(source))

    @staticmethod
    def _surface_name(source: str | Path | Surface | Atoms, surface: Surface) -> str:
        if isinstance(source, (str, Path)):
            return Path(source).stem
        return surface.atoms.get_chemical_formula()

    @staticmethod
    def _safe_name(value: object) -> str:
        return str(value).replace(" ", "_").replace("/", "-")

    @staticmethod
    def job_status(job_dir: str | Path) -> str:
        """Infer QE calculation status from its output file, if present."""
        output = Path(job_dir) / "pw.out"
        if not output.exists():
            return "Not Started"
        text = output.read_text(errors="ignore").lower()
        if "job done" in text:
            return "Completed"
        if any(marker in text for marker in ("error in routine", "error", "convergence not achieved")):
            return "Failed"
        return "Running"

    def _job_name(self, site_name: str, orientation: str, height: float | str, index: int) -> str:
        """Use the simple site name first, with deterministic suffixes as needed."""
        if index == 0:
            return self._safe_name(site_name)
        height_name = "auto" if height == "auto" else f"h{float(height):.2f}".replace(".", "p")
        return f"{self._safe_name(site_name)}_{self._safe_name(orientation)}_{height_name}"

    def generate(self, overwrite: bool = False) -> list[Path]:
        """Generate all placement structures and ready-to-run QE job folders."""
        records: list[dict[str, Any]] = []
        created: list[Path] = []
        work: list[tuple[object, Surface, str, str, object, str, object, int]] = []
        surface_names: dict[str, int] = {}
        for source in self.surfaces:
            surface = self._surface(source)
            base_name = self._surface_name(source, surface)
            surface_names[base_name] = surface_names.get(base_name, 0) + 1
            surface_name = base_name if surface_names[base_name] == 1 else f"{base_name}_{surface_names[base_name]}"
            sites = SiteFinder(surface).find_all()
            for formula in self.adsorbates:
                orientation_names = list(generate_orientations(Adsorbate(formula), self.orientations))
                index = 0
                for site in sites:
                    for orientation in orientation_names:
                        for height in self.heights:
                            work.append((source, surface, surface_name, formula, site, orientation, height, index))
                            index += 1

        total = len(work)
        for number, (_, surface, surface_name, formula, site, orientation, height, index) in enumerate(work, start=1):
            adsorbate_dir = self.output_dir / self._safe_name(surface_name) / formula
            name = self._job_name(str(site.name), orientation, height, index)
            job_dir = adsorbate_dir / name
            if job_dir.exists() and not overwrite:
                status = self.job_status(job_dir)
            else:
                if job_dir.exists():
                    import shutil

                    shutil.rmtree(job_dir)
                structure = place_adsorbate(surface, site, Adsorbate(formula), height=height, orientation=orientation)
                writer = QEJobWriter(self.qe_builder, adsorbate_dir, slurm=self.slurm)
                job_dir = writer.write_job(
                    structure,
                    name,
                    {"surface": surface_name, "adsorbate": formula, "site": str(site.name), "orientation": orientation, "height": structure.info["adsorption_height"]},
                )
                created.append(job_dir)
                status = "Not Started"
            records.append({"Surface": surface_name, "Adsorbate": formula, "Site": str(site.name), "Orientation": orientation, "Height": height, "Directory": str(job_dir), "Status": status})
            print(f"Generated {number} / {total} jobs")

        self.jobs = [Path(record["Directory"]) for record in records]
        self._write_summary(records)
        return self.jobs

    def _write_summary(self, records: list[dict[str, Any]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        site_counts: dict[str, int] = {}
        for record in records:
            site_counts[record["Site"]] = site_counts.get(record["Site"], 0) + 1
        summary = {
            "surface": [record["Surface"] for record in records if record][0] if records else None,
            "surfaces": sorted({record["Surface"] for record in records}),
            "adsorbates": self.adsorbates,
            "number_of_jobs": len(records),
            "site_counts": site_counts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "qe_settings": {"pseudo_dir": self.qe_builder.pseudo_dir, "ecutwfc": self.qe_builder.ecutwfc, "ecutrho": self.qe_builder.ecutrho, "kpts": self.qe_builder.kpts, "xc": self.qe_builder.xc},
        }
        (self.output_dir / "campaign.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with (self.output_dir / "campaign.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Surface", "Adsorbate", "Site", "Orientation", "Height", "Directory", "Status"])
            writer.writeheader()
            writer.writerows(records)
