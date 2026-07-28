"""Build temporary QE job folders for NH3 on a primitive WSe2 slab."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from ase import Atoms

from gal import Adsorbate, SiteFinder, Surface, generate_adsorption_structures
from gal.qe import QEInputBuilder, QEJobWriter


def load_wse2() -> Surface:
    return Surface(
        Atoms(
            "WSe2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 0.0, -1.6]],
            cell=[[3.28, 0.0, 0.0], [1.64, 2.8405, 0.0], [0.0, 0.0, 20.0]],
            pbc=(True, True, False),
        )
    )


if __name__ == "__main__":
    surface = load_wse2()
    sites = SiteFinder(surface).find_all()
    structures = generate_adsorption_structures(surface, Adsorbate("NH3"), sites=sites)
    builder = QEInputBuilder(pseudo_dir="./pseudo", ecutwfc=60, ecutrho=480, kpts=(6, 6, 1))

    with TemporaryDirectory(prefix="gal-qe-") as directory:
        writer = QEJobWriter(builder, Path(directory) / "jobs", slurm={"cores": 8, "walltime": "02:00:00"})
        job_directories = [
            writer.write_job(
                structure,
                f"{site.name}_{index}",
                {"surface": "WSe2", "adsorbate": "NH3", "site": str(site.name), "orientation": structure.info["orientation"], "height": structure.info["adsorption_height"]},
            )
            for index, (site, structure) in enumerate(zip((site for site in sites for _ in range(3)), structures))
        ]
        print(f"Detected sites: {len(sites)}")
        print(f"NH3 structures: {len(structures)}")
        print(f"QE job directories: {len(job_directories)}")
