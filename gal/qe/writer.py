"""Job-folder generation for Quantum ESPRESSO calculations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ase import Atoms

from .builder import QEInputBuilder
from .templates import render_slurm_script


class QEJobWriter:
    """Write isolated QE job folders containing input, metadata, and Slurm."""

    def __init__(self, builder: QEInputBuilder, jobs_dir: str | Path = "jobs", slurm: dict[str, Any] | None = None) -> None:
        self.builder = builder
        self.jobs_dir = Path(jobs_dir)
        self.slurm = slurm or {}

    def write_job(self, atoms: Atoms, name: str, metadata: dict[str, Any] | None = None) -> Path:
        """Create ``name`` folder and write ``pw.in``, ``submit.sh``, metadata."""
        job_dir = self.jobs_dir / name
        suffix = 2
        while job_dir.exists():
            job_dir = self.jobs_dir / f"{name}_{suffix}"
            suffix += 1
        job_dir.mkdir(parents=True)
        prefix = job_dir.name
        self.builder.build(atoms, prefix=prefix).write(job_dir / "pw.in")
        (job_dir / "submit.sh").write_text(render_slurm_script(job_name=prefix, **self.slurm))
        data = {
            "surface": metadata.get("surface") if metadata else None,
            "adsorbate": metadata.get("adsorbate") if metadata else None,
            "site": metadata.get("site") if metadata else None,
            "orientation": metadata.get("orientation") if metadata else None,
            "height": metadata.get("height") if metadata else None,
            "formula": atoms.get_chemical_formula(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            data.update(metadata)
        (job_dir / "metadata.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return job_dir
