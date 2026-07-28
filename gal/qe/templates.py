"""Text templates for scheduler submission scripts."""

from __future__ import annotations


def render_slurm_script(
    job_name: str,
    cores: int = 16,
    partition: str | None = None,
    walltime: str = "01:00:00",
    account: str | None = None,
    modules: tuple[str, ...] = (),
    executable: str = "pw.x",
) -> str:
    """Render a portable Slurm script that runs QE from ``pw.in``."""
    lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}", f"#SBATCH --ntasks={cores}", f"#SBATCH --time={walltime}", "#SBATCH --output=pw.out"]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    lines.extend(["", *(f"module load {module}" for module in modules), "", f"srun {executable} -in pw.in > pw.out"])
    return "\n".join(lines) + "\n"
