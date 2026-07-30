from pathlib import Path

from ase import Atoms
from ase.io import write

from gal import Surface, generate_campaign
from gal.qe import QEInputBuilder


def _surface() -> Atoms:
    return Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 12.0], pbc=(True, True, False))


def test_generate_campaign_accepts_atoms_and_single_adsorbate_without_qe(tmp_path):
    jobs = generate_campaign(_surface(), "H2", supercell=(1, 1, 1), qe=False, output_dir=tmp_path / "atoms")

    assert jobs
    assert (jobs[0] / "structure.xyz").exists()
    assert (jobs[0] / "metadata.json").exists()
    assert not (jobs[0] / "pw.in").exists()


def test_generate_campaign_accepts_surface_and_multiple_adsorbates(tmp_path):
    jobs = generate_campaign(Surface(_surface()), ["H2", "CO"], supercell=(1, 1, 1), output_dir=tmp_path / "surface", qe_builder=QEInputBuilder(kpts=(1, 1, 1)))

    assert jobs
    assert {job.parent.name for job in jobs} == {"H2", "CO"}
    assert all((job / "pw.in").exists() for job in jobs)


def test_generate_campaign_accepts_cif_filename(tmp_path):
    filename = tmp_path / "surface.cif"
    write(filename, _surface())

    jobs = generate_campaign(filename, ["CO"], supercell=(1, 1, 1), output_dir=tmp_path / "file", qe_builder=QEInputBuilder(kpts=(1, 1, 1)))

    assert jobs
    assert (tmp_path / "file" / "summary.csv").exists()
