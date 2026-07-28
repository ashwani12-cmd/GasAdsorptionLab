import json

from ase import Atoms

from gal.qe import QEInputBuilder, QEJobWriter
from gal.qe.pseudopotentials import pseudopotential_for


def _atoms() -> Atoms:
    return Atoms("MoS", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], cell=[3.0, 3.0, 15.0], pbc=(True, True, False))


def test_qe_input_builder_populates_pw_x_sections_and_overrides():
    builder = QEInputBuilder(pseudo_dir="./pseudo", ecutwfc=60, ecutrho=480, kpts=(6, 6, 1), pseudopotentials={"S": "custom-S.UPF"})
    text = builder.build(_atoms(), prefix="test").render()

    assert "nat = 2" in text
    assert "ntyp = 2" in text
    assert "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF" in text
    assert "custom-S.UPF" in text
    assert "CELL_PARAMETERS (angstrom)" in text
    assert "K_POINTS (automatic)" in text


def test_pseudopotential_mapping_supports_defaults_and_overrides():
    assert pseudopotential_for("Mo") == "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF"
    assert pseudopotential_for("S", {"S": "S.override.UPF"}) == "S.override.UPF"


def test_job_writer_creates_input_slurm_and_metadata(tmp_path):
    writer = QEJobWriter(
        QEInputBuilder(),
        jobs_dir=tmp_path / "jobs",
        slurm={"cores": 8, "partition": "compute", "walltime": "02:00:00", "account": "project", "modules": ("quantum-espresso",)},
    )
    job_dir = writer.write_job(_atoms(), "Top_NH3", {"surface": "MoS2", "adsorbate": "NH3", "site": "Top", "orientation": "n-down", "height": 2.1})

    assert (job_dir / "pw.in").exists()
    submit = (job_dir / "submit.sh").read_text()
    assert "#SBATCH --ntasks=8" in submit
    assert "#SBATCH --partition=compute" in submit
    assert "module load quantum-espresso" in submit
    metadata = json.loads((job_dir / "metadata.json").read_text())
    assert metadata["surface"] == "MoS2"
    assert metadata["orientation"] == "n-down"
    assert metadata["formula"] == "MoS"
    assert metadata["generated_at"]
