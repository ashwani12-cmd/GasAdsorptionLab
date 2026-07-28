from pathlib import Path

from ase import Atoms

from gal.adsorption import place_adsorbate
from gal.config import Config
from gal.gas import Gas
from gal.qe import QEInput
from gal.qe import QEInputBuilder
from gal.sites import Site, SiteFinder
from gal.surface import Surface
from gal.workflow import AdsorptionWorkflow


def test_adsorption_workflow_creates_summary_and_outputs(tmp_path):
    atoms = Atoms(
        symbols="Si2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )
    config = Config.from_dict({
        "qe": {"prefix": "test", "ecutwfc": 80, "ecutrho": 640, "occupations": "smearing", "conv_thr": 1.0e-8},
        "kpoints": {"scf": [1, 1, 1]},
    })

    workflow = AdsorptionWorkflow(config=config, output_dir=tmp_path)
    workflow.load_structure(atoms)
    workflow.detect_sites()
    workflow.generate_adsorption_configurations()
    workflow.write_qe_inputs()

    summary = workflow.run()

    assert summary["status"] == "completed"
    assert summary["sites_detected"] == 1
    assert summary["configurations_generated"] == 1
    assert summary["qe_inputs_written"] == 1
    assert (tmp_path / "qe").exists()
    assert (tmp_path / "qe" / "test.in").exists()


def test_adsorption_workflow_builds_qe_job_directories(tmp_path):
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 12.0], pbc=(True, True, False))
    workflow = AdsorptionWorkflow(config=Config.from_dict({"qe": {"prefix": "test", "ecutwfc": 60, "ecutrho": 480, "occupations": "smearing", "conv_thr": 1e-8}}), output_dir=tmp_path)
    workflow.load_structure(atoms)
    workflow.sites = [Site(name="Top", position=atoms.positions[0], neighbors=(0,))]
    workflow.generate_adsorption_configurations()

    directories = workflow.build_qe_jobs(QEInputBuilder(), slurm={"cores": 2})

    assert len(directories) == 1
    assert (directories[0] / "pw.in").exists()
    assert (directories[0] / "submit.sh").exists()
    assert (directories[0] / "metadata.json").exists()
