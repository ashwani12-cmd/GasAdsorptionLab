import csv
import json

from ase import Atoms
from ase.io import write
import yaml

from gal.campaign import AdsorptionCampaign
from gal.cli import main
from gal.qe import QEInputBuilder


def _surface() -> Atoms:
    return Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 12.0], pbc=(True, True, False))


def test_campaign_generates_qe_layout_and_summary_files(tmp_path, capsys):
    campaign = AdsorptionCampaign(_surface(), ["CO"], output_dir=tmp_path / "campaign", qe_builder=QEInputBuilder(kpts=(1, 1, 1)))
    jobs = campaign.generate()

    assert jobs
    assert all((job / "pw.in").exists() and (job / "submit.sh").exists() and (job / "metadata.json").exists() for job in jobs)
    summary = json.loads((tmp_path / "campaign" / "campaign.json").read_text())
    assert summary["adsorbates"] == ["CO"]
    assert summary["number_of_jobs"] == len(jobs)
    with (tmp_path / "campaign" / "campaign.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(jobs)
    assert set(rows[0]) == {"Surface", "Adsorbate", "Site", "Orientation", "Height", "Directory", "Status"}
    assert "Generated 1 /" in capsys.readouterr().out


def test_campaign_restart_skips_existing_jobs_and_status_tracking(tmp_path):
    campaign = AdsorptionCampaign(_surface(), ["H2"], output_dir=tmp_path / "campaign", orientations=("perpendicular",), heights=2.0)
    first = campaign.generate()
    (first[0] / "pw.out").write_text("JOB DONE.")

    second = campaign.generate()

    assert first == second
    assert campaign.job_status(first[0]) == "Completed"
    assert campaign.job_status(first[0].parent / "missing") == "Not Started"
    (first[-1] / "pw.out").write_text("Error in routine electrons")
    assert campaign.job_status(first[-1]) == "Failed"
    (first[-1] / "pw.out").write_text("iteration # 3")
    assert campaign.job_status(first[-1]) == "Running"


def test_campaign_supports_multiple_surfaces_adsorbates_and_heights(tmp_path):
    surface_two = _surface().copy()
    surface_two.positions[0, 0] = 0.2
    campaign = AdsorptionCampaign([_surface(), surface_two], ["H2", "CO"], output_dir=tmp_path / "campaign", orientations=("parallel",), heights=(2.0, 2.5))

    jobs = campaign.generate()

    assert jobs
    summary = json.loads((tmp_path / "campaign" / "campaign.json").read_text())
    assert summary["number_of_jobs"] == len(jobs)
    assert (tmp_path / "campaign" / "Si").exists()
    assert (tmp_path / "campaign" / "Si_2").exists()


def test_campaign_cli_reads_yaml_configuration(tmp_path):
    structure_path = tmp_path / "surface.xyz"
    write(structure_path, _surface())
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump({"campaign": {"surface": str(structure_path), "adsorbates": ["H2"], "output_dir": str(tmp_path / "campaign"), "orientations": ["perpendicular"], "heights": [2.0]}}))

    assert main(["campaign", str(config_path)]) == 0
    assert (tmp_path / "campaign" / "campaign.json").exists()
