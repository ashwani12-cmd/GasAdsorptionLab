import csv
import json
from pathlib import Path

import numpy as np
from ase.io import read

from gal.campaign import generate_wse2_co_campaign
from gal.qe import QEInputBuilder


def test_wse2_co_campaign_writes_six_complete_relax_jobs(tmp_path):
    cif_path = Path(__file__).parents[1] / "examples" / "WSe2.cif"
    output_dir = tmp_path / "campaigns" / "WSe2_CO"

    directories = generate_wse2_co_campaign(cif_path, output_dir, qe_builder=QEInputBuilder(kpts=(1, 1, 1)), supercell=(1, 1, 1))

    assert [directory.name for directory in directories] == ["00_Top", "01_Bridge", "02_Bridge", "03_Bridge", "04_FCC", "05_HCP"]
    assert len(directories) == 6
    for directory in directories:
        assert (directory / "structure.xyz").exists()
        assert (directory / "structure.cif").exists()
        assert (directory / "pw.in").exists()
        assert (directory / "submit.sh").exists()
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["adsorbate"] == "CO"
        assert metadata["orientation"] == "c-down"
        assert metadata["adsorption_position"]
        assert metadata["creation_time"]
        assert metadata["supercell"] == [1, 1, 1]
        assert "calculation = 'relax'" in (directory / "pw.in").read_text()

    with (output_dir / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert rows[0].keys() == {"Directory", "UniqueSiteID", "Site", "Adsorbate", "Supercell", "Structure", "Input"}
    assert rows[0]["Supercell"] == "1x1x1"
    assert rows[0]["UniqueSiteID"] == "WSe2-000"


def test_wse2_co_campaign_scales_surface_before_site_and_qe_generation(tmp_path):
    cif_path = Path(__file__).parents[1] / "examples" / "WSe2.cif"
    original = read(cif_path)
    directories = generate_wse2_co_campaign(cif_path, tmp_path / "campaign", qe_builder=QEInputBuilder(kpts=(1, 1, 1)), supercell=(3, 3, 1))

    assert directories
    structure = read(directories[0] / "structure.xyz")
    assert len(structure) == len(original) * 9 + 2
    assert (directories[0] / "pw.in").exists()
    np.testing.assert_allclose(structure.cell.array[0], original.cell.array[0] * 3)
    np.testing.assert_allclose(structure.cell.array[1], original.cell.array[1] * 3)
    np.testing.assert_allclose(structure.cell.array[2], original.cell.array[2])
