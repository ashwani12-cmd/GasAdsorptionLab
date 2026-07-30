import json

import pytest

from gal.cli import main
from gal.results import CampaignResults, QuantumEspressoParser, RY_TO_EV


QE_OUTPUT = """
iteration # 1
iteration # 4
!    total energy              =   -10.00000000 Ry
convergence has been achieved
the Fermi energy is 5.4321 ev
total magnetization = 1.25 Bohr mag/cell
P=   2.50
PWSCF        : 0h0m2.0s CPU    0h0m3.0s WALL
CELL_PARAMETERS (angstrom)
  3.000000 0.000000 0.000000
  0.000000 3.000000 0.000000
  0.000000 0.000000 12.000000

ATOMIC_POSITIONS (angstrom)
  Si 0.000000 0.000000 0.000000
  C  0.000000 0.000000 2.000000

JOB DONE.
"""


def _job(root, name, site, output=QE_OUTPUT):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({"site": site, "unique_site_id": f"id-{name}", "adsorbate": "CO", "supercell": [3, 3, 1], "orientation": "c-down", "height": 2.1}))
    if output is not None:
        (directory / "pw.out").write_text(output)
    return directory


def test_quantum_espresso_parser_extracts_available_result_fields(tmp_path):
    output = tmp_path / "pw.out"
    output.write_text(QE_OUTPUT)
    result = QuantumEspressoParser().parse(output)

    assert result.converged
    assert result.energy == pytest.approx(-10 * RY_TO_EV)
    assert result.scf_iterations == 4
    assert result.cpu_time == pytest.approx(2.0)
    assert result.wall_time == pytest.approx(3.0)
    assert result.pressure == pytest.approx(2.5)
    assert result.fermi_energy == pytest.approx(5.4321)
    assert result.magnetization == pytest.approx(1.25)
    assert result.atoms is not None and len(result.atoms) == 2


def test_campaign_results_merge_rank_and_export(tmp_path):
    _job(tmp_path, "Top", "Top")
    _job(tmp_path, "Bridge", "Bridge", output="iteration # 1\n! total energy = -9.0 Ry\n")
    results = CampaignResults.from_directory(tmp_path)

    assert list(results.dataframe["Site"]) == ["Bridge", "Top"]
    assert set(["Directory", "Site", "UniqueSiteID", "Adsorbate", "Supercell", "Energy", "Converged", "CPUTime", "WallTime", "Orientation", "Height"]) <= set(results.dataframe.columns)
    assert results.dataframe.loc[results.dataframe["Site"] == "Bridge", "Converged"].item() is False
    results.compute_adsorption_energy(clean_surface_energy=-100.0, gas_phase_energy=-20.0)
    ranked = results.rank_by_adsorption_energy()
    assert ranked.iloc[0]["Site"] == "Top"
    assert results.to_csv().exists()
    assert results.to_excel().exists()
    assert results.to_json().exists()
    assert results.plot_adsorption_energy().exists()
    optimized = results.export_optimized_structures()
    assert len(optimized) == 1
    assert optimized[0].exists()


def test_results_cli_exports_analysis_files(tmp_path, capsys):
    _job(tmp_path, "Top", "Top")

    assert main(["analyze", str(tmp_path), "--clean-surface-energy", "-100", "--gas-phase-energy", "-20"]) == 0
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.xlsx").exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "AdsorptionEnergy_vs_Site.png").exists()
    assert "Analyzed 1 calculations" in capsys.readouterr().out
