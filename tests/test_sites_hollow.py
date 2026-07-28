import numpy as np
from ase import Atoms

from gal.sites import SiteFinder


def test_find_hollow_returns_single_site_for_triangle():
    atoms = Atoms(
        symbols="Si3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
        ],
        cell=[[2.0, 0.0, 0.0], [1.0, np.sqrt(3.0), 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )

    finder = SiteFinder(atoms)
    sites = finder.find_hollow(cutoff=1.8)

    assert len(sites) == 1
    assert sites[0].name == "Hollow"
    assert sites[0].neighbors == (0, 1, 2)
    assert sites[0].surface_layer == 0
    assert sites[0].metadata["kind"] == "hollow"
    np.testing.assert_allclose(sites[0].position[2], 0.0)


def test_find_all_includes_hollow_sites():
    atoms = Atoms(
        symbols="Si3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
        ],
        cell=[[2.0, 0.0, 0.0], [1.0, np.sqrt(3.0), 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )

    finder = SiteFinder(atoms)
    sites = finder.find_all()

    hollow_sites = [site for site in sites if site.name == "Hollow"]
    bridge_sites = [site for site in sites if site.name == "Bridge"]
    top_se_sites = [site for site in sites if site.name == "Top_Se"]

    assert len(hollow_sites) == 1
    assert len(bridge_sites) >= 1
    assert len(top_se_sites) == 0
    assert [site.name for site in sites[:2]] == ["Bridge", "Bridge"]


def test_find_hollow_auto_cutoff_uses_first_neighbor_distance():
    atoms = Atoms(
        symbols="Si3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
        ],
        cell=[[2.0, 0.0, 0.0], [1.0, np.sqrt(3.0), 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )

    finder = SiteFinder(atoms)
    sites = finder.find_hollow(cutoff=None)

    assert len(sites) == 1
    assert sites[0].name == "Hollow"
    assert sites[0].metadata["kind"] == "hollow"


def test_find_hollow_manual_cutoff_override():
    atoms = Atoms(
        symbols="Si3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
        ],
        cell=[[2.0, 0.0, 0.0], [1.0, np.sqrt(3.0), 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )

    finder = SiteFinder(atoms)
    sites = finder.find_hollow(cutoff=0.5)

    assert len(sites) == 1
    assert sites[0].name == "Hollow"
    assert sites[0].metadata["kind"] == "hollow"
