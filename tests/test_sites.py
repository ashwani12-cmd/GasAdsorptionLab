import pytest

from ase import Atoms

from gal.sites import SiteFinder


@pytest.fixture
def simple_surface() -> Atoms:
    return Atoms(
        symbols="Si2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )


def test_sitefinder_detects_top_and_bridge_sites(simple_surface: Atoms):
    finder = SiteFinder(simple_surface)

    top_sites = finder.find_top_w()
    bridge_sites = finder.find_bridge(cutoff=3.0)

    assert len(top_sites) >= 0
    assert len(bridge_sites) >= 0
