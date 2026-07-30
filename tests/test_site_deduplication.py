from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from gal.sites import Site, SiteFinder


@pytest.mark.parametrize("supercell", [(1, 1, 1), (2, 2, 1), (3, 3, 1)])
def test_wse2_supercell_sites_are_periodically_unique(supercell):
    atoms = read(Path(__file__).parents[1] / "examples" / "WSe2.cif").repeat(supercell)
    finder = SiteFinder(atoms)
    sites = finder.find_all()
    unique = finder.find_unique_sites(sites)
    report = finder.site_deduplication_report(sites)

    assert sites
    assert len(unique) <= len(sites)
    assert report.total_sites == len(sites)
    assert report.unique_sites == len(unique)
    assert report.removed_duplicates == len(sites) - len(unique)


def test_deduplication_uses_periodic_fractional_coordinates_and_site_class():
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 10.0], pbc=(True, True, False))
    finder = SiteFinder(atoms)
    sites = [
        Site("Top", np.array([0.001, 0.0, 0.0])),
        Site("Top", np.array([3.001, 0.0, 0.0])),
        Site("Bridge", np.array([0.001, 0.0, 0.0])),
    ]

    unique = finder.find_unique_sites(sites)

    assert [site.name for site in unique] == ["Top", "Bridge"]
