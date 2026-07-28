"""Geometry benchmarks for the public universal site-finder API."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import graphene

from gal.sites import SiteFinder, SurfaceType


def _dichalcogenide(formula: str) -> Atoms:
    return Atoms(
        formula,
        positions=[[0.0, 0.0, 1.6], [0.0, 0.0, 0.0], [0.0, 0.0, -1.6]],
        cell=[[3.28, 0.0, 0.0], [1.64, 2.8405, 0.0], [0.0, 0.0, 20.0]],
        pbc=(True, True, False),
    )


def _fcc111(symbol: str) -> Atoms:
    return Atoms(
        f"{symbol}3",
        positions=[[0.0, 0.0, 2.0], [1.5, np.sqrt(3.0) / 2.0, 0.0], [1.0, np.sqrt(3.0) / 2.0, -2.0]],
        cell=[[3.0, 0.0, 0.0], [1.5, 3.0 * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 15.0]],
        pbc=(True, True, False),
    )


@pytest.mark.parametrize("formula", ["C2", "BN"])
def test_graphene_family_benchmark(formula: str):
    sites = SiteFinder(graphene(formula=formula, vacuum=8.0)).find_all()
    assert [site.name for site in sites].count("Top") == 2
    assert [site.name for site in sites].count("Bridge") == 3
    assert [site.name for site in sites].count("Hollow") == 1


@pytest.mark.parametrize("formula", ["MoS2", "WS2", "WSe2"])
def test_dichalcogenide_benchmark(formula: str):
    finder = SiteFinder(_dichalcogenide(formula))
    sites = finder.find_all()
    names = [site.name for site in sites]
    assert finder.detect_surface_type() is SurfaceType.HEXAGONAL_2D
    assert {"Top", "Bottom", "Bridge", "Bottom Bridge", "Hollow", "Bottom Hollow"} <= set(names)
    assert len({tuple(np.round(site.position, 8)) + (str(site.name),) for site in sites}) == len(sites)


@pytest.mark.parametrize("symbol", ["Cu", "Pt"])
def test_fcc111_benchmark(symbol: str):
    finder = SiteFinder(_fcc111(symbol))
    sites = finder.find_all()
    assert finder.detect_surface_type() is SurfaceType.FCC111
    assert [site.name for site in sites].count("Top") == 1
    assert [site.name for site in sites].count("Bridge") == 3
    assert {"FCC", "HCP"} <= {site.name for site in sites}


@pytest.mark.parametrize("builder", [lambda: graphene(vacuum=8.0), lambda: _dichalcogenide("MoS2"), lambda: _fcc111("Cu")])
def test_primitive_and_supercell_preserve_surface_family_and_unique_candidates(builder):
    primitive = builder()
    supercell = primitive.repeat((2, 2, 1))
    primitive_finder = SiteFinder(primitive)
    supercell_finder = SiteFinder(supercell)

    assert primitive_finder.detect_surface_type() is supercell_finder.detect_surface_type()
    for finder in (primitive_finder, supercell_finder):
        sites = finder.find_all()
        assert sites
        assert len({tuple(np.round(site.position, 8)) + (str(site.name),) for site in sites}) == len(sites)
