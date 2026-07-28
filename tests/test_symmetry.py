import numpy as np
from ase import Atoms

from gal.sites import SiteFinder
from gal.symmetry import SymmetryReducer


def test_reduce_sites_removes_symmetry_equivalent_sites():
    atoms = Atoms(
        symbols="Si2",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        cell=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )

    finder = SiteFinder(atoms)
    sites = [
        finder._make_site(
            name="Hollow",
            position=np.array([0.5, 0.5, 0.0]),
            neighbors=(0, 1),
            surface_layer=0,
            metadata={"kind": "hollow", "layer": 0},
        ),
        finder._make_site(
            name="Hollow",
            position=np.array([2.5, 0.5, 0.0]),
            neighbors=(0, 1),
            surface_layer=0,
            metadata={"kind": "hollow", "layer": 0},
        ),
    ]

    reducer = SymmetryReducer(atoms)
    reduced = reducer.reduce_sites(sites)

    assert len(reduced) == 1
    assert reduced[0].name == "Hollow"
