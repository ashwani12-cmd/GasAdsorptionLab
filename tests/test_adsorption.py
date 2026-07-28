import numpy as np
from ase import Atoms

from gal.adsorption import place_adsorbate
from gal.gas import Gas
from gal.sites import Site
from gal.surface import Surface


def test_place_adsorbate_returns_new_atoms_object():
    surface_atoms = Atoms(
        symbols="Si2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )
    surface = Surface(surface_atoms)

    site = Site(name="Top", position=np.array([0.0, 0.0, 0.0]), neighbors=(0,), metadata={"kind": "top"})
    gas = Gas("H2")

    adsorbate = place_adsorbate(surface, gas, site, adsorption_height=2.0, rotation=45.0)

    assert isinstance(adsorbate, Atoms)
    assert len(adsorbate) == len(surface.atoms) + len(gas.atoms)
    adsorbate_positions = adsorbate.get_positions()
    assert adsorbate_positions[-1][2] >= site.position[2] + 1.0
    assert adsorbate.get_chemical_symbols()[-1] == "H"
    assert adsorbate.info["adsorption_height"] == 2.0
    assert adsorbate.info["orientation"] == "custom"
