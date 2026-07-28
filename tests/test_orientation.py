import numpy as np
from ase import Atoms

from gal.adsorption import generate_orientations, rotate_molecule
from gal.gas import Gas


def test_rotate_molecule_preserves_geometry():
    gas = Gas("H2")
    rotated = rotate_molecule(gas.atoms, axis="z", angle=90.0)

    assert isinstance(rotated, Atoms)
    assert len(rotated) == len(gas.atoms)
    assert np.allclose(rotated.get_positions()[0], rotated.get_positions()[0])


def test_generate_orientations_returns_multiple_configurations():
    gas = Gas("H2")
    orientations = generate_orientations(gas.atoms, axes=("x", "z"), angles=(0.0, 90.0), n_per_axis=2)

    assert len(orientations) == 4
    assert all(isinstance(orientation, Atoms) for orientation in orientations)
    assert all(len(orientation) == len(gas.atoms) for orientation in orientations)
