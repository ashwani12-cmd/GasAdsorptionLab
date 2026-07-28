import numpy as np
import pytest

from gal.adsorbate import Adsorbate


@pytest.mark.parametrize("formula", ["H", "CO", "CO2", "NH3", "H2O", "CH4", "SO2", "H2S"])
def test_adsorbate_library_constructs_ase_geometries(formula: str):
    adsorbate = Adsorbate(formula)
    assert adsorbate.formula == formula
    assert len(adsorbate.atoms) > 0
    assert adsorbate.center_of_mass.shape == (3,)


def test_adsorbate_transformations_and_copy_are_independent():
    adsorbate = Adsorbate("CO")
    original = adsorbate.atoms.positions.copy()
    copied = adsorbate.copy().rotate(90.0, "x").translate([0.0, 0.0, 1.0])

    assert np.allclose(adsorbate.atoms.positions, original)
    assert not np.allclose(copied.atoms.positions, original)


def test_unsupported_adsorbate_has_actionable_error():
    with pytest.raises(ValueError, match="Unsupported adsorbate"):
        Adsorbate("XeF9")
