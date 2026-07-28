import numpy as np
from ase import Atoms

from gal.adsorbate import Adsorbate
from gal.placement import (
    estimate_adsorption_height,
    generate_adsorption_structures,
    generate_orientations,
    place_adsorbate,
)
from gal.sites import Site
from gal.surface import Surface


def _surface() -> Surface:
    return Surface(Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 12.0], pbc=(True, True, False)))


def _site() -> Site:
    return Site(name="Top", position=np.array([0.0, 0.0, 0.0]), neighbors=(0,), metadata={"kind": "top"})


def test_place_adsorbate_returns_combined_structure_at_explicit_height():
    surface = _surface()
    structure = place_adsorbate(surface, _site(), Adsorbate("CO"), height=2.0, orientation="c-down")

    assert len(structure) == len(surface.atoms) + 2
    assert np.min(structure.positions[-2:, 2]) == 2.0
    assert structure.info["orientation"] == "c-down"


def test_auto_height_uses_positive_atomic_radii_clearance():
    height = estimate_adsorption_height(_surface(), _site(), "CO")
    structure = place_adsorbate(_surface(), _site(), "CO", height="auto", orientation="o-down")

    assert height > 1.0
    assert np.isclose(np.min(structure.positions[-2:, 2]), height)


def test_default_orientations_cover_requested_molecules():
    assert set(generate_orientations("CO")) == {"c-down", "o-down", "parallel"}
    assert set(generate_orientations("NH3")) == {"n-down", "h-down", "tilted"}
    assert set(generate_orientations("H2O")) == {"o-down", "flat", "tilted"}
    assert set(generate_orientations("H2")) == {"perpendicular", "parallel"}


def test_batch_generation_combines_sites_and_default_orientations():
    structures = generate_adsorption_structures(_surface(), "NH3", sites=[_site()], height=2.0)

    assert len(structures) == 3
    assert all(len(structure) == 5 for structure in structures)
    assert {structure.info["orientation"] for structure in structures} == {"n-down", "h-down", "tilted"}
