import numpy as np
from ase import Atoms

from gal.sites import SiteFinder, SurfaceType


def test_hexagonal_engine_adds_bottom_face_candidates_for_layered_sheet():
    atoms = Atoms(
        "SeWSe",
        positions=[[0.0, 0.0, 1.6], [0.0, 0.0, 0.0], [0.0, 0.0, -1.6]],
        cell=[[3.28, 0.0, 0.0], [1.64, 2.8405, 0.0], [0.0, 0.0, 20.0]],
        pbc=(True, True, False),
    )
    sites = SiteFinder(atoms).find_all()
    names = [site.name for site in sites]

    assert "Top" in names
    assert "Bottom" in names
    assert "Bottom Bridge" in names
    assert "Bottom Hollow" in names


def test_fcc111_engine_labels_both_stacking_hollows_from_geometry():
    atoms = Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 2.0], [1.5, np.sqrt(3.0) / 2.0, 0.0], [1.0, np.sqrt(3.0) / 2.0, -2.0]],
        cell=[[3.0, 0.0, 0.0], [1.5, 3.0 * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 15.0]],
        pbc=(True, True, False),
    )
    finder = SiteFinder(atoms)

    assert finder.detect_surface_type() is SurfaceType.FCC111
    assert {site.name for site in finder.find_hollow()} == {"FCC", "HCP"}


def test_square_and_bcc_engines_expose_family_specific_sites():
    rocksalt = Atoms(
        "MgONaCl",
        positions=[[0.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 0.0, -1.0], [2.0, 2.0, -1.0]],
        cell=[4.0, 4.0, 12.0],
        pbc=(True, True, False),
    )
    bcc = Atoms("Fe", positions=[[0.0, 0.0, 0.0]], cell=[2.0, 3.0, 12.0], pbc=(True, True, False))

    assert "Fourfold" in {site.name for site in SiteFinder(rocksalt).find_all()}
    bcc_sites = SiteFinder(bcc).find_all()
    assert "Fourfold" in {site.name for site in bcc_sites}
    assert "Long Bridge" in {site.name for site in bcc_sites}
