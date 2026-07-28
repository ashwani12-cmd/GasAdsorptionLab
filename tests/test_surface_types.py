import numpy as np
from ase import Atoms

from gal.sites import SiteFinder, SiteType, SurfaceType


def test_site_type_keeps_legacy_values_and_exposes_universal_categories():
    assert SiteType.TOP.value == "Top"
    assert SiteType.HOLLOW.value == "Hollow"
    assert SiteType.BOTTOM_BRIDGE.value == "Bottom Bridge"
    assert SiteType.FCC.value == "FCC"
    assert SiteType.FOURFOLD.value == "Fourfold"


def test_geometry_classifies_hexagonal_layered_and_close_packed_surfaces():
    cell = [[3.0, 0.0, 0.0], [1.5, 3.0 * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 15.0]]
    layered = Atoms("HeNeAr", positions=[[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, -2.0]], cell=cell, pbc=(True, True, False))
    close_packed = Atoms("HeNeAr", positions=[[0.0, 0.0, 2.0], [1.5, np.sqrt(3.0) / 6.0, 0.0], [1.0, np.sqrt(3.0) / 3.0, -2.0]], cell=cell, pbc=(True, True, False))

    assert SiteFinder(layered).detect_surface_type() is SurfaceType.HEXAGONAL_2D
    assert SiteFinder(close_packed).detect_surface_type() is SurfaceType.FCC111


def test_geometry_classifies_square_surface_without_element_checks():
    rocksalt_like = Atoms(
        "HeNeArKr",
        positions=[[0.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 0.0, -1.0], [2.0, 2.0, -1.0]],
        cell=[4.0, 4.0, 12.0],
        pbc=(True, True, False),
    )
    assert SiteFinder(rocksalt_like).detect_surface_type() is SurfaceType.ROCKSALT001
