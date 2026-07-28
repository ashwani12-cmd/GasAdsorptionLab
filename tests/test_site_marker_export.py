import numpy as np
from ase import Atoms
from ase.io import read

from gal.sites import Site
from gal.visualize import export_site_markers


def _surface() -> Atoms:
    return Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[3.0, 3.0, 12.0], pbc=(True, True, False))


def _sites() -> list[Site]:
    return [
        Site(name="Top", position=np.array([0.0, 0.0, 0.0]), neighbors=(0,)),
        Site(name="Bridge", position=np.array([0.5, 0.0, 0.0]), neighbors=(0, 1)),
    ]


def test_visualization_position_returns_elevated_copy_without_mutation():
    site = _sites()[0]
    original = site.position.copy()

    position = site.visualization_position(2.5)

    np.testing.assert_allclose(site.position, original)
    np.testing.assert_allclose(position, [0.0, 0.0, 2.5])


def test_marker_export_creates_individual_files_and_trajectory(tmp_path):
    surface = _surface()
    sites = _sites()

    paths = export_site_markers(surface, sites, output_dir=tmp_path / "ovito_sites", marker_element="Ne", marker_height=2.5)

    assert len(paths) == len(sites)
    assert all(path.exists() for path in paths)
    for path, site in zip(paths, sites):
        exported = read(path)
        assert len(exported) == len(surface) + 1
        assert exported[-1].symbol == "Ne"
        assert exported.positions[-1, 2] > surface.positions[:, 2].max()
        np.testing.assert_allclose(exported.positions[-1], site.visualization_position(2.5))

    frames = read(tmp_path / "ovito_sites" / "all_sites.xyz", index=":")
    assert len(frames) == len(sites)
    assert all(len(frame) == len(surface) + 1 for frame in frames)
