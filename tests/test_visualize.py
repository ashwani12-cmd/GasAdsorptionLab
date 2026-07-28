import numpy as np
from ase import Atoms

from gal.sites import Site
from gal.visualize import plot_bridge_sites, plot_hollow_sites, plot_top_sites


def test_visualization_functions_return_ase_objects():
    atoms = Atoms(
        symbols="Si2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]],
        pbc=(True, True, False),
    )

    top_sites = [
        Site(name="Top", position=np.array([0.0, 0.0, 0.0]), neighbors=(0,), metadata={"kind": "top"})
    ]
    bridge_sites = [
        Site(name="Bridge", position=np.array([0.5, 0.0, 0.0]), neighbors=(0, 1), metadata={"kind": "bridge"})
    ]
    hollow_sites = [
        Site(name="Hollow", position=np.array([0.5, 0.5, 0.0]), neighbors=(0, 1), metadata={"kind": "hollow"})
    ]

    top_view = plot_top_sites(atoms, top_sites, show=False)
    bridge_view = plot_bridge_sites(atoms, bridge_sites, show=False)
    hollow_view = plot_hollow_sites(atoms, hollow_sites, show=False)

    assert isinstance(top_view, Atoms)
    assert isinstance(bridge_view, Atoms)
    assert isinstance(hollow_view, Atoms)

    assert len(top_view) == len(atoms) + 1
    assert len(bridge_view) == len(atoms) + 1
    assert len(hollow_view) == len(atoms) + 1

    assert top_view.get_array("site_labels")[0] == "Top_0"
    assert bridge_view.get_array("site_labels")[0] == "Bridge_0"
    assert hollow_view.get_array("site_labels")[0] == "Hollow_0"
