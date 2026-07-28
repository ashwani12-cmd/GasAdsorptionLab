"""Utilities for visualizing adsorption sites with ASE.

The module builds ASE-compatible atoms objects that can be rendered in
notebooks or through ASE's standard viewers. Each site type is plotted with
its own color and a label stored as an array on the returned atoms object.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from ase import Atoms
from ase.visualize import view

from .sites import Site

_SITE_COLORS = {
    "top": (0.12, 0.56, 1.0, 1.0),
    "bridge": (0.0, 0.7, 0.2, 1.0),
    "hollow": (0.95, 0.26, 0.21, 1.0),
}


def _append_site_marker(
    atoms: Atoms,
    site: Site,
    site_type: str,
    index: int,
) -> Atoms:
    """Return a copy of the atoms object with a site marker appended."""
    marker = Atoms(
        symbols="X",
        positions=[site.position],
        cell=atoms.cell.copy(),
        pbc=atoms.pbc,
    )
    marker.info["site_type"] = site_type
    marker.info["site_name"] = site.name
    marker.info["site_index"] = index

    combined = atoms.copy()
    combined.extend(marker)

    if "site_labels" not in combined.arrays:
        combined.arrays["site_labels"] = np.array([], dtype=str)
    labels = np.array([f"{site.name}_{index}"], dtype=str)
    combined.arrays["site_labels"] = np.concatenate([combined.arrays["site_labels"], labels])

    if "site_colors" not in combined.arrays:
        combined.arrays["site_colors"] = np.empty((0, 4), dtype=float)
    colors = np.array([_SITE_COLORS[site_type]], dtype=float)
    combined.arrays["site_colors"] = np.vstack([combined.arrays["site_colors"], colors])

    if "site_sizes" not in combined.arrays:
        combined.arrays["site_sizes"] = np.array([], dtype=float)
    sizes = np.array([20.0], dtype=float)
    combined.arrays["site_sizes"] = np.concatenate([combined.arrays["site_sizes"], sizes])
    return combined


def _plot_sites(atoms: Atoms, sites: Iterable[Site], site_type: str, show: bool = True) -> Atoms:
    """Append site markers for all sites of a given type to an ASE atoms object."""
    view_atoms = atoms.copy()
    for index, site in enumerate(sites):
        view_atoms = _append_site_marker(view_atoms, site, site_type, index)

    if show:
        view(view_atoms)

    return view_atoms


def plot_top_sites(atoms: Atoms, sites: Iterable[Site], show: bool = True) -> Atoms:
    """Visualize top adsorption sites with ASE.

    Parameters
    ----------
    atoms : ase.Atoms
        The underlying surface structure.
    sites : iterable[Site]
        Top-site objects to visualize.
    show : bool, optional
        Whether to open ASE's viewer immediately.

    Returns
    -------
    ase.Atoms
        An ASE atoms object containing the original structure and one marker
        for each plotted site.
    """
    return _plot_sites(atoms, sites, "top", show=show)


def plot_bridge_sites(atoms: Atoms, sites: Iterable[Site], show: bool = True) -> Atoms:
    """Visualize bridge adsorption sites with ASE."""
    return _plot_sites(atoms, sites, "bridge", show=show)


def plot_hollow_sites(atoms: Atoms, sites: Iterable[Site], show: bool = True) -> Atoms:
    """Visualize hollow adsorption sites with ASE."""
    return _plot_sites(atoms, sites, "hollow", show=show)


if __name__ == "__main__":
    from ase.build import bulk

    atoms = bulk("Si", cubic=True) * (1, 1, 1)
    site = Site(name="Top", position=np.array([0.0, 0.0, 0.0]), neighbors=(0,), metadata={"kind": "top"})
    plot_top_sites(atoms, [site], show=True)
