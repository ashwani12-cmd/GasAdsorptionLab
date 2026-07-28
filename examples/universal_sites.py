"""Run the universal site engine on representative crystal families."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from gal import SiteFinder


def wse2_like() -> Atoms:
    return Atoms(
        "WSe2",
        positions=[[0.0, 0.0, 1.6], [0.0, 0.0, 0.0], [0.0, 0.0, -1.6]],
        cell=[[3.28, 0.0, 0.0], [1.64, 2.8405, 0.0], [0.0, 0.0, 20.0]],
        pbc=(True, True, False),
    )


def cu111_like() -> Atoms:
    return Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 2.0], [1.5, np.sqrt(3.0) / 2.0, 0.0], [1.0, np.sqrt(3.0) / 2.0, -2.0]],
        cell=[[3.0, 0.0, 0.0], [1.5, 3.0 * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 15.0]],
        pbc=(True, True, False),
    )


def rocksalt001_like() -> Atoms:
    return Atoms(
        "MgONaCl",
        positions=[[0.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 0.0, -1.0], [2.0, 2.0, -1.0]],
        cell=[4.0, 4.0, 12.0],
        pbc=(True, True, False),
    )


if __name__ == "__main__":
    for label, atoms in {"WSe2-like": wse2_like(), "Cu(111)-like": cu111_like(), "rocksalt(001)-like": rocksalt001_like()}.items():
        finder = SiteFinder(atoms)
        print(f"{label}: {finder.detect_surface_type().value}")
        print([site.name for site in finder.find_all()])
