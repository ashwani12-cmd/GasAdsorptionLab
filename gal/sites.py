"""
gal/sites.py

Adsorption site detection.

Author: Ashwani Kushwaha
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.io import write
from ase.neighborlist import NeighborList


# ==========================================================
# Site
# ==========================================================

@dataclass
class Site:
    """
    Represents an adsorption site.
    """

    name: str
    position: np.ndarray
    element: str | None = None
    layer: int | None = None
    neighbors: tuple[int, ...] = ()


# ==========================================================
# Site Finder
# ==========================================================

class SiteFinder:
    """
    Detect adsorption sites on a surface.
    """

    def __init__(self, atoms: Atoms):
        self.atoms = atoms

    # ------------------------------------------------------

    def surface_atoms(self, tolerance: float = 0.5):
        """
        Return indices of atoms on the top surface.
        """

        z = self.atoms.positions[:, 2]
        zmax = np.max(z)

        return np.where(z > zmax - tolerance)[0]

    # ------------------------------------------------------

    def find_top_se(self, tolerance: float = 0.5):
        """
        Top Se adsorption sites.
        """

        sites = []

        zmax = np.max(self.atoms.positions[:, 2])

        for i, atom in enumerate(self.atoms):

            if atom.symbol != "Se":
                continue

            if atom.position[2] < zmax - tolerance:
                continue

            sites.append(
                Site(
                    name="Top_Se",
                    position=atom.position.copy(),
                    element="Se",
                    neighbors=(i,),
                )
            )

        return sites

    # ------------------------------------------------------

    def find_top_w(self):
        """
        Project upper-layer W atoms onto the surface.
        """

        sites = []

        z = self.atoms.positions[:, 2]

        z_surface = np.max(z)
        z_mid = 0.5 * (np.max(z) + np.min(z))

        for i, atom in enumerate(self.atoms):

            if atom.symbol != "W":
                continue

            if atom.position[2] < z_mid:
                continue

            pos = atom.position.copy()
            pos[2] = z_surface

            sites.append(
                Site(
                    name="Top_W",
                    position=pos,
                    element="W",
                    neighbors=(i,),
                )
            )

        return sites

    # ------------------------------------------------------

    def find_bridge(self, cutoff: float = 3.0):
        """
        Bridge adsorption sites.
        """

        top = set(self.surface_atoms())

        cutoffs = [cutoff / 2.0] * len(self.atoms)

        nl = NeighborList(
            cutoffs,
            self_interaction=False,
            bothways=True,
        )

        nl.update(self.atoms)

        z_surface = np.max(self.atoms.positions[:, 2])

        sites = []

        visited = set()

        for i in top:

            neigh, offsets = nl.get_neighbors(i)

            for j, offset in zip(neigh, offsets):

                if j not in top:
                    continue

                pair = tuple(sorted((i, j)))

                if pair in visited:
                    continue

                visited.add(pair)

                ri = self.atoms.positions[i]

                rj = (
                    self.atoms.positions[j]
                    + np.dot(offset, self.atoms.cell)
                )

                midpoint = 0.5 * (ri + rj)

                midpoint[2] = z_surface

                sites.append(
                    Site(
                        name="Bridge",
                        position=midpoint,
                        neighbors=(i, j),
                    )
                )

        return self.remove_duplicates(sites)

    # ------------------------------------------------------

    def remove_duplicates(self, sites, tol=1e-3):
        """
        Remove duplicate adsorption sites.
        """

        unique = []

        for site in sites:

            duplicate = False

            for other in unique:

                if np.linalg.norm(
                    site.position - other.position
                ) < tol:

                    duplicate = True
                    break

            if not duplicate:
                unique.append(site)

        return unique

    # ------------------------------------------------------

    def find_all(self):
        """
        Return every adsorption site.
        """

        sites = []

        sites.extend(self.find_top_se())
        sites.extend(self.find_top_w())
        sites.extend(self.find_bridge())

        return self.remove_duplicates(sites)

    # ------------------------------------------------------

    def write_xyz(
        self,
        filename="adsorption_sites.xyz",
        height=1.0,
    ):
        """
        Export the surface together with adsorption sites.

        Dummy atoms:
            He = Top Se
            Ne = Top W
            Ar = Bridge
            Kr = Hollow
        """

        mapping = {
            "Top_Se": "He",
            "Top_W": "Ne",
            "Bridge": "Ar",
            "Hollow": "Kr",
        }

        xyz = self.atoms.copy()

        for site in self.find_all():

            symbol = mapping.get(site.name, "He")

            pos = site.position.copy()
            pos[2] += height

            xyz += Atoms(
                symbols=symbol,
                positions=[pos],
            )

        write(filename, xyz)

        print(f"\nSaved adsorption sites to: {filename}")

        print(f"Total atoms in file: {len(xyz)}")
