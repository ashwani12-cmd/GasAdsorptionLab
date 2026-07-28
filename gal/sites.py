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


@dataclass
class Site:
    """Represent an adsorption site.

    Parameters
    ----------
    name : str
        Human-readable site label such as ``Top_Se`` or ``Hollow``.
    position : numpy.ndarray
        Cartesian position of the site.
    neighbors : tuple[int, ...], optional
        Indices of the atoms that define the site.
    surface_layer : int | None, optional
        Layer index associated with the site if known.
    metadata : dict[str, object], optional
        Additional site-specific information.
    """

    name: str
    position: np.ndarray
    neighbors: tuple[int, ...] = ()
    surface_layer: int | None = None
    metadata: dict[str, object] | None = None

    # Backward-compatible aliases for older code.
    element: str | None = None
    layer: int | None = None

    def __post_init__(self) -> None:
        """Populate backward-compatible aliases from the richer model."""
        if self.metadata is None:
            self.metadata = {}

        if self.element is None and self.metadata.get("element") is not None:
            self.element = self.metadata["element"]

        if self.layer is None and self.metadata.get("layer") is not None:
            self.layer = self.metadata["layer"]

        if self.surface_layer is None and self.layer is not None:
            self.surface_layer = self.layer


class SiteFinder:
    """Detect adsorption sites on a surface."""

    def __init__(self, atoms: Atoms) -> None:
        """Store the ASE atoms object used for site detection."""
        self.atoms = atoms

    def surface_atoms(self, tolerance: float = 0.5) -> np.ndarray:
        """
        Return indices of atoms on the top surface.

        Parameters
        ----------
        tolerance : float, optional
            Distance in Å from the highest z-position that still counts as
            part of the top surface.

        Returns
        -------
        numpy.ndarray
            Indices of atoms belonging to the top surface.
        """
        z_positions = self.atoms.positions[:, 2]
        z_max = np.max(z_positions)
        return np.where(z_positions > z_max - tolerance)[0]

    def _surface_height(self) -> float:
        """Return the maximum z-position of the current structure."""
        return float(np.max(self.atoms.positions[:, 2]))

    def _top_indices(self) -> list[int]:
        """Return sorted indices of top-surface atoms."""
        return sorted(int(index) for index in set(self.surface_atoms()))

    def _build_neighbor_list(self, cutoff: float) -> NeighborList:
        """Build a neighbor list using the requested cutoff."""
        cutoffs = [cutoff / 2.0] * len(self.atoms)
        neighbor_list = NeighborList(
            cutoffs,
            self_interaction=False,
            bothways=True,
        )
        neighbor_list.update(self.atoms)
        return neighbor_list

    def _top_neighbor_map(self, cutoff: float) -> dict[int, list[int]]:
        """Return a mapping from top atoms to their top-surface neighbors."""
        top_indices = self._top_indices()
        top_set = set(top_indices)
        neighbor_list = self._build_neighbor_list(cutoff)

        return {
            index: [
                neighbor
                for neighbor in neighbor_list.get_neighbors(index)[0]
                if neighbor in top_set and neighbor != index
            ]
            for index in top_indices
        }

    def _make_site(
        self,
        name: str,
        position: np.ndarray,
        neighbors: tuple[int, ...],
        element: str | None = None,
        layer: int | None = None,
        surface_layer: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Site:
        """Create a :class:`Site` instance with the supplied attributes."""
        return Site(
            name=name,
            position=position,
            neighbors=neighbors,
            surface_layer=surface_layer if surface_layer is not None else layer,
            metadata=metadata,
            element=element,
            layer=layer,
        )

    def find_top_se(self, tolerance: float = 0.5) -> list[Site]:
        """
        Find top Se adsorption sites.

        Parameters
        ----------
        tolerance : float, optional
            Distance in Å from the highest z-position that still counts as a
            top-layer Se atom.

        Returns
        -------
        list[Site]
            Top-site candidates for Se atoms.
        """
        sites: list[Site] = []
        z_max = np.max(self.atoms.positions[:, 2])

        for index, atom in enumerate(self.atoms):
            if atom.symbol != "Se":
                continue
            if atom.position[2] < z_max - tolerance:
                continue

            sites.append(
                self._make_site(
                    name="Top_Se",
                    position=atom.position.copy(),
                    neighbors=(index,),
                    element="Se",
                    surface_layer=0,
                    metadata={"element": "Se", "layer": 0},
                )
            )

        return sites

    def find_top_w(self) -> list[Site]:
        """
        Project upper-layer W atoms onto the surface.

        Returns
        -------
        list[Site]
            Top-site candidates for W atoms.
        """
        sites: list[Site] = []
        z_positions = self.atoms.positions[:, 2]
        z_surface = float(np.max(z_positions))
        z_mid = 0.5 * (np.max(z_positions) + np.min(z_positions))

        for index, atom in enumerate(self.atoms):
            if atom.symbol != "W":
                continue
            if atom.position[2] < z_mid:
                continue

            position = atom.position.copy()
            position[2] = z_surface

            sites.append(
                self._make_site(
                    name="Top_W",
                    position=position,
                    neighbors=(index,),
                    element="W",
                    surface_layer=0,
                    metadata={"element": "W", "layer": 0},
                )
            )

        return sites

    def find_bridge(self, cutoff: float = 3.0) -> list[Site]:
        """
        Find bridge adsorption sites between neighboring top-surface atoms.

        Parameters
        ----------
        cutoff : float, optional
            Neighbor-search cutoff in Å.

        Returns
        -------
        list[Site]
            Bridge-site candidates.
        """
        top_indices = self._top_indices()
        top_set = set(top_indices)
        neighbor_list = self._build_neighbor_list(cutoff)
        z_surface = self._surface_height()

        sites: list[Site] = []
        visited: set[tuple[int, int]] = set()

        for index in top_indices:
            neighbors, offsets = neighbor_list.get_neighbors(index)

            for neighbor_index, offset in zip(neighbors, offsets):
                if neighbor_index not in top_set:
                    continue

                pair = tuple(sorted((index, neighbor_index)))
                if pair in visited:
                    continue

                visited.add(pair)

                position_i = self.atoms.positions[index]
                position_j = self.atoms.positions[neighbor_index] + np.dot(
                    offset,
                    self.atoms.cell,
                )
                midpoint = 0.5 * (position_i + position_j)
                midpoint[2] = z_surface

                sites.append(
                    self._make_site(
                        name="Bridge",
                        position=midpoint,
                        neighbors=(index, neighbor_index),
                        surface_layer=0,
                        metadata={"kind": "bridge", "layer": 0},
                    )
                )

        return self.remove_duplicates(sites)

    def _get_first_neighbor_distance(self) -> float:
        """Estimate the first-neighbor distance from the top-surface atoms."""
        top_indices = self._top_indices()

        if len(top_indices) < 2:
            return 1.0

        distances: list[float] = []
        for i, index in enumerate(top_indices):
            for other_index in top_indices[i + 1 :]:
                distance = self.atoms.get_distance(index, other_index, mic=True)
                if distance > 0.0:
                    distances.append(distance)

        if not distances:
            return 1.0

        return float(np.min(distances))

    def _resolve_cutoff(self, cutoff: float | None) -> float:
        """Resolve the cutoff, using an automatic estimate when requested."""
        if cutoff is None:
            first_neighbor_distance = self._get_first_neighbor_distance()
            return max(first_neighbor_distance * 1.2, 1e-6)

        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")

        return float(cutoff)

    def find_hollow(self, cutoff: float | None = 3.0) -> list[Site]:
        """
        Find hollow adsorption sites from triangular arrangements of top atoms.

        The site position is taken as the centroid of three mutually
        neighboring top-surface atoms and projected onto the top surface plane.

        Parameters
        ----------
        cutoff : float | None, optional
            Neighbor-search cutoff in Å. If ``None``, the cutoff is inferred
            from the first-neighbor distance of the top-surface atoms. The
            default value of ``3.0`` preserves the historical behavior.

        Returns
        -------
        list[Site]
            Hollow-site candidates.
        """
        resolved_cutoff = self._resolve_cutoff(cutoff)
        top_indices = self._top_indices()
        top_neighbors = self._top_neighbor_map(resolved_cutoff)
        z_surface = self._surface_height()

        sites: list[Site] = []
        visited: set[tuple[int, int, int]] = set()

        for index in top_indices:
            for neighbor_index in top_neighbors[index]:
                for third_neighbor_index in top_neighbors[index]:
                    if neighbor_index >= third_neighbor_index:
                        continue
                    if third_neighbor_index not in top_neighbors[neighbor_index]:
                        continue
                    if neighbor_index not in top_neighbors[third_neighbor_index]:
                        continue

                    site_key = tuple(sorted((index, neighbor_index, third_neighbor_index)))
                    if site_key in visited:
                        continue

                    visited.add(site_key)

                    position_i = self.atoms.positions[index]
                    position_j = self.atoms.positions[neighbor_index]
                    position_k = self.atoms.positions[third_neighbor_index]
                    centroid = np.mean([position_i, position_j, position_k], axis=0)
                    centroid[2] = z_surface

                    sites.append(
                        self._make_site(
                            name="Hollow",
                            position=centroid,
                            neighbors=(index, neighbor_index, third_neighbor_index),
                            surface_layer=0,
                            metadata={"kind": "hollow", "layer": 0},
                        )
                    )

        return self.remove_duplicates(sites)

    def remove_duplicates(self, sites: list[Site], tol: float = 1e-3) -> list[Site]:
        """
        Remove duplicate adsorption sites.

        Parameters
        ----------
        sites : list[Site]
            Candidate sites to deduplicate.
        tol : float, optional
            Distance tolerance in Å below which two sites are considered
            duplicates.

        Returns
        -------
        list[Site]
            Unique sites sorted by insertion order.
        """
        unique: list[Site] = []

        for site in sites:
            duplicate = False
            for other in unique:
                if np.linalg.norm(site.position - other.position) < tol:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(site)

        return unique

    def find_all(self) -> list[Site]:
        """
        Return all adsorption sites detected on the surface.

        The method collects top, bridge, and hollow sites, removes duplicates,
        and orders the result by site type so the output is deterministic.

        Returns
        -------
        list[Site]
            All detected site candidates sorted by site type.
        """
        site_groups: list[list[Site]] = [
            self.find_top_se(),
            self.find_top_w(),
            self.find_bridge(),
            self.find_hollow(),
        ]

        sites: list[Site] = []
        for group in site_groups:
            sites.extend(group)

        unique_sites = self.remove_duplicates(sites)

        type_order = {
            "Top_Se": 0,
            "Top_W": 1,
            "Bridge": 2,
            "Hollow": 3,
        }

        return sorted(
            unique_sites,
            key=lambda site: (
                type_order.get(site.name, 999),
                site.name,
            ),
        )

    def write_xyz(
        self,
        filename: str = "adsorption_sites.xyz",
        height: float = 1.0,
    ) -> None:
        """
        Export the surface together with adsorption sites.

        Dummy atoms:
            He = Top Se
            Ne = Top W
            Ar = Bridge
            Kr = Hollow

        Parameters
        ----------
        filename : str, optional
            Output filename for the XYZ file.
        height : float, optional
            Vertical offset applied to each exported site.
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
            position = site.position.copy()
            position[2] += height
            xyz += Atoms(symbols=symbol, positions=[position])

        write(filename, xyz)
        print(f"\nSaved adsorption sites to: {filename}")
        print(f"Total atoms in file: {len(xyz)}")
