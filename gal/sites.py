"""Adsorption site detection on periodic surfaces.

The site finder implements a generic, periodic-aware approach for identifying
surface-top, bridge, and hollow adsorption sites. The implementation is
intended to work for both primitive cells and larger supercells, including
common 2D materials and metallic slabs.

Notes
-----
The algorithm assumes that the top surface layer is represented by atoms that
lie within a narrow vertical band near the maximum z coordinate. The default
surface tolerance is chosen to be robust for typical relaxed slabs and 2D
materials, but users may need to increase it for strongly corrugated or
highly distorted structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write


class SiteType(str, Enum):
    """Supported adsorption-site categories."""

    TOP = "Top"
    BRIDGE = "Bridge"
    HOLLOW = "Hollow"


@dataclass
class Site:
    """Represent an adsorption site.

    Parameters
    ----------
    name : str or SiteType
        Human-readable site label.
    position : numpy.ndarray
        Cartesian position of the site.
    neighbors : tuple[int, ...], optional
        Indices of the atoms that define the site.
    surface_layer : int | None, optional
        Layer index associated with the site if known.
    adsorption_height : float | None, optional
        Preferred adsorption height for this site.
    symmetry : str | None, optional
        Symmetry label or identifier for the site.
    metadata : dict[str, object], optional
        Additional site-specific information.
    """

    name: str | SiteType
    position: np.ndarray
    neighbors: tuple[int, ...] = ()
    surface_layer: int | None = None
    adsorption_height: float | None = None
    symmetry: str | None = None
    metadata: dict[str, Any] | None = None

    # Backward-compatible aliases for older code.
    element: str | None = None
    layer: int | None = None

    def __post_init__(self) -> None:
        """Populate backward-compatible aliases from the richer model."""
        self.position = np.asarray(self.position, dtype=float)
        if self.metadata is None:
            self.metadata = {}

        if self.element is None and self.metadata.get("element") is not None:
            self.element = self.metadata["element"]

        if self.layer is None and self.metadata.get("layer") is not None:
            self.layer = self.metadata["layer"]

        if self.surface_layer is None and self.layer is not None:
            self.surface_layer = self.layer


class SurfaceGraph:
    """Internal graph of periodic surface atoms and their neighbor relations."""

    def __init__(self, atoms: Atoms, surface_indices: list[int], cutoff: float) -> None:
        self.atoms = atoms
        self.surface_indices = list(surface_indices)
        self.cutoff = float(cutoff)
        self.nodes: list[int] = []
        self.neighbor_map: dict[int, set[int]] = {}
        self.edges: set[frozenset[int]] = set()
        self.triangles: set[frozenset[int]] = set()
        # These are deliberately kept private: the public graph attributes
        # above continue to use atom indices, while these records retain the
        # image coordinates needed to place sites in a primitive cell.
        self._image_nodes: list[tuple[int, tuple[int, int, int]]] = []
        self._image_positions: dict[tuple[int, tuple[int, int, int]], np.ndarray] = {}
        self._edge_records: list[tuple[tuple[int, tuple[int, int, int]], tuple[int, tuple[int, int, int]]]] = []
        self._triangle_records: list[
            tuple[
                tuple[int, tuple[int, int, int]],
                tuple[int, tuple[int, int, int]],
                tuple[int, tuple[int, int, int]],
            ]
        ] = []
        self._build()

    def _build(self) -> None:
        """Build a graph from surface atoms and their periodic images.

        A minimum-image pair search over only the atoms in the input cell
        cannot represent a bond from an atom to one of its own images.  That
        is exactly the common primitive-cell case (for example the single
        upper Se atom in primitive WSe2).  We therefore make a small 3x3x3
        image neighbourhood for periodic axes.  Public graph data remains
        indexed by the original atoms; private records retain image identity
        so bridge midpoints and triangle centroids are geometrically correct.
        """
        if not self.surface_indices:
            return

        self.nodes = list(self.surface_indices)
        for node in self.nodes:
            self.neighbor_map[node] = set()

        # Cells already containing a complete three-atom surface motif retain
        # the historical in-cell graph.  Besides preserving its one-site-per-
        # triangle behaviour, this avoids treating larger-cutoff second-shell
        # image cliques as new hollow-site motifs.  Image expansion is needed
        # precisely for underspecified primitive cells (one or two surface
        # atoms), where the in-cell graph cannot form an edge or a triangle.
        if len(self.nodes) >= 3:
            central = (0, 0, 0)
            central_nodes = [(node, central) for node in self.nodes]
            for node in central_nodes:
                self._image_nodes.append(node)
                self._image_positions[node] = self.atoms.positions[node[0]].copy()

            for left_index, left_node in enumerate(central_nodes):
                for right_node in central_nodes[left_index + 1 :]:
                    distance = np.linalg.norm(
                        self._image_positions[left_node] - self._image_positions[right_node]
                    )
                    if not 1e-8 < distance <= self.cutoff:
                        continue
                    left_atom, right_atom = left_node[0], right_node[0]
                    self.neighbor_map[left_atom].add(right_atom)
                    self.neighbor_map[right_atom].add(left_atom)
                    self.edges.add(frozenset((left_atom, right_atom)))
                    self._edge_records.append((left_node, right_node))

            for node in self.nodes:
                for first_neighbor in sorted(self.neighbor_map[node]):
                    for second_neighbor in sorted(self.neighbor_map[node]):
                        if first_neighbor >= second_neighbor:
                            continue
                        if second_neighbor in self.neighbor_map[first_neighbor]:
                            triangle = frozenset((node, first_neighbor, second_neighbor))
                            if len(triangle) == 3:
                                self.triangles.add(triangle)
                                self._triangle_records.append(
                                    tuple((atom, central) for atom in sorted(triangle))
                                )
            return

        translations = [
            range(-1, 2) if periodic else range(0, 1)
            for periodic in self.atoms.pbc
        ]
        for node in self.nodes:
            for a in translations[0]:
                for b in translations[1]:
                    for c in translations[2]:
                        image_node = (node, (a, b, c))
                        self._image_nodes.append(image_node)
                        self._image_positions[image_node] = (
                            self.atoms.positions[node]
                            + np.asarray((a, b, c), dtype=float) @ self.atoms.cell.array
                        )

        adjacency: list[set[int]] = [set() for _ in self._image_nodes]
        central_translation = (0, 0, 0)
        for left_index, left_node in enumerate(self._image_nodes):
            for right_index in range(left_index + 1, len(self._image_nodes)):
                right_node = self._image_nodes[right_index]
                distance = np.linalg.norm(
                    self._image_positions[left_node] - self._image_positions[right_node]
                )
                if not 1e-8 < distance <= self.cutoff:
                    continue

                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                left_atom, right_atom = left_node[0], right_node[0]
                self.neighbor_map[left_atom].add(right_atom)
                self.neighbor_map[right_atom].add(left_atom)
                self.edges.add(frozenset((left_atom, right_atom)))

                # A central-cell endpoint gives a complete set of bonds,
                # including atom-to-own-image bonds, without storing all
                # translational duplicates as placement candidates.
                if left_node[1] == central_translation or right_node[1] == central_translation:
                    self._edge_records.append((left_node, right_node))

        seen_triangles: set[tuple[int, int, int]] = set()
        for first_index, first_neighbors in enumerate(adjacency):
            for second_index in first_neighbors:
                if second_index <= first_index:
                    continue
                common_neighbors = first_neighbors & adjacency[second_index]
                for third_index in common_neighbors:
                    if third_index <= second_index:
                        continue
                    triangle_indices = (first_index, second_index, third_index)
                    if triangle_indices in seen_triangles:
                        continue
                    triangle_nodes = tuple(self._image_nodes[index] for index in triangle_indices)
                    if not any(node[1] == central_translation for node in triangle_nodes):
                        continue
                    seen_triangles.add(triangle_indices)
                    self._triangle_records.append(triangle_nodes)
                    self.triangles.add(frozenset(node[0] for node in triangle_nodes))


class SiteFinder:
    """Detect adsorption sites on a surface."""

    def __init__(self, atoms: Atoms) -> None:
        """Store the ASE atoms object used for site detection."""
        self.atoms = atoms
        self._graph_cache: dict[tuple[float | None, str | None], SurfaceGraph] = {}

    def surface_atoms(self, tolerance: float = 0.5) -> np.ndarray:
        """Return indices of top-surface atoms.

        Parameters
        ----------
        tolerance : float, optional
            Distance in Å from the highest z-position that still counts as part
            of the top surface layer.
        """
        z_positions = self.atoms.positions[:, 2]
        z_max = np.max(z_positions)
        return np.where(z_positions >= z_max - tolerance)[0]

    def _surface_height(self) -> float:
        """Return the highest z-position of the current structure."""
        return float(np.max(self.atoms.positions[:, 2]))

    def _resolve_tolerance(self, tolerance: float | None) -> float:
        """Resolve the surface-layer tolerance, using a default of 0.5 Å."""
        if tolerance is None:
            return 0.5
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        return float(tolerance)

    def _top_indices(self, tolerance: float | None = None, element: str | None = None) -> list[int]:
        """Return sorted indices of atoms belonging to the top surface layer."""
        resolved_tolerance = self._resolve_tolerance(tolerance)
        z_positions = self.atoms.positions[:, 2]
        z_max = np.max(z_positions)
        indices = [int(index) for index in np.where(z_positions >= z_max - resolved_tolerance)[0]]
        if element is not None:
            indices = [index for index in indices if self.atoms[index].symbol == element]
        return sorted(indices)

    def _get_graph(self, cutoff: float | None = None, element: str | None = None) -> SurfaceGraph:
        """Return a cached periodic surface graph for the current structure."""
        resolved_cutoff = self._resolve_cutoff(cutoff)
        key = (resolved_cutoff, element)
        if key not in self._graph_cache:
            self._graph_cache[key] = SurfaceGraph(
                atoms=self.atoms,
                surface_indices=self._top_indices(tolerance=0.5, element=element),
                cutoff=resolved_cutoff,
            )
        return self._graph_cache[key]

    def _top_neighbor_map(self, cutoff: float | None = None) -> dict[int, list[int]]:
        """Return a compatibility-style neighbor map for the current top surface."""
        graph = self._get_graph(cutoff=cutoff)
        return {
            index: sorted(neighbors)
            for index, neighbors in graph.neighbor_map.items()
        }

    def _estimate_neighbor_distance(self) -> float:
        """Estimate the nearest top-layer distance, including periodic images."""
        surface_indices = self._top_indices()
        if not surface_indices:
            return 1.0

        translations = [
            range(-1, 2) if periodic else range(0, 1)
            for periodic in self.atoms.pbc
        ]
        distances: list[float] = []
        for left_node in surface_indices:
            for right_node in surface_indices:
                for a in translations[0]:
                    for b in translations[1]:
                        for c in translations[2]:
                            if left_node == right_node and (a, b, c) == (0, 0, 0):
                                continue
                            displacement = (
                                self.atoms.positions[right_node]
                                + np.asarray((a, b, c), dtype=float) @ self.atoms.cell.array
                                - self.atoms.positions[left_node]
                            )
                            distance = float(np.linalg.norm(displacement))
                            if distance > 1e-8:
                                distances.append(distance)

        if not distances:
            return 1.0
        return float(np.min(distances))

    def _resolve_cutoff(self, cutoff: float | None) -> float:
        """Resolve the neighbor-search cutoff."""
        estimated = float(self._estimate_neighbor_distance() * 1.2)

        if cutoff is None:
            return estimated

        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")

        if cutoff < estimated:
            return estimated

        return float(cutoff)

    def _wrap_position(self, position: np.ndarray) -> np.ndarray:
        """Wrap a Cartesian position into the unit cell."""
        wrapped = self.atoms.cell.scaled_positions([position])[0]
        wrapped = np.mod(wrapped, 1.0)
        return self.atoms.cell.cartesian_positions([wrapped])[0]

    def _make_site(
        self,
        name: str | SiteType,
        position: np.ndarray,
        neighbors: tuple[int, ...],
        element: str | None = None,
        layer: int | None = None,
        surface_layer: int | None = None,
        metadata: dict[str, Any] | None = None,
        adsorption_height: float | None = None,
        symmetry: str | None = None,
    ) -> Site:
        """Create a :class:`Site` instance with the supplied attributes."""
        return Site(
            name=name,
            position=self._wrap_position(position),
            neighbors=neighbors,
            surface_layer=surface_layer if surface_layer is not None else layer,
            adsorption_height=adsorption_height,
            symmetry=symmetry,
            metadata=metadata,
            element=element,
            layer=layer,
        )

    def find_top(self, element: str | None = None, tolerance: float | None = None) -> list[Site]:
        """Return top-site candidates for the requested element.

        Parameters
        ----------
        element : str | None, optional
            If provided, only atoms with this elemental symbol are considered.
        tolerance : float | None, optional
            Surface-layer tolerance in Å. The default value of 0.5 Å is
            typically sufficient for relaxed 2D materials and slabs.
        """
        sites: list[Site] = []
        z_surface = self._surface_height()
        for index in self._top_indices(tolerance=tolerance, element=element):
            position = self.atoms.positions[index].copy()
            position[2] = z_surface
            metadata = {"kind": "top", "layer": 0}
            if element is not None:
                metadata["element"] = element
            sites.append(
                self._make_site(
                    name="Top" if element is None else f"Top_{element}",
                    position=position,
                    neighbors=(index,),
                    element=element,
                    surface_layer=0,
                    metadata=metadata,
                )
            )
        return sites

    def find_top_se(self, tolerance: float | None = None) -> list[Site]:
        """Backward-compatible wrapper for finding top Se sites."""
        return self.find_top(element="Se", tolerance=tolerance)

    def find_top_w(self, tolerance: float | None = None) -> list[Site]:
        """Backward-compatible wrapper for finding top W sites."""
        return self.find_top(element="W", tolerance=tolerance)

    def find_bridge(self, cutoff: float | None = None) -> list[Site]:
        """Find bridge sites from periodic neighbors of the top surface layer."""
        graph = self._get_graph(cutoff=cutoff)
        z_surface = self._surface_height()
        sites: list[Site] = []
        seen: set[tuple[tuple[int, ...], tuple[float, float, float]]] = set()

        for left_image, right_image in graph._edge_records:
            left_node, _ = left_image
            right_node, _ = right_image
            left_position = graph._image_positions[left_image]
            right_position = graph._image_positions[right_image]
            midpoint = 0.5 * (left_position + right_position)
            midpoint[2] = z_surface
            wrapped_midpoint = self._wrap_position(midpoint)
            wrapped_midpoint[2] = z_surface
            # Periodic images of the same atom still define a valid bridge in
            # a primitive cell; keep the public Site neighbor field expressed
            # in original atom indices.
            neighbors = tuple(sorted({left_node, right_node}))
            key = (neighbors, tuple(np.round(wrapped_midpoint, 6)))
            if key in seen:
                continue
            seen.add(key)
            sites.append(
                self._make_site(
                    name="Bridge",
                    position=wrapped_midpoint,
                    neighbors=neighbors,
                    surface_layer=0,
                    metadata={"kind": "bridge", "layer": 0},
                )
            )

        return self.remove_duplicates(sites)

    def find_hollow(self, cutoff: float | None = None) -> list[Site]:
        """Find hollow sites from periodic triangles in the top-surface graph."""
        graph = self._get_graph(cutoff=cutoff)
        z_surface = self._surface_height()
        sites: list[Site] = []
        seen: set[tuple[tuple[int, ...], tuple[float, float, float]]] = set()

        for triangle in graph._triangle_records:
            nodes = [node[0] for node in triangle]
            positions = [graph._image_positions[node] for node in triangle]
            centroid = np.mean(positions, axis=0)
            centroid[2] = z_surface
            wrapped_centroid = self._wrap_position(centroid)
            wrapped_centroid[2] = z_surface
            # As with bridges, a primitive-cell hollow can be bounded by
            # three periodic images of a single original atom.
            neighbors = tuple(sorted(set(nodes)))
            key = (neighbors, tuple(np.round(wrapped_centroid, 6)))
            if key in seen:
                continue
            seen.add(key)
            sites.append(
                self._make_site(
                    name="Hollow",
                    position=wrapped_centroid,
                    neighbors=neighbors,
                    surface_layer=0,
                    metadata={"kind": "hollow", "layer": 0},
                )
            )

        return self.remove_duplicates(sites)

    def remove_duplicates(self, sites: list[Site], tol: float = 1e-3) -> list[Site]:
        """Remove duplicate adsorption sites after wrapping positions into the cell."""
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
        """Return all detected site candidates sorted by site type."""
        sites = [*self.find_top(), *self.find_bridge(), *self.find_hollow()]
        unique_sites = self.remove_duplicates(sites)
        type_order = {
            "Top": 0,
            "Top_Se": 0,
            "Top_W": 0,
            "Bridge": 1,
            "Hollow": 2,
        }
        return sorted(
            unique_sites,
            key=lambda site: (
                type_order.get(str(site.name), 999),
                str(site.name),
            ),
        )

    def write_xyz(self, filename: str = "adsorption_sites.xyz", height: float = 1.0) -> None:
        """Export the surface together with adsorption sites."""
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
