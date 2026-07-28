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
    BOTTOM = "Bottom"
    BRIDGE = "Bridge"
    BOTTOM_BRIDGE = "Bottom Bridge"
    HOLLOW = "Hollow"
    BOTTOM_HOLLOW = "Bottom Hollow"
    FCC = "FCC"
    HCP = "HCP"
    FOURFOLD = "Fourfold"
    LONG_BRIDGE = "Long Bridge"


class SurfaceType(str, Enum):
    """Geometry-derived surface families understood by :class:`SiteFinder`."""

    HEXAGONAL_2D = "hexagonal_2d"
    FCC111 = "fcc111"
    FCC100 = "fcc100"
    BCC110 = "bcc110"
    BCC100 = "bcc100"
    ROCKSALT001 = "rocksalt001"
    PEROVSKITE001 = "perovskite001"
    UNKNOWN = "unknown"


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

    def visualization_position(self, height: float | None = None) -> np.ndarray:
        """Return an elevated copy of this site position for visualization.

        ``Site.position`` remains the adsorption point on the surface.  This
        helper is deliberately separate from placement and is intended for
        marker atoms in OVITO, VESTA, and ASE viewers.
        """
        offset = self.adsorption_height if height is None and self.adsorption_height is not None else height
        offset = 2.5 if offset is None else float(offset)
        position = self.position.copy()
        position[2] += offset
        return position


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


class _SiteEngine:
    """Base class for geometry-family site engines.

    Engines intentionally share the periodic graph implementation; only the
    interpretation of graph motifs differs between crystal families.
    """

    def __init__(self, finder: "SiteFinder") -> None:
        self.finder = finder

    def find_top(self, element: str | None, tolerance: float | None) -> list[Site]:
        return self.finder._find_top_generic(element, tolerance)

    def find_bridge(self, cutoff: float | None) -> list[Site]:
        return self.finder._find_bridge_generic(cutoff)

    def find_hollow(self, cutoff: float | None) -> list[Site]:
        return self.finder._find_hollow_generic(cutoff)

    def find_all(self) -> list[Site]:
        return [*self.find_top(None, None), *self.find_bridge(None), *self.find_hollow(None)]


class _Hexagonal2DEngine(_SiteEngine):
    """Layered hexagonal materials, including primitive dichalcogenides."""

    def find_all(self) -> list[Site]:
        sites = super().find_all()
        top_indices = self.finder._top_indices()
        # The honeycomb primitive cell has a six-membered ring rather than a
        # triangular three-atom face, so it is not represented by the generic
        # triangle graph.  Its unique ring centre is fixed by lattice
        # geometry and does not depend on the two atom species.
        if len(top_indices) == 2 and not any(site.name == SiteType.HOLLOW.value for site in sites):
            first, second = self.finder.atoms.cell.array[:2]
            position = self.finder.atoms.positions[top_indices[0]] + (first + second) / 3.0
            position[2] = self.finder._surface_height()
            sites.append(
                self.finder._make_site(
                    SiteType.HOLLOW.value,
                    position,
                    tuple(top_indices),
                    surface_layer=0,
                    metadata={"kind": "hollow", "coordination": 6, "layer": 0},
                )
            )
        layers = self.finder._layer_indices()
        if len(layers) < 2:
            return sites
        bottom = layers[-1]
        # A monatomic sheet has identical top and bottom geometry.  Returning
        # it once preserves the long-standing one-face API behaviour.
        if set(bottom) == set(layers[0]):
            return sites
        sites.extend(self.finder._find_layer_sites(bottom, "Bottom", "Bottom Bridge", "Bottom Hollow"))
        return sites


class _FCC111Engine(_SiteEngine):
    """Close-packed (111) detector with local FCC/HCP stacking labels."""

    def find_hollow(self, cutoff: float | None) -> list[Site]:
        sites = self.finder._find_hollow_generic(cutoff)
        layers = self.finder._layer_indices()
        below = layers[1] if len(layers) > 1 else []
        if not below:
            return sites
        scaled = self.finder.atoms.get_scaled_positions(wrap=True)
        for site in sites:
            site_scaled = self.finder.atoms.cell.scaled_positions([site.position])[0]
            nearest = min(
                np.linalg.norm((site_scaled[:2] - scaled[index, :2]) - np.rint(site_scaled[:2] - scaled[index, :2]))
                for index in below
            )
            site.name = (SiteType.HCP if nearest < 0.12 else SiteType.FCC).value
            site.metadata = {**(site.metadata or {}), "kind": str(site.name).lower(), "stacking_distance": nearest}
        return sites


class _SquareSurfaceEngine(_SiteEngine):
    """Square-net detector shared by FCC(100), BCC, and ionic (001) slabs."""

    def find_fourfold(self) -> list[Site]:
        top_indices = self.finder._top_indices()
        if not top_indices:
            return []
        first, second = self.finder.atoms.cell.array[:2]
        z_surface = self.finder._surface_height()
        # A checkerboard ionic (001) layer has two surface sublattices in the
        # conventional square cell; its empty fourfold centres are quarter-
        # cell offsets rather than the alternate sublattice itself.
        fraction = 0.25 if self.finder.detect_surface_type() is SurfaceType.ROCKSALT001 else 0.5
        sites: list[Site] = []
        for index in top_indices:
            position = self.finder.atoms.positions[index] + fraction * (first + second)
            position[2] = z_surface
            sites.append(
                self.finder._make_site(
                    SiteType.FOURFOLD.value,
                    position,
                    (index,),
                    surface_layer=0,
                    metadata={"kind": "fourfold", "layer": 0},
                )
            )
        return self.finder.remove_duplicates(sites)

    def find_all(self) -> list[Site]:
        return [*super().find_all(), *self.find_fourfold()]


class _BCCEngine(_SquareSurfaceEngine):
    """BCC family detector, adding second-neighbour long bridges."""

    def find_long_bridge(self) -> list[Site]:
        first_neighbor = self.finder._estimate_neighbor_distance()
        graph = self.finder._get_graph(cutoff=first_neighbor * 1.9)
        z_surface = self.finder._surface_height()
        sites: list[Site] = []
        for left_image, right_image in graph._edge_records:
            distance = np.linalg.norm(graph._image_positions[left_image] - graph._image_positions[right_image])
            if distance <= first_neighbor * 1.1:
                continue
            midpoint = 0.5 * (graph._image_positions[left_image] + graph._image_positions[right_image])
            midpoint[2] = z_surface
            sites.append(
                self.finder._make_site(
                    SiteType.LONG_BRIDGE.value,
                    midpoint,
                    tuple(sorted({left_image[0], right_image[0]})),
                    surface_layer=0,
                    metadata={"kind": "long_bridge", "layer": 0},
                )
            )
        return self.finder.remove_duplicates(sites)

    def find_all(self) -> list[Site]:
        return [*super().find_all(), *self.find_long_bridge()]


class SiteFinder:
    """Detect adsorption sites on a surface."""

    def __init__(self, atoms: Atoms | Any) -> None:
        """Store the ASE atoms object used for site detection."""
        self.atoms = atoms.atoms if hasattr(atoms, "atoms") else atoms
        self._graph_cache: dict[tuple[float | None, str | None], SurfaceGraph] = {}
        self._surface_type: SurfaceType | None = None

    def _layer_indices(self, tolerance: float = 0.5) -> list[list[int]]:
        """Group atoms into horizontal layers without using element identity."""
        ordered = sorted(range(len(self.atoms)), key=lambda index: self.atoms.positions[index, 2], reverse=True)
        layers: list[list[int]] = []
        reference_heights: list[float] = []
        for index in ordered:
            height = float(self.atoms.positions[index, 2])
            if not layers or abs(height - reference_heights[-1]) > tolerance:
                layers.append([index])
                reference_heights.append(height)
            else:
                layers[-1].append(index)
        return layers

    def _in_plane_geometry(self) -> tuple[float, float, float]:
        """Return in-plane vector lengths and their included angle in degrees."""
        first, second = self.atoms.cell.array[:2]
        first_length = float(np.linalg.norm(first))
        second_length = float(np.linalg.norm(second))
        if first_length < 1e-8 or second_length < 1e-8:
            return 0.0, 0.0, 0.0
        cosine = np.clip(np.dot(first, second) / (first_length * second_length), -1.0, 1.0)
        return first_length, second_length, float(np.degrees(np.arccos(cosine)))

    def _layer_offset(self, upper: list[int], lower: list[int]) -> float:
        """Return the smallest periodic in-plane offset between two layers."""
        if not upper or not lower:
            return float("inf")
        scaled = self.atoms.get_scaled_positions(wrap=True)
        offsets: list[float] = []
        for upper_index in upper:
            for lower_index in lower:
                delta = scaled[upper_index, :2] - scaled[lower_index, :2]
                delta -= np.rint(delta)
                offsets.append(float(np.linalg.norm(delta)))
        return min(offsets)

    def detect_surface_type(self) -> SurfaceType:
        """Classify the exposed geometry using lattice and layer topology.

        Classification intentionally depends only on geometry.  Chemical
        symbols are not consulted, allowing the same code path for alloys and
        materials whose elements are not known in advance.
        """
        if self._surface_type is not None:
            return self._surface_type

        if len(self.atoms) == 0 or not all(self.atoms.pbc[:2]):
            self._surface_type = SurfaceType.UNKNOWN
            return self._surface_type

        first_length, second_length, angle = self._in_plane_geometry()
        equal_lengths = np.isclose(first_length, second_length, rtol=0.08)
        is_hexagonal = equal_lengths and (np.isclose(angle, 60.0, atol=4.0) or np.isclose(angle, 120.0, atol=4.0))
        is_square = equal_lengths and np.isclose(angle, 90.0, atol=4.0)
        layers = self._layer_indices()

        if is_hexagonal:
            # Close-packed (111) slabs have laterally shifted consecutive
            # layers, whereas layered 2D compounds repeat the same registry
            # above and below their central plane.
            if len(layers) >= 3 and self._layer_offset(layers[0], layers[1]) > 0.08:
                self._surface_type = SurfaceType.FCC111
            else:
                self._surface_type = SurfaceType.HEXAGONAL_2D
        elif is_square:
            top_count = len(layers[0]) if layers else 0
            second_count = len(layers[1]) if len(layers) > 1 else 0
            if top_count >= 3 and second_count >= 2:
                self._surface_type = SurfaceType.PEROVSKITE001
            elif top_count >= 2:
                self._surface_type = SurfaceType.ROCKSALT001
            elif len(layers) >= 3 and self._layer_offset(layers[0], layers[1]) > 0.08:
                self._surface_type = SurfaceType.FCC100
            else:
                self._surface_type = SurfaceType.BCC100
        elif np.isclose(angle, 90.0, atol=4.0) and max(first_length, second_length) / min(first_length, second_length) > 1.25:
            self._surface_type = SurfaceType.BCC110
        else:
            self._surface_type = SurfaceType.UNKNOWN
        return self._surface_type

    def _engine(self) -> _SiteEngine:
        """Return the geometry-specialized engine for this surface."""
        surface_type = self.detect_surface_type()
        if surface_type is SurfaceType.HEXAGONAL_2D:
            return _Hexagonal2DEngine(self)
        if surface_type is SurfaceType.FCC111:
            return _FCC111Engine(self)
        if surface_type in {SurfaceType.BCC110, SurfaceType.BCC100}:
            return _BCCEngine(self)
        if surface_type in {
            SurfaceType.FCC100,
            SurfaceType.ROCKSALT001,
            SurfaceType.PEROVSKITE001,
        }:
            return _SquareSurfaceEngine(self)
        return _SiteEngine(self)

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
        for axis, periodic in enumerate(self.atoms.pbc):
            if periodic:
                wrapped[axis] %= 1.0
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

    def _find_top_generic(self, element: str | None = None, tolerance: float | None = None) -> list[Site]:
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

    def _find_bridge_generic(self, cutoff: float | None = None) -> list[Site]:
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

    def _find_hollow_generic(self, cutoff: float | None = None) -> list[Site]:
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

    def _find_layer_sites(
        self,
        indices: list[int],
        top_name: str,
        bridge_name: str,
        hollow_name: str,
    ) -> list[Site]:
        """Generate periodic graph sites on an explicitly selected layer."""
        if not indices:
            return []
        graph = SurfaceGraph(self.atoms, indices, self._resolve_cutoff(None))
        height = float(np.mean(self.atoms.positions[indices, 2]))
        sites: list[Site] = []
        for index in indices:
            position = self.atoms.positions[index].copy()
            position[2] = height
            sites.append(self._make_site(top_name, position, (index,), surface_layer=-1, metadata={"kind": "bottom", "layer": -1}))
        for left_image, right_image in graph._edge_records:
            position = 0.5 * (graph._image_positions[left_image] + graph._image_positions[right_image])
            position[2] = height
            sites.append(self._make_site(bridge_name, position, tuple(sorted({left_image[0], right_image[0]})), surface_layer=-1, metadata={"kind": "bottom_bridge", "layer": -1}))
        for triangle in graph._triangle_records:
            position = np.mean([graph._image_positions[node] for node in triangle], axis=0)
            position[2] = height
            sites.append(self._make_site(hollow_name, position, tuple(sorted({node[0] for node in triangle})), surface_layer=-1, metadata={"kind": "bottom_hollow", "layer": -1}))
        return self.remove_duplicates(sites)

    # Public API: these methods dispatch but retain their historical
    # signatures and return types.
    def find_top(self, element: str | None = None, tolerance: float | None = None) -> list[Site]:
        """Find top sites using the engine selected from surface geometry."""
        return self._engine().find_top(element, tolerance)

    def find_bridge(self, cutoff: float | None = None) -> list[Site]:
        """Find top-face bridge sites using the selected surface engine."""
        return self._engine().find_bridge(cutoff)

    def find_hollow(self, cutoff: float | None = None) -> list[Site]:
        """Find top-face hollow sites using the selected surface engine."""
        return self._engine().find_hollow(cutoff)

    def remove_duplicates(self, sites: list[Site], tol: float = 1e-3) -> list[Site]:
        """Remove duplicate adsorption sites after wrapping positions into the cell."""
        unique: list[Site] = []
        for site in sites:
            duplicate = False
            for other in unique:
                if site.name == other.name and np.linalg.norm(site.position - other.position) < tol:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(site)
        return unique

    def find_all(self) -> list[Site]:
        """Return all detected site candidates sorted by site type."""
        sites = self._engine().find_all()
        unique_sites = self.remove_duplicates(sites)
        type_order = {
            "Top": 0,
            "Top_Se": 0,
            "Top_W": 0,
            "Bridge": 1,
            "Bottom": 1,
            "Bottom Bridge": 2,
            "Hollow": 3,
            "Bottom Hollow": 4,
            "FCC": 3,
            "HCP": 3,
            "Fourfold": 3,
            "Long Bridge": 2,
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
            position = site.visualization_position(height)
            xyz += Atoms(symbols=symbol, positions=[position])

        write(filename, xyz)
        print(f"\nSaved adsorption sites to: {filename}")
        print(f"Total atoms in file: {len(xyz)}")
