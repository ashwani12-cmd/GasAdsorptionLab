"""Geometry-based placement of adsorbates on detected adsorption sites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from .adsorbate import Adsorbate
from .sites import Site

if TYPE_CHECKING:
    from .surface import Surface


Orientation = Literal["auto", "c-down", "o-down", "n-down", "h-down", "o-down", "flat", "tilted", "parallel", "perpendicular"]


def _surface_atoms(surface: "Surface | Atoms") -> Atoms:
    return surface.atoms if hasattr(surface, "atoms") else surface


def _as_adsorbate(adsorbate: Adsorbate | str) -> Adsorbate:
    return Adsorbate(adsorbate) if isinstance(adsorbate, str) else adsorbate


def _rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps one non-zero vector onto another."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        # A 180 degree rotation around an axis perpendicular to source.
        axis = np.cross(source, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(source, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / np.dot(cross, cross))


def _point_atom_down(atoms: Atoms, symbol: str) -> Atoms:
    """Orient the first matching atom along negative z relative to the COM."""
    oriented = atoms.copy()
    center = oriented.get_center_of_mass()
    index = oriented.get_chemical_symbols().index(symbol)
    vector = oriented.positions[index] - center
    if np.linalg.norm(vector) > 1e-12:
        rotation = _rotation_matrix(vector, np.array([0.0, 0.0, -1.0]))
        oriented.positions = (oriented.positions - center) @ rotation.T + center
    return oriented


def generate_orientations(adsorbate: Adsorbate | str, orientations: str | tuple[str, ...] = "auto") -> dict[str, Atoms]:
    """Return named default molecular orientations as independent ASE objects."""
    molecule = _as_adsorbate(adsorbate).atoms
    if orientations != "auto":
        requested = (orientations,) if isinstance(orientations, str) else orientations
        defaults = generate_orientations(adsorbate, "auto")
        return {name: defaults[name] for name in requested}

    symbols = molecule.get_chemical_symbols()
    formula = _as_adsorbate(adsorbate).formula
    center = molecule.get_center_of_mass()
    horizontal = molecule.copy()
    if len(molecule) > 1:
        axis = molecule.positions[-1] - molecule.positions[0]
        if np.linalg.norm(axis) > 1e-12:
            rotation = _rotation_matrix(axis, np.array([1.0, 0.0, 0.0]))
            horizontal.positions = (horizontal.positions - center) @ rotation.T + center

    if formula == "CO":
        return {"c-down": _point_atom_down(molecule, "C"), "o-down": _point_atom_down(molecule, "O"), "parallel": horizontal}
    if formula == "NH3":
        tilted = _point_atom_down(molecule, "N")
        tilted.rotate(30.0, "x", center="COM")
        return {"n-down": _point_atom_down(molecule, "N"), "h-down": _point_atom_down(molecule, "H"), "tilted": tilted}
    if formula == "H2O":
        tilted = _point_atom_down(molecule, "O")
        tilted.rotate(30.0, "x", center="COM")
        return {"o-down": _point_atom_down(molecule, "O"), "flat": horizontal, "tilted": tilted}
    if len(molecule) == 2 or formula in {"CO2", "NO", "H2"}:
        return {"perpendicular": molecule.copy(), "parallel": horizontal}
    return {"upright": molecule.copy()}


def estimate_adsorption_height(surface: "Surface | Atoms", site: Site, adsorbate: Adsorbate | str, offset: float = 0.35) -> float:
    """Estimate contact height from covalent radii plus a small clearance."""
    substrate = _surface_atoms(surface)
    molecule = _as_adsorbate(adsorbate).atoms
    indices = site.neighbors or tuple(range(len(substrate)))
    surface_radius = max(covalent_radii[substrate.numbers[index]] for index in indices)
    adsorbate_radius = max(covalent_radii[number] for number in molecule.numbers)
    return float(surface_radius + adsorbate_radius + offset)


def place_adsorbate(
    surface: "Surface | Atoms",
    site: Site,
    adsorbate: Adsorbate | str,
    height: float | str = "auto",
    orientation: str | Atoms = "auto",
) -> Atoms:
    """Place one oriented adsorbate above ``site`` and return combined atoms."""
    substrate = _surface_atoms(surface).copy()
    adsorbate_model = _as_adsorbate(adsorbate)
    if isinstance(orientation, Atoms):
        molecule = orientation.copy()
        orientation_name = "custom"
    else:
        orientation_name = next(iter(generate_orientations(adsorbate_model, orientation))) if orientation == "auto" else orientation
        molecule = generate_orientations(adsorbate_model, orientation_name)[orientation_name]
    if height == "auto":
        height_value = estimate_adsorption_height(substrate, site, adsorbate_model)
    elif isinstance(height, (int, float)):
        height_value = float(height)
    else:
        raise ValueError("height must be a number or 'auto'")

    molecule.translate(-molecule.get_center_of_mass())
    molecule.translate([site.position[0], site.position[1], site.position[2] + height_value - np.min(molecule.positions[:, 2])])
    combined = substrate + molecule
    combined.info.update({"adsorption_site": str(site.name), "orientation": orientation_name, "adsorption_height": height_value})
    return combined


def generate_adsorption_structures(
    surface: "Surface | Atoms",
    adsorbate: Adsorbate | str,
    sites: list[Site] | None = None,
    height: float | str = "auto",
    orientations: str | tuple[str, ...] = "auto",
) -> list[Atoms]:
    """Generate structures for every requested orientation at every site."""
    substrate = _surface_atoms(surface)
    if sites is None:
        from .sites import SiteFinder

        sites = SiteFinder(substrate).find_all()
    orientation_map = generate_orientations(adsorbate, orientations)
    return [
        place_adsorbate(substrate, site, adsorbate, height=height, orientation=orientation)
        for site in sites
        for orientation in orientation_map
    ]
