"""Utilities for placing gas molecules above adsorption sites."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from .gas import Gas
from .sites import Site
from .surface import Surface


def rotate_molecule(atoms: Atoms, axis: str = "z", angle: float = 90.0) -> Atoms:
    """Return a rotated copy of an ASE atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Molecule to rotate.
    axis : str, optional
        Rotation axis, one of ``"x"``, ``"y"``, or ``"z"``.
    angle : float, optional
        Rotation angle in degrees.

    Returns
    -------
    ase.Atoms
        Rotated copy of the molecule.
    """
    rotated = atoms.copy()
    theta = np.deg2rad(angle)

    if axis.lower() == "x":
        rotation_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
    elif axis.lower() == "y":
        rotation_matrix = np.array(
            [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]],
            dtype=float,
        )
    elif axis.lower() == "z":
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
    else:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")

    positions = rotated.get_positions()
    centered = positions - np.mean(positions, axis=0)
    rotated.set_positions(centered @ rotation_matrix.T + np.mean(positions, axis=0))
    return rotated


def generate_orientations(
    atoms: Atoms,
    axes: tuple[str, ...] = ("x", "y", "z"),
    angles: tuple[float, ...] = (0.0, 90.0, 180.0),
    n_per_axis: int | None = None,
) -> list[Atoms]:
    """Generate multiple rotated copies of a molecule.

    Parameters
    ----------
    atoms : ase.Atoms
        Molecule to rotate.
    axes : tuple[str, ...], optional
        Axes to sample for rotations.
    angles : tuple[float, ...], optional
        Rotation angles in degrees for each axis.
    n_per_axis : int | None, optional
        Optional explicit count of orientations to generate per axis.
        If provided, evenly spaced angles are derived from ``angles``.

    Returns
    -------
    list[ase.Atoms]
        Generated molecule orientations.
    """
    if n_per_axis is not None:
        angle_values = np.linspace(angles[0], angles[-1], n_per_axis)
    else:
        angle_values = list(angles)

    orientations: list[Atoms] = []
    for axis in axes:
        for angle in angle_values:
            orientations.append(rotate_molecule(atoms, axis=axis, angle=float(angle)))

    return orientations


def place_adsorbate(
    surface: Surface,
    gas: Gas,
    site: Site,
    adsorption_height: float = 2.0,
    rotation: float | None = None,
) -> Atoms:
    """Place a gas molecule above a site and return a new ASE atoms object.

    Parameters
    ----------
    surface : Surface
        Surface object containing the substrate atoms.
    gas : Gas
        Gas object whose molecule should be placed.
    site : Site
        Adsorption site defining the placement position.
    adsorption_height : float, optional
        Distance in Å above the site position where the molecule is centered.
    rotation : float | None, optional
        Rotation angle in degrees applied around the z-axis before placement.

    Returns
    -------
    ase.Atoms
        A new ASE atoms object containing the surface and the adsorbate.
    """
    substrate = surface.atoms.copy()
    molecule = gas.atoms.copy()

    if molecule is None:
        raise ValueError("Gas atoms object is empty")

    if rotation is not None:
        molecule = molecule.copy()
        theta = np.deg2rad(rotation)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        positions = molecule.get_positions()
        centered = positions - np.mean(positions, axis=0)
        rotated = centered @ rotation_matrix.T
        molecule.set_positions(rotated + np.mean(positions, axis=0))

    # Center the molecule on its geometric center and place it above the site.
    positions = molecule.get_positions()
    center = np.mean(positions, axis=0)
    molecule.translate(-center)

    molecule_positions = molecule.get_positions()
    max_z = np.max(molecule_positions[:, 2])
    molecule.translate(np.array([0.0, 0.0, adsorption_height - max_z]))

    position = np.array(site.position, dtype=float).copy()
    molecule.translate(position)

    combined = substrate + molecule
    return combined
