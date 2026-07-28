"""Symmetry utilities for adsorption-site reduction.

This module provides helpers for removing symmetry-equivalent adsorption
sites using :mod:`spglib`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

try:
    import spglib  # type: ignore
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    spglib = None

from .sites import Site


class SymmetryReducer:
    """Remove symmetry-equivalent adsorption sites from a structure.

    The reducer works by identifying the crystal symmetry of the underlying
    surface and mapping each candidate site into the symmetry-equivalent
    space using the crystal's space-group operations.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Store the ASE atoms object used for symmetry reduction."""
        self.atoms = atoms

    def _cell_and_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the cell and scaled positions for the structure."""
        cell = self.atoms.get_cell()
        scaled_positions = self.atoms.get_scaled_positions()
        return cell, scaled_positions

    def _space_group(self) -> int:
        """Return the space-group number for the current structure."""
        if spglib is None:
            return 1

        cell, scaled_positions = self._cell_and_positions()
        return spglib.get_space_group(cell, scaled_positions)[0]

    def _symmetry_operations(self) -> list[np.ndarray]:
        """Return symmetry operations for the current structure."""
        if spglib is None:
            return [np.eye(3, dtype=float)]

        cell, scaled_positions = self._cell_and_positions()
        return spglib.get_symmetry(cell, scaled_positions, symprec=1e-5)["rotations"]

    def _rotate_position(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Apply a symmetry rotation to a Cartesian or fractional position."""
        return np.dot(rotation, position)

    def _apply_symmetry(self, site: Site) -> list[np.ndarray]:
        """Return symmetry-equivalent positions for a site."""
        site_scaled = self.atoms.cell.scaled_positions(site.position)

        equivalent_positions: list[np.ndarray] = []
        for rotation in self._symmetry_operations():
            transformed = np.dot(rotation, site_scaled)
            transformed = transformed % 1.0
            equivalent_positions.append(self.atoms.cell.cartesian_positions(transformed))

        # Add lattice translations for the fallback case.
        if len(equivalent_positions) == 1:
            for translation in [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 1.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
                np.array([-1.0, -1.0, 0.0]),
            ]:
                equivalent_positions.append(
                    self.atoms.cell.cartesian_positions((site_scaled + translation) % 1.0)
                )

        return equivalent_positions

    def reduce_sites(self, sites: list[Site]) -> list[Site]:
        """Return one representative site for each symmetry-equivalent class.

        Parameters
        ----------
        sites : list[Site]
            Candidate adsorption sites to reduce.

        Returns
        -------
        list[Site]
            Unique symmetry-reduced sites.
        """
        if not sites:
            return []

        reduced_sites: list[Site] = []
        seen: list[np.ndarray] = []

        for site in sites:
            site_positions = self._apply_symmetry(site)
            equivalent = False

            for reduced_site in reduced_sites:
                reduced_scaled = self.atoms.cell.scaled_positions(reduced_site.position)
                for candidate_position in site_positions:
                    candidate_scaled = self.atoms.cell.scaled_positions(candidate_position)
                    delta = candidate_scaled - reduced_scaled
                    delta = np.mod(delta + 0.5, 1.0) - 0.5
                    if np.linalg.norm(delta) < 1e-3:
                        equivalent = True
                        break
                if equivalent:
                    break

            if equivalent:
                continue

            reduced_sites.append(site)
            seen.append(site.position)

        return reduced_sites
