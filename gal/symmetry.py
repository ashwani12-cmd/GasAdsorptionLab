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

if spglib is not None:
    # spglib 2.x asks callers to opt into exception-based error handling.
    try:  # pragma: no branch - module layout differs across spglib releases
        import spglib.error as _spglib_error
        import spglib.spg as _spglib_api

        _spglib_error.OLD_ERROR_HANDLING = False
        _spglib_api.OLD_ERROR_HANDLING = False
    except ImportError:  # pragma: no cover - old spglib releases
        pass

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

    def _spglib_cell(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the official spglib cell tuple for current API versions."""
        cell, scaled_positions = self._cell_and_positions()
        return cell, scaled_positions, self.atoms.numbers

    def _space_group(self) -> int:
        """Return the space-group number for the current structure."""
        if spglib is None:
            return 1

        cell = self._spglib_cell()
        # spglib 2.x returns a symbol string from get_spacegroup; this helper
        # is retained for compatibility and is not used in site reduction.
        try:
            result = spglib.get_spacegroup(cell)
        except AttributeError:  # pragma: no cover - old spglib releases
            result = spglib.get_space_group(*cell)
        return int(result.split("(")[-1].rstrip(")")) if result else 1

    def _symmetry_operations(self) -> list[np.ndarray]:
        """Return symmetry operations for the current structure."""
        if spglib is None:
            return [np.eye(3, dtype=float)]

        cell = self._spglib_cell()
        try:
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
            rotations = dataset.rotations if dataset is not None else None
        except AttributeError:  # pragma: no cover - compatibility with pre-2.x API
            symmetry = spglib.get_symmetry(*cell, symprec=1e-5)
            rotations = symmetry["rotations"] if symmetry is not None else None
        if rotations is None:
            return [np.eye(3, dtype=float)]
        return rotations

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
