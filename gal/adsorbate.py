"""Adsorbate models and a small library of starting molecular geometries."""

from __future__ import annotations

import numpy as np
from ase import Atoms


_GEOMETRIES: dict[str, tuple[str, list[list[float]]]] = {
    "H": ("H", [[0.0, 0.0, 0.0]]),
    "O": ("O", [[0.0, 0.0, 0.0]]),
    "N": ("N", [[0.0, 0.0, 0.0]]),
    "C": ("C", [[0.0, 0.0, 0.0]]),
    "CO": ("CO", [[0.0, 0.0, -0.564], [0.0, 0.0, 0.564]]),
    "CO2": ("OCO", [[0.0, 0.0, -1.160], [0.0, 0.0, 0.0], [0.0, 0.0, 1.160]]),
    "NH3": ("NHHH", [[0.0, 0.0, 0.10], [0.937, 0.0, -0.28], [-0.469, 0.812, -0.28], [-0.469, -0.812, -0.28]]),
    "NO": ("NO", [[0.0, 0.0, -0.575], [0.0, 0.0, 0.575]]),
    "NO2": ("NOO", [[0.0, 0.0, 0.0], [1.197, 0.0, 0.0], [-0.480, 1.097, 0.0]]),
    "H2": ("HH", [[0.0, 0.0, -0.371], [0.0, 0.0, 0.371]]),
    "H2O": ("OHH", [[0.0, 0.0, 0.0], [0.758, 0.0, 0.586], [-0.758, 0.0, 0.586]]),
    "CH4": ("CHHHH", [[0.0, 0.0, 0.0], [0.629, 0.629, 0.629], [0.629, -0.629, -0.629], [-0.629, 0.629, -0.629], [-0.629, -0.629, 0.629]]),
    "SO2": ("SOO", [[0.0, 0.0, 0.0], [1.430, 0.0, 0.0], [-0.682, 1.257, 0.0]]),
    "H2S": ("SHH", [[0.0, 0.0, 0.0], [0.961, 0.0, 0.755], [-0.961, 0.0, 0.755]]),
}


class Adsorbate:
    """A mutable ASE-backed adsorbate with reasonable initial geometry.

    The bundled geometries are starting structures based on standard gas-phase
    bond lengths; they are intentionally not geometry optimized.
    """

    def __init__(self, formula: str) -> None:
        try:
            symbols, positions = _GEOMETRIES[formula]
        except KeyError as exc:
            supported = ", ".join(_GEOMETRIES)
            raise ValueError(f"Unsupported adsorbate {formula!r}. Supported: {supported}") from exc
        self.formula = formula
        self.atoms = Atoms(symbols=symbols, positions=positions)

    @classmethod
    def from_atoms(cls, formula: str, atoms: Atoms) -> "Adsorbate":
        """Create an adsorbate from an existing ASE geometry.

        This supports legacy :class:`gal.gas.Gas` molecules that are not part
        of the built-in placement library while keeping the canonical model.
        """
        adsorbate = cls.__new__(cls)
        adsorbate.formula = formula
        adsorbate.atoms = atoms.copy()
        return adsorbate

    @property
    def center_of_mass(self) -> np.ndarray:
        """Return the current center of mass in Å."""
        return self.atoms.get_center_of_mass()

    def rotate(self, angle: float, axis: str | np.ndarray = "z") -> "Adsorbate":
        """Rotate in place around the center of mass and return ``self``."""
        self.atoms.rotate(angle, axis, center="COM")
        return self

    def translate(self, vector: np.ndarray | list[float]) -> "Adsorbate":
        """Translate in place and return ``self``."""
        self.atoms.translate(vector)
        return self

    def copy(self) -> "Adsorbate":
        """Return an independent copy of this adsorbate."""
        copied = self.__class__.__new__(self.__class__)
        copied.formula = self.formula
        copied.atoms = self.atoms.copy()
        return copied
