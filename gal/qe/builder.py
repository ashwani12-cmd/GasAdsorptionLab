"""High-level builder for QE pw.x calculations."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from .input import QEInput
from .pseudopotentials import pseudopotential_map


class QEInputBuilder:
    """Build QE input decks from atoms and concise calculation settings."""

    def __init__(
        self,
        pseudo_dir: str = "./pseudo",
        ecutwfc: float = 60,
        ecutrho: float = 480,
        kpts: tuple[int, int, int] = (6, 6, 1),
        xc: str = "PBE",
        pseudopotentials: dict[str, str] | None = None,
        occupations: str = "smearing",
        conv_thr: float = 1.0e-8,
        smearing: str = "mv",
        degauss: float = 0.01,
        vdw_corr: str | None = "grimme-d3",
        nspin: int = 1,
    ) -> None:
        self.pseudo_dir = pseudo_dir
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self.kpts = tuple(kpts)
        self.xc = xc
        self.pseudopotentials = pseudopotentials or {}
        self.occupations = occupations
        self.conv_thr = conv_thr
        self.smearing = smearing
        self.degauss = degauss
        self.vdw_corr = vdw_corr
        self.nspin = nspin

    def configuration(self, atoms: Atoms, prefix: str, calculation: str = "scf") -> dict[str, Any]:
        """Return the canonical configuration dictionary for an input deck."""
        qe: dict[str, Any] = {
            "prefix": prefix,
            "calculation": calculation,
            "pseudo_dir": self.pseudo_dir,
            "ecutwfc": self.ecutwfc,
            "ecutrho": self.ecutrho,
            "occupations": self.occupations,
            "conv_thr": self.conv_thr,
            "smearing": self.smearing,
            "degauss": self.degauss,
            "nspin": self.nspin,
        }
        if self.vdw_corr is not None:
            qe["vdw_corr"] = self.vdw_corr
        return {
            "qe": qe,
            "kpoints": {"scf": list(self.kpts)},
            "pseudopotentials": pseudopotential_map(atoms.get_chemical_symbols(), self.pseudopotentials),
            "xc": self.xc,
        }

    def build(self, atoms: Atoms, prefix: str = "adsorption", calculation: str = "scf") -> QEInput:
        """Build a serializable pw.x input object."""
        return QEInput(self.configuration(atoms, prefix, calculation), atoms=atoms)
