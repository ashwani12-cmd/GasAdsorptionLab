from __future__ import annotations

from ase import Atoms
from ase.build import make_supercell
from ase.io import read, write


class Surface:

    def __init__(self, atoms: Atoms):

        self.atoms = atoms

    # --------------------------------------------------

    @classmethod
    def from_mp(cls, mpid, api_key):

        try:
            from mp_api.client import MPRester
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as exc:  # pragma: no cover - exercised in minimal envs
            raise ImportError(
                "Materials Project support requires optional dependencies: mp_api and pymatgen"
            ) from exc

        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(mpid)

        atoms = AseAtomsAdaptor.get_atoms(structure)

        return cls(atoms)

    # --------------------------------------------------

    @classmethod
    def from_cif(cls, filename):

        atoms = read(filename)

        return cls(atoms)

    # --------------------------------------------------

    def supercell(self, a=1, b=1, c=1):

        P = [
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c]
        ]

        self.atoms = make_supercell(self.atoms, P)

        return self

    # --------------------------------------------------

    def add_vacuum(self, vacuum=20):

        cell = self.atoms.get_cell()

        cell[2, 2] += vacuum

        self.atoms.set_cell(cell)

        self.atoms.center(axis=2)

        return self

    # --------------------------------------------------

    def write(self, filename):

        write(filename, self.atoms)

    # --------------------------------------------------

    def info(self):

        print("=" * 60)
        print("Formula :", self.atoms.get_chemical_formula())
        print("Atoms   :", len(self.atoms))
        print("Cell")
        print(self.atoms.cell)
        print("=" * 60)
