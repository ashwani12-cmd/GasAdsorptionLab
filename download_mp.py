"""
===============================================================
GasAdsorptionLab
download_mp.py

Author : Ashwani Kushwaha & ChatGPT
Description
-----------
Download crystal structures from the Materials Project,
convert them to ASE Atoms objects, generate supercells,
add vacuum, and save CIF files.

Dependencies
------------
mp-api
ase
numpy
logging

===============================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from ase import Atoms
from ase.build import make_supercell
from ase.io import read, write

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from mp_api.client import MPRester


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)


class MPDownloader:
    """
    Download structures from Materials Project.
    """

    def __init__(self, api_key: str):

        self.api_key = api_key

    # ---------------------------------------------------------
    def download(
        self,
        mpid: str,
    ) -> Atoms:
        """
        Download a material using Materials Project ID.

        Parameters
        ----------
        mpid : str

            Example
            -------
            mp-1821

        Returns
        -------
        ASE Atoms
        """

        logger.info(f"Downloading {mpid}")

        with MPRester(self.api_key) as mpr:

            structure: Structure = mpr.get_structure_by_material_id(mpid)

        atoms = AseAtomsAdaptor.get_atoms(structure)

        logger.info("Download complete.")

        return atoms

    # ---------------------------------------------------------

    def save_cif(
        self,
        atoms: Atoms,
        filename: str,
    ):

        logger.info(f"Saving {filename}")

        write(filename, atoms)

    # ---------------------------------------------------------

    def build_supercell(
        self,
        atoms: Atoms,
        repeats: Tuple[int, int, int],
    ) -> Atoms:

        logger.info(
            f"Building supercell {repeats}"
        )

        P = [
            [repeats[0], 0, 0],
            [0, repeats[1], 0],
            [0, 0, repeats[2]],
        ]

        return make_supercell(atoms, P)

    # ---------------------------------------------------------

    def add_vacuum(
        self,
        atoms: Atoms,
        vacuum: float = 20.0,
        axis: int = 2,
    ) -> Atoms:

        logger.info(
            f"Adding {vacuum:.1f} Å vacuum"
        )

        cell = atoms.get_cell()

        cell[axis, axis] += vacuum

        atoms.set_cell(cell)

        atoms.center(axis=axis)

        return atoms

    # ---------------------------------------------------------

    def save_xyz(
        self,
        atoms: Atoms,
        filename: str,
    ):

        write(filename, atoms)

    # ---------------------------------------------------------

    def save_poscar(
        self,
        atoms: Atoms,
        filename: str = "POSCAR",
    ):

        write(filename, atoms, format="vasp")

    # ---------------------------------------------------------

    def load_local_cif(
        self,
        filename: str,
    ) -> Atoms:

        logger.info(f"Reading {filename}")

        return read(filename)

    # ---------------------------------------------------------

    def info(
        self,
        atoms: Atoms,
    ):

        print()

        print("=" * 60)

        print("Chemical Formula")

        print(atoms.get_chemical_formula())

        print()

        print("Number of atoms")

        print(len(atoms))

        print()

        print("Cell")

        print(atoms.cell)

        print()

        print("=" * 60)

