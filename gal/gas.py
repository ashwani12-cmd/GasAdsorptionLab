from pathlib import Path

from ase.io import read


class Gas:
    """Legacy XYZ-backed gas molecule loader."""

    def __init__(self, formula):

        path = (
            Path(__file__).parent
            / "data"
            / "gases"
            / f"{formula}.xyz"
        )

        self.atoms = read(path)

        self.formula = formula
