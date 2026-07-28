"""Find adsorption sites on a compact Cu(111)-like slab."""

import numpy as np
from ase import Atoms

from gal import SiteFinder


if __name__ == "__main__":
    atoms = Atoms("Cu3", positions=[[0.0, 0.0, 2.0], [1.5, np.sqrt(3.0) / 2.0, 0.0], [1.0, np.sqrt(3.0) / 2.0, -2.0]], cell=[[3.0, 0.0, 0.0], [1.5, 3.0 * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 15.0]], pbc=(True, True, False))
    sites = SiteFinder(atoms).find_all()
    print(f"Cu(111) sites: {len(sites)}")
    print([site.name for site in sites])
