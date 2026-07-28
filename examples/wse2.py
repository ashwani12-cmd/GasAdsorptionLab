"""Find adsorption sites on a primitive WSe2-like slab."""

from ase import Atoms

from gal import SiteFinder


if __name__ == "__main__":
    atoms = Atoms("WSe2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 0.0, -1.6]], cell=[[3.28, 0.0, 0.0], [1.64, 2.8405, 0.0], [0.0, 0.0, 20.0]], pbc=(True, True, False))
    sites = SiteFinder(atoms).find_all()
    print(f"WSe2 sites: {len(sites)}")
    print([site.name for site in sites])
