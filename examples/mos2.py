"""Find adsorption sites on a primitive MoS2-like slab."""

from ase import Atoms

from gal import SiteFinder


if __name__ == "__main__":
    atoms = Atoms("MoS2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 0.0, -1.6]], cell=[[3.18, 0.0, 0.0], [1.59, 2.753, 0.0], [0.0, 0.0, 20.0]], pbc=(True, True, False))
    sites = SiteFinder(atoms).find_all()
    print(f"MoS2 sites: {len(sites)}")
    print([site.name for site in sites])
