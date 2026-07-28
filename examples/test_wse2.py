from ase import Atoms
from gal import SiteFinder

atoms = Atoms(
    "WSe2",
    positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 0.0, -1.6]],
    cell=[[3.28, 0.0, 0.0], [1.64, 2.8405, 0.0], [0.0, 0.0, 20.0]],
    pbc=(True, True, False),
)

finder = SiteFinder(atoms)

sites = finder.find_all()

print(f"Total sites: {len(sites)}")

for site in sites:
    print(site.name, site.position)
