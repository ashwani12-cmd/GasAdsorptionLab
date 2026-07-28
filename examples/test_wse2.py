from ase.io import read
from gal.sites import SiteFinder

atoms = read("/home/ashwani/Downloads/WSe2.cif")

finder = SiteFinder(atoms)

sites = finder.find_all()

print(f"Total sites: {len(sites)}")

for site in sites:
    print(site.name, site.position)
