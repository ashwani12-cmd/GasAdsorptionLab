from ase.io import read

from gal import Surface, SiteFinder
from gal.visualize import export_site_markers

# -------------------------------------------------
# Load WSe2
# -------------------------------------------------

atoms = read("examples/WSe2.cif")

surface = Surface(atoms)
finder = SiteFinder(surface)
sites = finder.find_all()

print(f"Detected {len(sites)} adsorption sites")

paths = export_site_markers(surface.atoms, sites, output_dir="ovito_sites", marker_element="Ne", marker_height=2.5)
for path, site in zip(paths, sites):
    print(f"{path.name}: {site.visualization_position(2.5)}")

print("\nSaved individual files and ovito_sites/all_sites.xyz")
