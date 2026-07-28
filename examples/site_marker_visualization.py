"""Export elevated adsorption-site markers for OVITO, VESTA, or ASE viewers."""

from ase import Atoms

from gal import SiteFinder
from gal.visualize import export_site_markers


if __name__ == "__main__":
    surface = Atoms(
        "WSe2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 0.0, -1.6]],
        cell=[[3.28, 0.0, 0.0], [1.64, 2.8405, 0.0], [0.0, 0.0, 20.0]],
        pbc=(True, True, False),
    )
    sites = SiteFinder(surface).find_all()
    paths = export_site_markers(surface, sites, output_dir="ovito_sites", marker_element="Ne", marker_height=2.5)
    print(f"Exported {len(paths)} individual marker files")
    print("Trajectory: ovito_sites/all_sites.xyz")
