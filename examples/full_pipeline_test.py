from pathlib import Path
import numpy as np

from ase.io import read, write

from gal import (
    Surface,
    SiteFinder,
    Adsorbate,
    place_adsorbate,
    export_site_markers,
)

# =====================================================
# Configuration
# =====================================================

SURFACE_FILE = "examples/WSe2.cif"

ADSORBATES = [
    "CO",
    "NH3",
    "H2O",
]

OUTPUT = Path("pipeline_test")
OUTPUT.mkdir(exist_ok=True)

# =====================================================
# Load surface
# =====================================================

atoms = read(SURFACE_FILE)

print("=" * 60)
print("Surface loaded")
print("=" * 60)

print("Atoms :", len(atoms))
print("Cell")
print(atoms.cell)

# =====================================================
# Detect sites
# =====================================================

surface = Surface(atoms)

finder = SiteFinder(surface)

sites = finder.find_all()

print("\nDetected sites:", len(sites))

for i, site in enumerate(sites):
    print(
        f"{i:02d} "
        f"{site.name:8s} "
        f"{site.position}"
    )

# =====================================================
# Export marker files
# =====================================================

print("\nExporting marker files...")

export_site_markers(
    atoms,
    sites,
    output_dir=OUTPUT / "markers",
    marker_element="Ne",
    marker_height=2.5,
)

# =====================================================
# Place adsorbates
# =====================================================

ads_dir = OUTPUT / "adsorbates"
ads_dir.mkdir(exist_ok=True)

count = 0

for gas_name in ADSORBATES:

    gas = Adsorbate(gas_name)

    gas_dir = ads_dir / gas_name
    gas_dir.mkdir(exist_ok=True)

    print(f"\nTesting {gas_name}")

    for i, site in enumerate(sites):

        structure = place_adsorbate(
            surface=surface,
            site=site,
            adsorbate=gas,
            height="auto",
            orientation="auto",
        )

        outfile = gas_dir / f"{i:02d}_{site.name}.xyz"

        write(outfile, structure)

        count += 1

print("\nGenerated structures:", count)

# =====================================================
# Simple geometry sanity check
# =====================================================

print("\nChecking minimum interatomic distances")

for gas_name in ADSORBATES:

    gas_dir = ads_dir / gas_name

    for xyz in sorted(gas_dir.glob("*.xyz")):

        atoms = read(xyz)

        dmin = 1e9

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                d = atoms.get_distance(i, j, mic=True)
                dmin = min(dmin, d)

        status = "OK"

        if dmin < 0.6:
            status = "WARNING"

        print(
            f"{xyz.name:25s} "
            f"min distance = {dmin:6.3f} Å   {status}"
        )

print("\n" + "=" * 60)
print("Pipeline test completed successfully!")
print("=" * 60)
