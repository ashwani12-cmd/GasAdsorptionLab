from pathlib import Path

from ase.io import read, write

from gal import Surface, SiteFinder, Adsorbate, place_adsorbate

atoms = read("examples/WSe2.cif")

surface = Surface(atoms)

finder = SiteFinder(surface)

sites = finder.find_all()

print(f"Detected sites: {len(sites)}")

co = Adsorbate("CO")

outdir = Path("wse2_ovito")
outdir.mkdir(exist_ok=True)

for i, site in enumerate(sites):

    structure = place_adsorbate(
        surface=surface,
        site=site,
        adsorbate=co,
        height="auto",
        orientation="auto",
    )

    print(f"{i:02d}  {site.name:12s}  {site.position}")

    write(
        outdir / f"{i:02d}_{site.name.replace(' ', '_')}.xyz",
        structure,
    )

print(f"\nGenerated {len(sites)} structures in {outdir}")
