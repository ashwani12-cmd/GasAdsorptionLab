from gal import Config
from gal import Surface
from gal.sites import SiteFinder

cfg = Config()

surface = Surface.from_mp(
    cfg.get("material", "mp", "id"),
    cfg.get("materials_project", "api_key"),
)

surface.supercell(4, 4, 1)
surface.add_vacuum(20)

finder = SiteFinder(surface.atoms)

top_se = finder.find_top_se()
top_w = finder.find_top_w()
bridge = finder.find_bridge()

print("=" * 60)
print("Adsorption Site Detection")
print("=" * 60)

print(f"Top Se Sites : {len(top_se)}")
print(f"Top W  Sites : {len(top_w)}")
print(f"Bridge Sites : {len(bridge)}")

print("\nFirst Top Se")
print(top_se[0])

print("\nFirst Top W")
print(top_w[0])

print("\nFirst Bridge")
print(bridge[0])

finder.write_xyz("adsorption_sites.xyz")
