from gal import Adsorbate
from gal import SiteFinder

surface = Surface("MoS2.cif")

finder = SiteFinder(surface)

co = Adsorbate("CO")

structures = finder.place_adsorbate(
    co,
    height="auto",
    orientations="auto",
)
