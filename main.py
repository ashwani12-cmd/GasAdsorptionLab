from gal.config import Config
from gal.surface import Surface

config = Config()

api_key = config.get("materials_project", "api_key")

source = config.get("material", "source")

if source == "mp":
    mpid = config.get("material", "mp", "id")
    surface = Surface.from_mp(mpid, api_key)

elif source == "cif":
    cif = config.get("material", "cif", "file")
    surface = Surface.from_cif(cif)

else:
    raise ValueError(f"Unsupported material source: {source}")

supercell = config.get("surface", "supercell")
vacuum = config.get("surface", "vacuum")

surface.supercell(*supercell)

surface.add_vacuum(vacuum)

surface.info()

surface.write("output/WSe2_4x4.cif")
