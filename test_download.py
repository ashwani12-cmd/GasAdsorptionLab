from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from utils import load_config

config = load_config()

api_key = config["materials_project"]["api_key"]
mpid = config["material"]["mpid"]

print(f"Downloading {mpid}...")

with MPRester(api_key) as mpr:
    structure = mpr.get_structure_by_material_id(mpid)

atoms = AseAtomsAdaptor.get_atoms(structure)

print(atoms)
