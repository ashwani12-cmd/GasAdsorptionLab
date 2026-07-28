import pytest

pytest.importorskip("mp_api")
pytest.importorskip("pymatgen")

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from utils import load_config


def test_download_structure_is_optional():
    config = load_config()
    api_key = config["materials_project"]["api_key"]
    mpid = config["material"]["mpid"]

    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(mpid)

    atoms = AseAtomsAdaptor.get_atoms(structure)
    assert atoms is not None
