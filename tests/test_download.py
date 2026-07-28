import os

import pytest

if os.environ.get("GAL_RUN_NETWORK_TESTS") != "1":
    pytest.skip("Materials Project download test is opt-in; set GAL_RUN_NETWORK_TESTS=1", allow_module_level=True)

pytest.importorskip("mp_api")
pytest.importorskip("pymatgen")

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from utils import load_config


def test_download_structure_is_optional():
    config = load_config()
    api_key = os.environ.get("MP_API_KEY") or config.get("materials_project", {}).get("api_key")
    material = config.get("material", {})
    mpid = material.get("mpid") or material.get("mp", {}).get("id")
    if not api_key or not mpid:
        pytest.skip("Materials Project credentials or material id are not configured")

    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(mpid)

    atoms = AseAtomsAdaptor.get_atoms(structure)
    assert atoms is not None
