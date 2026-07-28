import pytest

from gal import Gas


@pytest.fixture
def gas() -> Gas:
    return Gas("NO2")


def test_gas_initializes_with_supported_formula(gas: Gas):
    assert gas.formula == "NO2"
    assert gas.atoms is not None
    assert len(gas.atoms) > 0
