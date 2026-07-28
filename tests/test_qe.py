from pathlib import Path

from gal.config import Config
from gal.qe import QEInput


def test_qe_input_writes_expected_sections(tmp_path):
    config = Config("config.yaml")
    qe = QEInput.from_config(config, prefix="test")

    output_file = tmp_path / "test.in"
    qe.write(output_file)

    text = output_file.read_text()

    assert "&CONTROL" in text
    assert "&SYSTEM" in text
    assert "&ELECTRONS" in text
    assert "ATOMIC_SPECIES" in text
    assert "ATOMIC_POSITIONS" in text
    assert "K_POINTS" in text
    assert "prefix = 'test'" in text
