from pathlib import Path
import yaml


def load_config(config_file="config.yaml"):
    """Load YAML configuration."""

    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"{config_file} not found.")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)
