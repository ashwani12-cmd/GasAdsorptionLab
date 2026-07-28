from pathlib import Path
import yaml


class Config:

    def __init__(self, filename="config.yaml"):

        with open(filename) as f:
            self.data = yaml.safe_load(f)

    @classmethod
    def from_dict(cls, data):
        """Create a config object from a plain dictionary."""
        instance = cls.__new__(cls)
        instance.data = data
        return instance

    def get(self, *keys, default=None):

        value = self.data

        for key in keys:

            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                if default is not None:
                    return default
                raise KeyError(" -> ".join(keys))

        return value
