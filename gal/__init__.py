"""
GasAdsorptionLab (GAL)

Automated Gas Adsorption Toolkit
for Quantum ESPRESSO
"""

__version__ = "0.1.0"

from .config import Config
from .gas import Gas
from .sites import SiteType, SurfaceType
from .workflow import AdsorptionWorkflow

__all__ = [
    "Surface",
    "Gas",
    "Config",
    "AdsorptionWorkflow",
    "SiteType",
    "SurfaceType",
]


def __getattr__(name: str):
    """Lazily import package symbols to avoid import-time dependency issues."""

    if name == "Surface":
        from .surface import Surface
        return Surface

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
