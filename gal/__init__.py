"""
GasAdsorptionLab (GAL)

Automated Gas Adsorption Toolkit
for Quantum ESPRESSO
"""

__version__ = "0.6.0"

from .config import Config
from .gas import Gas
from .workflow import AdsorptionWorkflow

__all__ = [
    "Surface",
    "SiteFinder",
    "Site",
    "SiteType",
    "SurfaceType",
    "Gas",
    "Config",
    "AdsorptionWorkflow",
    "Adsorbate",
    "place_adsorbate",
    "generate_adsorption_structures",
    "AdsorptionCampaign",
    "export_site_markers",
]


def __getattr__(name: str):
    """Lazily import package symbols to avoid import-time dependency issues."""

    if name == "Surface":
        from .surface import Surface
        return Surface

    if name in {"SiteFinder", "Site", "SiteType", "SurfaceType"}:
        from .sites import Site, SiteFinder, SiteType, SurfaceType
        return {
            "SiteFinder": SiteFinder,
            "Site": Site,
            "SiteType": SiteType,
            "SurfaceType": SurfaceType,
        }[name]

    if name == "Adsorbate":
        from .adsorbate import Adsorbate
        return Adsorbate

    if name in {"place_adsorbate", "generate_adsorption_structures"}:
        from .placement import generate_adsorption_structures, place_adsorbate
        return {
            "place_adsorbate": place_adsorbate,
            "generate_adsorption_structures": generate_adsorption_structures,
        }[name]

    if name == "AdsorptionCampaign":
        from .campaign import AdsorptionCampaign
        return AdsorptionCampaign

    if name == "export_site_markers":
        from .visualize import export_site_markers
        return export_site_markers

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
