"""Generate CO and NH3 adsorption structures for a primitive WSe2 slab."""

from __future__ import annotations

from ase import Atoms

from gal import Adsorbate, SiteFinder, Surface, generate_adsorption_structures


def load_wse2() -> Surface:
    """Return a compact WSe2 slab suitable for a self-contained example."""
    return Surface(
        Atoms(
            "WSe2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 0.0, -1.6]],
            cell=[[3.28, 0.0, 0.0], [1.64, 2.8405, 0.0], [0.0, 0.0, 20.0]],
            pbc=(True, True, False),
        )
    )


if __name__ == "__main__":
    surface = load_wse2()
    finder = SiteFinder(surface)
    sites = finder.find_all()

    co_structures = generate_adsorption_structures(surface, Adsorbate("CO"), sites=sites)
    nh3_structures = generate_adsorption_structures(surface, Adsorbate("NH3"), sites=sites)

    print(f"Detected sites: {len(sites)}")
    print(f"CO structures: {len(co_structures)}")
    print(f"NH3 structures: {len(nh3_structures)}")
