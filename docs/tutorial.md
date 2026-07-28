# Adsorption tutorial

`SiteFinder` classifies periodic surface geometry and dispatches to the
appropriate detector. Its public methods are `find_top`, `find_bridge`,
`find_hollow`, and `find_all`.

```python
from gal import Adsorbate, SiteFinder
from gal.placement import place_adsorbate

finder = SiteFinder(surface)
site = finder.find_all()[0]
structure = place_adsorbate(surface, site, Adsorbate("NH3"), height="auto", orientation="n-down")
```

The automatic height combines covalent radii and a small clearance. Explicit
numeric heights remain available for controlled scans.
