# Quick start

```python
from gal import Adsorbate, SiteFinder, Surface, generate_adsorption_structures

surface = Surface("MoS2.cif")
sites = SiteFinder(surface).find_all()
structures = generate_adsorption_structures(surface, Adsorbate("CO"), sites=sites)
print(len(structures))
```

Each result is an ASE `Atoms` object containing the surface and one adsorbate.
